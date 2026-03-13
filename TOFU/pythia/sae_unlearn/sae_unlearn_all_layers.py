"""
COMPREHENSIVE SAE-GUIDED UNLEARNING — Pythia-6.9b
==================================================
Tests ALL combinations of dict sizes × layers.

Model  : EleutherAI/pythia-6.9b
Layers : 32 transformer layers (indices 0–31)
d_model: 4096

Dict sizes : [4096, 8192, 16384, 32768, 65536]
Layers     : [0 … 31]
Total      : 5 × 32 = 160 configurations

Results saved to: ../results/comprehensive_pythia_6.9b/dict_{size}/layer_{idx}/

Fixes over GPT-2 original:
  - Critical: device was always 'cpu' even when CUDA available
  - Critical: hook used model.transformer.h — Pythia uses model.gpt_neox.layers
  - Critical: model.config.n_embd → model.config.hidden_size (Pythia attr name)
  - FP16 inference to fit Pythia-6.9b on A100 80GB
  - CUDA cache cleared after every config
  - Hidden states index corrected for Pythia
  - Resume support: skips configs whose CSV already exists
  - Batch size reduced to 4 (safe for 6.9B + SAE on 80GB)

Author: Amartya Bhattacharya
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import copy
import random
import time
import traceback
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

from get_dataset import get_tofudataset, tokenize_function
from get_model import get_gptmodel

# ── Reproducibility ────────────────────────────────────────────────────────────
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

class Config:
    """Pythia-6.9b experiment configuration"""

    # ── Model ──────────────────────────────────────────────────────────────────
    MODEL_NAME  = 'EleutherAI/pythia-6.9b'
    D_MODEL     = 4096     # Pythia-6.9b hidden dimension
    N_LAYERS    = 32       # Pythia-6.9b transformer layer count (indices 0–31)
    USE_FP16    = True     # fp16 inference: ~14GB VRAM vs ~28GB fp32

    # ── SAE sweep ──────────────────────────────────────────────────────────────
    DICT_SIZES  = [4096, 8192, 16384, 32768, 65536]
    LAYERS      = list(range(N_LAYERS))   # 0–31
    SAE_BASE_PATH = '../models'

    # ── Results ────────────────────────────────────────────────────────────────
    RESULTS_BASE_DIR = '../results/comprehensive_pythia_6.9b'

    # ── TOFU ───────────────────────────────────────────────────────────────────
    FORGET_SPLIT = 'forget10'
    RETAIN_SPLIT = 'retain90'
    MAX_SEQ_LEN  = 512
    BATCH_SIZE   = 4       # safe for 6.9B + SAE on 80GB A100

    # ── SAE-guided hyperparameters ─────────────────────────────────────────────
    TOP_K_FEATURES = 128   # contrastive features to suppress

    # ── Gradient ascent baseline ───────────────────────────────────────────────
    GA_STEPS         = 5
    GA_LR            = 1e-4
    GA_MAX_GRAD_NORM = 1.0

    # ── Device ─────────────────────────────────────────────────────────────────
    # BUG FIX: original always resolved to 'cpu'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Author groups (from ED calibration) ────────────────────────────────────
    HIGH_ED_AUTHORS = [
        'Rajeev Majumdar', 'Jun Chen', 'Basil Mahfouz',
        'Hsiao Yun', 'Carmen Montenegro', 'Behrouz Rohani',
        'Yeon Park', 'Kalkidan Abera', 'Takashi Nakamura'
    ]
    LOW_ED_AUTHORS = [
        'Raven Marais', 'Xin Lee', 'Adib Jarrah',
        'Moshe Ben', 'Hina Ameen', 'Elvin Mammadov',
        'Nikolai Abilov', 'Jad Ambrose', 'Patrick Sullivan', 'Aysha Al'
    ]

    @staticmethod
    def date_str() -> str:
        return datetime.now().strftime("%d_%b")


# ══════════════════════════════════════════════════════════════════════════════
# SPARSE AUTOENCODER
# ══════════════════════════════════════════════════════════════════════════════

class AnthropicSAE(nn.Module):
    """Anthropic-style SAE: encoder (Linear+ReLU) + decoder (Linear, no bias)"""

    def __init__(self, d_model: int, dict_size: int):
        super().__init__()
        self.d_model   = d_model
        self.dict_size = dict_size
        self.encoder   = nn.Linear(d_model, dict_size)
        self.decoder   = nn.Linear(dict_size, d_model, bias=False)
        nn.init.normal_(self.decoder.weight, std=0.02)

    def forward(self, x):
        acts  = F.relu(self.encoder(x))
        recon = self.decoder(acts)
        return recon, acts

    def encode(self, x) -> torch.Tensor:
        return F.relu(self.encoder(x))

    def decode(self, features) -> torch.Tensor:
        return self.decoder(features)


class StandardSAE(nn.Module):
    """Standard SAE with bias on decoder — used as fallback for some checkpoints"""

    def __init__(self, d_model: int, dict_size: int):
        super().__init__()
        self.d_model   = d_model
        self.dict_size = dict_size
        self.encoder   = nn.Linear(d_model, dict_size, bias=True)
        self.decoder   = nn.Linear(dict_size, d_model, bias=True)

    def forward(self, x):
        acts  = F.relu(self.encoder(x))
        recon = self.decoder(acts)
        return recon, acts

    def encode(self, x) -> torch.Tensor:
        return F.relu(self.encoder(x))

    def decode(self, features) -> torch.Tensor:
        return self.decoder(features)


def _load_sae_flexible(checkpoint: dict, d_model: int, dict_size: int, device: str):
    """
    Handle multiple SAE checkpoint formats:
      1. TransformerLens/SAELens format (W_enc, b_enc, W_dec keys)
      2. Standard format (encoder.weight, decoder.weight keys)
    """
    state_dict = checkpoint.get('state_dict', checkpoint)

    if 'W_enc' in state_dict:
        # TransformerLens / SAELens format
        mapped = {
            'encoder.weight': state_dict['W_enc'].T,
            'encoder.bias'  : state_dict['b_enc'],
            'decoder.weight': state_dict['W_dec'].T,
        }
        has_dec_bias = 'b_dec' in state_dict
        if has_dec_bias:
            sae = StandardSAE(d_model, dict_size).to(device)
            mapped['decoder.bias'] = state_dict['b_dec']
        else:
            sae = AnthropicSAE(d_model, dict_size).to(device)
        sae.load_state_dict(mapped, strict=True)

    elif 'encoder.weight' in state_dict:
        has_dec_bias = 'decoder.bias' in state_dict
        if has_dec_bias:
            sae = StandardSAE(d_model, dict_size).to(device)
        else:
            sae = AnthropicSAE(d_model, dict_size).to(device)
        sae.load_state_dict(state_dict, strict=True)

    else:
        raise ValueError(
            f"Unknown SAE checkpoint format. Keys: {list(state_dict.keys())}"
        )

    sae.eval()
    return sae


def load_sae(dict_size: int, layer_idx: int, device: str = None):
    """
    Load pre-trained SAE checkpoint.
    Expected path: {SAE_BASE_PATH}/dict_{dict_size}/layer_{layer_idx}.pt

    Checkpoint must contain:
        'd_model'    : int
        'dict_size'  : int
        'state_dict' : OrderedDict
    """
    if device is None:
        device = Config.DEVICE

    sae_path = Path(Config.SAE_BASE_PATH) / f"dict_{dict_size}" / f"layer_{layer_idx}.pt"

    if not sae_path.exists():
        raise FileNotFoundError(f"SAE checkpoint not found: {sae_path}")

    print(f"    Loading SAE: dict={dict_size}, layer={layer_idx}")
    checkpoint = torch.load(sae_path, map_location='cpu')

    d_model   = checkpoint.get('d_model',
                checkpoint.get('cfg', {}).get('d_in', Config.D_MODEL))
    dict_size_ = checkpoint.get('dict_size',
                 checkpoint.get('cfg', {}).get('d_sae', dict_size))

    if dict_size_ != dict_size:
        print(f"    ⚠️  Checkpoint dict_size={dict_size_}, expected={dict_size}")

    sae = _load_sae_flexible(checkpoint, d_model, dict_size_, device)
    print(f"    ✓ SAE loaded (d_model={d_model}, dict_size={dict_size_})")
    return sae


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_tofu_data(author_indices: list, tokenizer):
    """
    Build forget/retain DataLoaders for one author.

    Returns:
        (retain_loader, forget_loader)
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    forget_full = get_tofudataset(Config.FORGET_SPLIT)
    retain_ds   = get_tofudataset(Config.RETAIN_SPLIT)

    forget_ds = forget_full.select(author_indices)

    def _tokenize(ds):
        tok = ds.map(
            lambda x: tokenize_function(x, tokenizer, Config.MAX_SEQ_LEN),
            batched=True,
            remove_columns=ds.column_names
        )
        tok.set_format(type="torch", columns=["input_ids", "attention_mask"])
        return tok

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    forget_loader = DataLoader(
        _tokenize(forget_ds),
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collator
    )
    retain_loader = DataLoader(
        _tokenize(retain_ds),
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collator
    )
    return retain_loader, forget_loader


# ══════════════════════════════════════════════════════════════════════════════
# SAE-GUIDED UNLEARNING
# ══════════════════════════════════════════════════════════════════════════════

class SAEGuidedUnlearning:
    """
    SAE-guided feature suppression for Pythia-6.9b.

    Key Pythia difference from GPT-2:
        GPT-2  hook target: model.transformer.h[layer_idx]
        Pythia hook target: model.gpt_neox.layers[layer_idx]   ← this class
    """

    def __init__(
        self,
        model,
        sae,
        tokenizer,
        layer_idx: int,
        device: str,
        suppression_scale: float,
        interp_alpha: float,
        feature_multiplier: float
    ):
        self.model              = model
        self.sae                = sae
        self.tokenizer          = tokenizer
        self.layer_idx          = layer_idx
        self.device             = device
        self.suppression_scale  = suppression_scale
        self.alpha              = interp_alpha
        self.feature_multiplier = feature_multiplier

    def _pythia_layer(self):
        """
        Return the Pythia transformer block for self.layer_idx.

        Pythia-6.9b architecture:
            model.gpt_neox                  ← GPTNeoXModel
            model.gpt_neox.layers[i]        ← GPTNeoXLayer  (hook target)
            model.gpt_neox.layers[i].attention
            model.gpt_neox.layers[i].mlp
        """
        return self.model.gpt_neox.layers[self.layer_idx]

    def _get_activations(self, dataloader, max_batches: int = None) -> torch.Tensor:
        """
        Extract per-token hidden states at self.layer_idx via a forward hook.

        Pythia's GPTNeoXLayer returns a tuple: (hidden_state, *optional_cache)
        so the hook correctly extracts output[0].

        Returns:
            Tensor of shape (total_valid_tokens, d_model) on self.device.
        """
        all_token_acts = []

        # BUG FIX: use model.config.hidden_size, not model.config.n_embd
        # GPT-2 exposes n_embd; Pythia exposes hidden_size
        d_model = self.model.config.hidden_size

        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break

            inputs = {
                k: v.to(self.device)
                for k, v in batch.items()
                if k in ('input_ids', 'attention_mask')
            }
            captured = []

            def hook(module, inp, output):
                # GPTNeoXLayer output is a tuple: (hidden_state, ...)
                hidden = output[0] if isinstance(output, tuple) else output
                captured.append(hidden.detach().cpu())

            # BUG FIX: hook Pythia layer, not GPT-2 layer
            handle = self._pythia_layer().register_forward_hook(hook)

            with torch.no_grad():
                self.model(**inputs)

            handle.remove()

            if not captured:
                continue

            acts            = captured[0]                     # (B, T, D)
            attention_mask  = batch['attention_mask'].cpu()

            for i in range(acts.shape[0]):
                mask = attention_mask[i].bool()
                all_token_acts.append(acts[i][mask])          # (valid_T, D)

        if not all_token_acts:
            return torch.empty(0, d_model, device=self.device)

        return torch.cat(all_token_acts, dim=0).to(self.device)

    def identify_top_features_contrast(
        self,
        forget_loader,
        retain_loader,
        k: int = 50,
        max_batches: int = 10
    ) -> torch.Tensor:
        """
        Identify top-k SAE features most specific to the forget set.
        Ranked by: mean_forget_activation − mean_retain_activation.

        Returns:
            LongTensor of feature indices, shape (k,).
        """
        forget_acts = self._get_activations(forget_loader, max_batches)
        retain_acts = self._get_activations(retain_loader, max_batches)

        if forget_acts.shape[0] == 0 or retain_acts.shape[0] == 0:
            return torch.tensor([], dtype=torch.long, device=self.device)

        # Cast to fp32 before SAE (SAE weights are fp32)
        with torch.no_grad():
            f_feats = self.sae.encode(forget_acts.float())
            r_feats = self.sae.encode(retain_acts.float())

        contrast = f_feats.mean(dim=0) - r_feats.mean(dim=0)
        k        = min(k, contrast.numel())
        _, top_idx = torch.topk(contrast, k=k)
        return top_idx

    def _make_suppression_hook(self, target_features: torch.Tensor):
        """
        Build a forward hook that:
          1. Encodes hidden states with the SAE
          2. Suppresses target feature activations
          3. Decodes back and blends with original hidden state

        Uses soft blending (alpha) instead of hard replacement to preserve
        retain-set representations.
        """
        scale      = self.suppression_scale * self.feature_multiplier
        alpha      = self.alpha
        sae        = self.sae
        device     = self.device
        tgt_feats  = target_features

        def hook(module, inp, output):
            hidden = output[0] if isinstance(output, tuple) else output
            rest   = output[1:] if isinstance(output, tuple) else ()

            shape = hidden.shape
            flat  = hidden.reshape(-1, shape[-1]).float()   # (B*T, D) fp32

            # SAE encode → suppress → decode
            features                 = sae.encode(flat)
            features[:, tgt_feats]  *= scale
            recon                    = sae.decode(features).to(hidden.dtype)
            recon                    = recon.reshape(shape)

            # Soft blend: original + alpha * (suppressed - original)
            blended = hidden + alpha * (recon - hidden)

            return (blended,) + rest if rest else blended

        return hook

    def evaluate_loss(
        self,
        dataloader,
        hook_fn=None,
        max_batches: int = None
    ) -> float:
        """Compute average cross-entropy loss, optionally with suppression hook."""
        handle = None
        if hook_fn is not None:
            handle = self._pythia_layer().register_forward_hook(hook_fn)

        total_loss = 0.0
        n_samples  = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches and batch_idx >= max_batches:
                    break
                inputs      = {k: v.to(self.device) for k, v in batch.items()}
                outputs     = self.model(**inputs)
                batch_size  = inputs['input_ids'].shape[0]
                total_loss += outputs.loss.item() * batch_size
                n_samples  += batch_size

        if handle is not None:
            handle.remove()

        return total_loss / n_samples if n_samples > 0 else 0.0

    def run(self, forget_loader, retain_loader) -> dict | None:
        """
        Full SAE-guided unlearning run for one author.

        Returns dict with before/after losses, or None on failure.
        """
        target_features = self.identify_top_features_contrast(
            forget_loader, retain_loader,
            k=Config.TOP_K_FEATURES,
            max_batches=10
        )

        if len(target_features) == 0:
            print("    ⚠️  No contrastive features found")
            return None

        hook_fn = self._make_suppression_hook(target_features)

        forget_base = self.evaluate_loss(forget_loader)
        retain_base = self.evaluate_loss(retain_loader)
        forget_supp = self.evaluate_loss(forget_loader, hook_fn)
        retain_supp = self.evaluate_loss(retain_loader, hook_fn)

        return {
            'forget_baseline'   : forget_base,
            'forget_suppressed' : forget_supp,
            'forget_increase'   : forget_supp - forget_base,
            'retain_baseline'   : retain_base,
            'retain_suppressed' : retain_supp,
            'retain_change'     : retain_supp - retain_base,
            'perplexity_base'   : float(np.exp(retain_base)),
            'perplexity_supp'   : float(np.exp(retain_supp))
        }


# ══════════════════════════════════════════════════════════════════════════════
# GRADIENT ASCENT BASELINE
# ══════════════════════════════════════════════════════════════════════════════

def _compute_loss(model, dataloader, device: str) -> float:
    """Average cross-entropy loss over a DataLoader."""
    total_loss = 0.0
    n_samples  = 0
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            inputs      = {k: v.to(device) for k, v in batch.items()}
            outputs     = model(**inputs)
            batch_size  = inputs['input_ids'].shape[0]
            total_loss += outputs.loss.item() * batch_size
            n_samples  += batch_size

    return total_loss / n_samples if n_samples > 0 else 0.0


def run_gradient_ascent(model, tokenizer, forget_loader, retain_loader) -> dict:
    """
    Gradient ascent baseline: maximise forget-set loss for GA_STEPS steps.
    Operates on a deep copy — original model is untouched.
    """
    device     = Config.DEVICE
    model_copy = copy.deepcopy(model).to(device)
    optimizer  = torch.optim.Adam(model_copy.parameters(), lr=Config.GA_LR)

    forget_before = _compute_loss(model_copy, forget_loader, device)
    retain_before = _compute_loss(model_copy, retain_loader, device)

    model_copy.train()
    forget_iter = iter(forget_loader)

    for _ in range(Config.GA_STEPS):
        try:
            batch = next(forget_iter)
        except StopIteration:
            forget_iter = iter(forget_loader)
            batch = next(forget_iter)

        inputs  = {k: v.to(device) for k, v in batch.items()}
        outputs = model_copy(**inputs)
        loss    = -outputs.loss                    # gradient ascent = negate loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model_copy.parameters(), max_norm=Config.GA_MAX_GRAD_NORM
        )
        optimizer.step()

    model_copy.eval()
    forget_after = _compute_loss(model_copy, forget_loader, device)
    retain_after = _compute_loss(model_copy, retain_loader, device)

    del model_copy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'forget_before'     : forget_before,
        'forget_after'      : forget_after,
        'forget_increase'   : forget_after - forget_before,
        'retain_before'     : retain_before,
        'retain_after'      : retain_after,
        'retain_change'     : retain_after - retain_before,
        'perplexity_before' : float(np.exp(retain_before)),
        'perplexity_after'  : float(np.exp(retain_after))
    }


# ══════════════════════════════════════════════════════════════════════════════
# PER-AUTHOR RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_single_author(
    author_name: str,
    author_indices: list,
    group: str,
    model,
    sae,
    tokenizer,
    layer_idx: int,
    dict_size: int,
    dict_config: dict
) -> dict | None:
    """
    Run SAE-guided unlearning + gradient ascent baseline for one author.

    Returns:
        Combined results dict, or None on failure.
    """
    print(f"      [{group}] {author_name}")

    try:
        retain_loader, forget_loader = load_tofu_data(author_indices, tokenizer)
    except Exception as e:
        print(f"        ❌ Data loading failed: {e}")
        return None

    # SAE-guided
    sae_unlearner = SAEGuidedUnlearning(
        model              = model,
        sae                = sae,
        tokenizer          = tokenizer,
        layer_idx          = layer_idx,
        device             = Config.DEVICE,
        suppression_scale  = dict_config['suppression_scale'],
        interp_alpha       = dict_config['interp_alpha'],
        feature_multiplier = dict_config['feature_multiplier']
    )

    try:
        sae_results = sae_unlearner.run(forget_loader, retain_loader)
        if sae_results is None:
            return None
    except Exception as e:
        print(f"        ❌ SAE unlearning failed: {e}")
        return None

    # Gradient ascent baseline
    try:
        ga_results = run_gradient_ascent(model, tokenizer, forget_loader, retain_loader)
    except Exception as e:
        print(f"        ❌ GA baseline failed: {e}")
        return None

    # Selectivity: forget increase / |retain change|
    eps     = 1e-8
    ga_sel  = ga_results['forget_increase']  / (abs(ga_results['retain_change'])  + eps)
    sae_sel = sae_results['forget_increase'] / (abs(sae_results['retain_change']) + eps)

    print(f"        GA  — forget Δ: {ga_results['forget_increase']:+.4f}  "
          f"retain Δ: {ga_results['retain_change']:+.4f}  "
          f"selectivity: {ga_sel:.2f}")
    print(f"        SAE — forget Δ: {sae_results['forget_increase']:+.4f}  "
          f"retain Δ: {sae_results['retain_change']:+.4f}  "
          f"selectivity: {sae_sel:.2f}")

    return {
        'author'                : author_name,
        'group'                 : group,
        'dict_size'             : dict_size,
        'layer'                 : layer_idx,
        # SAE metrics
        'sae_forget_increase'   : sae_results['forget_increase'],
        'sae_retain_change'     : sae_results['retain_change'],
        'sae_perplexity_change' : sae_results['perplexity_supp'] - sae_results['perplexity_base'],
        'sae_selectivity'       : sae_sel,
        # GA metrics
        'ga_forget_increase'    : ga_results['forget_increase'],
        'ga_retain_change'      : ga_results['retain_change'],
        'ga_perplexity_change'  : ga_results['perplexity_after'] - ga_results['perplexity_before'],
        'ga_selectivity'        : ga_sel,
        # Comparative
        'retain_benefit'        : ga_results['retain_change'] - sae_results['retain_change'],
        'selectivity_gain'      : sae_sel - ga_sel
    }


# ══════════════════════════════════════════════════════════════════════════════
# PER-CONFIG RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_single_config(
    dict_size: int,
    layer_idx: int,
    model,
    tokenizer,
    author_to_samples: dict,
    dict_config: dict,
    date_str: str,
    resume: bool = True
) -> pd.DataFrame | None:
    """
    Run all authors for ONE (dict_size, layer_idx) configuration.

    Args:
        resume: If True, skip configs whose result CSV already exists.
    """
    results_dir = (
        Path(Config.RESULTS_BASE_DIR)
        / f"dict_{dict_size}"
        / f"layer_{layer_idx:02d}"
    )
    output_path = results_dir / f"results_{date_str}.csv"

    if resume and output_path.exists():
        print(f"  ↩  Skipping (exists): {output_path}")
        return pd.read_csv(output_path)

    print(f"\n  {'='*76}")
    print(f"  CONFIG: dict={dict_size}  layer={layer_idx}  model=pythia-6.9b")
    print(f"  {'='*76}")

    # Load SAE for this config
    try:
        sae = load_sae(dict_size, layer_idx, Config.DEVICE)
    except Exception as e:
        print(f"    ❌ SAE load failed: {e}")
        return None

    all_results = []

    for group, authors in [
        ('high_ed', Config.HIGH_ED_AUTHORS),
        ('low_ed',  Config.LOW_ED_AUTHORS)
    ]:
        print(f"\n    {group.upper()} AUTHORS:")
        for author in authors:
            if author not in author_to_samples:
                print(f"      ⚠️  {author} not in mapping, skipping")
                continue
            result = run_single_author(
                author_name    = author,
                author_indices = author_to_samples[author],
                group          = group,
                model          = model,
                sae            = sae,
                tokenizer      = tokenizer,
                layer_idx      = layer_idx,
                dict_size      = dict_size,
                dict_config    = dict_config
            )
            if result:
                all_results.append(result)

    # Clean up SAE
    del sae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not all_results:
        print(f"\n    ❌ No results for dict={dict_size} layer={layer_idx}")
        return None

    df = pd.DataFrame(all_results)
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, float_format='%.6f')

    print(f"\n    💾 Saved: {output_path}  ({len(df)} authors)")

    high = df[df['group'] == 'high_ed']['retain_benefit']
    low  = df[df['group'] == 'low_ed']['retain_benefit']
    print(f"    📊 retain_benefit  high-ED: {high.mean():.4f} ± {high.std():.4f}")
    print(f"                       low-ED:  {low.mean():.4f} ± {low.std():.4f}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def create_summary_statistics(df: pd.DataFrame, date_str: str):
    """Save grouped summary CSVs."""
    base = Path(Config.RESULTS_BASE_DIR)

    for groupby_cols, suffix in [
        (['dict_size'],         'by_dict_size'),
        (['layer'],             'by_layer'),
        (['dict_size', 'layer'], 'by_dict_and_layer'),
        (['group', 'dict_size', 'layer'], 'by_group')
    ]:
        summary = df.groupby(groupby_cols).agg({
            'sae_retain_change' : ['mean', 'std'],
            'ga_retain_change'  : ['mean', 'std'],
            'retain_benefit'    : ['mean', 'std', 'count'],
            'selectivity_gain'  : ['mean', 'std']
        }).round(4)
        path = base / f"summary_{suffix}_{date_str}.csv"
        summary.to_csv(path)
        print(f"  ✓ {path.name}")


def create_heatmap_data(df: pd.DataFrame, date_str: str):
    """Save heatmap CSVs for visualization."""
    base = Path(Config.RESULTS_BASE_DIR)

    for dict_size in Config.DICT_SIZES:
        sub = df[df['dict_size'] == dict_size]
        if sub.empty:
            continue
        hmap = sub.groupby('layer').agg({
            'sae_retain_change' : 'mean',
            'ga_retain_change'  : 'mean',
            'retain_benefit'    : 'mean',
            'selectivity_gain'  : 'mean'
        }).round(4)
        path = base / f"heatmap_dict_{dict_size}_{date_str}.csv"
        hmap.to_csv(path)
        print(f"  ✓ {path.name}")

    overall = df.pivot_table(
        index='dict_size', columns='layer',
        values='retain_benefit', aggfunc='mean'
    ).round(4)
    path = base / f"heatmap_overall_{date_str}.csv"
    overall.to_csv(path)
    print(f"  ✓ {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(
    dict_sizes: list = None,
    layers: list = None,
    resume: bool = True
):
    """
    Sweep ALL (dict_size, layer) combinations for Pythia-6.9b.

    Args:
        dict_sizes : Override Config.DICT_SIZES if provided.
        layers     : Override Config.LAYERS if provided.
        resume     : Skip configs whose result CSV already exists.
    """
    dict_sizes = dict_sizes or Config.DICT_SIZES
    layers     = layers     or Config.LAYERS
    date_str   = Config.date_str()

    total_configs = len(dict_sizes) * len(layers)

    print("\n" + "=" * 80)
    print("COMPREHENSIVE SAE-GUIDED UNLEARNING — Pythia-6.9b")
    print("=" * 80)
    print(f"Device      : {Config.DEVICE}")
    if Config.DEVICE == 'cuda':
        print(f"GPU         : {torch.cuda.get_device_name(0)}")
    print(f"FP16        : {Config.USE_FP16}")
    print(f"Dict sizes  : {dict_sizes}")
    print(f"Layers      : {layers}")
    print(f"Total cfgs  : {total_configs}")
    print(f"Authors     : {len(Config.HIGH_ED_AUTHORS)} high-ED + "
          f"{len(Config.LOW_ED_AUTHORS)} low-ED")
    print(f"Date string : {date_str}")
    print(f"Resume      : {resume}")

    # Load per-config hyperparameters
    df_config = pd.read_json('sae_unlearning_configs.json')

    # ── Load shared resources ─────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("LOADING SHARED RESOURCES")
    print(f"{'='*80}")

    print("\n[1/3] Loading Pythia-6.9b...")
    model, tokenizer = get_gptmodel(Config.MODEL_NAME)
    model = model.to(Config.DEVICE)

    if Config.USE_FP16 and Config.DEVICE == 'cuda':
        model = model.half()
        print("      → converted to fp16")

    model.eval()
    print(f"      → loaded on {Config.DEVICE}")

    print("\n[2/3] Loading author mapping...")
    with open('../data/tofu_author_mapping.json', 'r') as f:
        author_to_samples = json.load(f)['author_to_samples']
    print(f"      → {len(author_to_samples)} authors")

    print("\n[3/3] Creating output directory...")
    Path(Config.RESULTS_BASE_DIR).mkdir(parents=True, exist_ok=True)

    # ── Main sweep ────────────────────────────────────────────────────────────
    results_by_dict_size = {d: [] for d in dict_sizes}
    completed  = 0
    start_time = time.time()

    print(f"\n{'='*80}")
    print("RUNNING ALL CONFIGURATIONS")
    print(f"{'='*80}")

    for dict_size in dict_sizes:
        dict_config = df_config[dict_size]

        print(f"\n{'#'*80}")
        print(f"DICT SIZE: {dict_size}")
        print(f"{'#'*80}")

        for layer_idx in layers:
            completed += 1
            print(f"\n  CONFIG {completed}/{total_configs}: "
                  f"dict={dict_size}  layer={layer_idx}")

            try:
                df = run_single_config(
                    dict_size          = dict_size,
                    layer_idx          = layer_idx,
                    model              = model,
                    tokenizer          = tokenizer,
                    author_to_samples  = author_to_samples,
                    dict_config        = dict_config,
                    date_str           = date_str,
                    resume             = resume
                )
                if df is not None:
                    results_by_dict_size[dict_size].append(df)

            except Exception as e:
                print(f"  ❌ Config failed: {e}")
                traceback.print_exc()

            # Progress
            elapsed   = time.time() - start_time
            rate      = completed / max(elapsed, 1e-6)
            eta       = (total_configs - completed) / rate
            print(f"\n  📊 {completed}/{total_configs} "
                  f"({100*completed/total_configs:.1f}%)  "
                  f"⏱ {elapsed/60:.1f}min / ETA {eta/60:.1f}min")

    # ── Consolidate and save ──────────────────────────────────────────────────
    base      = Path(Config.RESULTS_BASE_DIR)
    all_dfs   = []

    print(f"\n{'='*80}")
    print("SAVING CONSOLIDATED RESULTS")
    print(f"{'='*80}")

    for dict_size, dfs in results_by_dict_size.items():
        if not dfs:
            print(f"  ⚠️  No results for dict_size={dict_size}")
            continue
        combined = pd.concat(dfs, ignore_index=True)
        all_dfs.append(combined)
        path = base / f"dict_{dict_size}_all_layers_{date_str}.csv"
        combined.to_csv(path, index=False, float_format='%.6f')
        print(f"  💾 dict={dict_size}: {path.name}  ({len(combined)} rows, "
              f"{combined['layer'].nunique()} layers)")

    if all_dfs:
        all_results = pd.concat(all_dfs, ignore_index=True)
        all_results.to_csv(
            base / f"all_results_{date_str}.csv",
            index=False, float_format='%.6f'
        )
        print(f"\n  💾 all_results_{date_str}.csv  ({len(all_results)} total rows)")

        print(f"\n{'='*80}")
        print("SUMMARY STATISTICS")
        print(f"{'='*80}")
        create_summary_statistics(all_results, date_str)

        print(f"\n{'='*80}")
        print("HEATMAP DATA")
        print(f"{'='*80}")
        create_heatmap_data(all_results, date_str)

    # ── Final report ─────────────────────────────────────────────────────────
    total_time = time.time() - start_time
    n_success  = sum(len(v) for v in results_by_dict_size.values())

    print(f"\n{'='*80}")
    print("✅ COMPLETE — Pythia-6.9b SAE-Guided Unlearning")
    print(f"{'='*80}")
    print(f"Configs : {n_success}/{total_configs} successful")
    print(f"Time    : {total_time/60:.1f} min ({total_time/3600:.2f} hr)")
    print(f"Output  : {base}")

    return results_by_dict_size


if __name__ == "__main__":
    results = main(
        dict_sizes = [8192, 16384],
        layers     = [27, 28, 29, 30, 31],
        resume     = True
    )