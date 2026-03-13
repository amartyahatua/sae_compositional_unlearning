"""
Compute ED/Gini/L0/Entropy for ALL Dict Sizes × ALL Layers
============================================================
Pythia-6.9b specific rewrite.

Model  : EleutherAI/pythia-6.9b
Layers : 32 transformer layers (indices 0–31)
d_model: 4096

Dict sizes : [4096, 8192, 16384, 32768, 65536]
Layers     : [0 … 31]
Total      : 5 × 32 = 160 configurations

Fixes over original:
  - Critical bug: device was always 'cpu' even when CUDA available
  - CUDA cache now cleared after every config
  - Resume support: skips already-completed CSVs
  - Pythia hidden_states index corrected (0 = embedding, 1–32 = layers)
  - Half-precision (fp16) inference to fit Pythia-6.9b on A100 80GB
  - Configurable batch size with sensible Pythia default (4)
  - Summary filename no longer has a hardcoded date
  - Author dataloader rebuilt cleanly per config
"""

import torch
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
import os
from pathlib import Path
import time
import torch.nn.functional as F
from datetime import datetime

from get_dataset import get_tofudataset, tokenize_function
from get_model import get_gptmodel


# ── Pythia-6.9b constants ──────────────────────────────────────────────────────
PYTHIA_MODEL_ID   = "EleutherAI/pythia-6.9b"
PYTHIA_D_MODEL    = 4096          # hidden dimension of pythia-6.9b
PYTHIA_N_LAYERS   = 32            # transformer layers: indices 0–31
MAX_SEQ_LEN       = 512
BATCH_SIZE        = 4             # safe default for 6.9B + SAE on 80GB A100
USE_FP16          = True          # half precision for Pythia inference


# ─────────────────────────────────────────────────────────────────────────────
# SAE
# ─────────────────────────────────────────────────────────────────────────────

class SparseAutoencoder(torch.nn.Module):
    """
    Standard L1-SAE.
    Encoder: Linear → ReLU
    Decoder: Linear (no bias), optionally normalized
    """

    def __init__(self, d_model: int, dict_size: int):
        super().__init__()
        self.d_model   = d_model
        self.dict_size = dict_size
        self.encoder   = torch.nn.Linear(d_model, dict_size)
        self.decoder   = torch.nn.Linear(dict_size, d_model, bias=False)
        torch.nn.init.normal_(self.decoder.weight, std=0.02)

    def forward(self, x):
        acts  = F.relu(self.encoder(x))
        recon = self.decoder(acts)
        return recon, acts

    @torch.no_grad()
    def encode(self, x) -> torch.Tensor:
        return F.relu(self.encoder(x))

    @torch.no_grad()
    def decode(self, acts) -> torch.Tensor:
        return self.decoder(acts)

    @torch.no_grad()
    def normalize_decoder(self):
        W = self.decoder.weight
        self.decoder.weight.copy_(
            W / W.norm(dim=0, keepdim=True).clamp(min=1e-6)
        )


# ─────────────────────────────────────────────────────────────────────────────
# SAE loading
# ─────────────────────────────────────────────────────────────────────────────

def load_sae(layer_idx: int, dict_size: int, sae_dir: str = '../models') -> SparseAutoencoder:
    """
    Load a trained SAE checkpoint.

    Expected checkpoint format:
        {
            'd_model'    : int,
            'dict_size'  : int,
            'state_dict' : OrderedDict
        }

    Checkpoint path: {sae_dir}/dict_{dict_size}/layer_{layer_idx}.pt
    """
    checkpoint_path = Path(sae_dir) / f"dict_{dict_size}" / f"layer_{layer_idx}.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SAE checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    d_model   = checkpoint['d_model']
    dict_size_ = checkpoint['dict_size']

    sae = SparseAutoencoder(d_model, dict_size_)
    sae.load_state_dict(checkpoint['state_dict'])
    sae.eval()
    return sae


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction — Pythia hidden states
# ─────────────────────────────────────────────────────────────────────────────

def get_author_features(
    model,
    sae: SparseAutoencoder,
    dataloader: DataLoader,
    layer_idx: int,
    device: str
) -> torch.Tensor | None:
    """
    Extract SAE feature activations for all tokens of one author.

    Pythia hidden_states layout:
        hidden_states[0]           → embedding output (before any transformer block)
        hidden_states[1]           → after transformer block 0
        hidden_states[2]           → after transformer block 1
        ...
        hidden_states[layer_idx+1] → after transformer block layer_idx  ← what we want

    Args:
        model      : Pythia model (eval mode, fp16 if USE_FP16)
        sae        : Trained SAE for this (dict_size, layer_idx)
        dataloader : DataLoader for one author's tokenized samples
        layer_idx  : Which transformer layer to hook (0-indexed, 0–31)
        device     : 'cuda' or 'cpu'

    Returns:
        Tensor of shape (total_valid_tokens, dict_size), on CPU.
        None if no tokens were processed.
    """
    model.eval()
    sae.eval()
    all_features = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Forward pass — request all hidden states
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

            # hidden_states[layer_idx + 1] = output of transformer block layer_idx
            # Shape: (batch_size, seq_len, d_model)
            layer_acts = outputs.hidden_states[layer_idx + 1]

            # Cast to fp32 before SAE (SAE weights are fp32)
            layer_acts = layer_acts.float()

            # SAE encode: (batch_size, seq_len, dict_size)
            features = sae.encode(layer_acts)

            # Keep only valid (non-padding) tokens
            for i in range(features.shape[0]):
                mask = attention_mask[i].bool()          # (seq_len,)
                valid_features = features[i][mask]       # (n_valid, dict_size)
                all_features.append(valid_features.cpu())

    if len(all_features) == 0:
        return None

    return torch.cat(all_features, dim=0)   # (total_tokens, dict_size)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_ed_95(features: torch.Tensor) -> int:
    """
    Effective Dimensionality at 95% threshold.
    Number of top features needed to account for 95% of total mean activation.

    Higher ED → features are spread across many dimensions (less concentrated).
    Lower ED  → a few features dominate (more concentrated).
    """
    mean_acts  = features.mean(dim=0).cpu().numpy()
    sorted_acts = np.sort(mean_acts)[::-1]
    cumsum     = np.cumsum(sorted_acts)
    total      = cumsum[-1]

    if total == 0:
        return 0

    ed_95 = int(np.argmax(cumsum >= 0.95 * total) + 1)
    return ed_95


def compute_gini_coefficient(features: torch.Tensor) -> float:
    """
    Gini coefficient of mean feature activations.
    Range: [0, 1]
      0 → perfectly uniform (all features equally active)
      1 → perfectly concentrated (one feature does everything)

    Higher Gini → sparser, more discriminative feature usage.
    """
    mean_acts   = features.mean(dim=0).cpu().numpy()
    sorted_acts = np.sort(mean_acts)           # ascending sort for Gini formula
    n           = len(sorted_acts)

    if sorted_acts.sum() == 0:
        return 0.0

    cumsum = np.cumsum(sorted_acts)
    gini   = (
        (2 * np.sum(np.arange(1, n + 1) * sorted_acts))
        / (n * cumsum[-1])
    ) - (n + 1) / n

    return float(np.clip(gini, 0.0, 1.0))


def compute_l0_sparsity(features: torch.Tensor, threshold: float = 0.0) -> float:
    """
    L0 sparsity: average number of active (> threshold) features per token.
    Ranges from 0 to dict_size.

    Lower L0 → sparser representations (fewer features fire per token).
    """
    l0 = (features > threshold).sum(dim=1).float().mean().item()
    return float(l0)


def compute_entropy(features: torch.Tensor) -> float:
    """
    Shannon entropy of the mean activation distribution.
    Treats normalized mean activations as a probability distribution.

    Lower entropy → more concentrated / selective feature usage.
    Higher entropy → more uniform / spread feature usage.
    """
    mean_acts = features.mean(dim=0).cpu().numpy()
    total     = mean_acts.sum()

    if total == 0:
        return 0.0

    probs    = mean_acts / total
    probs    = probs[probs > 0]          # remove zeros before log
    entropy  = float(-np.sum(probs * np.log(probs)))
    return entropy


# ─────────────────────────────────────────────────────────────────────────────
# Per-config calibration
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_one_config(
    dict_size: int,
    layer_idx: int,
    model,
    tokenizer,
    author_to_samples: dict,
    forget_ds,
    collator,
    device: str,
    output_dir: Path,
    batch_size: int = BATCH_SIZE,
    resume: bool = True
) -> pd.DataFrame | None:
    """
    Compute ED/Gini/L0/Entropy for all authors under one (dict_size, layer_idx) config.

    Args:
        dict_size        : SAE dictionary size
        layer_idx        : Pythia transformer layer index (0–31)
        model            : Loaded Pythia model
        tokenizer        : Pythia tokenizer
        author_to_samples: {author_name: [dataset_indices]}
        forget_ds        : TOFU forget split HuggingFace Dataset
        collator         : DataCollatorForLanguageModeling
        device           : 'cuda' or 'cpu'
        output_dir       : Root output directory
        batch_size       : Inference batch size
        resume           : If True, skip configs whose CSV already exists

    Returns:
        DataFrame with per-author metrics, or None if SAE not found / no results.
    """
    subfolder   = output_dir / f"dict_{dict_size}"
    output_path = subfolder / f"calibration_dict{dict_size}_layer{layer_idx}.csv"

    # Resume: skip if already done
    if resume and output_path.exists():
        print(f"  ↩  Skipping (already exists): {output_path}")
        return pd.read_csv(output_path)

    print(f"\n{'='*80}")
    print(f"CONFIG: dict_size={dict_size}  |  layer={layer_idx}  |  model=pythia-6.9b")
    print(f"{'='*80}")

    # Load SAE for this config
    try:
        sae = load_sae(layer_idx, dict_size).to(device)
        sae.eval()
    except FileNotFoundError as e:
        print(f"  ⚠️  SAE not found, skipping: {e}")
        return None
    except Exception as e:
        print(f"  ❌ Error loading SAE: {e}")
        return None

    results = []

    for author_name, author_indices in tqdm(
        author_to_samples.items(),
        desc=f"D{dict_size} L{layer_idx}",
        leave=False
    ):
        try:
            # Build per-author DataLoader
            author_subset    = forget_ds.select(author_indices)
            author_tokenized = author_subset.map(
                lambda x: tokenize_function(x, tokenizer, MAX_SEQ_LEN),
                batched=True,
                remove_columns=author_subset.column_names
            )
            author_tokenized.set_format(
                type="torch",
                columns=["input_ids", "attention_mask"]
            )
            author_loader = DataLoader(
                author_tokenized,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collator
            )

            # Extract SAE features for all valid tokens
            features = get_author_features(
                model, sae, author_loader, layer_idx, device
            )

            if features is None or features.shape[0] == 0:
                print(f"  ⚠️  {author_name}: no valid tokens, skipping")
                continue

            # Compute metrics
            results.append({
                'author'           : author_name,
                'num_samples'      : len(author_indices),
                'num_tokens'       : features.shape[0],
                'effective_dim_95' : compute_ed_95(features),
                'gini_coefficient' : compute_gini_coefficient(features),
                'l0_sparsity'      : compute_l0_sparsity(features),
                'entropy'          : compute_entropy(features),
                'dict_size'        : dict_size,
                'layer'            : layer_idx
            })

        except Exception as e:
            print(f"  ❌ Error for {author_name}: {e}")
            continue

    # Clean up SAE and free GPU memory
    del sae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if len(results) == 0:
        print(f"  ❌ No results for dict_size={dict_size}, layer={layer_idx}")
        return None

    df = pd.DataFrame(results)

    # Save
    subfolder.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, float_format='%.6f')

    print(f"  ✅ Saved: {output_path}")
    print(f"     Authors : {len(df)}")
    print(f"     Tokens  : {df['num_tokens'].sum():,}")
    print(f"     ED-95   : [{df['effective_dim_95'].min()}, {df['effective_dim_95'].max()}]")
    print(f"     Gini    : [{df['gini_coefficient'].min():.4f}, {df['gini_coefficient'].max():.4f}]")
    print(f"     L0      : {df['l0_sparsity'].mean():.2f} avg active features/token")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main sweep
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_all_configs(
    dict_sizes: list = None,
    layers: list = None,
    output_dir: str = '../results/calibration_comprehensive',
    sae_dir: str = '../models',
    author_mapping_path: str = '../data/tofu_author_mapping.json',
    tofu_split: str = 'forget10',
    batch_size: int = BATCH_SIZE,
    resume: bool = True
) -> pd.DataFrame | None:
    """
    Sweep ALL (dict_size, layer) combinations for Pythia-6.9b.

    Args:
        dict_sizes          : SAE dictionary sizes to sweep.
                              Default: [4096, 8192, 16384, 32768, 65536]
        layers              : Pythia layer indices to sweep (0–31).
                              Default: all 32 layers
        output_dir          : Root directory for results.
        sae_dir             : Directory containing SAE checkpoints.
        author_mapping_path : Path to tofu_author_mapping.json
        tofu_split          : TOFU dataset split ('forget10', 'forget05', etc.)
        batch_size          : Inference batch size per author DataLoader.
        resume              : Skip configs whose output CSV already exists.

    Returns:
        Summary DataFrame across all configs, or None.
    """

    # ── defaults ──────────────────────────────────────────────────────────────
    if dict_sizes is None:
        dict_sizes = [4096, 8192, 16384, 32768, 65536]
    if layers is None:
        layers = list(range(PYTHIA_N_LAYERS))   # 0–31

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── device ────────────────────────────────────────────────────────────────
    # BUG FIX: original code had 'cpu' on both branches of the ternary
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    total_configs = len(dict_sizes) * len(layers)

    print("\n" + "=" * 80)
    print("COMPREHENSIVE CALIBRATION — Pythia-6.9b")
    print("=" * 80)
    print(f"Model      : {PYTHIA_MODEL_ID}")
    print(f"d_model    : {PYTHIA_D_MODEL}")
    print(f"Device     : {device}")
    print(f"FP16       : {USE_FP16}")
    print(f"Dict sizes : {dict_sizes}")
    print(f"Layers     : {layers}")
    print(f"Batch size : {batch_size}")
    print(f"Total cfgs : {total_configs}")
    print(f"Resume     : {resume}")
    print(f"Output     : {output_dir}")

    # ── Load shared resources (once) ─────────────────────────────────────────
    print("\n[1/4] Loading Pythia-6.9b...")
    model, tokenizer = get_gptmodel(PYTHIA_MODEL_ID)
    model = model.to(device)

    # Half precision inference — Pythia-6.9b at fp16 uses ~14GB on A100
    if USE_FP16 and device == 'cuda':
        model = model.half()
        print("      → converted to fp16")

    model.eval()
    print(f"      → model loaded on {device}")

    print("\n[2/4] Loading author mapping...")
    with open(author_mapping_path, 'r') as f:
        author_data = json.load(f)
    author_to_samples = author_data['author_to_samples']
    print(f"      → {len(author_to_samples)} authors found")

    print(f"\n[3/4] Loading TOFU split: {tofu_split}...")
    forget_ds = get_tofudataset(tofu_split)
    print(f"      → {len(forget_ds)} samples")

    print("\n[4/4] Building collator...")
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ── Main sweep ────────────────────────────────────────────────────────────
    completed   = 0
    successful  = 0
    failed      = 0
    all_summaries = []
    start_time  = time.time()

    for dict_size in dict_sizes:

        print(f"\n{'#' * 80}")
        print(f"DICT SIZE: {dict_size}")
        print(f"{'#' * 80}")

        for layer_idx in layers:

            df = calibrate_one_config(
                dict_size      = dict_size,
                layer_idx      = layer_idx,
                model          = model,
                tokenizer      = tokenizer,
                author_to_samples = author_to_samples,
                forget_ds      = forget_ds,
                collator       = collator,
                device         = device,
                output_dir     = output_dir,
                batch_size     = batch_size,
                resume         = resume
            )

            completed += 1

            if df is not None:
                successful += 1
                all_summaries.append({
                    'dict_size'    : dict_size,
                    'layer'        : layer_idx,
                    'num_authors'  : len(df),
                    'ed_mean'      : df['effective_dim_95'].mean(),
                    'ed_std'       : df['effective_dim_95'].std(),
                    'ed_min'       : df['effective_dim_95'].min(),
                    'ed_max'       : df['effective_dim_95'].max(),
                    'gini_mean'    : df['gini_coefficient'].mean(),
                    'gini_std'     : df['gini_coefficient'].std(),
                    'gini_min'     : df['gini_coefficient'].min(),
                    'gini_max'     : df['gini_coefficient'].max(),
                    'l0_mean'      : df['l0_sparsity'].mean(),
                    'entropy_mean' : df['entropy'].mean()
                })
            else:
                failed += 1

            # Progress
            elapsed   = time.time() - start_time
            rate      = completed / max(elapsed, 1e-6)
            remaining = total_configs - completed
            eta       = remaining / rate

            print(
                f"\n  📊 Progress: {completed}/{total_configs} "
                f"({100 * completed / total_configs:.1f}%)  "
                f"✅ {successful}  ❌ {failed}  "
                f"⏱ {elapsed/60:.1f}min elapsed / {eta/60:.1f}min ETA"
            )

            # Checkpoint summary after every config
            if all_summaries:
                _save_summary(all_summaries, output_dir)

    # ── Final report ─────────────────────────────────────────────────────────
    elapsed_total = time.time() - start_time

    print(f"\n{'=' * 80}")
    print("✅ CALIBRATION COMPLETE — Pythia-6.9b")
    print(f"{'=' * 80}")
    print(f"Total time   : {elapsed_total/60:.1f} min ({elapsed_total/3600:.2f} hr)")
    print(f"Configs done : {completed}")
    print(f"Successful   : {successful}")
    print(f"Failed       : {failed}")
    print(f"Results in   : {output_dir}")

    if not all_summaries:
        return None

    summary_df = _save_summary(all_summaries, output_dir)

    # Top configs by discriminability
    print("\n📊 TOP 5 CONFIGS BY ED VARIANCE (want HIGH variance):")
    top_ed = summary_df.nlargest(5, 'ed_std')[
        ['dict_size', 'layer', 'ed_std', 'ed_min', 'ed_max']
    ]
    print(top_ed.to_string(index=False))

    print("\n📊 TOP 5 CONFIGS BY GINI VARIANCE (want HIGH variance):")
    top_gini = summary_df.nlargest(5, 'gini_std')[
        ['dict_size', 'layer', 'gini_std', 'gini_min', 'gini_max']
    ]
    print(top_gini.to_string(index=False))

    return summary_df


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _save_summary(all_summaries: list, output_dir: Path) -> pd.DataFrame:
    """Save rolling summary CSV — no hardcoded date in filename."""
    summary_df  = pd.DataFrame(all_summaries)
    summary_path = output_dir / 'summary_statistics.csv'
    summary_df.to_csv(summary_path, index=False, float_format='%.6f')
    return summary_df


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Quick sweep example — last 5 layers, two dict sizes
    # Change dict_sizes and layers to None for full 160-config sweep
    summary = calibrate_all_configs(
        dict_sizes = [8192, 16384],
        layers     = [27, 28, 29, 30, 31],
        resume     = True
    )