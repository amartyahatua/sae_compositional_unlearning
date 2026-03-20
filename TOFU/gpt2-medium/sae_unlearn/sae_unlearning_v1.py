"""
SAE-GUIDED UNLEARNING — GPT-2 Small & Medium
=============================================

Two ablation modes:
  --ablation_mode zero   : zero out selected SAE features
  --ablation_mode mean   : replace selected features with retain-set mean

Configurable feature selection:
  --top_k_features N     : number of features to ablate per author
  --contrast_threshold T : minimum contrast score to include a feature
                           (0.0 = top-K only, >0 = threshold gate first)

Author: Amartya Hatua
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import numpy as np
import pandas as pd
import json
import copy
import time
import random
import argparse
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

from get_dataset import get_tofudataset, tokenize_function
from get_model import get_gptmodel

# ── Reproducibility ────────────────────────────────────────────────────────────
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    # ── Model ──────────────────────────────────────────────────────────────────
    MODEL_NAME   = 'gpt2'          # 'gpt2' | 'gpt2-medium'
    N_LAYERS     = 12              # 12 for small, 24 for medium

    # ── SAE sweep ──────────────────────────────────────────────────────────────
    DICT_SIZES   = [65536]
    LAYERS       = [10]            # layer 10 for small, 23 for medium

    SAE_BASE_PATH   = '../models'
    RESULTS_BASE_DIR = '../results/sae_unlearning'

    # ── TOFU ───────────────────────────────────────────────────────────────────
    FORGET_SPLIT = 'forget10'
    RETAIN_SPLIT = 'retain90'

    # ── Feature selection (overridable via CLI) ─────────────────────────────────
    TOP_K_FEATURES     = 128    # how many features to ablate
    CONTRAST_THRESHOLD = 0.0    # min contrast score; 0.0 = pure top-K

    # ── Ablation mode (overridable via CLI) ─────────────────────────────────────
    ABLATION_MODE = 'zero'      # 'zero' | 'mean'

    # ── GA baseline ────────────────────────────────────────────────────────────
    GA_STEPS         = 5
    GA_LEARNING_RATE = 1e-4

    # ── Device ─────────────────────────────────────────────────────────────────
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Authors ────────────────────────────────────────────────────────────────
    HIGH_ED_AUTHORS = [
        'Rajeev Majumdar', 'Jun Chen', 'Basil Mahfouz',
        'Hsiao Yun', 'Carmen Montenegro', 'Behrouz Rohani',
        'Yeon Park', 'Kalkidan Abera', 'Takashi Nakamura',
    ]
    LOW_ED_AUTHORS = [
        'Raven Marais', 'Xin Lee', 'Adib Jarrah',
        'Moshe Ben', 'Hina Ameen', 'Elvin Mammadov',
        'Nikolai Abilov', 'Jad Ambrose', 'Patrick Sullivan', 'Aysha Al',
    ]

    @staticmethod
    def date_str():
        return datetime.now().strftime("%d_%b")


# ==============================================================================
# SAE MODELS
# ==============================================================================

class AnthropicSAE(nn.Module):
    def __init__(self, d_model, dict_size):
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

    def encode(self, x):  return F.relu(self.encoder(x))
    def decode(self, z):  return self.decoder(z)


class SparseAutoencoder(nn.Module):
    def __init__(self, d_model, dict_size):
        super().__init__()
        self.d_model   = d_model
        self.dict_size = dict_size
        self.encoder   = nn.Linear(d_model, dict_size, bias=True)
        self.decoder   = nn.Linear(dict_size, d_model, bias=True)

    def forward(self, x):
        acts  = F.relu(self.encoder(x))
        recon = self.decoder(acts)
        return recon, acts

    def encode(self, x):  return F.relu(self.encoder(x))
    def decode(self, z):  return self.decoder(z)


def get_model_d_model(model_name):
    """Return the hidden dim for a given GPT-2 variant."""
    return {
        'gpt2':         768,
        'gpt2-medium':  1024,
        'gpt2-large':   1280,
        'gpt2-xl':      1600,
    }.get(model_name, 768)


def load_sae(dict_size, layer_idx, device):
    path = f"{Config.SAE_BASE_PATH}/dict_{dict_size}/layer_{layer_idx}.pt"
    print(f"    Loading SAE: dict={dict_size}, layer={layer_idx}  [{path}]")

    ckpt       = torch.load(path, map_location=device)
    state_dict = ckpt.get('state_dict', ckpt)

    # ── d_model: always trust the running model, not the checkpoint ──────────
    # Checkpoint may have been saved for a different model size.
    # Using the wrong d_model causes the exact "mat1 and mat2 shapes cannot
    # be multiplied (NxACTUAL and CHECKPOINT_DIMxDICT)" error.
    model_d_model      = get_model_d_model(Config.MODEL_NAME)
    checkpoint_d_model = ckpt.get('d_model', ckpt.get('cfg', {}).get('d_in', None))

    if checkpoint_d_model is not None and checkpoint_d_model != model_d_model:
        print(f"    ⚠  d_model mismatch: checkpoint={checkpoint_d_model}, "
              f"model={model_d_model}. Using model dim ({model_d_model}).")

    d_model = model_d_model  # always use the running model's dim

    sae = AnthropicSAE(d_model, dict_size).to(device)

    if 'W_enc' in state_dict:
        new_sd = {
            'encoder.weight': state_dict['W_enc'].T,
            'encoder.bias':   state_dict['b_enc'],
            'decoder.weight': state_dict['W_dec'].T,
        }
        try:
            sae.load_state_dict(new_sd, strict=True)
        except Exception:
            sae = SparseAutoencoder(d_model, dict_size).to(device)
            new_sd['decoder.bias'] = state_dict.get('b_dec', torch.zeros(d_model))
            sae.load_state_dict(new_sd, strict=True)
    elif 'encoder.weight' in state_dict:
        if 'decoder.bias' in state_dict:
            sae = SparseAutoencoder(d_model, dict_size).to(device)
        sae.load_state_dict(state_dict, strict=True)
    else:
        raise ValueError(f"Unknown SAE checkpoint format. Keys: {list(state_dict.keys())}")

    sae.eval()
    print(f"    SAE loaded")
    return sae


# ==============================================================================
# DATA
# ==============================================================================

def load_tofu_data(author_indices, tokenizer, batch_size=8, max_length=512):
    forget_full = get_tofudataset(Config.FORGET_SPLIT)
    retain_ds   = get_tofudataset(Config.RETAIN_SPLIT)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    forget_ds = forget_full.select(author_indices)

    def tok(x): return tokenize_function(x, tokenizer, max_length)

    retain_tok = retain_ds.map(tok, batched=True)
    forget_tok = forget_ds.map(tok, batched=True)

    for ds in [retain_tok, forget_tok]:
        ds.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    forget_loader = DataLoader(forget_tok, batch_size=batch_size,
                               shuffle=False, collate_fn=collator)
    retain_loader = DataLoader(retain_tok, batch_size=batch_size,
                               shuffle=False, collate_fn=collator)
    return forget_loader, retain_loader


# ==============================================================================
# METRICS
# ==============================================================================

def compute_loss(model, loader, device, max_batches=None):
    """
    Average cross-entropy loss over loader.
    Computes loss manually from logits to avoid ForCausalLMLoss
    dimension errors in newer transformers versions.
    """
    total, n = 0.0, 0
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Get raw logits only — no labels passed to avoid loss_type issues
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask)
            logits  = outputs.logits  # (B, T, vocab_size)

            # Causal shift: predict position t+1 from position t
            shift_logits = logits[:, :-1, :].contiguous()  # (B, T-1, vocab)
            shift_labels = input_ids[:, 1:].contiguous()   # (B, T-1)

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction='mean',
            )
            total += loss.item() * input_ids.shape[0]
            n     += input_ids.shape[0]

    return total / n if n > 0 else 0.0


def loss_to_ppl(loss): return float(np.exp(loss))


# ==============================================================================
# ACTIVATION EXTRACTION
# ==============================================================================

def extract_activations(model, sae, loader, layer_idx, device, max_batches=10):
    """Extract per-token SAE feature activations from a given layer."""
    all_feats = []

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        inputs   = {k: v.to(device) for k, v in batch.items()
                    if k in ['input_ids', 'attention_mask']}
        captured = []

        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            captured.append(h.detach().cpu())

        handle = model.transformer.h[layer_idx].register_forward_hook(hook)
        with torch.no_grad():
            model(**inputs)
        handle.remove()

        if not captured:
            continue

        acts = captured[0]                      # (B, T, D)
        mask = batch['attention_mask'].cpu()    # (B, T)

        for b in range(acts.shape[0]):
            valid = acts[b][mask[b].bool()]     # (valid_T, D)
            with torch.no_grad():
                feats = sae.encode(valid.to(device))
            all_feats.append(feats.cpu())

    if not all_feats:
        return torch.empty(0, sae.dict_size)
    return torch.cat(all_feats, dim=0)          # (N_tokens, dict_size)


# ==============================================================================
# FEATURE SELECTION
# ==============================================================================

def select_features(forget_feats, retain_feats, top_k, threshold):
    """
    Select features to ablate using contrast score: mean_forget - mean_retain.

    Steps:
      1. Compute contrast per feature.
      2. If threshold > 0: keep only features where contrast >= threshold.
      3. Among remaining, take top-K by contrast score.

    Returns indices tensor on CPU.
    """
    f_mean = forget_feats.mean(dim=0)   # (dict_size,)
    r_mean = retain_feats.mean(dim=0)
    contrast = f_mean - r_mean           # higher = more forget-specific

    # Gate by threshold first
    if threshold > 0.0:
        above = (contrast >= threshold).nonzero(as_tuple=True)[0]
        if len(above) == 0:
            print(f"      ⚠ No features above threshold {threshold:.4f} — "
                  f"falling back to pure top-K")
        else:
            contrast_filtered = torch.full_like(contrast, -1e9)
            contrast_filtered[above] = contrast[above]
            contrast = contrast_filtered

    k = min(top_k, (contrast > -1e8).sum().item())
    _, top_idx = torch.topk(contrast, k=k)

    # Diagnostics
    selected_contrast = contrast[top_idx]
    print(f"      Features selected : {k}")
    print(f"      Contrast  min/mean/max : "
          f"{selected_contrast.min():.4f} / "
          f"{selected_contrast.mean():.4f} / "
          f"{selected_contrast.max():.4f}")

    return top_idx.cpu()


# ==============================================================================
# ABLATION HOOKS
# ==============================================================================

def _patch_hidden(h, sae, target_features, device, retain_means=None):
    """
    Core patching logic shared by both hooks.

    h              : (B, T, D) hidden state tensor
    target_features: 1D LongTensor of feature indices (CPU)
    retain_means   : (dict_size,) tensor — if provided, use mean ablation
                     if None, use zero ablation
    Returns patched hidden state same shape as h.
    """
    orig_shape = h.shape                        # (B, T, D)
    flat = h.reshape(-1, h.shape[-1])           # (B*T, D)

    with torch.no_grad():
        feats = sae.encode(flat)                # (B*T, dict_size)
        idx   = target_features.to(device)     # (K,)  — always 1D

        if retain_means is None:
            # Zero ablation
            feats[:, idx] = 0.0
        else:
            # Mean ablation — broadcast means across all tokens
            feats[:, idx] = retain_means[idx].unsqueeze(0)  # (1, K) → broadcast

        recon = sae.decode(feats)               # (B*T, D)

    return recon.reshape(orig_shape)            # (B, T, D)


def make_zero_hook(sae, target_features, device):
    """
    Hook that zeros out selected SAE features.
    Returns ONLY the modified hidden state; preserves all other tuple elements.
    """
    def hook(module, inp, out):
        # GPT-2 Block returns (hidden_states, present) or
        # (hidden_states, present, attentions) depending on config.
        # We only modify hidden_states (out[0]).
        if isinstance(out, tuple):
            h    = out[0]                               # (B, T, D)
            rest = out[1:]
            patched = _patch_hidden(h, sae, target_features, device)
            return (patched,) + rest
        else:
            return _patch_hidden(out, sae, target_features, device)
    return hook


def make_mean_hook(sae, target_features, retain_feature_means, device):
    """
    Hook that replaces selected SAE features with retain-set mean activations.
    Gentler than zero ablation — better retain PPL preservation.
    """
    means = retain_feature_means.to(device)    # (dict_size,)

    def hook(module, inp, out):
        if isinstance(out, tuple):
            h    = out[0]
            rest = out[1:]
            patched = _patch_hidden(h, sae, target_features, device,
                                    retain_means=means)
            return (patched,) + rest
        else:
            return _patch_hidden(out, sae, target_features, device,
                                 retain_means=means)
    return hook


# ==============================================================================
# SAE-GUIDED UNLEARNING
# ==============================================================================

def run_sae_unlearning(model, sae, forget_loader, retain_loader,
                       layer_idx, device, top_k, threshold, ablation_mode):
    """
    Full SAE-guided unlearning pipeline.

    ablation_mode: 'zero' | 'mean'
    Returns dict with PPL and loss metrics.
    """

    # ── Step 1: Extract activations ───────────────────────────────────────────
    print(f"      Extracting forget activations...")
    forget_feats = extract_activations(model, sae, forget_loader,
                                       layer_idx, device, max_batches=10)
    print(f"      Extracting retain activations...")
    retain_feats = extract_activations(model, sae, retain_loader,
                                       layer_idx, device, max_batches=10)

    if forget_feats.shape[0] == 0 or retain_feats.shape[0] == 0:
        print("      ❌ Empty activations — skipping")
        return None

    # ── Step 2: Select features ───────────────────────────────────────────────
    print(f"      Selecting features (top_k={top_k}, threshold={threshold}, "
          f"mode={ablation_mode})...")
    target_features = select_features(forget_feats, retain_feats, top_k, threshold)

    if len(target_features) == 0:
        print("      ❌ No features selected — skipping")
        return None

    # ── Step 3: Build hook ────────────────────────────────────────────────────
    if ablation_mode == 'zero':
        hook_fn = make_zero_hook(sae, target_features, device)

    elif ablation_mode == 'mean':
        retain_means = retain_feats.mean(dim=0)   # (dict_size,)
        hook_fn = make_mean_hook(sae, target_features, retain_means, device)
    else:
        raise ValueError(f"Unknown ablation_mode: {ablation_mode}")

    # ── Step 4: Evaluate before ───────────────────────────────────────────────
    forget_loss_before  = compute_loss(model, forget_loader, device)
    retain_loss_before  = compute_loss(model, retain_loader, device)
    forget_ppl_before   = loss_to_ppl(forget_loss_before)
    retain_ppl_before   = loss_to_ppl(retain_loss_before)

    # ── Step 5: Evaluate after (with hook) ────────────────────────────────────
    handle = model.transformer.h[layer_idx].register_forward_hook(hook_fn)
    forget_loss_after  = compute_loss(model, forget_loader, device)
    retain_loss_after  = compute_loss(model, retain_loader, device)
    handle.remove()

    forget_ppl_after   = loss_to_ppl(forget_loss_after)
    retain_ppl_after   = loss_to_ppl(retain_loss_after)

    # ── Step 6: Compute metrics ───────────────────────────────────────────────
    forget_ppl_increase = forget_ppl_after  - forget_ppl_before
    retain_ppl_change   = retain_ppl_after  - retain_ppl_before
    forget_loss_increase = forget_loss_after - forget_loss_before
    retain_loss_change   = retain_loss_after  - retain_loss_before

    eps = 1e-8
    selectivity = forget_ppl_increase / (abs(retain_ppl_change) + eps)

    return {
        # PPL metrics (primary)
        'forget_ppl_before':   forget_ppl_before,
        'forget_ppl_after':    forget_ppl_after,
        'forget_ppl_increase': forget_ppl_increase,
        'retain_ppl_before':   retain_ppl_before,
        'retain_ppl_after':    retain_ppl_after,
        'retain_ppl_change':   retain_ppl_change,
        'selectivity':         selectivity,
        # Loss metrics (secondary)
        'forget_loss_before':   forget_loss_before,
        'forget_loss_after':    forget_loss_after,
        'forget_loss_increase': forget_loss_increase,
        'retain_loss_change':   retain_loss_change,
        # Feature info
        'n_features_ablated':   len(target_features),
        'ablation_mode':        ablation_mode,
    }


# ==============================================================================
# GRADIENT ASCENT BASELINE
# ==============================================================================

def compute_loss_with_grad(model, batch, device):
    """
    Compute causal LM loss with gradients enabled.
    Used inside GA training loop — does NOT use torch.no_grad().
    """
    input_ids      = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)

    outputs      = model(input_ids=input_ids, attention_mask=attention_mask)
    logits       = outputs.logits                           # (B, T, vocab)
    shift_logits = logits[:, :-1, :].contiguous()          # (B, T-1, vocab)
    shift_labels = input_ids[:, 1:].contiguous()           # (B, T-1)

    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction='mean',
    )


def run_gradient_ascent(model, forget_loader, retain_loader, device):
    """
    Gradient ascent baseline.
    Deep-copies the base model and re-enables gradients on the copy
    so GA training works even when the base model is frozen.
    """
    model_copy = copy.deepcopy(model).to(device)

    # Re-enable gradients — base model is frozen but copy must be trainable
    for param in model_copy.parameters():
        param.requires_grad_(True)

    optimizer = torch.optim.Adam(model_copy.parameters(),
                                 lr=Config.GA_LEARNING_RATE)

    # Baseline (no_grad eval)
    forget_loss_before = compute_loss(model_copy, forget_loader, device)
    retain_loss_before = compute_loss(model_copy, retain_loader, device)

    # Ascent
    model_copy.train()
    forget_iter = iter(forget_loader)
    for _ in range(Config.GA_STEPS):
        try:
            batch = next(forget_iter)
        except StopIteration:
            forget_iter = iter(forget_loader)
            batch = next(forget_iter)

        loss = -compute_loss_with_grad(model_copy, batch, device)  # maximize

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_copy.parameters(), 1.0)
        optimizer.step()

    # After (no_grad eval)
    model_copy.eval()
    forget_loss_after = compute_loss(model_copy, forget_loader, device)
    retain_loss_after = compute_loss(model_copy, retain_loader, device)

    del model_copy
    torch.cuda.empty_cache()

    forget_ppl_before   = loss_to_ppl(forget_loss_before)
    retain_ppl_before   = loss_to_ppl(retain_loss_before)
    forget_ppl_after    = loss_to_ppl(forget_loss_after)
    retain_ppl_after    = loss_to_ppl(retain_loss_after)
    forget_ppl_increase = forget_ppl_after  - forget_ppl_before
    retain_ppl_change   = retain_ppl_after  - retain_ppl_before

    eps = 1e-8
    selectivity = forget_ppl_increase / (abs(retain_ppl_change) + eps)

    return {
        'forget_ppl_before':   forget_ppl_before,
        'forget_ppl_after':    forget_ppl_after,
        'forget_ppl_increase': forget_ppl_increase,
        'retain_ppl_before':   retain_ppl_before,
        'retain_ppl_after':    retain_ppl_after,
        'retain_ppl_change':   retain_ppl_change,
        'selectivity':         selectivity,
    }


# ==============================================================================
# PER-AUTHOR RUNNER
# ==============================================================================

def run_author(author_name, author_indices, group,
               model, sae, tokenizer, layer_idx, dict_size,
               top_k, threshold, ablation_mode, device):

    print(f"\n      [{group}] {author_name}")

    try:
        forget_loader, retain_loader = load_tofu_data(author_indices, tokenizer)
    except Exception as e:
        print(f"        ❌ Data load failed: {e}")
        return None

    # SAE unlearning
    try:
        sae_res = run_sae_unlearning(
            model, sae, forget_loader, retain_loader,
            layer_idx, device, top_k, threshold, ablation_mode
        )
        if sae_res is None:
            return None
    except Exception as e:
        print(f"        ❌ SAE unlearning failed: {e}")
        return None

    # GA baseline
    try:
        ga_res = run_gradient_ascent(model, forget_loader, retain_loader, device)
    except Exception as e:
        print(f"        ❌ GA failed: {e}")
        return None

    # ── Debug print ───────────────────────────────────────────────────────────
    print(f"\n        --- DEBUG ---")
    print(f"        SAE forget PPL : {sae_res['forget_ppl_before']:.2f} → "
          f"{sae_res['forget_ppl_after']:.2f}  "
          f"(+{sae_res['forget_ppl_increase']:.2f})")
    print(f"        SAE retain PPL : {sae_res['retain_ppl_before']:.2f} → "
          f"{sae_res['retain_ppl_after']:.2f}  "
          f"({sae_res['retain_ppl_change']:+.2f})")
    print(f"        SAE selectivity: {sae_res['selectivity']:.4f}")
    print(f"        GA  forget PPL : {ga_res['forget_ppl_before']:.2f} → "
          f"{ga_res['forget_ppl_after']:.2f}  "
          f"(+{ga_res['forget_ppl_increase']:.2f})")
    print(f"        GA  retain PPL : {ga_res['retain_ppl_before']:.2f} → "
          f"{ga_res['retain_ppl_after']:.2f}  "
          f"({ga_res['retain_ppl_change']:+.2f})")
    print(f"        GA  selectivity: {ga_res['selectivity']:.4f}")
    print(f"        --- END DEBUG ---")

    return {
        'author':         author_name,
        'group':          group,
        'dict_size':      dict_size,
        'layer':          layer_idx,
        'ablation_mode':  ablation_mode,
        'top_k_features': top_k,
        'threshold':      threshold,

        # SAE PPL
        'sae_forget_ppl_before':   sae_res['forget_ppl_before'],
        'sae_forget_ppl_after':    sae_res['forget_ppl_after'],
        'sae_forget_ppl_increase': sae_res['forget_ppl_increase'],
        'sae_retain_ppl_before':   sae_res['retain_ppl_before'],
        'sae_retain_ppl_after':    sae_res['retain_ppl_after'],
        'sae_retain_ppl_change':   sae_res['retain_ppl_change'],
        'sae_selectivity':         sae_res['selectivity'],
        'sae_n_features':          sae_res['n_features_ablated'],

        # GA PPL
        'ga_forget_ppl_before':   ga_res['forget_ppl_before'],
        'ga_forget_ppl_after':    ga_res['forget_ppl_after'],
        'ga_forget_ppl_increase': ga_res['forget_ppl_increase'],
        'ga_retain_ppl_before':   ga_res['retain_ppl_before'],
        'ga_retain_ppl_after':    ga_res['retain_ppl_after'],
        'ga_retain_ppl_change':   ga_res['retain_ppl_change'],
        'ga_selectivity':         ga_res['selectivity'],

        # Benefit = how much better SAE retain is vs GA retain
        'retain_ppl_benefit': ga_res['retain_ppl_change'] - sae_res['retain_ppl_change'],
        'selectivity_gain':   sae_res['selectivity'] - ga_res['selectivity'],
    }


# ==============================================================================
# CONFIG RUNNER
# ==============================================================================

def run_config(dict_size, layer_idx, model, tokenizer, author_to_samples,
               top_k, threshold, ablation_mode, date_str):

    print(f"\n{'=' * 76}")
    print(f"  CONFIG  dict={dict_size}  layer={layer_idx}  "
          f"mode={ablation_mode}  top_k={top_k}  threshold={threshold}")
    print(f"{'=' * 76}")

    out_dir = (Path(Config.RESULTS_BASE_DIR)
               / f"dict_{dict_size}"
               / f"layer_{layer_idx:02d}"
               / ablation_mode)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        sae = load_sae(dict_size, layer_idx, Config.DEVICE)
    except Exception as e:
        print(f"  ❌ SAE load failed: {e}")
        return None

    all_authors = (
        [(a, 'high_ed') for a in Config.HIGH_ED_AUTHORS] +
        [(a, 'low_ed')  for a in Config.LOW_ED_AUTHORS]
    )

    results = []
    for author_name, group in all_authors:
        if author_name not in author_to_samples:
            print(f"    ⚠ {author_name} not in mapping — skipping")
            continue

        row = run_author(
            author_name, author_to_samples[author_name], group,
            model, sae, tokenizer, layer_idx, dict_size,
            top_k, threshold, ablation_mode, Config.DEVICE
        )
        if row:
            results.append(row)

    if not results:
        return None

    df = pd.DataFrame(results)
    out_path = out_dir / f"results_topk{top_k}_thresh{threshold}_{date_str}.csv"
    df.to_csv(out_path, index=False, float_format='%.6f')
    print(f"\n  Saved: {out_path}")

    # Quick summary
    print(f"\n  SUMMARY (n={len(df)})")
    print(f"  SAE  forget PPL increase : {df['sae_forget_ppl_increase'].mean():.2f} "
          f"± {df['sae_forget_ppl_increase'].std():.2f}")
    print(f"  SAE  retain PPL change   : {df['sae_retain_ppl_change'].mean():.2f} "
          f"± {df['sae_retain_ppl_change'].std():.2f}")
    print(f"  SAE  selectivity         : {df['sae_selectivity'].mean():.4f} "
          f"± {df['sae_selectivity'].std():.4f}")
    print(f"  GA   forget PPL increase : {df['ga_forget_ppl_increase'].mean():.2f} "
          f"± {df['ga_forget_ppl_increase'].std():.2f}")
    print(f"  GA   retain PPL change   : {df['ga_retain_ppl_change'].mean():.2f} "
          f"± {df['ga_retain_ppl_change'].std():.2f}")
    print(f"  GA   selectivity         : {df['ga_selectivity'].mean():.4f} "
          f"± {df['ga_selectivity'].std():.4f}")
    print(f"  Selectivity gain (SAE-GA): {df['selectivity_gain'].mean():.4f}")

    del sae
    torch.cuda.empty_cache()
    return df


# ==============================================================================
# MAIN
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="SAE-guided unlearning — GPT-2 Small/Medium"
    )

    # Model
    parser.add_argument('--model_name',    default=Config.MODEL_NAME)
    parser.add_argument('--n_layers',      type=int, default=Config.N_LAYERS)

    # SAE sweep
    parser.add_argument('--dict_sizes',    type=int, nargs='+',
                        default=Config.DICT_SIZES)
    parser.add_argument('--layers',        type=int, nargs='+',
                        default=Config.LAYERS)
    parser.add_argument('--sae_base_path', default=Config.SAE_BASE_PATH)
    parser.add_argument('--results_dir',   default=Config.RESULTS_BASE_DIR)

    # ── Feature selection ─────────────────────────────────────────────────────
    parser.add_argument('--top_k_features',     type=int,   default=Config.TOP_K_FEATURES,
                        help='Number of SAE features to ablate per author '
                             '(try 64, 128, 256, 512, 1024)')
    parser.add_argument('--contrast_threshold', type=float, default=Config.CONTRAST_THRESHOLD,
                        help='Min contrast score to include a feature. '
                             '0.0 = pure top-K. '
                             'Positive values (e.g. 0.01) gate features '
                             'before taking top-K.')

    # ── Ablation mode ─────────────────────────────────────────────────────────
    parser.add_argument('--ablation_mode', choices=['zero', 'mean'],
                        default=Config.ABLATION_MODE,
                        help='"zero": set selected features to 0. '
                             '"mean": replace with retain-set mean activation.')

    # GA
    parser.add_argument('--ga_steps', type=int, default=Config.GA_STEPS)
    parser.add_argument('--ga_lr',    type=float, default=Config.GA_LEARNING_RATE)

    return parser.parse_args()


def main():
    args = parse_args()

    # Apply CLI overrides to Config
    Config.MODEL_NAME        = args.model_name
    Config.N_LAYERS          = args.n_layers
    Config.DICT_SIZES        = args.dict_sizes
    Config.LAYERS            = args.layers
    Config.SAE_BASE_PATH     = args.sae_base_path
    Config.RESULTS_BASE_DIR  = args.results_dir
    Config.TOP_K_FEATURES    = args.top_k_features
    Config.CONTRAST_THRESHOLD = args.contrast_threshold
    Config.ABLATION_MODE     = args.ablation_mode
    Config.GA_STEPS          = args.ga_steps
    Config.GA_LEARNING_RATE  = args.ga_lr

    date_str = Config.date_str()

    print("\n" + "=" * 80)
    print("SAE-GUIDED UNLEARNING")
    print("=" * 80)
    print(f"  Model           : {Config.MODEL_NAME}")
    print(f"  Dict sizes      : {Config.DICT_SIZES}")
    print(f"  Layers          : {Config.LAYERS}")
    print(f"  Ablation mode   : {Config.ABLATION_MODE}")
    print(f"  Top-K features  : {Config.TOP_K_FEATURES}")
    print(f"  Threshold       : {Config.CONTRAST_THRESHOLD}")
    print(f"  Device          : {Config.DEVICE}")
    if Config.DEVICE == 'cuda':
        print(f"  GPU             : {torch.cuda.get_device_name(0)}")

    # Load model
    print("\nLoading model...")
    model, tokenizer = get_gptmodel(Config.MODEL_NAME)
    model.to(Config.DEVICE)
    model.eval()

    # Freeze base model — never modify it
    for param in model.parameters():
        param.requires_grad_(False)

    # Load author mapping
    with open('../data/tofu_author_mapping.json', 'r') as f:
        author_to_samples = json.load(f)['author_to_samples']
    print(f"Authors loaded: {len(author_to_samples)}")

    total     = len(Config.DICT_SIZES) * len(Config.LAYERS)
    completed = 0
    all_dfs   = []
    t0        = time.time()

    for dict_size in Config.DICT_SIZES:
        for layer_idx in Config.LAYERS:
            completed += 1
            print(f"\n{'#' * 80}")
            print(f"[{completed}/{total}]")

            df = run_config(
                dict_size, layer_idx, model, tokenizer, author_to_samples,
                Config.TOP_K_FEATURES, Config.CONTRAST_THRESHOLD,
                Config.ABLATION_MODE, date_str
            )
            if df is not None:
                all_dfs.append(df)

            elapsed   = time.time() - t0
            remaining = (elapsed / completed) * (total - completed)
            print(f"\n  Progress {completed}/{total} — "
                  f"elapsed {elapsed/60:.1f} min — "
                  f"ETA {remaining/60:.1f} min")

    # Save combined
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        out = (Path(Config.RESULTS_BASE_DIR) /
               f"all_results_{Config.ABLATION_MODE}_"
               f"topk{Config.TOP_K_FEATURES}_"
               f"thresh{Config.CONTRAST_THRESHOLD}_{date_str}.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out, index=False, float_format='%.6f')
        print(f"\n  Combined results saved: {out}")

    total_time = time.time() - t0
    print(f"\n{'=' * 80}")
    print(f"DONE — {total_time/60:.1f} minutes")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
