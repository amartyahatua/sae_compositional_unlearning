"""
RECOVER MSE FROM ALREADY-TRAINED SAE CHECKPOINTS
=================================================
Loads saved SAE weights, extracts fresh activations from the base model,
and computes reconstruction quality metrics — no retraining needed.

Usage:
    python recover_sae_metrics.py
"""

import torch
import torch.nn.functional as F
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from get_dataset import get_tofudataset, tokenize_function
from get_model import get_gptmodel


# ============================================================
# CONFIG — match your training config
# ============================================================

class Config:
    MODEL_NAME    = 'gpt2-medium'
    D_MODEL       = 1024             # GPT-2 Medium hidden dim
    DICT_SIZES    = [65536]          # as per your Medium training
    LAYERS        = list(range(24))  # 0-23 for Medium
    SAE_BASE_PATH = '../models'
    RESULTS_PATH  = '../results/sae_training_simple/recovered_metrics_medium.csv'
    MAX_LENGTH    = 512
    BATCH_SIZE    = 8
    MAX_BATCHES   = 50
    DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================
# SAE MODEL (same architecture as training)
# ============================================================

class AnthropicSAE(torch.nn.Module):
    def __init__(self, d_model, dict_size):
        super().__init__()
        self.encoder = torch.nn.Linear(d_model, dict_size)
        self.decoder = torch.nn.Linear(dict_size, d_model, bias=False)

    def forward(self, x):
        acts = F.relu(self.encoder(x))
        recon = self.decoder(acts)
        return recon, acts


# ============================================================
# ACTIVATION EXTRACTION
# ============================================================

@torch.no_grad()
def extract_activations(model, loader, layer_idx, device, max_batches=50):
    model.eval()
    all_acts = []

    for i, batch in enumerate(tqdm(loader, desc=f'Layer {layer_idx}', leave=False)):
        if i >= max_batches:
            break
        input_ids = batch['input_ids'].to(device)
        attn_mask = batch['attention_mask'].to(device)

        out = model(input_ids=input_ids,
                    attention_mask=attn_mask,
                    output_hidden_states=True)

        h = out.hidden_states[layer_idx + 1]  # (B, T, D)

        for b in range(h.size(0)):
            valid = attn_mask[b].bool()
            all_acts.append(h[b, valid].cpu())

    return torch.cat(all_acts, dim=0)   # (N_tokens, D)


# ============================================================
# METRIC COMPUTATION
# ============================================================

@torch.no_grad()
def compute_metrics(sae, activations, device, sample_size=4096):
    sae.eval()

    # Use a random sample for efficiency
    n = min(sample_size, activations.shape[0])
    idx = torch.randperm(activations.shape[0])[:n]
    x   = activations[idx].to(device)

    recon, acts = sae(x)

    rel_mse   = (F.mse_loss(recon, x) / x.var()).item()
    cosine    = F.cosine_similarity(x, recon, dim=-1).mean().item()
    l0_frac   = (acts > 0).float().mean().item()
    l0_mean   = (acts > 0).float().sum(dim=-1).mean().item()
    dead_pct  = ((acts > 0).float().sum(dim=0) == 0).float().mean().item()

    # Relative MSE quality gate
    if rel_mse < 0.1:
        quality = 'excellent'
    elif rel_mse < 0.2:
        quality = 'acceptable'
    else:
        quality = 'poor'

    return {
        'train_rel_mse':  rel_mse,
        'train_cosine':   cosine,
        'train_l0_frac':  l0_frac,
        'train_l0_mean':  l0_mean,
        'train_dead_pct': dead_pct,
        'quality':        quality,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print('=' * 70)
    print('RECOVERING SAE METRICS FROM SAVED CHECKPOINTS')
    print('=' * 70)
    print(f'  Model      : {Config.MODEL_NAME}')
    print(f'  Dict sizes : {Config.DICT_SIZES}')
    print(f'  Layers     : {Config.LAYERS}')
    print(f'  Device     : {Config.DEVICE}')

    # ── Load base model ───────────────────────────────────────────────────────
    print('\nLoading base model...')
    model, tokenizer = get_gptmodel(Config.MODEL_NAME)
    model.to(Config.DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # ── Prepare dataloader ────────────────────────────────────────────────────
    print('Preparing data...')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = get_tofudataset('retain90')
    dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, Config.MAX_LENGTH),
        batched=True
    )
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    loader   = DataLoader(dataset, batch_size=Config.BATCH_SIZE,
                          shuffle=False, collate_fn=collator)

    all_results = []
    sae_base = Path(Config.SAE_BASE_PATH)

    # ── Loop over all checkpoints ─────────────────────────────────────────────
    for dict_size in Config.DICT_SIZES:
        print(f'\n{"="*70}')
        print(f'Dict size: {dict_size}')
        print(f'{"="*70}')

        # Cache activations per layer — extract once, reuse across dict sizes
        layer_acts_cache = {}

        for layer_idx in Config.LAYERS:
            ckpt_path = sae_base / f'dict_{dict_size}' / f'layer_{layer_idx}.pt'

            if not ckpt_path.exists():
                print(f'  ⚠ Missing: {ckpt_path}')
                continue

            # Extract activations for this layer (cache to avoid re-extraction)
            if layer_idx not in layer_acts_cache:
                acts = extract_activations(
                    model, loader, layer_idx, Config.DEVICE, Config.MAX_BATCHES
                )
                layer_acts_cache[layer_idx] = acts
            else:
                acts = layer_acts_cache[layer_idx]

            # Load SAE
            ckpt = torch.load(ckpt_path, map_location=Config.DEVICE)
            sae  = AnthropicSAE(Config.D_MODEL, dict_size).to(Config.DEVICE)
            sae.load_state_dict(ckpt['state_dict'])
            sae.eval()

            # Compute metrics
            metrics = compute_metrics(sae, acts, Config.DEVICE)

            row = {
                'dict_size': dict_size,
                'layer':     layer_idx,
                **metrics,
            }
            all_results.append(row)

            print(f'  Layer {layer_idx:2d}: '
                  f'RelMSE={metrics["train_rel_mse"]:.4f} [{metrics["quality"]}]  '
                  f'Cos={metrics["train_cosine"]:.4f}  '
                  f'L0={metrics["train_l0_frac"]*100:.1f}%  '
                  f'Dead={metrics["train_dead_pct"]*100:.1f}%')

            # Update checkpoint with metrics (optional — adds metadata to .pt file)
            ckpt.update({
                'train_rel_mse':  metrics['train_rel_mse'],
                'train_cosine':   metrics['train_cosine'],
                'train_l0_frac':  metrics['train_l0_frac'],
                'train_l0_mean':  metrics['train_l0_mean'],
                'train_dead_pct': metrics['train_dead_pct'],
            })
            torch.save(ckpt, ckpt_path)

            del sae
            torch.cuda.empty_cache()

    # ── Save results ──────────────────────────────────────────────────────────
    df = pd.DataFrame(all_results)
    out = Path(Config.RESULTS_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, float_format='%.6f')

    print(f'\n{"="*70}')
    print('SUMMARY')
    print(f'{"="*70}')
    print(f'\nTotal SAEs evaluated: {len(df)}')
    for q in ['excellent', 'acceptable', 'poor']:
        n = (df['quality'] == q).sum()
        print(f'  {q:10s}: {n}')

    print(f'\nBest per dict size (lowest RelMSE):')
    for d in Config.DICT_SIZES:
        sub = df[df['dict_size'] == d]
        if len(sub) == 0:
            continue
        best = sub.loc[sub['train_rel_mse'].idxmin()]
        print(f'  dict={d:6d}: layer={int(best["layer"]):2d}  '
              f'RelMSE={best["train_rel_mse"]:.4f}  [{best["quality"]}]')

    print(f'\nSaved to: {out}')


if __name__ == '__main__':
    main()