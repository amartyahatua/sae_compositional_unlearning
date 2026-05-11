"""
RECOVER MSE FROM ALREADY-TRAINED SAE CHECKPOINTS (Llama 3.2-1B)
================================================================
Loads saved SAE weights, extracts fresh activations from Llama 3.2-1B,
and computes reconstruction quality metrics — no retraining needed.

Architecture differences vs. GPT-2 Large:
    GPT-2 Large : 36 layers,  d_model = 1280
    Llama 3.2-1B: 16 layers,  d_model = 2048

Note: Llama 3.2 is a *gated* model on Hugging Face. You'll need to:
    1. Accept the license at https://huggingface.co/meta-llama/Llama-3.2-1B
    2. Run `huggingface-cli login` with your HF token

Usage:
    python recover_sae_metrics_llama.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling

from get_dataset import get_tofudataset, tokenize_function


# ============================================================
# CONFIG — Llama 3.2-1B specifics
# ============================================================

class Config:
    # Model
    MODEL_NAME    = 'meta-llama/Llama-3.2-1B'     # or your TOFU-finetuned path
    D_MODEL       = 2048                          # Llama 3.2-1B hidden dimension
    N_LAYERS      = 16                            # Llama 3.2-1B transformer layers

    # SAE sweep
    DICT_SIZES    = [65536]                       # match your trained checkpoints
    LAYERS        = list(range(N_LAYERS))         # 0–15

    # Paths
    SAE_BASE_PATH = 'model'                       # where checkpoints live
    RESULTS_PATH  = '../results/sae_training_simple/recovered_metrics_llama32_1b.csv'

    # Tokenization / batching
    MAX_LENGTH    = 1024                          # Llama 3.2 supports 8K+, but 1024 is plenty for TOFU
    BATCH_SIZE    = 2                             # smaller than GPT-2 Large because Llama 3.2-1B is heavier
    MAX_BATCHES   = 50

    # Hardware
    DEVICE        = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    MODEL_DTYPE   = torch.bfloat16 if torch.cuda.is_available() else torch.float32


# ============================================================
# SAE MODEL (same architecture as training)
# ============================================================

class AnthropicSAE(nn.Module):
    """Standard SAE architecture."""

    def __init__(self, d_model, dict_size):
        super().__init__()
        self.d_model = d_model
        self.dict_size = dict_size

        self.encoder = nn.Linear(d_model, dict_size, bias=True)
        self.decoder = nn.Linear(dict_size, d_model, bias=True)

    def forward(self, x):
        pre_activation = self.encoder(x)
        feature_acts = torch.relu(pre_activation)
        x_reconstruct = self.decoder(feature_acts)
        return x_reconstruct, feature_acts


# ============================================================
# MODEL LOADING (Llama 3.2-1B)
# ============================================================

def load_llama_model(model_name, dtype, device):
    """
    Load Llama 3.2-1B and its tokenizer.

    Uses AutoModelForCausalLM directly so this works whether you point
    MODEL_NAME at the HF hub ('meta-llama/Llama-3.2-1B') or a local
    TOFU-finetuned checkpoint directory.
    """
    print(f'Loading tokenizer: {model_name}')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f'Loading model: {model_name}  (dtype={dtype})')
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer


# ============================================================
# ACTIVATION EXTRACTION
# ============================================================

@torch.no_grad()
def extract_activations(model, loader, layer_idx, device, max_batches=50):
    """
    Pull residual-stream activations after transformer block `layer_idx`.

    hidden_states convention (same in GPT-2 and Llama):
        hidden_states[0]           = embeddings
        hidden_states[layer_idx+1] = output of block layer_idx
    """
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

        h = out.hidden_states[layer_idx + 1]      # (B, T, D)

        # Cast back to float32 before saving — bf16 activations lose precision
        # in the SAE forward pass, which inflates RelMSE artificially.
        h = h.to(torch.float32)

        for b in range(h.size(0)):
            valid = attn_mask[b].bool()
            all_acts.append(h[b, valid].cpu())

    return torch.cat(all_acts, dim=0)             # (N_tokens, D_model)


# ============================================================
# METRIC COMPUTATION
# ============================================================

@torch.no_grad()
def compute_metrics(sae, activations, device, sample_size=4096):
    sae.eval()

    n = min(sample_size, activations.shape[0])
    idx = torch.randperm(activations.shape[0])[:n]
    x   = activations[idx].to(device).to(torch.float32)

    recon, acts = sae(x)

    rel_mse   = (F.mse_loss(recon, x) / x.var()).item()
    cosine    = F.cosine_similarity(x, recon, dim=-1).mean().item()
    l0_frac   = (acts > 0).float().mean().item()
    l0_mean   = (acts > 0).float().sum(dim=-1).mean().item()
    dead_pct  = ((acts > 0).float().sum(dim=0) == 0).float().mean().item()

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


def clear_device_cache(device):
    if device == 'cuda':
        torch.cuda.empty_cache()
    elif device == 'mps':
        torch.mps.empty_cache()


# ============================================================
# MAIN
# ============================================================

def main():
    print('=' * 70)
    print('RECOVERING SAE METRICS FROM SAVED CHECKPOINTS — Llama 3.2-1B')
    print('=' * 70)
    print(f'  Model      : {Config.MODEL_NAME}')
    print(f'  D_model    : {Config.D_MODEL}')
    print(f'  Layers     : {Config.LAYERS}')
    print(f'  Dict sizes : {Config.DICT_SIZES}')
    print(f'  Device     : {Config.DEVICE}')
    print(f'  Model dtype: {Config.MODEL_DTYPE}')

    # ── Load base model ───────────────────────────────────────────────────────
    model, tokenizer = load_llama_model(
        Config.MODEL_NAME, Config.MODEL_DTYPE, Config.DEVICE
    )

    # Sanity check that the model dimension matches what the SAEs expect
    detected_d_model = model.config.hidden_size
    assert detected_d_model == Config.D_MODEL, (
        f'Mismatch: model.config.hidden_size={detected_d_model} '
        f'but Config.D_MODEL={Config.D_MODEL}. Update D_MODEL.'
    )
    detected_n_layers = model.config.num_hidden_layers
    assert detected_n_layers == Config.N_LAYERS, (
        f'Mismatch: model.config.num_hidden_layers={detected_n_layers} '
        f'but Config.N_LAYERS={Config.N_LAYERS}. Update N_LAYERS.'
    )
    print(f'  Confirmed  : d_model={detected_d_model}, n_layers={detected_n_layers}')

    # ── Prepare dataloader ────────────────────────────────────────────────────
    print('\nPreparing data...')
    dataset = get_tofudataset('retain90')
    dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, Config.MAX_LENGTH),
        batched=True,
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
            ckpt_path = sae_base / f'dict_{dict_size}' / f'sae_layer_{layer_idx}.pt'

            if not ckpt_path.exists():
                print(f'  ⚠ Missing: {ckpt_path}')
                continue

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

            metrics = compute_metrics(sae, acts, Config.DEVICE)

            row = {
                'model':     Config.MODEL_NAME,
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

            # Cache metrics back into the checkpoint
            ckpt.update({
                'train_rel_mse':  metrics['train_rel_mse'],
                'train_cosine':   metrics['train_cosine'],
                'train_l0_frac':  metrics['train_l0_frac'],
                'train_l0_mean':  metrics['train_l0_mean'],
                'train_dead_pct': metrics['train_dead_pct'],
            })
            torch.save(ckpt, ckpt_path)

            del sae
            clear_device_cache(Config.DEVICE)

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