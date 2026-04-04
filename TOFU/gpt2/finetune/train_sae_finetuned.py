"""
TRAIN SAEs ON FINETUNED GPT-2 ACTIVATIONS
==========================================

Extracts activations from your TOFU-finetuned GPT-2 and trains SAEs
for all dict sizes. Produces checkpoints in the same format as your
existing pipeline: {save_dir}/dict_{size}/layer_{layer}.pt

USAGE:
    # GPT-2 Small, layer 10, all dict sizes
    python train_sae_finetuned.py \
        --model_path ./models/gpt2_tofu_finetuned \
        --layer 10 \
        --save_dir ./models_finetuned

    # Single dict size for quick test
    python train_sae_finetuned.py \
        --model_path ./models/gpt2_tofu_finetuned \
        --layer 10 \
        --dict_sizes 16384 \
        --save_dir ./models_finetuned

    # GPT-2 Medium, layer 20
    python train_sae_finetuned.py \
        --model_path ./models/gpt2-medium_tofu_finetuned \
        --layer 20 \
        --d_model 1024 \
        --save_dir ./models_finetuned_medium

Author: Amartya Hatua
"""

import argparse
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from scipy.stats import pearsonr


# ══════════════════════════════════════════════════════════════
# SAE — same architecture as your existing training
# ══════════════════════════════════════════════════════════════

class AnthropicSAE(nn.Module):
    def __init__(self, d_model, dict_size):
        super().__init__()
        self.encoder = nn.Linear(d_model, dict_size)
        self.decoder = nn.Linear(dict_size, d_model, bias=False)
        nn.init.normal_(self.decoder.weight, std=0.02)

    def forward(self, x):
        acts = F.relu(self.encoder(x))
        recon = self.decoder(acts)
        return recon, acts

    def encode(self, x):
        return F.relu(self.encoder(x))

    def decode(self, z):
        return self.decoder(z)

    @torch.no_grad()
    def normalize_decoder(self):
        """Normalize decoder columns to unit norm."""
        w = self.decoder.weight.data
        norms = w.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.decoder.weight.data = w / norms


# ══════════════════════════════════════════════════════════════
# ACTIVATION EXTRACTION
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_activations(model, loader, layer_idx, device, max_batches=None):
    """
    Extract hidden states from a specific layer.
    Returns: (N_tokens, d_model) tensor on CPU.
    """
    model.eval()
    all_acts = []

    for i, batch in enumerate(tqdm(loader, desc=f"Extracting layer {layer_idx}")):
        if max_batches is not None and i >= max_batches:
            break

        input_ids = batch['input_ids'].to(device)
        attn_mask = batch['attention_mask'].to(device)

        out = model(input_ids=input_ids,
                    attention_mask=attn_mask,
                    output_hidden_states=True)

        # hidden_states[0] = embeddings, [k] = output of layer k
        h = out.hidden_states[layer_idx + 1]  # (B, T, D)

        for b in range(h.size(0)):
            valid = attn_mask[b].bool()
            all_acts.append(h[b, valid].cpu())

    activations = torch.cat(all_acts, dim=0)
    print(f"  Extracted {activations.shape[0]:,} tokens, dim={activations.shape[1]}")
    return activations


# ══════════════════════════════════════════════════════════════
# SAE TRAINING
# ══════════════════════════════════════════════════════════════

def train_sae(activations, d_model, dict_size, device,
              l1_coeff=5e-5, lr=3e-4, batch_size=256,
              epochs=8, warmup_epochs=1):
    """
    Train SAE on extracted activations.
    Matches your existing training pipeline exactly.
    """
    print(f"\n  Training SAE: d_model={d_model}, dict_size={dict_size}")
    print(f"  Tokens: {activations.shape[0]:,}")

    # Adaptive L1
    l1_coeff_scaled = l1_coeff * (4096 / dict_size)

    sae = AnthropicSAE(d_model, dict_size).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)

    loader = DataLoader(
        TensorDataset(activations),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device == "cuda" else False,
    )

    for epoch in range(epochs):
        sae.train()
        epoch_loss = 0
        n_batches = 0

        for (x,) in loader:
            x = x.to(device)

            recon, acts = sae(x)
            recon_loss = F.mse_loss(recon, x)
            l1_loss = acts.abs().sum(dim=-1).mean() / dict_size

            loss = recon_loss if epoch < warmup_epochs else recon_loss + l1_coeff_scaled * l1_loss

            opt.zero_grad()
            loss.backward()
            opt.step()
            sae.normalize_decoder()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches

        # Report every 2 epochs
        if (epoch + 1) % 2 == 0 or epoch == 0:
            with torch.no_grad():
                n_sample = min(8192, activations.shape[0])
                x_sample = activations[:n_sample].to(device)
                recon, acts_sample = sae(x_sample)
                rel_mse = F.mse_loss(recon, x_sample) / x_sample.var()
                cos = F.cosine_similarity(x_sample, recon, dim=-1).mean()
                l0 = (acts_sample > 0).float().sum(dim=-1).mean()
                frac_active = (acts_sample > 0).float().mean()
                del x_sample, recon, acts_sample

            print(f"    Epoch {epoch + 1}/{epochs}: Loss={avg_loss:.4f}, "
                  f"RelMSE={rel_mse:.4f}, Cos={cos:.4f}, "
                  f"L0={l0:.1f}/{dict_size} ({frac_active * 100:.1f}%)")

    return sae


# ══════════════════════════════════════════════════════════════
# NULL INTERVENTION TEST
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def null_intervention_test(sae, activations, device, max_tokens=500000):
    """
    Null test: encode → decode → re-encode.
    ICC < 0.05 = PASS (SAE faithfully reconstructs without distortion).
    """
    sae.eval()
    n = min(max_tokens, activations.shape[0])
    x = activations[:n].to(device)

    # First pass
    z1 = sae.encode(x)
    recon = sae.decode(z1)

    # Second pass (re-encode the reconstruction)
    z2 = sae.encode(recon)

    # ICC: correlation between z1 and z2 feature activations
    z1_flat = z1.cpu().numpy().flatten()
    z2_flat = z2.cpu().numpy().flatten()

    # Subsample for speed
    if len(z1_flat) > 1_000_000:
        idx = np.random.choice(len(z1_flat), 1_000_000, replace=False)
        z1_flat = z1_flat[idx]
        z2_flat = z2_flat[idx]

    icc, _ = pearsonr(z1_flat, z2_flat)
    icc_val = 1.0 - icc  # ICC ≈ 1 - pearson_r (simplified)

    # Reconstruction quality
    rel_mse = (F.mse_loss(recon, x) / x.var()).item()
    cosine = F.cosine_similarity(x, recon, dim=-1).mean().item()

    passed = icc_val < 0.05

    return {
        'icc': icc_val,
        'passed': passed,
        'rel_mse': rel_mse,
        'cosine': cosine,
    }


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Train SAEs on finetuned GPT-2 activations")

    parser.add_argument("--model_path", type=str,
                        default="./models/gpt2_tofu_finetuned",
                        help="Path to TOFU-finetuned GPT-2")
    parser.add_argument("--layer", type=int, default=10,
                        help="Layer to extract activations from")
    parser.add_argument("--d_model", type=int, default=768,
                        help="Hidden dim (768 for Small, 1024 for Medium)")
    parser.add_argument("--dict_sizes", type=int, nargs='+',
                        default=[16384, 32768, 65536])
    parser.add_argument("--save_dir", type=str, default="./models_finetuned",
                        help="Output dir. Files: {save_dir}/dict_{size}/layer_{layer}.pt")

    # Training params
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--l1_coeff", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Max batches for activation extraction (None=all)")

    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"{'='*60}")
    print("TRAIN SAEs ON FINETUNED MODEL ACTIVATIONS")
    print(f"{'='*60}")
    print(f"  Model: {args.model_path}")
    print(f"  Layer: {args.layer}")
    print(f"  d_model: {args.d_model}")
    print(f"  Dict sizes: {args.dict_sizes}")
    print(f"  Save dir: {args.save_dir}")
    print(f"  Device: {args.device}")

    # ── Load finetuned model ──
    print(f"\n1. Loading finetuned model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    model.to(args.device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    n_layers = model.config.n_layer
    d_model = model.config.n_embd
    print(f"  Loaded: {n_layers} layers, d_model={d_model}")

    if d_model != args.d_model:
        print(f"  WARNING: --d_model={args.d_model} but model has {d_model}. Using {d_model}.")
        args.d_model = d_model

    # ── Prepare data ──
    print(f"\n2. Loading TOFU retain90 for activation extraction...")
    dataset = load_dataset("locuslab/TOFU", "retain90", split="train")

    def tokenize_fn(examples):
        texts = [
            f"Answer the following question:\n### Question:\n{q}\n\n### Answer:\n{a}"
            for q, a in zip(examples["question"], examples["answer"])
        ]
        return tokenizer(texts, truncation=True, max_length=512, padding="max_length")

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    loader = DataLoader(tokenized, batch_size=8, shuffle=False, collate_fn=collator)
    print(f"  {len(dataset)} samples")

    # ── Extract activations ──
    print(f"\n3. Extracting activations from layer {args.layer}...")
    activations = extract_activations(
        model, loader, args.layer, args.device, args.max_batches
    )
    print(f"  Shape: {activations.shape}")

    # Free model memory
    del model
    torch.cuda.empty_cache()
    print("  Model freed from GPU")

    # ── Train SAEs ──
    all_results = []

    for dict_size in args.dict_sizes:
        print(f"\n{'='*60}")
        print(f"DICT SIZE: {dict_size}")
        print(f"{'='*60}")

        # Train
        sae = train_sae(
            activations, args.d_model, dict_size, args.device,
            l1_coeff=args.l1_coeff, lr=args.lr,
            batch_size=args.batch_size, epochs=args.epochs,
        )

        # Null test
        print(f"\n  Running null test...")
        null_result = null_intervention_test(sae, activations, args.device)
        status = "PASS" if null_result['passed'] else "FAIL"
        print(f"  Null test: ICC={null_result['icc']:.4f} [{status}]")
        print(f"  RelMSE={null_result['rel_mse']:.4f}, Cos={null_result['cosine']:.4f}")

        # Save checkpoint — same format as your existing pipeline
        save_path = Path(args.save_dir) / f"dict_{dict_size}"
        save_path.mkdir(parents=True, exist_ok=True)
        ckpt_path = save_path / f"layer_{args.layer}.pt"

        checkpoint = {
            'state_dict': sae.state_dict(),
            'd_model': args.d_model,
            'dict_size': dict_size,
            'layer': args.layer,
            'model_path': args.model_path,
            'null_icc': null_result['icc'],
            'null_passed': null_result['passed'],
            'train_rel_mse': null_result['rel_mse'],
            'train_cosine': null_result['cosine'],
        }
        torch.save(checkpoint, ckpt_path)
        print(f"  Saved: {ckpt_path}")

        all_results.append({
            'dict_size': dict_size,
            'layer': args.layer,
            'null_icc': null_result['icc'],
            'null_passed': null_result['passed'],
            'rel_mse': null_result['rel_mse'],
            'cosine': null_result['cosine'],
        })

        del sae
        torch.cuda.empty_cache()

    # ── Summary ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Dict':>8s} | {'ICC':>8s} | {'RelMSE':>8s} | {'Cosine':>8s} | {'Status':>6s}")
    print(f"{'-'*50}")
    for r in all_results:
        status = "PASS" if r['null_passed'] else "FAIL"
        print(f"{r['dict_size']:>8d} | {r['null_icc']:>8.4f} | "
              f"{r['rel_mse']:>8.4f} | {r['cosine']:>8.4f} | {status:>6s}")

    # Save summary
    summary_path = os.path.join(args.save_dir, "sae_training_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Summary: {summary_path}")

    print(f"\n  NEXT STEP:")
    print(f"    python causal_ablation.py \\")
    print(f"        --model_path {args.model_path} \\")
    print(f"        --sae_path {args.save_dir} \\")
    print(f"        --sae_layer {args.layer} \\")
    print(f"        --sweep")


if __name__ == "__main__":
    main()