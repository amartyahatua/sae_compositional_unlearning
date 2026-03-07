"""
SAE TRAINING FOR PYTHIA 6.9B WITH NULL INTERVENTION TEST
==========================================================

Adapted from GPT-2 Medium SAE training script.
Key differences:
  - Model: EleutherAI/pythia-6.9b (GPTNeoXForCausalLM)
  - d_model: 4096 (vs 1024 for GPT-2 Medium)
  - Layers: 32 (vs 24 for GPT-2 Medium)
  - Architecture: model.gpt_neox.layers[i] (vs model.transformer.h[i])
  - Loading: bfloat16 to save VRAM

Usage:
    python train_sae_pythia_6.9b.py

Author: Amartya Hatua
Date: March 2026
"""

import os
import json
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from typing import Dict
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    GPTNeoXForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from tqdm import tqdm

from get_dataset import get_tofudataset, tokenize_function

torch.cuda.empty_cache()


# ============================================================
# CONFIGURATION
# ============================================================
class Config:
    # Model
    MODEL_NAME = "EleutherAI/pythia-6.9b"
    D_MODEL = 4096
    N_LAYERS = 32

    # Training configurations
    DICT_SIZES = [4096, 8192, 16384, 32768, 65536]
    DICT_SIZES = [8192, 16384]

    LAYERS = list(range(32))  # All 32 Pythia 6.9B layers
    LAYERS = [27, 28, 29, 30, 31]  # All 32 Pythia 6.9B layers


    # Training hyperparameters
    L1_COEFFICIENT = 5e-5
    LR = 3e-4
    BATCH_SIZE = 128       # Smaller than GPT-2 due to larger d_model
    EPOCHS = 20
    WARMUP_EPOCHS = 1

    # Dataset
    MAX_BATCHES = 300      # Fewer batches — each has more features (4096 vs 1024)
    DATA_BATCH_SIZE = 4    # Smaller batch for 6.9B model inference
    MAX_LENGTH = 512

    # Paths
    SAVE_DIR = "../models_pythia"
    RESULTS_DIR = "../results/sae_training_pythia_6.9b"

    # Null intervention test
    NULL_TEST_BATCHES = 30  # Fewer batches — each is more expensive

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Memory optimization
    USE_BF16 = True        # Load model in bfloat16
    GRADIENT_CHECKPOINTING = False  # Not needed for SAE training (no model grads)


# ============================================================
# SAE MODEL
# ============================================================

class AnthropicSAE(nn.Module):
    def __init__(self, d_model, dict_size):
        super().__init__()
        self.d_model = d_model
        self.dict_size = dict_size
        self.encoder = nn.Linear(d_model, dict_size)
        self.decoder = nn.Linear(dict_size, d_model, bias=False)
        nn.init.normal_(self.decoder.weight, std=0.02)

    def forward(self, x):
        acts = F.relu(self.encoder(x))
        recon = self.decoder(acts)
        return recon, acts

    def encode(self, x):
        return F.relu(self.encoder(x))

    @torch.no_grad()
    def normalize_decoder(self):
        W = self.decoder.weight
        self.decoder.weight.copy_(W / W.norm(dim=0, keepdim=True).clamp(min=1e-6))


# ============================================================
# MODEL LOADING
# ============================================================

def load_pythia_model():
    """Load Pythia 6.9B in bfloat16 for memory efficiency."""
    print(f"\nLoading {Config.MODEL_NAME}...")

    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if Config.USE_BF16 else torch.float32
    model = GPTNeoXForCausalLM.from_pretrained(
        Config.MODEL_NAME,
        torch_dtype=dtype,
        device_map=Config.DEVICE,
    )
    model.eval()

    # Print memory usage
    mem_gb = torch.cuda.memory_allocated() / 1e9
    print(f"✓ Model loaded: {Config.MODEL_NAME}")
    print(f"  d_model: {model.config.hidden_size}")
    print(f"  n_layers: {model.config.num_hidden_layers}")
    print(f"  dtype: {dtype}")
    print(f"  VRAM used: {mem_gb:.1f} GB")

    return model, tokenizer


# ============================================================
# ACTIVATION EXTRACTION
# ============================================================

@torch.no_grad()
def extract_layer_activations(model, dataloader, layer_idx, device, max_batches):
    """
    Extract activations from a specific layer of Pythia 6.9B.

    Pythia uses GPTNeoX architecture:
      - model.gpt_neox.layers[i] for transformer blocks
      - output_hidden_states=True gives all layer outputs

    Returns activations in float32 for SAE training.
    """
    model.eval()
    acts = []

    for i, batch in enumerate(tqdm(dataloader, desc=f"Extracting Layer {layer_idx}")):
        if i >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            output_hidden_states=True,
        )

        # hidden_states[0] = embeddings, hidden_states[i+1] = output of layer i
        h = outputs.hidden_states[layer_idx + 1]  # (B, T, d_model)

        # Convert to float32 for SAE training
        h = h.float()

        for b in range(h.size(0)):
            valid = attn_mask[b].bool()
            acts.append(h[b, valid].cpu())

        # Free memory
        del outputs, h
        torch.cuda.empty_cache()

    if len(acts) == 0:
        return torch.empty(0, Config.D_MODEL)

    return torch.cat(acts, dim=0)  # (N_tokens, d_model)


# ============================================================
# SAE TRAINING
# ============================================================

def train_sae(
    activations,
    d_model,
    dict_size,
    device,
    l1_coeff=5e-5,
    lr=3e-4,
    batch_size=128,
    epochs=20,
    warmup_epochs=1,
):
    """Train SAE on extracted activations."""
    print(f"  Training SAE: d_model={d_model}, dict_size={dict_size}")
    print(f"  Activations: {activations.shape[0]:,} tokens")
    print(f"  SAE params: {(d_model * dict_size * 2 + dict_size + d_model):,}")

    sae = AnthropicSAE(d_model, dict_size).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)

    # Create dataloader from activations
    loader = DataLoader(activations, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        sae.train()
        epoch_loss = 0
        n_batches = 0

        for x in loader:
            x = x.to(device)

            recon, acts = sae(x)
            recon_loss = F.mse_loss(recon, x)
            l1_loss = acts.abs().sum(dim=-1).mean() / dict_size

            loss = recon_loss if epoch < warmup_epochs else recon_loss + l1_coeff * l1_loss

            opt.zero_grad()
            loss.backward()
            opt.step()
            sae.normalize_decoder()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches

        # Epoch summary (every 5 epochs to reduce spam)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            with torch.no_grad():
                n_sample = min(2048, activations.shape[0])
                x_sample = activations[:n_sample].to(device)
                recon, _ = sae(x_sample)
                rel_mse = F.mse_loss(recon, x_sample) / x_sample.var()
                cos = F.cosine_similarity(x_sample, recon, dim=-1).mean()
                del x_sample, recon

            print(f"    Epoch {epoch + 1}/{epochs}: Loss={avg_loss:.4f}, "
                  f"RelMSE={rel_mse:.3f}, Cos={cos:.3f}")

    return sae


# ============================================================
# NULL INTERVENTION TEST
# ============================================================

@torch.no_grad()
def null_intervention_test(
    sae: AnthropicSAE,
    model: GPTNeoXForCausalLM,
    dataloader: DataLoader,
    layer_idx: int,
    device: str,
) -> Dict:
    """
    Null intervention test: encode -> decode -> re-encode (without suppression).
    Measures ICC, cosine similarity, reconstruction MSE, and L0 sparsity.
    """
    print(f"\n  Null Intervention Test — Layer {layer_idx}")

    sae.eval()
    model.eval()

    icc_values = []
    cosine_sims = []
    recon_mses = []
    l0_sparsities = []

    batches_processed = 0

    for batch in tqdm(dataloader, desc=f"  Null test L{layer_idx}", leave=False):
        if batches_processed >= Config.NULL_TEST_BATCHES:
            break

        tokens = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)

        outputs = model(
            input_ids=tokens,
            attention_mask=attn_mask,
            output_hidden_states=True,
        )

        acts = outputs.hidden_states[layer_idx + 1].float()  # (B, T, d_model)

        for i in range(acts.shape[0]):
            mask = attn_mask[i].bool()
            valid_acts = acts[i][mask]  # (seq, d_model)

            # Null intervention: encode -> decode -> re-encode
            z_original = sae.encode(valid_acts)
            recon = sae.decoder(z_original)
            z_reencoded = sae.encode(recon)

            # ICC
            feature_change = (z_reencoded - z_original).abs().mean().item()
            feature_magnitude = z_original.abs().mean().item()
            icc = feature_change / (feature_magnitude + 1e-10)
            icc_values.append(icc)

            # Cosine similarity
            cosine = F.cosine_similarity(valid_acts, recon, dim=-1).mean().item()
            cosine_sims.append(cosine)

            # Reconstruction MSE
            mse = F.mse_loss(recon, valid_acts).item()
            recon_mses.append(mse)

            # L0 sparsity
            l0 = (z_original > 0).float().mean().item()
            l0_sparsities.append(l0)

        del outputs, acts
        torch.cuda.empty_cache()
        batches_processed += 1

    if len(icc_values) == 0:
        return {"layer": layer_idx, "null_icc": float("nan"), "overall_pass": False}

    results = {
        "layer": layer_idx,
        "null_icc": float(np.mean(icc_values)),
        "null_cosine_similarity": float(np.mean(cosine_sims)),
        "null_reconstruction_mse": float(np.mean(recon_mses)),
        "null_l0_sparsity": float(np.mean(l0_sparsities)),
        "n_batches": batches_processed,
    }

    results["icc_pass"] = results["null_icc"] < 0.05
    results["recon_pass"] = results["null_reconstruction_mse"] < 0.1
    results["overall_pass"] = results["icc_pass"] and results["recon_pass"]

    status = "✅ PASS" if results["overall_pass"] else "❌ FAIL"
    print(f"    ICC={results['null_icc']:.6f}, MSE={results['null_reconstruction_mse']:.6f}, "
          f"Cos={results['null_cosine_similarity']:.4f}, L0={100 * results['null_l0_sparsity']:.1f}% — {status}")

    return results


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_training_dataloader(tokenizer):
    """Prepare dataloader for SAE training using TOFU full dataset."""
    print("\nPreparing training data...")

    dataset = get_tofudataset("full")

    dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, Config.MAX_LENGTH),
        batched=True,
    )
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    loader = DataLoader(
        dataset,
        batch_size=Config.DATA_BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
    )

    print(f"✓ Training dataloader: {len(dataset)} samples, batch_size={Config.DATA_BATCH_SIZE}")
    return loader


def prepare_null_test_dataloader(tokenizer):
    """Prepare dataloader for null intervention testing."""
    print("Preparing null test data...")

    dataset = get_tofudataset("retain90")

    dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, Config.MAX_LENGTH),
        batched=True,
    )
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    loader = DataLoader(
        dataset,
        batch_size=Config.DATA_BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
    )

    print(f"✓ Null test dataloader ready")
    return loader


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

def main():
    print("\n" + "=" * 80)
    print("SAE TRAINING — PYTHIA 6.9B")
    print("=" * 80)

    print(f"\nConfiguration:")
    print(f"  Model: {Config.MODEL_NAME}")
    print(f"  d_model: {Config.D_MODEL}")
    print(f"  Layers: {len(Config.LAYERS)} ({Config.LAYERS[0]}-{Config.LAYERS[-1]})")
    print(f"  Dict sizes: {Config.DICT_SIZES}")
    print(f"  Total SAEs: {len(Config.DICT_SIZES) * len(Config.LAYERS)}")
    print(f"  Device: {Config.DEVICE}")
    print(f"  bf16: {Config.USE_BF16}")

    # Estimate time
    total_saes = len(Config.DICT_SIZES) * len(Config.LAYERS)
    print(f"\n  Estimated time: ~{total_saes * 12 / 60:.0f} hours "
          f"({total_saes} SAEs × ~12 min each)")

    # Setup directories
    results_dir = Path(Config.RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, tokenizer = load_pythia_model()

    # Prepare data
    train_loader = prepare_training_dataloader(tokenizer)
    test_loader = prepare_null_test_dataloader(tokenizer)

    # Track results
    all_results = []

    total = len(Config.DICT_SIZES) * len(Config.LAYERS)
    current = 0

    for dict_size in Config.DICT_SIZES:
        print("\n" + "=" * 80)
        print(f"DICT SIZE: {dict_size}")
        print("=" * 80)

        # Adaptive L1 coefficient
        l1_coeff = Config.L1_COEFFICIENT * (16384 / dict_size)

        save_dir = Path(Config.SAVE_DIR) / f"dict_{dict_size}"
        save_dir.mkdir(parents=True, exist_ok=True)

        for layer_idx in Config.LAYERS:
            current += 1

            print(f"\n{'#' * 80}")
            print(f"[{current}/{total}] Layer {layer_idx} | Dict {dict_size}")
            print(f"{'#' * 80}")

            checkpoint_path = save_dir / f"layer_{layer_idx}.pt"

            # Skip if already trained
            if checkpoint_path.exists():
                print(f"  ⚠️ Checkpoint exists: {checkpoint_path}, skipping...")

                # Still run null test if needed
                try:
                    sae = AnthropicSAE(Config.D_MODEL, dict_size).to(Config.DEVICE)
                    ckpt = torch.load(checkpoint_path, map_location=Config.DEVICE)
                    sae.load_state_dict(ckpt['state_dict'])
                    sae.eval()

                    null_results = null_intervention_test(
                        sae, model, test_loader, layer_idx, Config.DEVICE
                    )

                    combined = {'dict_size': dict_size, 'layer': layer_idx, **null_results}
                    all_results.append(combined)

                    del sae
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"    ✗ Error loading/testing: {e}")

                continue

            try:
                # ── Extract activations ──
                print(f"\n  Extracting activations...")
                acts = extract_layer_activations(
                    model, train_loader, layer_idx, Config.DEVICE, Config.MAX_BATCHES
                )
                print(f"  ✓ Extracted {acts.shape[0]:,} tokens × {acts.shape[1]} dims")

                if acts.shape[0] < 1000:
                    print(f"  ✗ Too few activations ({acts.shape[0]}), skipping")
                    continue

                # ── Train SAE ──
                print(f"\n  Training SAE...")
                sae = train_sae(
                    acts,
                    d_model=Config.D_MODEL,
                    dict_size=dict_size,
                    device=Config.DEVICE,
                    l1_coeff=l1_coeff,
                    lr=Config.LR,
                    batch_size=Config.BATCH_SIZE,
                    epochs=Config.EPOCHS,
                    warmup_epochs=Config.WARMUP_EPOCHS,
                )

                # ── Save checkpoint ──
                torch.save({
                    "layer": layer_idx,
                    "dict_size": dict_size,
                    "d_model": Config.D_MODEL,
                    "model_name": Config.MODEL_NAME,
                    "state_dict": sae.state_dict(),
                }, checkpoint_path)
                print(f"  ✓ Saved: {checkpoint_path}")

                # Free activation memory before null test
                del acts
                gc.collect()
                torch.cuda.empty_cache()

                # ── Null intervention test ──
                null_results = null_intervention_test(
                    sae, model, test_loader, layer_idx, Config.DEVICE
                )

                # ── Save results ──
                combined = {'dict_size': dict_size, 'layer': layer_idx, **null_results}
                all_results.append(combined)

                # Save individual result
                with open(save_dir / f"layer_{layer_idx}_null_test.json", 'w') as f:
                    json.dump(combined, f, indent=2)

                # Save incremental progress
                progress_df = pd.DataFrame(all_results)
                progress_df.to_csv(results_dir / "training_progress.csv", index=False)

                # Cleanup
                del sae
                gc.collect()
                torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                print(f"\n  ❌ OOM at Layer {layer_idx}, Dict {dict_size}")
                print(f"     Try reducing MAX_BATCHES or BATCH_SIZE")
                gc.collect()
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"\n  ❌ Error: {e}")
                import traceback
                traceback.print_exc()
                gc.collect()
                torch.cuda.empty_cache()

    # ── Final summary ──
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE — FINAL SUMMARY")
    print("=" * 80)

    if all_results:
        results_df = pd.DataFrame(all_results)
        final_path = results_dir / "final_results.csv"
        results_df.to_csv(final_path, index=False)

        print(f"\n  Total configurations: {len(results_df)}")
        if 'overall_pass' in results_df.columns:
            passed = results_df['overall_pass'].sum()
            print(f"  Null test PASS: {passed}/{len(results_df)} ({100 * passed / len(results_df):.1f}%)")

        # Summary by dict size
        print(f"\n  By dict size:")
        for ds in Config.DICT_SIZES:
            subset = results_df[results_df['dict_size'] == ds]
            if len(subset) > 0 and 'null_icc' in subset.columns:
                mean_icc = subset['null_icc'].mean()
                n_pass = subset['overall_pass'].sum() if 'overall_pass' in subset.columns else 0
                print(f"    Dict {ds:>6d}: ICC={mean_icc:.4f}, "
                      f"Pass={n_pass}/{len(subset)}")

        print(f"\n  💾 Results: {final_path}")
    else:
        print("\n  ⚠️ No results collected")

    print(f"  📁 SAEs: {Config.SAVE_DIR}/dict_*/layer_*.pt")

    # Print VRAM summary
    if torch.cuda.is_available():
        mem_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"  🔥 Peak VRAM: {mem_gb:.1f} GB")


if __name__ == "__main__":
    main()