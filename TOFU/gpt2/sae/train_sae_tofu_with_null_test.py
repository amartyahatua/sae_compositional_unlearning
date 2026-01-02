"""
SIMPLIFIED SAE TRAINING WITH NULL INTERVENTION TEST
====================================================

Uses manual SAE training (your original working code) + fixed null intervention test.
No dependency on SAELens API version issues.

Usage:
    python train_sae_simple.py

Author: Amartya Hatua
Date: December 2024
"""

import os
import json

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from typing import Dict
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling, AutoTokenizer
from tqdm import tqdm

from transformer_lens import HookedTransformer
from get_dataset import get_tofudataset, tokenize_function
from get_model import get_gptmodel
from get_dataset import get_tofudataset



# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    # Model
    MODEL_NAME = "gpt2"
    D_MODEL = 768

    # Training configurations
    # DICT_SIZES = [4096, 8192, 16384, 32768, 65536]
    DICT_SIZES = [65536]
    LAYERS = list(range(12))  # All 12 GPT-2 layers
    LAYERS = [8]

    # Training hyperparameters
    L1_COEFFICIENT = 5e-5
    LR = 3e-4
    BATCH_SIZE = 256
    EPOCHS = 8
    WARMUP_EPOCHS = 1

    # Dataset
    MAX_BATCHES = 500  # More data than before
    DATA_BATCH_SIZE = 8
    MAX_LENGTH = 512

    # Paths
    SAVE_DIR = "../models"
    RESULTS_DIR = "../results/sae_training_simple"

    # Null intervention test
    NULL_TEST_BATCHES = 50

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# SAE MODEL (YOUR WORKING VERSION)
# ============================================================

class AnthropicSAE(torch.nn.Module):
    def __init__(self, d_model, dict_size):
        super().__init__()
        self.encoder = torch.nn.Linear(d_model, dict_size)
        self.decoder = torch.nn.Linear(dict_size, d_model, bias=False)
        torch.nn.init.normal_(self.decoder.weight, std=0.02)

    def forward(self, x):
        acts = F.relu(self.encoder(x))
        recon = self.decoder(acts)
        return recon, acts

    @torch.no_grad()
    def normalize_decoder(self):
        W = self.decoder.weight
        self.decoder.weight.copy_(W / W.norm(dim=0, keepdim=True).clamp(min=1e-6))


# ============================================================
# ACTIVATION EXTRACTION
# ============================================================

@torch.no_grad()
def extract_layer_activations(model, dataloader, layer_idx, device, max_batches):
    """Extract activations from a specific layer"""
    model.eval()
    acts = []

    for i, batch in enumerate(tqdm(dataloader, desc=f"Extracting Layer {layer_idx}")):
        if i >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)

        out = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            output_hidden_states=True,
        )

        h = out.hidden_states[layer_idx + 1]  # (B, T, d_model)

        for b in range(h.size(0)):
            valid = attn_mask[b].bool()
            acts.append(h[b, valid].cpu())

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
    batch_size=256,
    epochs=8,
    warmup_epochs=1,
):
    """Train SAE on activations"""
    sae = AnthropicSAE(d_model, dict_size).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)

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

        # Epoch summary
        avg_loss = epoch_loss / n_batches

        with torch.no_grad():
            x_sample = activations[:2048].to(device)
            recon, _ = sae(x_sample)
            rel_mse = F.mse_loss(recon, x_sample) / x_sample.var()
            cos = F.cosine_similarity(x_sample, recon, dim=-1).mean()

        print(f"  Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, RelMSE={rel_mse:.3f}, Cos={cos:.3f}")

    return sae


# ============================================================
# NULL INTERVENTION TEST (FIXED!)
# ============================================================

@torch.no_grad()
def null_intervention_test(
    sae: AnthropicSAE,
    model,
    dataloader: DataLoader,
    layer_idx: int,
    device: str,
) -> Dict:
    """
    FIXED null intervention test using TransformerLens.
    Tests: encode -> decode -> re-encode (without suppression)
    """
    print(f"\n{'='*80}")
    print(f"NULL INTERVENTION TEST - Layer {layer_idx}")
    print(f"{'='*80}")

    sae.eval()
    model.eval()

    # Collect metrics
    icc_values = []
    cosine_sims = []
    recon_mses = []
    l0_sparsities = []

    batches_processed = 0

    for batch in tqdm(dataloader, desc=f"Layer {layer_idx} null test"):
        if batches_processed >= Config.NULL_TEST_BATCHES:
            break

        tokens = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)

        # Get activations using output_hidden_states
        outputs = model(
            input_ids=tokens,
            attention_mask=attn_mask,
            output_hidden_states=True
        )

        acts = outputs.hidden_states[layer_idx + 1]  # [batch, seq, d_model]

        # Flatten valid tokens
        for i in range(acts.shape[0]):
            mask = attn_mask[i].bool()
            valid_acts = acts[i][mask]  # [valid_seq, d_model]

            # FIXED: NULL INTERVENTION - encode -> decode -> re-encode
            z_original = F.relu(sae.encoder(valid_acts))
            recon = sae.decoder(z_original)
            z_reencoded = F.relu(sae.encoder(recon))  # ← CRITICAL: Must re-encode!

            # METRIC 1: ICC (Intervention Coupling Coefficient)
            feature_change = (z_reencoded - z_original).abs().mean().item()
            feature_magnitude = z_original.abs().mean().item()
            icc = feature_change / (feature_magnitude + 1e-10)
            icc_values.append(icc)

            # METRIC 2: Cosine similarity
            cosine = F.cosine_similarity(valid_acts, recon, dim=-1).mean().item()
            cosine_sims.append(cosine)

            # METRIC 3: Reconstruction MSE
            mse = F.mse_loss(recon, valid_acts).item()
            recon_mses.append(mse)

            # METRIC 4: L0 sparsity
            l0 = (z_original > 0).float().mean().item()
            l0_sparsities.append(l0)

        batches_processed += 1

    # Compute statistics
    results = {
        "layer": layer_idx,
        "null_icc": float(sum(icc_values) / len(icc_values)),
        "null_cosine_similarity": float(sum(cosine_sims) / len(cosine_sims)),
        "null_reconstruction_mse": float(sum(recon_mses) / len(recon_mses)),
        "null_l0_sparsity": float(sum(l0_sparsities) / len(l0_sparsities)),
        "n_batches": batches_processed,
    }

    # Pass/fail criteria
    results["icc_pass"] = results["null_icc"] < 0.05
    results["recon_pass"] = results["null_reconstruction_mse"] < 0.1
    results["overall_pass"] = results["icc_pass"] and results["recon_pass"]

    # Print results
    print(f"\n📊 Results:")
    print(f"   ICC:        {results['null_icc']:.6f} {'✅ PASS' if results['icc_pass'] else '❌ FAIL'}")
    print(f"   MSE:        {results['null_reconstruction_mse']:.6f} {'✅ PASS' if results['recon_pass'] else '❌ FAIL'}")
    print(f"   Cosine:     {results['null_cosine_similarity']:.6f}")
    print(f"   L0:         {100*results['null_l0_sparsity']:.2f}%")
    print(f"   Status:     {'✅ OVERALL PASS' if results['overall_pass'] else '❌ OVERALL FAIL'}")

    return results


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_training_dataloader():
    """Prepare dataloader for SAE training"""
    print("\nPreparing training data...")

    dataset = get_tofudataset("full")

    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, Config.MAX_LENGTH),
        batched=True
    )
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    loader = DataLoader(dataset, batch_size=Config.DATA_BATCH_SIZE, shuffle=True, collate_fn=collator)

    print(f"✓ Training dataloader ready")
    return loader


def prepare_null_test_dataloader():
    """Prepare dataloader for null intervention testing"""
    print("\nPreparing null test data...")

    dataset = get_tofudataset("retain90")

    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, Config.MAX_LENGTH),
        batched=True
    )
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collator)

    print(f"✓ Null test dataloader ready")
    return loader


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

def main():
    """Main function"""
    print("\n" + "🔥"*40)
    print("SIMPLIFIED SAE TRAINING WITH NULL INTERVENTION TEST")
    print("🔥"*40)

    print(f"\nConfiguration:")
    print(f"  Model: {Config.MODEL_NAME}")
    print(f"  Dict sizes: {Config.DICT_SIZES}")
    print(f"  Layers: {Config.LAYERS}")
    print(f"  Total SAEs: {len(Config.DICT_SIZES) * len(Config.LAYERS)}")
    print(f"  Device: {Config.DEVICE}")

    # Setup results directory
    results_dir = Path(Config.RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("\nLoading model...")
    model, tokenizer = get_gptmodel(Config.MODEL_NAME)
    model.to(Config.DEVICE)
    model.eval()
    print(f"✓ Model loaded")

    # Prepare data
    train_loader = prepare_training_dataloader()
    test_loader = prepare_null_test_dataloader()

    # Track results
    all_results = []

    # Main training loop
    total = len(Config.DICT_SIZES) * len(Config.LAYERS)
    current = 0

    for dict_size in Config.DICT_SIZES:
        print("\n" + "="*80)
        print(f"DICT SIZE: {dict_size}")
        print("="*80)

        # Adaptive L1
        l1_coeff = Config.L1_COEFFICIENT * (16384 / dict_size)

        save_dir = Path(Config.SAVE_DIR) / f"dict_{dict_size}"
        save_dir.mkdir(parents=True, exist_ok=True)

        for layer_idx in Config.LAYERS:
            current += 1

            print(f"\n{'#'*80}")
            print(f"[{current}/{total}] Layer {layer_idx} | Dict {dict_size}")
            print(f"{'#'*80}")

            checkpoint_path = save_dir / f"layer_{layer_idx}.pt"

            try:
                # Check if already trained
                if checkpoint_path.exists():
                    print(f"⚠️  Checkpoint exists, loading...")
                    checkpoint = torch.load(checkpoint_path)
                    sae = AnthropicSAE(Config.D_MODEL, dict_size).to(Config.DEVICE)
                    sae.load_state_dict(checkpoint['state_dict'])
                    sae.eval()
                else:
                    # Extract activations
                    print(f"\nExtracting activations...")
                    acts = extract_layer_activations(
                        model, train_loader, layer_idx, Config.DEVICE, Config.MAX_BATCHES
                    )
                    print(f"✓ Extracted {acts.shape[0]} tokens")

                    # Train SAE
                    print(f"\nTraining SAE...")
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

                # Save
                torch.save({
                    "layer": layer_idx,
                    "dict_size": dict_size,
                    "d_model": Config.D_MODEL,
                    "state_dict": sae.state_dict(),
                }, checkpoint_path)
                print(f"✓ Saved to {checkpoint_path}")

                del acts
                torch.cuda.empty_cache()

                # Run null intervention test
                null_results = null_intervention_test(
                    sae, model, test_loader, layer_idx, Config.DEVICE
                )

                # Save results
                combined_results = {
                    'dict_size': dict_size,
                    'layer': layer_idx,
                    **null_results
                }

                all_results.append(combined_results)

                # Save individual result
                with open(save_dir / f"layer_{layer_idx}_null_test.json", 'w') as f:
                    json.dump(combined_results, f, indent=2)

                # Save incremental progress
                progress_df = pd.DataFrame(all_results)
                progress_df.to_csv(results_dir / "training_progress.csv", index=False)

                del sae
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()

    # Final summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE - FINAL SUMMARY")
    print("="*80)

    results_df = pd.DataFrame(all_results)
    final_path = results_dir / "final_results.csv"
    results_df.to_csv(final_path, index=False)

    print(f"\nTotal configurations: {len(results_df)}")
    passed = results_df['overall_pass'].sum()
    print(f"Null test PASS: {passed}/{len(results_df)} ({100*passed/len(results_df):.1f}%)")

    print(f"\n💾 Results saved to: {final_path}")
    print(f"📁 SAEs saved to: {Config.SAVE_DIR}/dict_*/layer_*.pt")


if __name__ == "__main__":
    main()