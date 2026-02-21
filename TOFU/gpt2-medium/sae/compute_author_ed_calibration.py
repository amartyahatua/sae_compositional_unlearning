"""
Compute ED/Gini/L0/Entropy for ALL Dict Sizes × ALL Layers
============================================================

Systematically tests all combinations to find optimal configuration
Saves each result in separate subfolder organized by dict_size

Dict sizes: [4096, 8192, 16384, 32768, 65536]
Layers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
Total: 5 × 12 = 60 configurations
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

from get_dataset import get_tofudataset, tokenize_function
from get_model import get_gptmodel


class SparseAutoencoder(torch.nn.Module):
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
    def encode(self, x):
        """Get SAE activations without reconstruction"""
        return F.relu(self.encoder(x))

    # OPTIONAL: Also add decode for consistency
    @torch.no_grad()
    def decode(self, acts):
        """Reconstruct from activations"""
        return self.decoder(acts)

    @torch.no_grad()
    def normalize_decoder(self):
        W = self.decoder.weight
        self.decoder.weight.copy_(W / W.norm(dim=0, keepdim=True).clamp(min=1e-6))


def load_sae(layer_idx, dict_size, sae_dir='../models'):
    """Load SAE checkpoint"""
    checkpoint_path = f'{sae_dir}/dict_{dict_size}/layer_{layer_idx}.pt'

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"SAE not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    d_model = checkpoint['d_model']
    dict_size_loaded = checkpoint['dict_size']

    sae = SparseAutoencoder(d_model, dict_size_loaded)
    sae.load_state_dict(checkpoint['state_dict'])
    return sae


def get_author_features(model, sae, dataloader, layer_idx, device):
    """Extract SAE features for author"""
    model.eval()
    sae.eval()
    all_features = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Get layer activations
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                          output_hidden_states=True)
            layer_acts = outputs.hidden_states[layer_idx + 1]

            # Get SAE features
            features = sae.encode(layer_acts)

            # Only keep valid tokens
            for i in range(features.shape[0]):
                mask = attention_mask[i].bool()
                valid_features = features[i][mask]
                all_features.append(valid_features.cpu())

    if len(all_features) == 0:
        return None

    return torch.cat(all_features, dim=0)


def compute_ed_95(features):
    """Compute ED at 95% threshold"""
    mean_acts = features.mean(dim=0).cpu().numpy()
    sorted_acts = np.sort(mean_acts)[::-1]
    cumsum = np.cumsum(sorted_acts)
    total = cumsum[-1]

    if total == 0:
        return 0

    ed_95 = np.argmax(cumsum >= 0.95 * total) + 1
    return int(ed_95)


def compute_gini_coefficient(features):
    """Compute Gini coefficient of feature activations"""
    mean_acts = features.mean(dim=0).cpu().numpy()

    # Sort
    sorted_acts = np.sort(mean_acts)
    n = len(sorted_acts)

    # Handle edge case
    if sorted_acts.sum() == 0:
        return 0.0

    # Gini formula
    cumsum = np.cumsum(sorted_acts)
    gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_acts)) / (n * cumsum[-1]) - (n + 1) / n

    return float(gini)


def compute_l0_sparsity(features, threshold=0.0):
    """Compute L0 sparsity (average active features per token)"""
    l0 = (features > threshold).sum(dim=1).float().mean().item()
    return l0


def compute_entropy(features):
    """Compute entropy of mean feature activations"""
    mean_acts = features.mean(dim=0).cpu().numpy()

    # Normalize to probability distribution
    if mean_acts.sum() == 0:
        return 0.0

    probs = mean_acts / mean_acts.sum()
    probs = probs[probs > 0]  # Remove zeros

    entropy = -np.sum(probs * np.log(probs))
    return float(entropy)


def calibrate_one_config(dict_size, layer_idx, model, tokenizer, author_to_samples,
                         forget_ds, collator, device, output_dir):
    """
    Calibrate all authors for ONE configuration

    Args:
        dict_size: SAE dictionary size
        layer_idx: Layer index
        ... (other shared resources)
        output_dir: Where to save results

    Returns:
        DataFrame with results, or None if failed
    """

    print(f"\n{'='*80}")
    print(f"CONFIG: dict_size={dict_size}, layer={layer_idx}")
    print(f"{'='*80}")

    # Load SAE
    try:
        sae = load_sae(layer_idx, dict_size).to(device)
        sae.eval()
    except FileNotFoundError as e:
        print(f"⚠️  SAE not found, skipping: {e}")
        return None
    except Exception as e:
        print(f"❌ Error loading SAE: {e}")
        return None

    # Results storage
    results = []

    # Process each author
    print(f"Processing {len(author_to_samples)} authors...")

    for author_name, author_indices in tqdm(author_to_samples.items(), desc=f"D{dict_size} L{layer_idx}"):

        # Create dataloader for this author
        try:
            author_subset = forget_ds.select(author_indices)
            author_tokenized = author_subset.map(
                lambda x: tokenize_function(x, tokenizer, 512),
                batched=True
            )
            author_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
            author_loader = DataLoader(
                author_tokenized,
                batch_size=8,
                shuffle=False,
                collate_fn=collator
            )

            # Extract features
            features = get_author_features(model, sae, author_loader, layer_idx, device)

            if features is None or features.shape[0] == 0:
                print(f"  ⚠️  {author_name}: No features extracted")
                continue

            # Compute all metrics
            ed_95 = compute_ed_95(features)
            gini = compute_gini_coefficient(features)
            l0 = compute_l0_sparsity(features)
            entropy = compute_entropy(features)

            results.append({
                'author': author_name,
                'num_samples': len(author_indices),
                'num_tokens': features.shape[0],
                'effective_dim_95': ed_95,
                'gini_coefficient': gini,
                'l0_sparsity': l0,
                'entropy': entropy,
                'dict_size': dict_size,
                'layer': layer_idx
            })

        except Exception as e:
            print(f"  ❌ Error processing {author_name}: {e}")
            continue

    # Clean up
    del sae
    torch.cuda.empty_cache()

    # Check if we got results
    if len(results) == 0:
        print(f"❌ No results for dict_size={dict_size}, layer={layer_idx}")
        return None

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save to subfolder
    subfolder = output_dir / f"dict_{dict_size}"
    subfolder.mkdir(parents=True, exist_ok=True)

    output_path = subfolder / f"calibration_dict{dict_size}_layer{layer_idx}.csv"
    df.to_csv(output_path, index=False, float_format='%.6f')

    print(f"✅ Saved: {output_path}")
    print(f"   Authors: {len(df)}")
    print(f"   ED range: [{df['effective_dim_95'].min()}, {df['effective_dim_95'].max()}]")
    print(f"   Gini range: [{df['gini_coefficient'].min():.4f}, {df['gini_coefficient'].max():.4f}]")

    return df


def calibrate_all_configs():
    """
    Main function: Calibrate ALL dict_sizes × ALL layers
    """

    print("\n" + "="*80)
    print("COMPREHENSIVE CALIBRATION: ALL DICT_SIZES × ALL LAYERS")
    print("="*80)

    # Configuration
    DICT_SIZES = [4096, 8192, 16384, 32768, 65536]
    LAYERS = list(range(12))  # 0-11
    OUTPUT_DIR = Path('../results/calibration_comprehensive')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Dict sizes: {DICT_SIZES}")
    print(f"Layers: {LAYERS}")
    print(f"Total configs: {len(DICT_SIZES) * len(LAYERS)}")
    print(f"Output: {OUTPUT_DIR}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load shared resources (once)
    print("\n1. Loading shared resources...")
    print("   - GPT-2 model")
    model, tokenizer = get_gptmodel('gpt2-medium')
    model = model.to(device)
    model.eval()

    print("   - Author mapping")
    with open('../data/tofu_author_mapping.json', 'r') as f:
        author_data = json.load(f)
    author_to_samples = author_data['author_to_samples']

    print("   - TOFU dataset")
    forget_ds = get_tofudataset("forget10")

    print("   - Collator")
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Track progress
    total_configs = len(DICT_SIZES) * len(LAYERS)
    completed = 0
    successful = 0
    failed = 0

    start_time = time.time()

    # Summary storage
    all_summaries = []

    # MAIN LOOP: For each dict_size
    for dict_size in DICT_SIZES:

        print(f"\n{'#'*80}")
        print(f"DICT_SIZE: {dict_size}")
        print(f"{'#'*80}")

        # For each layer
        for layer_idx in LAYERS:

            # Run calibration
            df = calibrate_one_config(
                dict_size, layer_idx,
                model, tokenizer, author_to_samples, forget_ds, collator,
                device, OUTPUT_DIR
            )

            completed += 1

            if df is not None:
                successful += 1

                # Store summary stats
                all_summaries.append({
                    'dict_size': dict_size,
                    'layer': layer_idx,
                    'num_authors': len(df),
                    'ed_mean': df['effective_dim_95'].mean(),
                    'ed_std': df['effective_dim_95'].std(),
                    'ed_min': df['effective_dim_95'].min(),
                    'ed_max': df['effective_dim_95'].max(),
                    'gini_mean': df['gini_coefficient'].mean(),
                    'gini_std': df['gini_coefficient'].std(),
                    'gini_min': df['gini_coefficient'].min(),
                    'gini_max': df['gini_coefficient'].max(),
                    'l0_mean': df['l0_sparsity'].mean(),
                    'entropy_mean': df['entropy'].mean()
                })
            else:
                failed += 1

            # Progress update
            elapsed = time.time() - start_time
            rate = completed / elapsed
            remaining = total_configs - completed
            eta = remaining / rate if rate > 0 else 0

            print(f"\n📊 Overall Progress: {completed}/{total_configs} ({100*completed/total_configs:.1f}%)")
            print(f"   ✅ Successful: {successful}")
            print(f"   ❌ Failed: {failed}")
            print(f"   ⏱️  Elapsed: {elapsed/60:.1f} min | ETA: {eta/60:.1f} min")

            # Save checkpoint summary after each layer
            if len(all_summaries) > 0:
                summary_df = pd.DataFrame(all_summaries)
                summary_path = OUTPUT_DIR / 'summary_statistics.csv'
                summary_df.to_csv(summary_path, index=False, float_format='%.6f')

    # Final summary
    elapsed_total = time.time() - start_time

    print(f"\n{'='*80}")
    print("✅ COMPREHENSIVE CALIBRATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {elapsed_total/60:.1f} minutes ({elapsed_total/3600:.2f} hours)")
    print(f"Configs tested: {completed}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nResults saved in: {OUTPUT_DIR}")
    print(f"  - Individual CSVs: {OUTPUT_DIR}/dict_*/calibration_*.csv")
    print(f"  - Summary stats: {OUTPUT_DIR}/summary_statistics_7_Jan.csv")

    # Create final summary
    if len(all_summaries) > 0:
        summary_df = pd.DataFrame(all_summaries)

        # Find best configs by variance
        print(f"\n📊 TOP CONFIGS BY VARIANCE:")

        print("\nTop 5 by ED std (want HIGH variance):")
        top_ed = summary_df.nlargest(5, 'ed_std')[['dict_size', 'layer', 'ed_std', 'ed_min', 'ed_max']]
        print(top_ed.to_string(index=False))

        print("\nTop 5 by Gini std (want HIGH variance):")
        top_gini = summary_df.nlargest(5, 'gini_std')[['dict_size', 'layer', 'gini_std', 'gini_min', 'gini_max']]
        print(top_gini.to_string(index=False))

        # Save summary
        summary_path = OUTPUT_DIR / 'summary_statistics.csv'
        summary_df.to_csv(summary_path, index=False, float_format='%.6f')
        print(f"\n💾 Summary saved: {summary_path}")

        return summary_df

    return None


if __name__ == "__main__":
    summary = calibrate_all_configs()