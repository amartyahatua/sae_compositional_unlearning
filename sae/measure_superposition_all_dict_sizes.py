import torch
import json
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os
import time
import random
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from get_dataset import get_tofudataset, tokenize_function
from get_model import get_gptmodel


class SparseAutoencoder(torch.nn.Module):
    """Standard SAE architecture"""

    def __init__(self, d_model, dict_size):
        super().__init__()
        self.d_model = d_model
        self.dict_size = dict_size
        self.encoder = torch.nn.Linear(d_model, dict_size, bias=True)
        self.decoder = torch.nn.Linear(dict_size, d_model, bias=True)

    def forward(self, x):
        pre_activation = self.encoder(x)
        feature_acts = torch.relu(pre_activation)
        x_reconstruct = self.decoder(feature_acts)
        return x_reconstruct, feature_acts


def load_sae(layer_idx, dict_size, sae_dir='../models'):
    """Load trained SAE for a specific layer and dict_size"""
    checkpoint_path = f'{sae_dir}/saes_gpt2_{dict_size}/sae_layer_{layer_idx}.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    d_model = checkpoint['d_model']
    dict_size_loaded = checkpoint['dict_size']

    sae = SparseAutoencoder(d_model, dict_size_loaded)
    sae.load_state_dict(checkpoint['state_dict'])
    return sae


def get_sae_feature_activations(model, sae, dataloader, layer_idx, device, max_batches=None):
    """
    Extract SAE feature activations for a dataset
    Returns: tensor of shape (num_tokens, dict_size) with feature activations
    """
    model.eval()
    sae.eval()
    all_features = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Extracting features", leave=False)
        for batch_idx, batch in enumerate(pbar):
            if max_batches and batch_idx >= max_batches:
                break

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Get model activations
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            layer_activation = outputs.hidden_states[layer_idx + 1]  # (batch, seq, d_model)

            # Get SAE feature activations
            _, feature_acts = sae(layer_activation)  # (batch, seq, dict_size)

            # Only keep non-padded tokens
            for i in range(feature_acts.shape[0]):
                mask = attention_mask[i].bool()
                valid_features = feature_acts[i][mask]  # (valid_seq, dict_size)
                all_features.append(valid_features.cpu())

            # Update progress bar
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    'tokens': sum(f.shape[0] for f in all_features),
                    'mem_GB': f"{torch.cuda.memory_allocated() / 1e9:.1f}" if torch.cuda.is_available() else "0"
                })

    # Concatenate all features
    all_features = torch.cat(all_features, dim=0)  # (total_tokens, dict_size)
    return all_features


def compute_superposition_metrics_multithreshold(author_features, retain_features, thresholds=[0.0, 0.001, 0.01, 0.1]):
    """
    Compute superposition metrics at multiple thresholds

    Args:
        author_features: (num_author_tokens, dict_size)
        retain_features: (num_retain_tokens, dict_size)
        thresholds: list of activation thresholds to test

    Returns:
        dict with metrics for each threshold + statistics
    """

    # Compute mean and std for each feature (for Gaussian analysis later)
    author_mean = author_features.mean(dim=0)  # (dict_size,)
    author_std = author_features.std(dim=0)  # (dict_size,)
    retain_mean = retain_features.mean(dim=0)
    retain_std = retain_features.std(dim=0)

    # Compute metrics for each threshold
    threshold_metrics = {}

    for threshold in thresholds:
        # 1. Identify active features
        author_active = (author_features > threshold).float().mean(dim=0)  # (dict_size,)
        retain_active = (retain_features > threshold).float().mean(dim=0)

        # Features active for >1% of tokens
        author_mask = author_active > 0.01
        retain_mask = retain_active > 0.01

        # 2. Jaccard similarity
        intersection = (author_mask & retain_mask).sum().item()
        union = (author_mask | retain_mask).sum().item()
        jaccard = intersection / union if union > 0 else 0

        # 3. Overlap percentage
        author_feature_count = author_mask.sum().item()
        overlap_pct = intersection / author_feature_count if author_feature_count > 0 else 0

        # 4. L0 sparsity (average active features per token)
        author_l0 = (author_features > threshold).sum(dim=1).float().mean().item()
        retain_l0 = (retain_features > threshold).sum(dim=1).float().mean().item()

        threshold_metrics[f'threshold_{threshold}'] = {
            'jaccard_similarity': jaccard,
            'overlap_percentage': overlap_pct,
            'num_author_features': author_feature_count,
            'num_retain_features': retain_mask.sum().item(),
            'num_shared_features': intersection,
            'author_l0_sparsity': author_l0,
            'retain_l0_sparsity': retain_l0
        }

    # 5. Cosine similarity (threshold-independent)
    cosine_sim = torch.nn.functional.cosine_similarity(
        author_mean.unsqueeze(0),
        retain_mean.unsqueeze(0)
    ).item()

    # 6. L2 distance (threshold-independent)
    l2_distance = torch.norm(author_mean - retain_mean).item()

    # 7. Active features with their statistics (Option B output)
    # Get features that are active for >1% of author tokens (using lowest threshold)
    active_mask = (author_features > thresholds[0]).float().mean(dim=0) > 0.01
    active_indices = torch.where(active_mask)[0].cpu().numpy().tolist()

    active_feature_stats = {
        'active_feature_indices': active_indices,
        'active_feature_means': author_mean[active_mask].cpu().numpy().tolist(),
        'active_feature_stds': author_std[active_mask].cpu().numpy().tolist(),
        'num_active_features': len(active_indices)
    }

    return {
        'threshold_metrics': threshold_metrics,
        'cosine_similarity': cosine_sim,
        'l2_distance': l2_distance,
        'author_mean_activation': author_mean.cpu().numpy().tolist(),
        'author_std_activation': author_std.cpu().numpy().tolist(),
        'retain_mean_activation': retain_mean.cpu().numpy().tolist(),
        'retain_std_activation': retain_std.cpu().numpy().tolist(),
        'active_features': active_feature_stats
    }


def measure_superposition_all_dict_sizes():
    """Main function to measure superposition for all dict_sizes and all authors"""

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Dictionary sizes to test
    dict_sizes = [4096, 8192, 16384, 32768, 65536]

    # Thresholds to test
    thresholds = [0.0, 0.001, 0.01, 0.1]

    # Samples per author
    samples_per_author = 20

    # Load author mapping for forget10
    with open('../data/tofu_author_mapping.json', 'r') as f:
        author_data = json.load(f)

    author_to_samples = author_data['author_to_samples']
    authors = list(author_to_samples.keys())

    print(f"\n{'=' * 70}")
    print(f"MEASURING SUPERPOSITION FOR ALL DICTIONARY SIZES")
    print("=" * 70)
    print(f"Dictionary sizes: {dict_sizes}")
    print(f"Authors: {len(authors)}")
    print(f"Samples per author: {samples_per_author}")
    print(f"Thresholds: {thresholds}")
    print(f"Layers: 0-11")

    # Load model
    print("\n1) Loading GPT-2 model...")
    model, tokenizer = get_gptmodel('gpt2')
    model = model.to(device)
    model.eval()

    # Load datasets
    print("\n2) Loading datasets...")
    forget_ds = get_tofudataset("forget10")
    retain_ds = get_tofudataset("retain90")

    max_length = 512
    batch_size = 8

    # Tokenize retain set (same for all)
    print("\n3) Tokenizing retain set...")
    retain_tokenized = retain_ds.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True
    )
    retain_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    retain_loader = DataLoader(retain_tokenized, batch_size=batch_size, shuffle=False, collate_fn=collator)

    # Pre-sample author indices for reproducibility
    print("\n4) Sampling author indices...")
    author_sampled_indices = {}
    for author in authors:
        author_indices = author_to_samples[author]
        if len(author_indices) > samples_per_author:
            sampled = random.sample(author_indices, samples_per_author)
        else:
            sampled = author_indices
        author_sampled_indices[author] = sampled
        print(f"  {author:20s}: {len(sampled):2d} samples (from {len(author_indices)})")

    # Store results for all dict_sizes
    all_results = {}

    # Track timing
    total_start = time.time()
    total_ops = len(dict_sizes) * 12 * len(authors)
    completed_ops = 0

    # OUTER LOOP: For each dictionary size
    for dict_idx, dict_size in enumerate(dict_sizes, 1):
        dict_start = time.time()

        print(f"\n{'#' * 70}")
        print(f"DICTIONARY SIZE: {dict_size} [{dict_idx}/{len(dict_sizes)}]")
        print(f"{'#' * 70}")

        # Initialize results structure for this dict_size
        all_results[f'sae_{dict_size}'] = {
            'dict_size': dict_size,
            'authors': authors,
            'layers': list(range(12)),
            'thresholds': thresholds,
            'samples_per_author': samples_per_author,
            'superposition_scores': {}
        }

        # MIDDLE LOOP: For each layer
        for layer_idx in range(12):
            layer_start = time.time()

            print(f"\n{'=' * 70}")
            print(f"DICT_SIZE={dict_size}, LAYER {layer_idx} [{layer_idx + 1}/12]")
            print(f"{'=' * 70}")

            # Load SAE for this layer and dict_size
            print(f"Loading SAE...")
            sae = load_sae(layer_idx, dict_size).to(device)
            sae.eval()

            # Extract retain set features
            print(f"Extracting retain set features...")
            retain_features = get_sae_feature_activations(
                model, sae, retain_loader, layer_idx, device, max_batches=50
            )
            print(f"  Retain features: {retain_features.shape} ({retain_features.shape[0]:,} tokens)")

            # BATCHED PROCESSING: Process all authors for this layer
            print(f"Processing {len(authors)} authors...")

            for author_idx, author in enumerate(tqdm(authors, desc=f"Authors (L{layer_idx})")):
                # Get sampled indices for this author
                sampled_indices = author_sampled_indices[author]

                # Verify indices are valid
                if max(sampled_indices) >= len(forget_ds):
                    print(f"⚠️  WARNING: {author} has invalid indices, skipping")
                    continue

                # Create dataloader for this author
                author_subset = forget_ds.select(sampled_indices)
                author_tokenized = author_subset.map(
                    lambda x: tokenize_function(x, tokenizer, max_length),
                    batched=True
                )
                author_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
                author_loader = DataLoader(author_tokenized, batch_size=batch_size, shuffle=False, collate_fn=collator)

                # Extract author features
                author_features = get_sae_feature_activations(
                    model, sae, author_loader, layer_idx, device
                )

                # Compute metrics with multiple thresholds
                metrics = compute_superposition_metrics_multithreshold(
                    author_features, retain_features, thresholds
                )

                # Store results
                if author not in all_results[f'sae_{dict_size}']['superposition_scores']:
                    all_results[f'sae_{dict_size}']['superposition_scores'][author] = {}

                all_results[f'sae_{dict_size}']['superposition_scores'][author][str(layer_idx)] = metrics

                completed_ops += 1

            # Clear memory
            del sae, retain_features
            torch.cuda.empty_cache()

            layer_time = time.time() - layer_start
            print(f"  ⏱️  Layer {layer_idx} completed in {layer_time / 60:.1f} min")

            # Calculate ETA
            elapsed = time.time() - total_start
            avg_time_per_op = elapsed / completed_ops if completed_ops > 0 else 0
            remaining_ops = total_ops - completed_ops
            eta_seconds = avg_time_per_op * remaining_ops
            print(f"  📊 Progress: {completed_ops}/{total_ops} ops ({100 * completed_ops / total_ops:.1f}%)")
            print(f"  ⏰ ETA: {eta_seconds / 3600:.1f} hours")

            # Save checkpoint after each layer
            checkpoint_path = f'../data/superposition_all_dict_sizes_checkpoint.json'
            print(f"  💾 Saving checkpoint...")
            with open(checkpoint_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"  ✅ Checkpoint saved")

        dict_time = time.time() - dict_start
        print(f"\n✅ Dict_size {dict_size} completed in {dict_time / 3600:.2f} hours")

    # Save final results
    output_path = '../data/superposition_all_dict_sizes_COMPLETE.json'
    print(f"\n💾 Saving final results...")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    total_time = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"✅ COMPLETE! Total time: {total_time / 3600:.2f} hours")
    print(f"Saved to: {output_path}")
    print("=" * 70)

    # Print summary statistics
    print(f"\nSummary:")
    for dict_size in dict_sizes:
        key = f'sae_{dict_size}'
        if key in all_results:
            num_authors = len(all_results[key]['superposition_scores'])
            num_layers = len(all_results[key]['layers'])
            print(f"  {key}: {num_authors} authors × {num_layers} layers")

    print(f"\nTotal measurements: {len(dict_sizes) * len(authors) * 12:,}")
    print(f"Thresholds tested: {thresholds}")
    print(f"Samples per author: {samples_per_author}")


if __name__ == "__main__":
    measure_superposition_all_dict_sizes()