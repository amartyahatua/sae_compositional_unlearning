# measure_superposition_forget10.py
import torch
import json
import numpy as np
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from get_dataset import get_tofudataset, tokenize_function
from get_model import get_gptmodel
from train_saes_all_layers import SparseAutoencoder


def load_sae(layer_idx, sae_dir='../models/saes_gpt2'):
    """Load trained SAE for a specific layer"""
    checkpoint = torch.load(f'{sae_dir}/sae_layer_{layer_idx}.pt', map_location='cpu')
    d_model = checkpoint['d_model']
    dict_size = checkpoint['dict_size']

    sae = SparseAutoencoder(d_model, dict_size)
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
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Layer {layer_idx} features", leave=False)):
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

    # Concatenate all features
    all_features = torch.cat(all_features, dim=0)  # (total_tokens, dict_size)
    return all_features


def compute_superposition_metrics(author_features, retain_features, threshold=0.01):
    """
    Compute superposition metrics between author and retain set features
    """
    # 1. Identify which features are active (above threshold)
    author_active = (author_features > threshold).float().mean(dim=0)  # (dict_size,)
    retain_active = (retain_features > threshold).float().mean(dim=0)  # (dict_size,)

    # 2. Jaccard similarity
    author_mask = author_active > 0.01
    retain_mask = retain_active > 0.01

    intersection = (author_mask & retain_mask).sum().item()
    union = (author_mask | retain_mask).sum().item()
    jaccard = intersection / union if union > 0 else 0

    # 3. Feature overlap percentage
    author_feature_count = author_mask.sum().item()
    overlap_pct = intersection / author_feature_count if author_feature_count > 0 else 0

    # 4. Cosine similarity
    author_mean = author_features.mean(dim=0)
    retain_mean = retain_features.mean(dim=0)
    cosine_sim = torch.nn.functional.cosine_similarity(
        author_mean.unsqueeze(0),
        retain_mean.unsqueeze(0)
    ).item()

    # 5. L2 distance
    l2_distance = torch.norm(author_mean - retain_mean).item()

    return {
        'jaccard_similarity': jaccard,
        'overlap_percentage': overlap_pct,
        'cosine_similarity': cosine_sim,
        'l2_distance': l2_distance,
        'num_author_features': author_feature_count,
        'num_retain_features': retain_mask.sum().item(),
        'num_shared_features': intersection
    }


def measure_author_superposition():
    """Main function to measure superposition for forget10 authors"""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load author mapping for forget10
    mapping_file = '../data/tofu_author_mapping_forget10.json'

    if not os.path.exists(mapping_file):
        print(f"❌ Error: {mapping_file} not found!")
        print("Run create_forget10_mapping.py first to generate the mapping.")
        return

    with open(mapping_file, 'r') as f:
        author_data = json.load(f)

    author_to_samples = author_data['author_to_samples']
    authors = list(author_to_samples.keys())

    print(f"\n{'=' * 70}")
    print(f"MEASURING SUPERPOSITION FOR {len(authors)} AUTHORS (FORGET10)")
    print("=" * 70)

    # Load model
    print("\n1) Loading GPT-2 model...")
    model, tokenizer = get_gptmodel('gpt2')
    model = model.to(device)
    model.eval()

    # Load datasets - IMPORTANT: Use forget10 and retain90!
    print("\n2) Loading datasets...")
    forget_ds = get_tofudataset("forget10")  # Changed from forget05
    retain_ds = get_tofudataset("retain90")  # Changed from retain95

    print(f"   Forget10 size: {len(forget_ds)}")
    print(f"   Retain90 size: {len(retain_ds)}")

    max_length = 512
    batch_size = 2

    # Tokenize retain set
    retain_tokenized = retain_ds.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True
    )
    retain_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    retain_loader = DataLoader(retain_tokenized, batch_size=batch_size, shuffle=False, collate_fn=collator)

    # Store results
    results = {
        'authors': authors,
        'layers': list(range(12)),
        'superposition_scores': {}
    }

    # For each layer
    for layer_idx in range(12):
        print(f"\n{'=' * 70}")
        print(f"LAYER {layer_idx}")
        print(f"{'=' * 70}")

        # Load SAE for this layer
        print(f"Loading SAE for layer {layer_idx}...")
        sae = load_sae(layer_idx).to(device)
        sae.eval()

        # Extract retain set features
        print(f"Extracting retain set features...")
        retain_features = get_sae_feature_activations(
            model, sae, retain_loader, layer_idx, device, max_batches=25
        )
        print(f"  Retain features shape: {retain_features.shape}")

        # For each author
        for author_idx, author in enumerate(authors, 1):
            print(f"\n  [{author_idx}/{len(authors)}] Author: {author}")

            # Get author's sample indices
            author_indices = author_to_samples[author]
            print(f"      Samples: {len(author_indices)}, indices range: {min(author_indices)}-{max(author_indices)}")

            # Verify indices are valid
            if max(author_indices) >= len(forget_ds):
                print(f"      ⚠️  WARNING: Max index {max(author_indices)} >= dataset size {len(forget_ds)}")
                print(f"      Skipping {author}")
                continue

            # Create dataloader for this author
            author_subset = forget_ds.select(author_indices)
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
            print(f"      Author features shape: {author_features.shape}")

            # Compute metrics
            metrics = compute_superposition_metrics(author_features, retain_features)

            # Store results
            if author not in results['superposition_scores']:
                results['superposition_scores'][author] = {}
            results['superposition_scores'][author][str(layer_idx)] = metrics

            print(f"      Jaccard: {metrics['jaccard_similarity']:.4f}, "
                  f"Overlap: {metrics['overlap_percentage']:.4f}, "
                  f"Cosine: {metrics['cosine_similarity']:.4f}")

        # Clear memory
        del sae, retain_features
        torch.cuda.empty_cache()

        # Save checkpoint after each layer
        checkpoint_path = '../data/superposition_scores_forget10_checkpoint.json'
        with open(checkpoint_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  💾 Checkpoint saved (Layer {layer_idx} complete)")

    # Save final results
    output_path = '../data/superposition_scores_forget10.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"✅ Superposition analysis complete!")
    print(f"Saved to: {output_path}")
    print("=" * 70)

    # Summary
    print(f"\nSummary (Layer 11 Jaccard):")
    for author in authors:
        if author in results['superposition_scores']:
            jaccard = results['superposition_scores'][author]['11']['jaccard_similarity']
            print(f"  {author:<40s}: {jaccard:.4f}")


if __name__ == "__main__":
    measure_author_superposition()