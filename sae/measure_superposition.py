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
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Layer {layer_idx} features")):
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

    Args:
        author_features: (num_author_tokens, dict_size) - SAE activations for author samples
        retain_features: (num_retain_tokens, dict_size) - SAE activations for retain samples
        threshold: activation threshold for considering a feature "active"

    Returns:
        dict with superposition metrics
    """

    # 1. Identify which features are active (above threshold)
    author_active = (author_features > threshold).float().mean(dim=0)  # (dict_size,)
    retain_active = (retain_features > threshold).float().mean(dim=0)  # (dict_size,)

    # 2. Jaccard similarity: intersection over union of active features
    author_mask = author_active > 0.01  # Features active for >1% of author tokens
    retain_mask = retain_active > 0.01  # Features active for >1% of retain tokens

    intersection = (author_mask & retain_mask).sum().item()
    union = (author_mask | retain_mask).sum().item()
    jaccard = intersection / union if union > 0 else 0

    # 3. Feature overlap percentage: what % of author features are also in retain?
    author_feature_count = author_mask.sum().item()
    overlap_pct = intersection / author_feature_count if author_feature_count > 0 else 0

    # 4. Cosine similarity of mean activation patterns
    author_mean = author_features.mean(dim=0)
    retain_mean = retain_features.mean(dim=0)
    cosine_sim = torch.nn.functional.cosine_similarity(
        author_mean.unsqueeze(0),
        retain_mean.unsqueeze(0)
    ).item()

    # 5. L2 distance between mean activations
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
    """Main function to measure superposition for all authors across all layers"""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load author mapping
    with open('../data/tofu_author_mapping.json', 'r') as f:
        author_data = json.load(f)

    author_to_samples = author_data['author_to_samples']
    authors = list(author_to_samples.keys())

    print(f"\n{'=' * 70}")
    print(f"MEASURING SUPERPOSITION FOR {len(authors)} AUTHORS")
    print("=" * 70)

    # Load model
    print("\n1) Loading GPT-2 model...")
    model, tokenizer = get_gptmodel('gpt2')
    model = model.to(device)
    model.eval()

    # Load datasets
    print("\n2) Loading datasets...")
    forget_ds = get_tofudataset("forget05")
    retain_ds = get_tofudataset("retain95")

    max_length = 512
    batch_size = 2

    # Tokenize retain set (we'll use this for all authors)
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
        'superposition_scores': {}  # author -> layer -> metrics
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

        # Extract retain set features (same for all authors)
        print(f"Extracting retain set features...")
        retain_features = get_sae_feature_activations(
            model, sae, retain_loader, layer_idx, device, max_batches=25  # Sample retain set
        )
        print(f"  Retain features shape: {retain_features.shape}")

        # For each author
        for author in authors:
            print(f"\n  Author: {author}")

            # Get author's sample indices
            author_indices = author_to_samples[author]

            # Create dataloader for just this author's samples
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
            print(f"    Author features shape: {author_features.shape}")

            # Compute superposition metrics
            metrics = compute_superposition_metrics(author_features, retain_features)

            # Store results
            if author not in results['superposition_scores']:
                results['superposition_scores'][author] = {}
            results['superposition_scores'][author][layer_idx] = metrics

            print(f"    Jaccard similarity: {metrics['jaccard_similarity']:.4f}")
            print(f"    Overlap percentage: {metrics['overlap_percentage']:.4f}")
            print(f"    Cosine similarity: {metrics['cosine_similarity']:.4f}")

        # Clear memory
        del sae, retain_features
        torch.cuda.empty_cache()

    # Save results
    output_path = '../data/superposition_scores.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"✅ Superposition analysis complete!")
    print(f"Saved to: {output_path}")
    print("=" * 70)

    # Print summary statistics
    print(f"\nSummary across all layers:")
    for author in authors:
        avg_jaccard = np.mean([
            results['superposition_scores'][author][str(layer)]['jaccard_similarity']
            for layer in range(12)
        ])
        print(f"  {author:30s}: avg Jaccard = {avg_jaccard:.4f}")


if __name__ == "__main__":
    measure_author_superposition()