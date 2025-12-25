import torch
import json
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os

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


def measure_superposition_all_dict_sizes():
    """Main function to measure superposition for all dict_sizes and all authors"""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Dictionary sizes to test
    dict_sizes = [4096, 8192, 16384, 32768, 65536]

    # Load author mapping for forget10
    with open('../data/tofu_author_mapping_forget10.json', 'r') as f:
        author_data = json.load(f)

    author_to_samples = author_data['author_to_samples']
    authors = list(author_to_samples.keys())

    print(f"\n{'=' * 70}")
    print(f"MEASURING SUPERPOSITION FOR ALL DICTIONARY SIZES")
    print("=" * 70)
    print(f"Dictionary sizes: {dict_sizes}")
    print(f"Authors: {len(authors)}")
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
    batch_size = 2

    # Tokenize retain set (same for all authors)
    retain_tokenized = retain_ds.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True
    )
    retain_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    retain_loader = DataLoader(retain_tokenized, batch_size=batch_size, shuffle=False, collate_fn=collator)

    # Store results for all dict_sizes
    all_results = {}

    # OUTER LOOP: For each dictionary size
    for dict_size in dict_sizes:
        print(f"\n{'#' * 70}")
        print(f"DICTIONARY SIZE: {dict_size}")
        print(f"{'#' * 70}")

        # Initialize results structure for this dict_size
        all_results[f'sae_{dict_size}'] = {
            'dict_size': dict_size,
            'authors': authors,
            'layers': list(range(12)),
            'superposition_scores': {}
        }

        # MIDDLE LOOP: For each layer
        for layer_idx in range(12):
            print(f"\n{'=' * 70}")
            print(f"DICT_SIZE={dict_size}, LAYER {layer_idx}")
            print(f"{'=' * 70}")

            # Load SAE for this layer and dict_size
            print(f"Loading SAE (dict_size={dict_size}, layer={layer_idx})...")
            sae = load_sae(layer_idx, dict_size).to(device)
            sae.eval()

            # Extract retain set features
            print(f"Extracting retain set features...")
            retain_features = get_sae_feature_activations(
                model, sae, retain_loader, layer_idx, device, max_batches=25
            )
            print(f"  Retain features shape: {retain_features.shape}")

            # INNER LOOP: For each author
            for author_idx, author in enumerate(authors, 1):
                print(f"\n  [{author_idx}/{len(authors)}] Author: {author}")

                # Get author's sample indices
                author_indices = author_to_samples[author]
                print(f"      Samples: {len(author_indices)}")

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
                if author not in all_results[f'sae_{dict_size}']['superposition_scores']:
                    all_results[f'sae_{dict_size}']['superposition_scores'][author] = {}

                all_results[f'sae_{dict_size}']['superposition_scores'][author][str(layer_idx)] = metrics

                print(f"      Jaccard: {metrics['jaccard_similarity']:.4f}, "
                      f"Cosine: {metrics['cosine_similarity']:.4f}, "
                      f"Features: {metrics['num_author_features']}")

            # Clear memory
            del sae, retain_features
            torch.cuda.empty_cache()

            # Save checkpoint after each layer
            checkpoint_path = f'../data/superposition_all_dict_sizes_checkpoint.json'
            with open(checkpoint_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\n  💾 Checkpoint saved (dict_size={dict_size}, layer={layer_idx} complete)")

        print(f"\n{'#' * 70}")
        print(f"✅ COMPLETED ALL LAYERS FOR DICT_SIZE={dict_size}")
        print(f"{'#' * 70}")

    # Save final results
    output_path = '../data/superposition_all_dict_sizes_COMPLETE.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"✅ Superposition analysis complete for ALL dictionary sizes!")
    print(f"Saved to: {output_path}")
    print("=" * 70)

    # Print summary statistics
    print(f"\nSummary:")
    for dict_size in dict_sizes:
        key = f'sae_{dict_size}'
        num_authors = len(all_results[key]['superposition_scores'])
        print(f"  {key}: {num_authors} authors × 12 layers")

    print(f"\nTotal measurements: {len(dict_sizes)} dict_sizes × {len(authors)} authors × 12 layers")
    print(f"                  = {len(dict_sizes) * len(authors) * 12} total metric sets")


if __name__ == "__main__":
    measure_superposition_all_dict_sizes()