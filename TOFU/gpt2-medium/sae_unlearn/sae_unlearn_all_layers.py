"""
COMPREHENSIVE SAE-GUIDED UNLEARNING - ALL LAYERS × ALL DICT SIZES
===================================================================

Tests ALL combinations:
- Dict sizes: [4096, 8192, 16384, 32768, 65536]  # adjust in Config as needed
- Layers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
- Total: len(DICT_SIZES) × 12 configurations

Each config saves results to: ../results/comprehensive_gpt2_small/dict_{size}/layer_{idx}/

Author: Amartya Hatua
Date: January 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import numpy as np
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
import copy
from scipy import stats
import random
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
import time
from datetime import datetime

from get_dataset import get_tofudataset, tokenize_function
from get_model import get_gptmodel

# Set seed
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)
# torch.cuda.empty_cache()

# ==============================================================================
# CONFIGURATION
# ==============================================================================
df_config = pd.read_json('sae_unlearning_configs.json')


class Config:
    """Configuration for comprehensive_gpt2_small experiments"""

    # Model
    MODEL_NAME = 'gpt2-medium'
    N_LAYERS = 24

    # SAE configurations - ALL COMBINATIONS
    DICT_SIZES = [4096, 8192, 16384, 32768, 65536]
    # DICT_SIZES = [32768]
    LAYERS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

    # SAE paths
    # Expecting files like: ../models/dict_{dict_size}/layer_{layer_idx}.pt
    SAE_BASE_PATH = '../models'

    # Results directory
    RESULTS_BASE_DIR = '../results/comprehensive_gpt2_medium'

    # TOFU dataset
    FORGET_SPLIT = 'forget10'
    RETAIN_SPLIT = 'retain90'

    # SAE-Guided parameters
    TOP_K_FEATURES = 128  # more targeted; tune {32, 64, 128}

    # Gradient Ascent parameters
    GA_STEPS = 5
    GA_LEARNING_RATE = 1e-4

    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Authors (from ED calibration)
    HIGH_ED_AUTHORS = [
        'Rajeev Majumdar', 'Jun Chen', 'Basil Mahfouz',
        'Hsiao Yun', 'Carmen Montenegro', 'Behrouz Rohani',
        'Yeon Park', 'Kalkidan Abera', 'Takashi Nakamura'
    ]

    LOW_ED_AUTHORS = [
        'Raven Marais', 'Xin Lee', 'Adib Jarrah',
        'Moshe Ben', 'Hina Ameen', 'Elvin Mammadov',
        'Nikolai Abilov', 'Jad Ambrose', 'Patrick Sullivan', 'Aysha Al'
    ]

    @staticmethod
    def get_date_string():
        """Get current date in '23_Jan' format"""
        return datetime.now().strftime("%d_%b")


# ==============================================================================
# SPARSE AUTOENCODER CLASSES
# ==============================================================================

class AnthropicSAE(nn.Module):
    """Anthropic-style SAE with encode/decode methods"""

    def __init__(self, d_model, dict_size):
        super().__init__()
        self.d_model = d_model
        self.dict_size = dict_size
        self.encoder = nn.Linear(d_model, dict_size)
        self.decoder = nn.Linear(dict_size, d_model, bias=False)
        torch.nn.init.normal_(self.decoder.weight, std=0.02)

    def forward(self, x):
        acts = F.relu(self.encoder(x))
        recon = self.decoder(acts)
        return recon, acts

    def encode(self, x):
        return F.relu(self.encoder(x))

    def decode(self, features):
        return self.decoder(features)


class SparseAutoencoder(nn.Module):
    """Standard SAE architecture (fallback)"""

    def __init__(self, d_model, dict_size):
        super().__init__()
        self.d_model = d_model
        self.dict_size = dict_size
        self.encoder = nn.Linear(d_model, dict_size, bias=True)
        self.decoder = nn.Linear(dict_size, d_model, bias=True)

    def forward(self, x):
        pre_activation = self.encoder(x)
        feature_acts = F.relu(pre_activation)
        x_reconstruct = self.decoder(feature_acts)
        return x_reconstruct, feature_acts

    def encode(self, x):
        return F.relu(self.encoder(x))

    def decode(self, features):
        return self.decoder(features)


def load_sae_flexible(checkpoint, d_model, dict_size, device='cuda'):
    """Load SAE handling different formats"""

    sae = AnthropicSAE(d_model, dict_size).to(device)
    state_dict = checkpoint.get('state_dict', checkpoint)

    if 'W_enc' in state_dict:
        # TransformerLens/SAELens format
        new_state_dict = {
            'encoder.weight': state_dict['W_enc'].T,
            'encoder.bias': state_dict['b_enc'],
            'decoder.weight': state_dict['W_dec'].T,
        }
        try:
            sae.load_state_dict(new_state_dict, strict=True)
        except Exception:
            sae = SparseAutoencoder(d_model, dict_size).to(device)
            new_state_dict['decoder.bias'] = state_dict.get('b_dec', torch.zeros(d_model))
            sae.load_state_dict(new_state_dict, strict=True)

    elif 'encoder.weight' in state_dict:
        # Standard format
        has_decoder_bias = 'decoder.bias' in state_dict
        if not has_decoder_bias:
            sae.load_state_dict(state_dict, strict=True)
        else:
            sae = SparseAutoencoder(d_model, dict_size).to(device)
            sae.load_state_dict(state_dict, strict=True)
    else:
        raise ValueError(f"Unknown SAE format. Keys: {list(state_dict.keys())}")

    sae.eval()
    return sae


def load_sae(dict_size, layer_idx, device='cuda'):
    """Load pre-trained SAE for specific dict size and layer"""

    # Example expected path: ../models/dict_4096/layer_3.pt
    sae_path = f"{Config.SAE_BASE_PATH}/dict_{dict_size}/layer_{layer_idx}.pt"

    print(f"    Loading SAE: dict={dict_size}, layer={layer_idx}")
    print(f"    Path: {sae_path}")

    checkpoint = torch.load(sae_path, map_location=device)

    # Extract metadata
    d_model = checkpoint.get('d_model', checkpoint.get('cfg', {}).get('d_in', 1024))
    dict_size_checkpoint = checkpoint.get('dict_size', checkpoint.get('cfg', {}).get('d_sae', dict_size))

    # Verify dict size matches
    if dict_size_checkpoint != dict_size:
        print(f"    ⚠️  Warning: Checkpoint dict_size={dict_size_checkpoint}, expected={dict_size}")

    sae = load_sae_flexible(checkpoint, d_model, dict_size, device)
    print(f"    ✓ SAE loaded")

    return sae


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_tofu_data(author_indices, tokenizer):
    """Load TOFU datasets for a given author's forget subset"""
    BATCH_SIZE = 8
    MAX_LENGTH = 512

    forget_full = get_tofudataset(Config.FORGET_SPLIT)
    retain_ds = get_tofudataset(Config.RETAIN_SPLIT)

    # Ensure tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Select author's forget samples
    forget_ds = forget_full.select(author_indices)

    # Tokenize
    retain_tokenized = retain_ds.map(
        lambda x: tokenize_function(x, tokenizer, MAX_LENGTH),
        batched=True
    )
    forget_tokenized = forget_ds.map(
        lambda x: tokenize_function(x, tokenizer, MAX_LENGTH),
        batched=True
    )

    retain_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    forget_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    forget_loader = DataLoader(
        forget_tokenized, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator
    )
    retain_loader = DataLoader(
        retain_tokenized, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator
    )

    return retain_loader, forget_loader


# ==============================================================================
# SAE-GUIDED UNLEARNING
# ==============================================================================

class SAEGuidedUnlearning:
    """SAE-Guided feature suppression"""

    def __init__(self, model, sae, tokenizer, layer_idx, device, suppression_scale, interp_alpha, feature_multiplier):
        self.model = model
        self.sae = sae
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self.device = device
        self.suppression_scale = suppression_scale
        self.alpha = interp_alpha
        self.feature_multiplier = feature_multiplier

    def get_activations(self, dataloader, max_batches=None):
        """Extract layer activations as per-token vectors"""
        all_token_acts = []

        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break

            inputs = {
                k: v.to(self.device)
                for k, v in batch.items()
                if k in ['input_ids', 'attention_mask']
            }

            captured = []

            def hook(module, input, output):
                # Handle both tensor and tuple outputs
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                captured.append(hidden.detach().cpu())

            handle = self.model.transformer.h[self.layer_idx].register_forward_hook(hook)

            with torch.no_grad():
                _ = self.model(**inputs)

            handle.remove()

            if len(captured) == 0:
                continue

            acts = captured[0]
            attention_mask = batch['attention_mask'].cpu()

            for i in range(acts.shape[0]):
                mask = attention_mask[i].bool()
                valid_acts = acts[i][mask]
                for j in range(valid_acts.shape[0]):
                    all_token_acts.append(valid_acts[j])

        if len(all_token_acts) == 0:
            return torch.empty(0, self.model.config.n_embd).to(self.device)

        all_acts = torch.stack(all_token_acts, dim=0).to(self.device)
        return all_acts

    def identify_top_features_contrast(self, forget_loader, retain_loader,
                                       k=50, max_batches=10):
        """
        Identify top-k SAE features that are most specific to the forget set:
        rank by mean_forget - mean_retain.
        """
        forget_acts = self.get_activations(forget_loader, max_batches=max_batches)
        retain_acts = self.get_activations(retain_loader, max_batches=max_batches)

        if forget_acts.shape[0] == 0 or retain_acts.shape[0] == 0:
            return torch.tensor([], dtype=torch.long, device=self.device)

        with torch.no_grad():
            f_feats = self.sae.encode(forget_acts)
            r_feats = self.sae.encode(retain_acts)

        f_mean = f_feats.mean(dim=0)
        r_mean = r_feats.mean(dim=0)
        contrast = f_mean - r_mean  # positive => forget-specific

        k = min(k, contrast.numel())
        _, top_idx = torch.topk(contrast, k=k)
        return top_idx.to(self.device)

    def create_suppression_hook(self, target_features):
        """Create hook that suppresses specific SAE features and blends back"""

        suppression_scale = self.suppression_scale
        alpha = self.alpha

        def hook(module, input, output):
            # Handle both tensor and tuple outputs
            if isinstance(output, tuple):
                hidden = output[0]
                rest = output[1:]
            else:
                hidden = output
                rest = ()

            shape = hidden.shape
            flat = hidden.reshape(-1, shape[-1])

            # Encode to SAE features
            features = self.sae.encode(flat)

            # Suppress target features by multiplying by suppression_scale
            if features.shape[-1] > 0 and len(target_features) > 0:
                features[:, target_features] *= suppression_scale * self.feature_multiplier

            # Decode back to hidden space
            recon = self.sae.decode(features)
            recon = recon.reshape(shape)

            # Blend instead of hard replacement
            reconstructed = hidden + self.alpha * (recon - hidden)

            if isinstance(output, tuple):
                return (reconstructed,) + rest
            else:
                return reconstructed

        return hook

    def evaluate_loss(self, dataloader, hook_fn=None, max_batches=None):
        """Evaluate average loss with optional intervention hook"""
        handle = None
        if hook_fn is not None:
            handle = self.model.transformer.h[self.layer_idx].register_forward_hook(hook_fn)

        total_loss = 0.0
        n_samples = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches and batch_idx >= max_batches:
                    break

                inputs = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**inputs)

                total_loss += outputs.loss.item() * inputs['input_ids'].shape[0]
                n_samples += inputs['input_ids'].shape[0]

        if handle is not None:
            handle.remove()

        return total_loss / n_samples if n_samples > 0 else 0.0

    def run(self, forget_loader, retain_loader):
        """Run SAE-guided unlearning"""

        # Identify top-K contrastive features (forget vs retain)
        target_features = self.identify_top_features_contrast(
            forget_loader, retain_loader,
            k=Config.TOP_K_FEATURES,
            max_batches=10
        )

        if len(target_features) == 0:
            return None

        hook_fn = self.create_suppression_hook(target_features)

        # Evaluate baseline and suppressed losses
        forget_base = self.evaluate_loss(forget_loader)
        retain_base = self.evaluate_loss(retain_loader)
        forget_supp = self.evaluate_loss(forget_loader, hook_fn)
        retain_supp = self.evaluate_loss(retain_loader, hook_fn)

        results = {
            'forget_baseline': forget_base,
            'forget_suppressed': forget_supp,
            'forget_increase': forget_supp - forget_base,
            'retain_baseline': retain_base,
            'retain_suppressed': retain_supp,
            'retain_change': retain_supp - retain_base,
            'perplexity_base': np.exp(retain_base),
            'perplexity_supp': np.exp(retain_supp)
        }

        return results


# ==============================================================================
# GRADIENT ASCENT BASELINE
# ==============================================================================

def run_gradient_ascent(model, tokenizer, forget_loader, retain_loader, device):
    """Run gradient ascent baseline"""

    model_copy = copy.deepcopy(model)
    model_copy.to(device)
    optimizer = torch.optim.Adam(model_copy.parameters(), lr=Config.GA_LEARNING_RATE)

    model_copy.eval()
    with torch.no_grad():
        forget_before = compute_loss(model_copy, forget_loader, device)
        retain_before = compute_loss(model_copy, retain_loader, device)

    model_copy.train()
    forget_iter = iter(forget_loader)

    for step in range(Config.GA_STEPS):
        try:
            batch = next(forget_iter)
        except StopIteration:
            forget_iter = iter(forget_loader)
            batch = next(forget_iter)

        inputs = {k: v.to(device) for k, v in batch.items()}
        outputs = model_copy(**inputs)
        loss = -outputs.loss  # maximize forget loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_copy.parameters(), max_norm=1.0)
        optimizer.step()

    model_copy.eval()
    with torch.no_grad():
        forget_after = compute_loss(model_copy, forget_loader, device)
        retain_after = compute_loss(model_copy, retain_loader, device)

    del model_copy
    torch.cuda.empty_cache()

    results = {
        'forget_before': forget_before,
        'forget_after': forget_after,
        'forget_increase': forget_after - forget_before,
        'retain_before': retain_before,
        'retain_after': retain_after,
        'retain_change': retain_after - retain_before,
        'perplexity_before': np.exp(retain_before),
        'perplexity_after': np.exp(retain_after)
    }

    return results


def compute_loss(model, dataloader, device):
    """Compute average loss"""
    if dataloader is None:
        return 0.0

    total_loss = 0.0
    n_samples = 0
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            batch_size = inputs['input_ids'].shape[0]
            total_loss += outputs.loss.item() * batch_size
            n_samples += batch_size

    return total_loss / n_samples if n_samples > 0 else 0.0


# ==============================================================================
# EXPERIMENT RUNNER
# ==============================================================================

def run_single_author(author_name, author_indices, group, model, sae, tokenizer,
                      layer_idx, dict_size, dict_config):
    """Run experiment for one author"""

    print(f"      Processing {author_name}...")

    # Load data
    try:
        retain_loader, forget_loader = load_tofu_data(author_indices, tokenizer)
    except Exception as e:
        print(f"        ❌ Failed to load data: {e}")
        return None

    # Initialize SAE unlearner
    sae_unlearner = SAEGuidedUnlearning(
        model, sae, tokenizer,
        layer_idx=layer_idx,
        device=Config.DEVICE,
        suppression_scale=dict_config['suppression_scale'],
        interp_alpha=dict_config['interp_alpha'],
        feature_multiplier=dict_config['feature_multiplier']
    )

    # Run SAE-guided
    try:
        sae_results = sae_unlearner.run(forget_loader, retain_loader)
        if sae_results is None:
            print(f"        ❌ SAE unlearning returned None")
            return None
    except Exception as e:
        print(f"        ❌ SAE unlearning failed: {e}")
        return None

    # Run gradient ascent
    try:
        ga_results = run_gradient_ascent(
            model, tokenizer,
            forget_loader, retain_loader,
            Config.DEVICE
        )
    except Exception as e:
        print(f"        ❌ GA failed: {e}")
        return None

    # Combine results
    combined = {
        'author': author_name,
        'group': group,
        'dict_size': dict_size,
        'layer': layer_idx,
        'sae_forget_increase': sae_results['forget_increase'],
        'sae_retain_change': sae_results['retain_change'],
        'sae_perplexity_change': sae_results['perplexity_supp'] - sae_results['perplexity_base'],
        'ga_forget_increase': ga_results['forget_increase'],
        'ga_retain_change': ga_results['retain_change'],
        'ga_perplexity_change': ga_results['perplexity_after'] - ga_results['perplexity_before'],
        'retain_benefit': ga_results['retain_change'] - sae_results['retain_change']
    }

    # Debug: print forget / retain changes and selectivity
    print("\n--- DEBUG (per-author) ---")
    print(f" GA Forget Change   : {ga_results['forget_increase']}")
    print(f" GA Retain Change   : {ga_results['retain_change']}")
    print(f" SAE Forget Change  : {sae_results['forget_increase']}")
    print(f" SAE Retain Change  : {sae_results['retain_change']}")

    eps = 1e-8
    ga_sel = ga_results['forget_increase'] / (abs(ga_results['retain_change']) + eps)
    sae_sel = sae_results['forget_increase'] / (abs(sae_results['retain_change']) + eps)

    print(f" GA Selectivity     : {ga_sel}")
    print(f" SAE Selectivity    : {sae_sel}")
    print("--- END DEBUG ---\n")

    return combined


def run_single_config(dict_size, layer_idx, model, tokenizer, author_to_samples, dict_config, date_str):
    """Run experiments for ONE configuration (dict_size × layer)"""

    print(f"\n  {'=' * 76}")
    print(f"  CONFIG: Dict={dict_size}, Layer={layer_idx}")
    print(f"  {'=' * 76}")

    # Setup results directory
    results_dir = Path(Config.RESULTS_BASE_DIR) / f"dict_{dict_size}" / f"layer_{layer_idx:02d}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load SAE
    try:
        sae = load_sae(dict_size, layer_idx, Config.DEVICE)
    except Exception as e:
        print(f"    ❌ Failed to load SAE: {e}")
        return None

    # Collect results
    all_results = []

    # HIGH-ED AUTHORS
    print(f"\n    HIGH-ED AUTHORS:")
    for author in Config.HIGH_ED_AUTHORS:
        if author not in author_to_samples:
            print(f"      ⚠️  Skipping {author} - not in mapping")
            continue

        author_indices = author_to_samples[author]
        result = run_single_author(
            author, author_indices, 'high_ed',
            model, sae, tokenizer, layer_idx, dict_size, dict_config
        )
        if result:
            all_results.append(result)

    # LOW-ED AUTHORS
    print(f"\n    LOW-ED AUTHORS:")
    for author in Config.LOW_ED_AUTHORS:
        if author not in author_to_samples:
            print(f"      ⚠️  Skipping {author} - not in mapping")
            continue

        author_indices = author_to_samples[author]
        result = run_single_author(
            author, author_indices, 'low_ed',
            model, sae, tokenizer, layer_idx, dict_size, dict_config
        )
        if result:
            all_results.append(result)

    # Save results with date
    if len(all_results) > 0:
        df = pd.DataFrame(all_results)
        output_path = results_dir / f"results_{date_str}.csv"
        df.to_csv(output_path, index=False, float_format='%.6f')
        print(f"\n    💾 Saved: {output_path}")

        # Quick stats
        high = df[df['group'] == 'high_ed']['retain_benefit']
        low = df[df['group'] == 'low_ed']['retain_benefit']

        print(f"\n    📊 SUMMARY:")
        print(f"      High-ED: {high.mean():.4f} ± {high.std():.4f}")
        print(f"      Low-ED:  {low.mean():.4f} ± {low.std():.4f}")

        # Cleanup
        del sae
        torch.cuda.empty_cache()

        return df
    else:
        print(f"\n    ❌ No results collected")
        return None


# ==============================================================================
# SUMMARY / HEATMAP HELPERS
# ==============================================================================

def create_summary_statistics(df, date_str):
    """Create summary tables with date in filename"""

    results_base = Path(Config.RESULTS_BASE_DIR)

    print(f"\n{'=' * 80}")
    print("CREATING SUMMARY STATISTICS")
    print(f"{'=' * 80}")

    # Summary by dict_size
    summary_by_dict = df.groupby('dict_size').agg({
        'sae_retain_change': ['mean', 'std'],
        'ga_retain_change': ['mean', 'std'],
        'retain_benefit': ['mean', 'std']
    }).round(4)
    summary_by_dict.to_csv(results_base / f"summary_by_dict_size_{date_str}.csv")
    print(f"\n✓ Saved: summary_by_dict_size_{date_str}.csv")

    # Summary by layer
    summary_by_layer = df.groupby('layer').agg({
        'sae_retain_change': ['mean', 'std'],
        'ga_retain_change': ['mean', 'std'],
        'retain_benefit': ['mean', 'std']
    }).round(4)
    summary_by_layer.to_csv(results_base / f"summary_by_layer_{date_str}.csv")
    print(f"✓ Saved: summary_by_layer_{date_str}.csv")

    # Summary by dict_size × layer
    summary_by_both = df.groupby(['dict_size', 'layer']).agg({
        'retain_benefit': ['mean', 'std', 'count']
    }).round(4)
    summary_by_both.to_csv(results_base / f"summary_by_dict_and_layer_{date_str}.csv")
    print(f"✓ Saved: summary_by_dict_and_layer_{date_str}.csv")

    # Summary by group
    summary_by_group = df.groupby(['group', 'dict_size', 'layer']).agg({
        'retain_benefit': ['mean', 'std']
    }).round(4)
    summary_by_group.to_csv(results_base / f"summary_by_group_{date_str}.csv")
    print(f"✓ Saved: summary_by_group_{date_str}.csv")


def create_heatmap_data(df, date_str):
    """Create heatmap data for visualization with date in filename"""

    results_base = Path(Config.RESULTS_BASE_DIR)

    print(f"\n{'=' * 80}")
    print("CREATING HEATMAP DATA")
    print(f"{'=' * 80}")

    # For each dict size, create layer × effectiveness matrix
    for dict_size in Config.DICT_SIZES:
        dict_df = df[df['dict_size'] == dict_size]

        if len(dict_df) == 0:
            continue

        # Pivot table: layer × metric
        heatmap_data = dict_df.groupby('layer').agg({
            'sae_retain_change': 'mean',
            'ga_retain_change': 'mean',
            'retain_benefit': 'mean'
        }).round(4)

        output_path = results_base / f"heatmap_dict_{dict_size}_{date_str}.csv"
        heatmap_data.to_csv(output_path)
        print(f"✓ Saved: heatmap_dict_{dict_size}_{date_str}.csv")

    # Overall heatmap: dict_size × layer
    heatmap_overall = df.pivot_table(
        index='dict_size',
        columns='layer',
        values='retain_benefit',
        aggfunc='mean'
    ).round(4)

    output_path = results_base / f"heatmap_overall_{date_str}.csv"
    heatmap_overall.to_csv(output_path)
    print(f"✓ Saved: heatmap_overall_{date_str}.csv")

    print("\n📊 Heatmap data ready for visualization!")


# ==============================================================================
# MAIN - RUN ALL CONFIGURATIONS
# ==============================================================================

def main():
    """Main entry point - runs ALL dict_size × layer combinations"""

    print("\n" + "=" * 80)
    print("COMPREHENSIVE SAE-GUIDED UNLEARNING")
    print("=" * 80)
    print(f"Device: {Config.DEVICE}")
    if Config.DEVICE == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Get date string for file naming
    date_str = Config.get_date_string()
    print(f"Date string: {date_str}")

    # Calculate total configs
    total_configs = len(Config.DICT_SIZES) * len(Config.LAYERS)
    print(f"\nDict sizes: {Config.DICT_SIZES}")
    print(f"Layers: {Config.LAYERS}")
    print(f"Total configurations: {total_configs}")

    print(f"\nAuthors: {len(Config.HIGH_ED_AUTHORS + Config.LOW_ED_AUTHORS)}")
    print(f"  High-ED: {len(Config.HIGH_ED_AUTHORS)}")
    print(f"  Low-ED: {len(Config.LOW_ED_AUTHORS)}")

    # Load shared resources
    print(f"\n{'=' * 80}")
    print("LOADING SHARED RESOURCES")
    print(f"{'=' * 80}")

    # Load model
    print("\nLoading model...")
    model, tokenizer = get_gptmodel(Config.MODEL_NAME)
    model.to(Config.DEVICE)
    model.eval()
    print(f"✓ Model loaded: {Config.MODEL_NAME}")

    # Load author mapping
    print("\nLoading author mapping...")
    with open('../data/tofu_author_mapping.json', 'r') as f:
        author_data = json.load(f)
    author_to_samples = author_data['author_to_samples']
    print(f"✓ Author mapping loaded ({len(author_to_samples)} authors)")

    # Create results structure - organize by dict_size
    results_by_dict_size = {dict_size: [] for dict_size in Config.DICT_SIZES}

    # Progress tracking
    completed = 0
    start_time = time.time()

    # MAIN LOOP: For each dict_size × layer combination
    print(f"\n{'=' * 80}")
    print("RUNNING ALL CONFIGURATIONS")
    print(f"{'=' * 80}")

    for dict_size in Config.DICT_SIZES:
        dict_config = df_config[dict_size]
        for layer_idx in Config.LAYERS:
            config_name = f"dict_{dict_size}_layer_{layer_idx:02d}"

            print(f"\n{'#' * 80}")
            print(f"CONFIG {completed + 1}/{total_configs}: {config_name}")
            print(f"{'#' * 80}")

            try:
                # Run this configuration
                results = run_single_config(
                    dict_size, layer_idx,
                    model, tokenizer,
                    author_to_samples,
                    dict_config,
                    date_str
                )

                if results is not None:
                    results_by_dict_size[dict_size].append(results)
                    print(f"\n  ✅ {config_name} COMPLETE!")
                else:
                    print(f"\n  ⚠️  {config_name} completed with no results")

            except Exception as e:
                print(f"\n  ❌ {config_name} FAILED: {e}")
                import traceback
                traceback.print_exc()

            # Update progress
            completed += 1
            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            remaining = total_configs - completed
            eta = remaining / rate if rate > 0 else 0

            print(f"\n{'=' * 80}")
            print(f"PROGRESS: {completed}/{total_configs} ({100 * completed / total_configs:.1f}%)")
            print(f"Elapsed: {elapsed / 60:.1f} min | ETA: {eta / 60:.1f} min")
            print(f"{'=' * 80}")

    # Save results per dict_size
    print(f"\n{'=' * 80}")
    print("SAVING RESULTS PER DICT_SIZE")
    print(f"{'=' * 80}")

    results_base = Path(Config.RESULTS_BASE_DIR)
    all_results = []

    for dict_size in Config.DICT_SIZES:
        dict_results = results_by_dict_size[dict_size]

        if len(dict_results) > 0:
            # Combine all layers for this dict_size
            combined_df = pd.concat(dict_results, ignore_index=True)
            all_results.append(combined_df)

            # Save per dict_size file
            dict_path = results_base / f"dict_{dict_size}_all_layers_{date_str}.csv"
            combined_df.to_csv(dict_path, index=False, float_format='%.6f')
            print(f"\n💾 Saved dict_size {dict_size}: {dict_path}")
            print(f"   Rows: {len(combined_df)} | Layers: {combined_df['layer'].nunique()}")
        else:
            print(f"\n⚠️  No results for dict_size {dict_size}")

    # Create comprehensive summary if we have any results
    if len(all_results) > 0:
        # Combine all results
        combined_df = pd.concat(all_results, ignore_index=True)

        # Save comprehensive results
        summary_path = results_base / f"all_results_{date_str}.csv"
        combined_df.to_csv(summary_path, index=False, float_format='%.6f')
        print(f"\n💾 Saved comprehensive results: {summary_path}")
        print(f"   Total rows: {len(combined_df)}")

        # Create summary statistics
        create_summary_statistics(combined_df, date_str)

        # Create heatmap data
        create_heatmap_data(combined_df, date_str)

    # Final summary
    print(f"\n{'=' * 80}")
    print("✅ ALL CONFIGURATIONS COMPLETE")
    print(f"{'=' * 80}")

    success_count = sum(len(dfs) for dfs in results_by_dict_size.values())
    print(f"\nSuccessful: {success_count}/{total_configs}")

    total_time = time.time() - start_time
    print(f"Total time: {total_time / 60:.1f} minutes ({total_time / 3600:.2f} hours)")

    return results_by_dict_size


if __name__ == "__main__":
    results = main()