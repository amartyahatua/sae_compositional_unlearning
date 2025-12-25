"""
PHASE 3: SAE-GUIDED UNLEARNING - COMPLETE IMPLEMENTATION
========================================================

Purpose: Validate causal relationship between feature distribution and unlearning difficulty
Method: Compare SAE-guided unlearning vs gradient ascent on high-ED vs low-ED authors
Expected: High-ED authors show better retain preservation with SAE guidance

Author: Amartya Hatua
Date: December 2024
"""

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import numpy as np
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
import copy
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import random
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

from get_dataset import get_tofudataset, data_preparation, tokenize_function
from get_model import get_gptmodel
# Set seed for reproducibility
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)


# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_MODEL = "gpt2"

class Config:
    """Configuration for Phase 3 experiments"""

    # Models
    MODEL_NAME = 'gpt2'
    SAE_PATH = '../models/saes_gpt2_16384/sae_layer_8.pt'
    TARGET_LAYER = 8

    # TOFU dataset
    FORGET_SPLIT = 'forget10'
    RETAIN_SPLIT = 'retain90'

    # Experiment settings
    N_AUTHOR_SAMPLES = 20
    N_FORGET_EVAL = 50
    N_RETAIN_EVAL = 100

    # SAE-Guided parameters
    TOP_K_FEATURES = 50
    SUPPRESSION_FACTOR = 0.5

    # Gradient Ascent parameters
    GA_STEPS = 13
    GA_LEARNING_RATE = 1e-5
    GA_BATCH_SIZE = 4

    # Output
    RESULTS_DIR = '../results/phase3'

    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Authors to test
    HIGH_ED_AUTHORS = ['Carmen Montenegro', 'Basil Mahfouz', 'Patrick Sullivan']
    LOW_ED_AUTHORS = ['Nikolai Abilov', 'Hina Ameen', 'Xin Lee']



# ==============================================================================
# DATA LOADING FUNCTIONS (YOUR EXISTING FUNCTIONS)
# ==============================================================================


def data_preparation(dataset):
    """Format dataset with Alpaca prompt template"""
    alpaca_prompt = """Answer the following question:
    ### Question:
    {}

    ### Answer:
    {}"""

    def formatting_prompts_func(examples):
        texts = [alpaca_prompt.format(question, answer) for question, answer in
                 zip(examples["question"], examples["answer"])]
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)
    return dataset


# def get_tofudataset(dataset_name):
#     """Download TOFU dataset from HuggingFace"""
#     print(f"  Loading locuslab/TOFU split: {dataset_name}")
#     dataset = load_dataset("locuslab/TOFU", dataset_name, split="train")
#     processed_dataset = data_preparation(dataset)
#     return processed_dataset


# def get_gptmodel(model_name):
#     """Helper to load model and tokenizer"""
#     tokenizer = GPT2Tokenizer.from_pretrained(model_name)
#     tokenizer.pad_token = tokenizer.eos_token
#     model = GPT2LMHeadModel.from_pretrained(model_name)
#     model.config.pad_token_id = tokenizer.eos_token_id
#     return model, tokenizer



# ==============================================================================
# SPARSE AUTOENCODER (YOUR TRAINING CLASS + HELPER METHODS)
# ==============================================================================

class SparseAutoencoder(nn.Module):
    """Standard SAE architecture - matches your training code with helper methods"""

    def __init__(self, d_model, dict_size):
        super().__init__()
        self.d_model = d_model
        self.dict_size = dict_size

        # Encoder: d_model -> dict_size
        self.encoder = nn.Linear(d_model, dict_size, bias=True)
        # Decoder: dict_size -> d_model
        self.decoder = nn.Linear(dict_size, d_model, bias=True)

    def forward(self, x):
        """Full forward pass (used in training)"""
        # x: (batch, seq_len, d_model) or (batch, d_model)
        pre_activation = self.encoder(x)
        feature_acts = torch.relu(pre_activation)
        x_reconstruct = self.decoder(feature_acts)
        return x_reconstruct, feature_acts

    def encode(self, x):
        """
        Encode activations to sparse features
        Args:
            x: [N, d_model] tensor
        Returns:
            features: [N, dict_size] sparse feature activations
        """
        pre_activation = self.encoder(x)
        return torch.relu(pre_activation)

    def decode(self, features):
        """
        Decode sparse features back to activations
        Args:
            features: [N, dict_size] sparse features
        Returns:
            reconstruction: [N, d_model] reconstructed activations
        """
        return self.decoder(features)



def load_sae(sae_path, device='cuda'):
    """Load pre-trained SAE with robust checkpoint handling"""
    print(f"  Loading SAE from {sae_path}...")

    # Load checkpoint
    checkpoint = torch.load(sae_path, map_location=device)

    # Determine checkpoint format
    if 'state_dict' in checkpoint:
        # Nested format with metadata
        d_model = checkpoint.get('d_model', 768)
        dict_size = checkpoint.get('dict_size', 16384)
        state_dict = checkpoint['state_dict']
        print(f"    → Nested checkpoint format")
    elif 'encoder.weight' in checkpoint:
        # Direct state_dict
        d_model = checkpoint['encoder.weight'].shape[1]
        dict_size = checkpoint['encoder.weight'].shape[0]
        state_dict = checkpoint
        print(f"    → Direct state_dict format")
    else:
        raise ValueError(f"Unknown checkpoint format. Keys: {list(checkpoint.keys())}")

    print(f"    → d_model={d_model}, dict_size={dict_size}")

    # Initialize and load
    sae = SparseAutoencoder(d_model=d_model, dict_size=dict_size)
    sae.load_state_dict(state_dict)
    sae.to(device)
    sae.eval()

    print(f"  ✓ SAE loaded successfully")
    return sae


# ==============================================================================
# DATA EXTRACTION FROM TOFU
# ==============================================================================

def extract_author_samples(dataset, author_name, max_samples=None):
    """Extract all samples mentioning specific author"""
    samples = []

    for item in dataset:
        text = item.get('text', '')
        # Check if author name is in the text
        if author_name.lower() in text.lower():
            samples.append(text)
            if max_samples and len(samples) >= max_samples:
                break

    return samples

def causal_collator(features, tokenizer, pad_to_multiple_of=None):
    """
    inputs: features: list of dicts with keys "input_ids" and "attention_mask"
    returns: dict of tensors {input_ids, attention_mask, labels}
    labels have padding positions set to -100 so they won't contribute to loss.
    """
    input_ids = [f["input_ids"] for f in features]
    attention_mask = [f["attention_mask"] for f in features]

    # convert to plain lists if torch tensors
    input_ids = [list(x.cpu().numpy()) if hasattr(x, "cpu") else list(x) for x in input_ids]
    attention_mask = [list(x.cpu().numpy()) if hasattr(x, "cpu") else list(x) for x in attention_mask]

    batch_max = max(len(x) for x in input_ids)
    if pad_to_multiple_of is not None and batch_max % pad_to_multiple_of != 0:
        batch_max = ((batch_max + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    padded_input_ids = []
    padded_attention = []
    labels = []

    for ids, am in zip(input_ids, attention_mask):
        pad_len = batch_max - len(ids)
        padded_ids = ids + [pad_id] * pad_len
        padded_am = am + [0] * pad_len
        lbl = padded_ids.copy()
        for i in range(len(lbl)):
            if padded_am[i] == 0:
                lbl[i] = -100
        padded_input_ids.append(padded_ids)
        padded_attention.append(padded_am)
        labels.append(lbl)

    batch = {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(padded_attention, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
    return batch

def prepare_dataloader(self, dataset_name, tokenizer, max_length, batch_size):
    ds = get_tofudataset(dataset_name)

    # remove all columns except those needed by tokenize_function; safe approach:
    # first map tokenize_function, then set format to input_ids/attention_mask
    tokenized = ds.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=[c for c in ds.column_names if c not in ["question", "answer", "text", "prompt", "input"]]
    )
    # Ensure columns exist
    if "input_ids" not in tokenized.column_names or "attention_mask" not in tokenized.column_names:
        # If tokenize_function already removed columns differently, try mapping without remove_columns
        tokenized = ds.map(lambda x: tokenize_function(x, tokenizer, max_length), batched=True)

    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

    return DataLoader(
        tokenized,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda features: causal_collator(features, tokenizer, pad_to_multiple_of=8)
    )




def load_tofu_data(author_indices):
    """Load TOFU datasets"""
    BATCH_SIZE = 8
    EVAL_BATCH_SIZE = 8
    MAX_LENGTH = 512
    print("\n3) Loading datasets...")
    forget_full = get_tofudataset("forget10")
    retain_ds = get_tofudataset("retain90")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Select only this author's samples
    forget_ds = forget_full.select(author_indices)
    print(f"Forget size: {len(forget_ds)}, Retain size: {len(retain_ds)}")

    # Tokenize
    retain_tokenized = retain_ds.map(
        lambda x: tokenize_function(x, tokenizer, MAX_LENGTH),
        batched=True
    )
    forget_tokenized = forget_ds.map(
        lambda x: tokenize_function(x, tokenizer, MAX_LENGTH),
        batched=True
    )

    # Verify columns exist
    for ds in (forget_tokenized, retain_tokenized):
        if "input_ids" not in ds.column_names or "attention_mask" not in ds.column_names:
            raise RuntimeError("tokenize_function must produce 'input_ids' and 'attention_mask' columns.")

    forget_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    retain_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    forget_loader = DataLoader(
        forget_tokenized, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator
    )
    retain_loader = DataLoader(
        retain_tokenized, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator
    )

    return retain_loader, forget_loader

# ==============================================================================
# SAE-GUIDED UNLEARNING
# ==============================================================================

class SAEGuidedUnlearning:
    """SAE-Guided feature suppression for unlearning"""

    def __init__(self, model, sae, tokenizer, layer_idx, device):
        self.model = model
        self.sae = sae
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self.device = device
        self.suppression_factor = Config.SUPPRESSION_FACTOR

    def get_activations(self, dataloader, max_batches=None):
        """
        Extract layer activations from DataLoader

        Args:
            dataloader: DataLoader yielding batches with 'input_ids' and 'attention_mask'
            max_batches: Optional limit on number of batches to process

        Returns:
            all_acts: [total_tokens, hidden_dim] tensor of activations
        """
        all_token_acts = []

        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break

            # Extract inputs (filter out 'labels' if present)
            inputs = {
                k: v.to(self.device)
                for k, v in batch.items()
                if k in ['input_ids', 'attention_mask']
            }

            # Hook to capture activations
            captured = []

            def hook(module, input, output):
                captured.append(output[0].detach().cpu())

            handle = self.model.transformer.h[self.layer_idx].register_forward_hook(hook)

            with torch.no_grad():
                _ = self.model(**inputs)

            handle.remove()

            # captured[0] shape: [batch_size, seq_len, hidden_dim]
            acts = captured[0]  # Keep on CPU to save memory
            attention_mask = batch['attention_mask'].cpu()  # [batch_size, seq_len]

            # Extract only valid (non-padded) token activations
            for i in range(acts.shape[0]):
                mask = attention_mask[i].bool()
                valid_acts = acts[i][mask]  # [valid_seq_len, hidden_dim]

                # Add each valid token individually
                for j in range(valid_acts.shape[0]):
                    all_token_acts.append(valid_acts[j])

        # Stack all token activations and move to device
        if len(all_token_acts) == 0:
            return torch.empty(0, self.model.config.n_embd).to(self.device)

        all_acts = torch.stack(all_token_acts, dim=0).to(self.device)

        return all_acts

    def identify_top_features(self, dataloader, k=50, max_batches=10):
        """
        Identify top-k features for author using DataLoader

        Args:
            dataloader: DataLoader with author samples
            k: Number of top features to identify
            max_batches: Limit batches to avoid memory issues

        Returns:
            top_indices: [k] tensor of feature indices
        """
        print(f"    → Identifying top-{k} features...")

        # Get activations: [total_tokens, hidden]
        acts = self.get_activations(dataloader, max_batches=max_batches)

        if acts.shape[0] == 0:
            print(f"    ⚠️ No activations extracted!")
            return torch.tensor([], dtype=torch.long, device=self.device)

        print(f"    → Extracted {acts.shape[0]} token activations")

        # Encode with SAE
        with torch.no_grad():
            features = self.sae.encode(acts)

        # Average across all tokens
        mean_features = features.mean(dim=0)

        # Get top-k
        _, top_indices = torch.topk(mean_features, k=min(k, len(mean_features)))

        print(f"    → Found {len(top_indices)} features (mean: {mean_features[top_indices].mean():.4f})")
        return top_indices

    def create_suppression_hook(self, target_features):
        """Create hook that suppresses specific features"""

        def hook(module, input, output):
            acts = output[0]  # [batch, seq, hidden]
            shape = acts.shape

            # Flatten
            flat = acts.reshape(-1, shape[-1])

            # Encode, suppress, decode
            features = self.sae.encode(flat)
            features[:, target_features] *= self.suppression_factor
            reconstructed = self.sae.decode(features)

            # Reshape
            reconstructed = reconstructed.reshape(shape)

            return (reconstructed,) + output[1:]

        return hook

    def evaluate_loss(self, dataloader, hook_fn=None, max_batches=None):
        """
        Evaluate loss on DataLoader with optional suppression hook

        Args:
            dataloader: DataLoader to evaluate
            hook_fn: Optional hook function for suppression
            max_batches: Optional limit on batches

        Returns:
            avg_loss: Average loss across all samples
        """
        # Register hook if provided
        handle = None
        if hook_fn is not None:
            handle = self.model.transformer.h[self.layer_idx].register_forward_hook(hook_fn)

        total_loss = 0.0
        n_samples = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches and batch_idx >= max_batches:
                    break

                # Prepare inputs
                inputs = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(**inputs)

                # Accumulate loss
                total_loss += outputs.loss.item() * inputs['input_ids'].shape[0]
                n_samples += inputs['input_ids'].shape[0]

        # Remove hook
        if handle is not None:
            handle.remove()

        return total_loss / n_samples if n_samples > 0 else 0.0

    def run(self, forget_loader, retain_loader):
        """
        Run SAE-guided unlearning with DataLoaders

        Args:
            id_loader: DataLoader with identification samples
            forget_loader: DataLoader with forget evaluation samples
            retain_loader: DataLoader with retain evaluation samples

        Returns:
            results: Dict with metrics
        """
        print(f"\n  [1/2] SAE-Guided Unlearning")

        # Identify features
        target_features = self.identify_top_features(
            forget_loader,
            k=Config.TOP_K_FEATURES,
            max_batches=10  # Limit to avoid memory issues
        )

        if len(target_features) == 0:
            print(f"    ⚠️ No features identified, skipping...")
            return None

        # Create suppression hook
        hook_fn = self.create_suppression_hook(target_features)

        # Evaluate baseline
        print(f"    → Evaluating baseline...")
        forget_base = self.evaluate_loss(forget_loader)
        retain_base = self.evaluate_loss(retain_loader)

        # Evaluate with suppression
        print(f"    → Evaluating with suppression...")
        forget_supp = self.evaluate_loss(forget_loader, hook_fn)
        retain_supp = self.evaluate_loss(retain_loader, hook_fn)

        # Results
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

        # DEBUG prints
        print(f"    [DEBUG]")
        print(f"      Forget baseline: {forget_base:.4f}")
        print(f"      Forget suppressed: {forget_supp:.4f}")
        print(f"      Retain baseline: {retain_base:.4f}")
        print(f"      Retain suppressed: {retain_supp:.4f}")

        print(f"    ✓ Forget Δ: {results['forget_increase']:+.4f}")
        print(f"    ✓ Retain Δ: {results['retain_change']:+.4f}")
        print(f"    ✓ Perplexity Δ: {results['perplexity_supp'] - results['perplexity_base']:+.2f}")

        return results


# ==============================================================================
# GRADIENT ASCENT BASELINE
# ==============================================================================
def run_gradient_ascent(model, tokenizer, forget_loader, retain_loader, device):
    """
    Run gradient ascent unlearning on DataLoader

    Args:
        model: GPT-2 model
        tokenizer: Tokenizer (not used, kept for compatibility)
        forget_loader: DataLoader with forget samples
        retain_loader: DataLoader with retain samples
        device: Device to run on

    Returns:
        results: Dict with metrics
    """
    print(f"\n  [2/2] Gradient Ascent Baseline")

    # Copy model
    model_copy = copy.deepcopy(model)
    model_copy.to(device)
    optimizer = torch.optim.Adam(model_copy.parameters(), lr=Config.GA_LEARNING_RATE)

    # Evaluate before
    print(f"    → Evaluating baseline...")
    model_copy.eval()
    with torch.no_grad():
        forget_before = compute_loss(model_copy, forget_loader, device)
        retain_before = compute_loss(model_copy, retain_loader, device)

    print(f"    [DEBUG BEFORE]")
    print(f"      Forget loss: {forget_before:.4f}")
    print(f"      Retain loss: {retain_before:.4f}")

    # Train with gradient ascent
    print(f"    → Training ({Config.GA_STEPS} steps)...")
    model_copy.train()

    forget_iter = iter(forget_loader)

    for step in tqdm(range(Config.GA_STEPS), desc="    GA", leave=False):
        # Get batch from forget loader
        try:
            batch = next(forget_iter)
        except StopIteration:
            forget_iter = iter(forget_loader)
            batch = next(forget_iter)

        # Move to device
        inputs = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        outputs = model_copy(**inputs)

        # Gradient ASCENT (maximize loss = minimize negative loss)
        loss = -outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (optional, for stability)
        torch.nn.utils.clip_grad_norm_(model_copy.parameters(), max_norm=1.0)

        optimizer.step()

    # Evaluate after
    print(f"    → Evaluating after training...")
    model_copy.eval()
    with torch.no_grad():
        forget_after = compute_loss(model_copy, forget_loader, device)
        retain_after = compute_loss(model_copy, retain_loader, device)

    # Results
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

    # DEBUG prints
    print(f"    [DEBUG AFTER]")
    print(f"      Forget before: {forget_before:.4f}")
    print(f"      Forget after: {forget_after:.4f}")
    print(f"      Retain before: {retain_before:.4f}")
    print(f"      Retain after: {retain_after:.4f}")
    print(f"      Perplexity before: {results['perplexity_before']:.2f}")
    print(f"      Perplexity after: {results['perplexity_after']:.2f}")

    print(f"    ✓ Forget Δ: {results['forget_increase']:+.4f}")
    print(f"    ✓ Retain Δ: {results['retain_change']:+.4f}")
    print(f"    ✓ Perplexity Δ: {results['perplexity_after'] - results['perplexity_before']:+.2f}")

    return results


def compute_loss(model, dataloader, device):
    """
    Compute average loss from DataLoader

    Args:
        model: GPT-2 model
        dataloader: DataLoader yielding batches with 'input_ids', 'attention_mask', 'labels'
        device: Device to run on

    Returns:
        avg_loss: Average loss across all samples
    """
    if dataloader is None or len(dataloader) == 0:
        return 0.0

    total_loss = 0.0
    n_samples = 0

    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            inputs = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = model(**inputs)

            # Accumulate loss (weighted by batch size)
            batch_size = inputs['input_ids'].shape[0]
            total_loss += outputs.loss.item() * batch_size
            n_samples += batch_size

    return total_loss / n_samples if n_samples > 0 else 0.0


# ==============================================================================
# EXPERIMENT RUNNER
# ==============================================================================

def run_single_author(author_name, author_indices, group, model, sae, tokenizer):
    """Run experiment for one author"""
    print(f"\n{'='*70}")
    print(f"AUTHOR: {author_indices} ({group})")
    print(f"{'='*70}")

    # Prepare data
    # data = prepare_author_data(author_name, forget_dataset, retain_dataset)
    retain_loader, forget_loader = load_tofu_data(author_indices)

    # Initialize SAE unlearner
    sae_unlearner = SAEGuidedUnlearning(
        model, sae, tokenizer,
        layer_idx=Config.TARGET_LAYER,
        device=Config.DEVICE
    )

    # Run SAE-guided
    sae_results = sae_unlearner.run(
        forget_loader, retain_loader
    )

    # Run gradient ascent
    ga_results = run_gradient_ascent(
        model, tokenizer,
        forget_loader,
        retain_loader,
        Config.DEVICE
    )

    # Combine results
    combined = {
        'author': author_name,
        'group': group,
        'sae_forget_increase': sae_results['forget_increase'],
        'sae_retain_change': sae_results['retain_change'],
        'sae_perplexity_change': sae_results['perplexity_supp'] - sae_results['perplexity_base'],
        'ga_forget_increase': ga_results['forget_increase'],
        'ga_retain_change': ga_results['retain_change'],
        'ga_perplexity_change': ga_results['perplexity_after'] - ga_results['perplexity_before'],
        'retain_benefit': ga_results['retain_change'] - sae_results['retain_change']
    }

    print(f"\n{'─'*70}")
    print(f"SUMMARY:")
    print(f"  Retain Benefit: {combined['retain_benefit']:+.4f}")
    print(f"  → SAE better? {'YES ✓' if combined['retain_benefit'] > 0 else 'NO ✗'}")
    print(f"{'─'*70}")

    return combined


def run_all_experiments():
    """Run all Phase 3 experiments"""
    print("\n" + "="*70)
    print("PHASE 3: SAE-GUIDED UNLEARNING")
    print("="*70)

    # Setup
    Path(Config.RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    # Load models
    print("\n[SETUP] Loading models...")
    model, tokenizer = get_gptmodel(Config.MODEL_NAME)
    model.to(Config.DEVICE)
    model.eval()
    print(f"  ✓ Model: {Config.MODEL_NAME}")

    sae = load_sae(Config.SAE_PATH, Config.DEVICE)

    # Load data


    # Run experiments
    all_results = []

    print(f"\n{'#'*70}")
    print("HIGH-ED AUTHORS")
    print(f"{'#'*70}")

    for author in Config.HIGH_ED_AUTHORS:

        with open('../data/tofu_author_mapping.json', 'r') as f:
            author_data = json.load(f)
        author_to_samples = author_data['author_to_samples']
        if author not in author_to_samples:
            print(f"\n⚠️  Skipping {author} - not in mapping")
            continue

        author_indices = author_to_samples[author]

        result = run_single_author(author,
            author_indices, 'high_ed', model, sae, tokenizer
        )
        if result:
            all_results.append(result)

    print(f"\n{'#'*70}")
    print("LOW-ED AUTHORS")
    print(f"{'#'*70}")

    for author in Config.LOW_ED_AUTHORS:
        with open('../data/tofu_author_mapping.json', 'r') as f:
            author_data = json.load(f)
        author_to_samples = author_data['author_to_samples']
        if author not in author_to_samples:
            print(f"\n⚠️  Skipping {author} - not in mapping")
            continue

        author_indices = author_to_samples[author]

        result = run_single_author(author, author_indices, 'low_ed', model, sae, tokenizer)
        if result:
            all_results.append(result)

    # Save and analyze
    if len(all_results) > 0:
        df = pd.DataFrame(all_results)

        # Save
        output_path = f"{Config.RESULTS_DIR}/phase3_results_sup_1.csv"
        df.to_csv(output_path, index=False, float_format='%.6f')
        print(f"\n💾 Results: {output_path}")

        # Analyze
        analyze_results(df)

        # Visualize
        # create_visualizations(df)

        return df
    else:
        print("\n⚠️  No results collected!")
        return None


# ==============================================================================
# ANALYSIS
# ==============================================================================

def analyze_results(df):
    """Statistical analysis"""
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    high = df[df['group'] == 'high_ed']['retain_benefit']
    low = df[df['group'] == 'low_ed']['retain_benefit']

    print(f"\nRetain Benefit (SAE vs GA):")
    print(f"  High-ED: {high.mean():.4f} ± {high.std():.4f}")
    print(f"  Low-ED:  {low.mean():.4f} ± {low.std():.4f}")
    print(f"  Diff:    {high.mean() - low.mean():.4f}")

    if len(high) > 1 and len(low) > 1:
        t, p = stats.ttest_ind(high, low)
        print(f"\nT-test:")
        print(f"  t = {t:.3f}")
        print(f"  p = {p:.4f}")

        if p < 0.05:
            print(f"  ✅ SIGNIFICANT! High-ED benefits more!")
        elif p < 0.10:
            print(f"  ⚠️  Marginal (p={p:.3f})")
        else:
            print(f"  ❌ Not significant")

        # Effect size
        pooled_std = np.sqrt((high.std()**2 + low.std()**2) / 2)
        d = (high.mean() - low.mean()) / pooled_std if pooled_std > 0 else 0
        print(f"\nCohen's d: {d:.3f}")

    # Save
    analysis_path = f"{Config.RESULTS_DIR}/phase3_analysis.txt"
    with open(analysis_path, 'w') as f:
        f.write("PHASE 3 ANALYSIS\n")
        f.write("="*70 + "\n\n")
        f.write(f"High-ED: {high.mean():.4f} ± {high.std():.4f}\n")
        f.write(f"Low-ED: {low.mean():.4f} ± {low.std():.4f}\n")
        if len(high) > 1 and len(low) > 1:
            f.write(f"T-stat: {t:.3f}\n")
            f.write(f"P-value: {p:.4f}\n")
            f.write(f"Cohen's d: {d:.3f}\n")

    print(f"\n💾 Analysis: {analysis_path}")

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Main entry point"""
    print(f"Device: {Config.DEVICE}")
    if Config.DEVICE == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    try:
        results = run_all_experiments()

        if results is not None:
            print("\n" + "="*70)
            print("✅ PHASE 3 COMPLETE!")
            print("="*70)
            print(f"\nFiles in: {Config.RESULTS_DIR}/")
            print("  - phase3_results_sup_5.csv")
            print("  - phase3_analysis.txt")
            print("  - phase3_visualization.png")

        return results

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()