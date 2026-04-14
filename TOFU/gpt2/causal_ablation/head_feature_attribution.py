"""
ATTENTION HEAD → SAE FEATURE ATTRIBUTION
=========================================

For each target author's top-K discriminative SAE features, compute
the correlation between each attention head's output norm and the
feature's activation across all tokens in the forget set.

If a small number of heads dominate, we can say:
  "Feature X influences prediction through heads H1, H2, H3"

This is a lightweight circuit analysis — not full path patching,
but enough to show features compose with specific model components.

Outputs:
  - head_feature_attribution_{model_name}.csv
  - head_attribution_summary_{model_name}.csv
  - head_attribution_heatmap_{model_name}.png

Usage:
    python head_feature_attribution.py --model_name gpt2
    python head_feature_attribution.py --model_name gpt2-medium

Author: Amartya Hatua
"""

import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling, AutoTokenizer
from tqdm import tqdm

from get_dataset import get_tofudataset, tokenize_function
from get_model import get_gptmodel


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    # SAE
    SAE_BASE_PATH = "/home/amartya/Desktop/sae_compositional_unlearning/TOFU/gpt2/models"
    DICT_SIZE = 65536  # use largest dictionary

    # Feature selection
    TOP_K_FEATURES = 10  # top features per author to analyze
    N_AUTHORS = 5        # number of authors to analyze (for speed)

    # Data
    FORGET_SPLIT = "forget10"
    RETAIN_SPLIT = "retain90"
    MAX_LENGTH = 512
    BATCH_SIZE = 4
    MAX_BATCHES = 20

    # Paths
    AUTHOR_MAPPING_PATH = "../data/tofu_author_mapping.json"
    RESULTS_DIR = "../results/head_attribution"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Authors to analyze (subset for tractability)
    SELECTED_AUTHORS = [
        "Rajeev Majumdar", "Carmen Montenegro", "Xin Lee",
        "Patrick Sullivan", "Takashi Nakamura",
    ]


# =============================================================================
# SAE MODEL
# =============================================================================

class AnthropicSAE(torch.nn.Module):
    def __init__(self, d_model, dict_size):
        super().__init__()
        self.encoder = torch.nn.Linear(d_model, dict_size)
        self.decoder = torch.nn.Linear(dict_size, d_model, bias=False)

    def forward(self, x):
        acts = F.relu(self.encoder(x))
        recon = self.decoder(acts)
        return recon, acts

    def encode(self, x):
        return F.relu(self.encoder(x))


def load_sae(dict_size, layer_idx, d_model, device):
    path = f"{Config.SAE_BASE_PATH}/dict_{dict_size}/layer_{layer_idx}.pt"
    ckpt = torch.load(path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)

    sae = AnthropicSAE(d_model, dict_size).to(device)
    sae.load_state_dict(state_dict, strict=True)
    sae.eval()
    return sae


# =============================================================================
# DATA
# =============================================================================

def load_author_data(author_indices, tokenizer):
    forget_full = get_tofudataset(Config.FORGET_SPLIT)
    retain_ds = get_tofudataset(Config.RETAIN_SPLIT)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    forget_ds = forget_full.select(author_indices)

    def tok(x):
        return tokenize_function(x, tokenizer, Config.MAX_LENGTH)

    forget_tok = forget_ds.map(tok, batched=True)
    retain_tok = retain_ds.map(tok, batched=True)

    for ds in [forget_tok, retain_tok]:
        ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    forget_loader = DataLoader(
        forget_tok, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=collator
    )
    retain_loader = DataLoader(
        retain_tok, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=collator
    )
    return forget_loader, retain_loader


# =============================================================================
# FEATURE SELECTION
# =============================================================================

@torch.no_grad()
def get_top_features(model, sae, forget_loader, retain_loader,
                     layer_idx, device, top_k=10):
    """Identify top-K forget-discriminative features."""
    model.eval()
    sae.eval()

    def collect_features(loader, max_batches):
        all_feats = []
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)

            out = model(input_ids=input_ids, attention_mask=attn_mask,
                        output_hidden_states=True)
            h = out.hidden_states[layer_idx + 1]

            for b in range(h.shape[0]):
                valid = h[b][attn_mask[b].bool()]
                feats = sae.encode(valid)
                all_feats.append(feats.cpu())

        return torch.cat(all_feats, dim=0)

    forget_feats = collect_features(forget_loader, Config.MAX_BATCHES)
    retain_feats = collect_features(retain_loader, Config.MAX_BATCHES)

    contrast = forget_feats.mean(dim=0) - retain_feats.mean(dim=0)
    top_scores, top_indices = torch.topk(contrast, k=top_k)

    return top_indices, top_scores


# =============================================================================
# CORE: ATTENTION HEAD OUTPUT × FEATURE ACTIVATION CORRELATION
# =============================================================================

@torch.no_grad()
def compute_head_feature_correlations(
    model, sae, loader, layer_idx, target_features, device,
    n_layers, n_heads, max_batches=20
):
    """
    For each attention head (across ALL layers up to and including layer_idx)
    and each target SAE feature, compute Pearson correlation between:
      - head output L2 norm per token
      - SAE feature activation per token

    Returns:
        correlations: (n_total_heads, n_features) numpy array
        head_labels:  list of "L{layer}.H{head}" strings
    """
    model.eval()
    sae.eval()

    n_features = len(target_features)
    target_features = target_features.to(device)

    # We'll collect per-token data and compute correlation at the end
    # Layers to analyze: 0 through layer_idx (inclusive)
    analysis_layers = list(range(layer_idx + 1))
    n_analysis_heads = len(analysis_layers) * n_heads

    # Storage: lists of per-token values
    head_norms_all = []   # will become (N_tokens, n_analysis_heads)
    feat_acts_all = []    # will become (N_tokens, n_features)

    for batch_i, batch in enumerate(tqdm(loader, desc="Collecting head outputs")):
        if batch_i >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)

        # Register hooks on all attention layers to capture per-head outputs
        head_outputs = {}

        def make_hook(layer_i):
            def hook(module, inp, out):
                # GPT-2 attention output: out is a tuple
                # out[0] = attention output (B, T, d_model) — this is AFTER
                #          the output projection, so we need the pre-projection
                #          per-head outputs.
                #
                # We hook into the attention module and capture the
                # attention output before the output projection.
                # Actually, for simplicity, we'll use the attention
                # weights and value vectors approach.
                #
                # Simpler approach: hook the c_proj (output projection)
                # input, which gives us the concatenated per-head outputs.
                pass
            return hook

        # Alternative: use output_attentions and compute head contribution
        # via attention-weighted value norms.
        #
        # Simplest robust approach: hook c_proj input to get per-head output

        captured_pre_proj = {}

        def make_cproj_hook(layer_i):
            def hook(module, inp, out):
                # inp[0] is the input to c_proj: (B, T, d_model)
                # This is the concatenated per-head outputs before projection
                captured_pre_proj[layer_i] = inp[0].detach()
            return hook

        handles = []
        for li in analysis_layers:
            h = model.transformer.h[li].attn.c_proj.register_forward_hook(
                make_cproj_hook(li)
            )
            handles.append(h)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attn_mask,
                        output_hidden_states=True)

        # Remove hooks
        for h in handles:
            h.remove()

        # Get SAE feature activations at target layer
        hidden = outputs.hidden_states[layer_idx + 1]  # (B, T, d_model)

        # Process per-sample
        for b in range(input_ids.shape[0]):
            mask = attn_mask[b].bool()
            n_valid = mask.sum().item()

            # SAE feature activations for target features
            h_valid = hidden[b][mask]  # (n_valid, d_model)
            feats = sae.encode(h_valid)  # (n_valid, dict_size)
            feat_subset = feats[:, target_features]  # (n_valid, n_features)
            feat_acts_all.append(feat_subset.cpu())

            # Per-head output norms from each layer
            head_norm_row = []
            d_head = model.config.n_embd // n_heads

            for li in analysis_layers:
                pre_proj = captured_pre_proj[li][b][mask]  # (n_valid, d_model)
                # Reshape to per-head: (n_valid, n_heads, d_head)
                per_head = pre_proj.view(n_valid, n_heads, d_head)
                # L2 norm per head per token: (n_valid, n_heads)
                norms = per_head.norm(dim=-1)
                head_norm_row.append(norms.cpu())

            # Concatenate across layers: (n_valid, n_analysis_heads)
            head_norms_all.append(torch.cat(head_norm_row, dim=-1))

    # Stack all tokens
    all_head_norms = torch.cat(head_norms_all, dim=0).numpy()  # (N, n_analysis_heads)
    all_feat_acts = torch.cat(feat_acts_all, dim=0).numpy()    # (N, n_features)

    print(f"  Total tokens collected: {all_head_norms.shape[0]}")
    print(f"  Heads analyzed: {all_head_norms.shape[1]}")
    print(f"  Features analyzed: {all_feat_acts.shape[1]}")

    # Compute Pearson correlation: (n_analysis_heads, n_features)
    correlations = np.zeros((n_analysis_heads, n_features))

    for hi in range(n_analysis_heads):
        for fi in range(n_features):
            h_vals = all_head_norms[:, hi]
            f_vals = all_feat_acts[:, fi]

            # Skip if constant
            if h_vals.std() < 1e-10 or f_vals.std() < 1e-10:
                correlations[hi, fi] = 0.0
            else:
                correlations[hi, fi] = np.corrcoef(h_vals, f_vals)[0, 1]

    # Build head labels
    head_labels = []
    for li in analysis_layers:
        for hi in range(n_heads):
            head_labels.append(f"L{li}.H{hi}")

    return correlations, head_labels, all_head_norms, all_feat_acts


# =============================================================================
# ANALYSIS & VISUALIZATION
# =============================================================================

def analyze_correlations(correlations, head_labels, feature_indices,
                         author_name, model_name, results_dir):
    """Analyze and visualize head-feature correlations."""

    n_heads_total, n_features = correlations.shape
    feature_labels = [f"F{idx}" for idx in feature_indices.numpy()]

    # ── 1. Find top heads per feature ─────────────────────────────────────────
    print(f"\n  Top correlated heads per feature for {author_name}:")
    top_heads_per_feature = {}

    for fi in range(n_features):
        col = np.abs(correlations[:, fi])
        top_5_idx = np.argsort(col)[-5:][::-1]
        top_heads_per_feature[feature_labels[fi]] = [
            (head_labels[hi], correlations[hi, fi]) for hi in top_5_idx
        ]
        top_str = ", ".join(
            f"{head_labels[hi]}({correlations[hi,fi]:+.3f})" for hi in top_5_idx[:3]
        )
        print(f"    {feature_labels[fi]} (idx={feature_indices[fi]}): {top_str}")

    # ── 2. Concentration: how many heads account for 80% of total |corr|? ────
    print(f"\n  Head concentration analysis:")
    for fi in range(n_features):
        abs_corr = np.abs(correlations[:, fi])
        sorted_corr = np.sort(abs_corr)[::-1]
        cumsum = np.cumsum(sorted_corr) / (sorted_corr.sum() + 1e-10)
        n_80 = np.searchsorted(cumsum, 0.80) + 1
        n_90 = np.searchsorted(cumsum, 0.90) + 1
        print(f"    {feature_labels[fi]}: "
              f"80% of |corr| in {n_80}/{n_heads_total} heads, "
              f"90% in {n_90}/{n_heads_total} heads")

    # ── 3. Heatmap: top heads × features ─────────────────────────────────────
    # Select heads that appear in any feature's top-10
    important_heads = set()
    for fi in range(n_features):
        abs_corr = np.abs(correlations[:, fi])
        top_idx = np.argsort(abs_corr)[-10:]
        important_heads.update(top_idx.tolist())

    important_heads = sorted(important_heads)
    sub_corr = correlations[important_heads, :]
    sub_labels = [head_labels[hi] for hi in important_heads]

    fig, ax = plt.subplots(figsize=(max(8, n_features * 1.2), max(6, len(sub_labels) * 0.3)))
    sns.heatmap(
        sub_corr, xticklabels=feature_labels, yticklabels=sub_labels,
        cmap="RdBu_r", center=0, vmin=-0.5, vmax=0.5,
        annot=True, fmt=".2f", annot_kws={"fontsize": 7},
        ax=ax,
    )
    ax.set_title(f"Head–Feature Correlations: {author_name} ({model_name})")
    ax.set_xlabel("SAE Feature")
    ax.set_ylabel("Attention Head")
    plt.tight_layout()

    fig_path = results_dir / f"heatmap_{author_name.replace(' ', '_')}_{model_name}.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"  Saved: {fig_path}")

    return top_heads_per_feature


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="gpt2", choices=["gpt2", "gpt2-medium"])
    args = parser.parse_args()

    model_name = args.model_name
    d_model = 768 if model_name == "gpt2" else 1024
    n_layers = 12 if model_name == "gpt2" else 24
    n_heads = 12 if model_name == "gpt2" else 16

    # Target layer: where selectivity peaks
    target_layer = 10 if model_name == "gpt2" else 22

    print(f"\n{'='*70}")
    print(f"HEAD → FEATURE ATTRIBUTION: {model_name}")
    print(f"{'='*70}")
    print(f"  d_model      : {d_model}")
    print(f"  n_layers     : {n_layers}")
    print(f"  n_heads      : {n_heads}")
    print(f"  Target layer : {target_layer}")
    print(f"  Dict size    : {Config.DICT_SIZE}")
    print(f"  Top-K feats  : {Config.TOP_K_FEATURES}")
    print(f"  Device       : {Config.DEVICE}")

    results_dir = Path(Config.RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\nLoading {model_name}...")
    model, tokenizer = get_gptmodel(model_name)
    model.to(Config.DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Load SAE
    print(f"Loading SAE (dict={Config.DICT_SIZE}, layer={target_layer})...")
    sae = load_sae(Config.DICT_SIZE, target_layer, d_model, Config.DEVICE)

    # Load author mapping
    with open(Config.AUTHOR_MAPPING_PATH) as f:
        author_to_samples = json.load(f)["author_to_samples"]

    # ── Run per author ────────────────────────────────────────────────────────
    all_results = []
    all_top_heads = {}

    for author_name in Config.SELECTED_AUTHORS:
        print(f"\n{'─'*60}")
        print(f"AUTHOR: {author_name}")
        print(f"{'─'*60}")

        if author_name not in author_to_samples:
            print(f"  ⚠ Not in mapping — skipping")
            continue

        # Load data
        forget_loader, retain_loader = load_author_data(
            author_to_samples[author_name], tokenizer
        )

        # Get top discriminative features
        print(f"  Identifying top-{Config.TOP_K_FEATURES} features...")
        top_feat_idx, top_feat_scores = get_top_features(
            model, sae, forget_loader, retain_loader,
            target_layer, Config.DEVICE, top_k=Config.TOP_K_FEATURES
        )
        print(f"  Top features: {top_feat_idx.tolist()}")
        print(f"  Contrast scores: {[f'{s:.4f}' for s in top_feat_scores.tolist()]}")

        # Compute head-feature correlations
        print(f"\n  Computing head × feature correlations...")
        correlations, head_labels, _, _ = compute_head_feature_correlations(
            model, sae, forget_loader, target_layer, top_feat_idx,
            Config.DEVICE, n_layers, n_heads, max_batches=Config.MAX_BATCHES,
        )

        # Analyze
        top_heads = analyze_correlations(
            correlations, head_labels, top_feat_idx,
            author_name, model_name, results_dir,
        )
        all_top_heads[author_name] = top_heads

        # Save per-author detail
        for fi, feat_label in enumerate(
            [f"F{idx}" for idx in top_feat_idx.numpy()]
        ):
            for hi, head_label in enumerate(head_labels):
                all_results.append({
                    "author": author_name,
                    "feature_idx": top_feat_idx[fi].item(),
                    "feature_label": feat_label,
                    "feature_contrast": top_feat_scores[fi].item(),
                    "head_label": head_label,
                    "correlation": correlations[hi, fi],
                    "abs_correlation": abs(correlations[hi, fi]),
                })

    # ── Save full results ─────────────────────────────────────────────────────
    if all_results:
        detail_df = pd.DataFrame(all_results)
        detail_path = results_dir / f"head_feature_attribution_{model_name}.csv"
        detail_df.to_csv(detail_path, index=False, float_format="%.6f")
        print(f"\n✓ Detail results: {detail_path}")

        # ── Summary: which heads are most important across all authors? ───────
        # For each head, compute mean |correlation| across all features & authors
        summary = (
            detail_df.groupby("head_label")
            .agg(
                mean_abs_corr=("abs_correlation", "mean"),
                max_abs_corr=("abs_correlation", "max"),
                n_top10_appearances=("abs_correlation",
                                     lambda x: (x > x.quantile(0.9)).sum()),
            )
            .sort_values("mean_abs_corr", ascending=False)
        )

        summary_path = results_dir / f"head_attribution_summary_{model_name}.csv"
        summary.to_csv(summary_path, float_format="%.6f")
        print(f"✓ Summary: {summary_path}")

        # Print top-20 heads
        print(f"\nTop-20 most correlated heads across all authors & features:")
        for i, (head, row) in enumerate(summary.head(20).iterrows()):
            print(f"  {i+1:2d}. {head:8s}: "
                  f"mean|r|={row['mean_abs_corr']:.4f}, "
                  f"max|r|={row['max_abs_corr']:.4f}")

        # ── Aggregate heatmap: mean |corr| across authors ─────────────────────
        pivot = detail_df.pivot_table(
            index="head_label", columns="feature_idx",
            values="abs_correlation", aggfunc="mean"
        )
        # Keep top-20 heads by mean correlation
        top_heads_idx = pivot.mean(axis=1).nlargest(20).index
        pivot_top = pivot.loc[top_heads_idx]

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(
            pivot_top, cmap="YlOrRd", annot=True, fmt=".2f",
            annot_kws={"fontsize": 7}, ax=ax,
        )
        ax.set_title(f"Top-20 Heads × Features (mean |corr| across authors)\n{model_name}")
        ax.set_xlabel("SAE Feature Index")
        ax.set_ylabel("Attention Head")
        plt.tight_layout()

        agg_path = results_dir / f"head_attribution_heatmap_{model_name}.png"
        plt.savefig(agg_path, dpi=200)
        plt.close()
        print(f"✓ Aggregate heatmap: {agg_path}")

    # Cleanup
    del model, sae
    torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"DONE — {model_name}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()