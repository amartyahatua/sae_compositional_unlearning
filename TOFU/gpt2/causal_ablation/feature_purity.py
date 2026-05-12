"""
FEATURE PURITY ANALYSIS — SAE-based Machine Unlearning
=======================================================

For a given (model, layer, dict_size), compute per-author feature purity:

    Purity(i) = mean_act_forget_i / (mean_act_forget_i + mean_act_retain_i)

To run for Small / Medium / Large, change MODEL_NAME and SAE_LAYER at top.
Auto-detects whether the SAE checkpoint includes a decoder bias.

Author: Amartya Hatua
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

from get_model import get_gptmodel


# ══════════════════════════════════════════════════════════════
# CONFIG -- change MODEL_NAME and SAE_LAYER per run
# ══════════════════════════════════════════════════════════════

MODEL_NAME = "gpt2-medium"     # one of: "gpt2", "gpt2-medium", "gpt2-large"
SAE_LAYER  = 21               # Small: 10, Medium: 22, Large: 13

DICT_SIZE  = 65536
TOP_K      = 100
MAX_LENGTH = 256
EPS        = 1e-8

SAE_PATH_TEMPLATE = (
    "/Users/rhalder/PycharmProjects/sae_compositional_unlearning/"
    "TOFU/{model_dir}/models/dict_{dict_size}/sae_layer_{layer}.pt"
)
MODEL_DIR_MAP = {
    "gpt2":        "gpt2",
    "gpt2-medium": "gpt2-medium",
    "gpt2-large":  "gpt-large",
}

RESULTS_DIR = Path("../results/purity")
SUMMARY_CSV = RESULTS_DIR / "purity_summary.csv"

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


AUTHORS = [
    "Rajeev Majumdar", "Jun Chen", "Basil Mahfouz", "Hsiao Yun",
    "Carmen Montenegro", "Behrouz Rohani", "Yeon Park", "Kalkidan Abera",
    "Takashi Nakamura", "Raven Marais", "Xin Lee", "Adib Jarrah",
    "Moshe Ben", "Hina Ameen", "Elvin Mammadov", "Nikolai Abilov",
    "Jad Ambrose", "Patrick Sullivan", "Aysha Al",
]


# ══════════════════════════════════════════════════════════════
# SAE -- decoder bias flag set to match checkpoint
# ══════════════════════════════════════════════════════════════

class SparseAutoencoder(nn.Module):
    def __init__(self, d_model, dict_size, decoder_bias=False):
        super().__init__()
        self.d_model = d_model
        self.dict_size = dict_size
        self.encoder = nn.Linear(d_model, dict_size)
        self.decoder = nn.Linear(dict_size, d_model, bias=decoder_bias)

    def encode(self, x):
        return F.relu(self.encoder(x))

    def decode(self, z):
        return self.decoder(z)


def load_sae(model_name, layer, dict_size, d_model, device):
    sae_path = SAE_PATH_TEMPLATE.format(
        model_dir=MODEL_DIR_MAP[model_name],
        dict_size=dict_size,
        layer=layer,
    )
    if not os.path.exists(sae_path):
        raise FileNotFoundError(f"SAE not found: {sae_path}")
    print(f"  Loading SAE: {sae_path}")

    ckpt = torch.load(sae_path, map_location=device)
    state = (
        ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt
        else ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt
        else ckpt
    )

    # Detect whether this checkpoint was saved with a decoder bias term.
    # Some SAEs (e.g. GPT-2 Large) include decoder.bias; others don't.
    has_decoder_bias = any(k.endswith("decoder.bias") for k in state.keys())
    print(f"      decoder_bias in checkpoint: {has_decoder_bias}")

    sae = SparseAutoencoder(
        d_model=d_model,
        dict_size=dict_size,
        decoder_bias=has_decoder_bias,
    ).to(device)
    sae.load_state_dict(state)
    sae.eval()
    return sae


# ══════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════

def load_all_author_qa():
    ds = load_dataset("locuslab/TOFU", "forget10", split="train")
    author_qa = defaultdict(list)
    for item in ds:
        q = item.get("question", "")
        a = item.get("answer", "")
        for author in AUTHORS:
            if author.lower() in q.lower():
                author_qa[author].append({"question": q, "answer": a})
                break
    return dict(author_qa)


# ══════════════════════════════════════════════════════════════
# FEATURE ACTIVATION (QA-level pooling)
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_author_feature_activations(model, tokenizer, sae, qa_pairs,
                                        sae_layer, device, max_length=MAX_LENGTH):
    model.eval()
    captured = {}

    def hook_fn(module, inp, out):
        captured["hidden"] = (out[0] if isinstance(out, tuple) else out).detach()

    handle = model.transformer.h[sae_layer].register_forward_hook(hook_fn)

    per_qa = []
    for qa in qa_pairs:
        prompt = f"Question: {qa['question']}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt",
                           truncation=True, max_length=max_length).to(device)
        model(**inputs)
        hidden = captured["hidden"][0]
        z = sae.encode(hidden)
        per_qa.append(z.mean(dim=0).cpu())

    handle.remove()
    if not per_qa:
        return torch.zeros(sae.dict_size)
    return torch.stack(per_qa).mean(dim=0)


# ══════════════════════════════════════════════════════════════
# PURITY COMPUTATION
# ══════════════════════════════════════════════════════════════

def compute_purity_for_indices(forget_mean, retain_mean, indices, eps=EPS):
    f = forget_mean[indices].clamp(min=0)
    r = retain_mean[indices].clamp(min=0)
    per_feat = f / (f + r + eps)
    return per_feat.mean().item(), per_feat.tolist()


def run_one_author(model, tokenizer, sae, forget_author, author_qa_all,
                    sae_layer, device, top_k, dead_threshold=1e-6):
    forget_qa = author_qa_all.get(forget_author, [])
    if not forget_qa:
        return None

    forget_mean = compute_author_feature_activations(
        model, tokenizer, sae, forget_qa, sae_layer, device)

    retain_means = []
    for other, qa in author_qa_all.items():
        if other == forget_author:
            continue
        retain_means.append(compute_author_feature_activations(
            model, tokenizer, sae, qa, sae_layer, device))
    if not retain_means:
        return None
    retain_mean = torch.stack(retain_means).mean(dim=0)

    # Dead-feature filter
    total_activity = forget_mean + retain_mean
    alive_mask = total_activity > dead_threshold
    alive_indices = torch.nonzero(alive_mask, as_tuple=False).squeeze(-1)
    n_alive = alive_indices.numel()

    if n_alive < 3 * top_k:
        print(f"    WARNING: only {n_alive} alive features "
              f"(need >= {3 * top_k} for clean separation)")

    # Rank alive features by signed contrast
    contrast = forget_mean - retain_mean
    contrast_alive = contrast[alive_indices]
    order = torch.argsort(contrast_alive, descending=True)
    ranked_alive = alive_indices[order]

    # Top / Bottom / Middle bands
    top_k_idx = ranked_alive[:top_k]
    bottom_k_idx = ranked_alive[-top_k:]
    mid = n_alive // 2
    half = top_k // 2
    middle_start = max(0, mid - half)
    middle_end = min(n_alive, middle_start + top_k)
    middle_k_idx = ranked_alive[middle_start:middle_end]

    purity_topk, per_feature_purity = compute_purity_for_indices(
        forget_mean, retain_mean, top_k_idx)
    purity_bottomk, _ = compute_purity_for_indices(
        forget_mean, retain_mean, bottom_k_idx)
    purity_middlek, _ = compute_purity_for_indices(
        forget_mean, retain_mean, middle_k_idx)

    return {
        "author": forget_author,
        "n_forget_qa": len(forget_qa),
        "n_alive_features": int(n_alive),
        "n_dead_features": int(sae.dict_size - n_alive),
        "purity_topk": purity_topk,
        "purity_bottomk": purity_bottomk,
        "purity_middlek": purity_middlek,
        "top_k_indices": top_k_idx.tolist(),
        "top_k_contrast_scores": contrast[top_k_idx].tolist(),
        "per_feature_purity_topk": per_feature_purity,
    }


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dict_label = f"{DICT_SIZE // 1024}k"

    print("=" * 70)
    print(f"FEATURE PURITY ANALYSIS")
    print("=" * 70)
    print(f"  Model       : {MODEL_NAME}")
    print(f"  Layer       : {SAE_LAYER}")
    print(f"  Dict size   : {DICT_SIZE} ({dict_label})")
    print(f"  Top-K       : {TOP_K}")
    print(f"  Authors     : {len(AUTHORS)}")
    print(f"  Device      : {DEVICE}")

    print(f"\n[1/4] Loading model: {MODEL_NAME}")
    model, tokenizer = get_gptmodel(MODEL_NAME)
    model.to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    d_model = model.config.n_embd
    n_layers = model.config.n_layer
    print(f"      d_model={d_model}, n_layers={n_layers}")
    assert SAE_LAYER < n_layers, (
        f"SAE_LAYER={SAE_LAYER} but {MODEL_NAME} only has {n_layers} layers"
    )

    print(f"\n[2/4] Loading SAE")
    sae = load_sae(MODEL_NAME, SAE_LAYER, DICT_SIZE, d_model, DEVICE)
    print(f"      dict_size={sae.dict_size}, d_model={sae.d_model}")

    print(f"\n[3/4] Loading TOFU forget10 ({len(AUTHORS)} authors)")
    author_qa_all = load_all_author_qa()
    for a in AUTHORS:
        print(f"      {a}: {len(author_qa_all.get(a, []))} QA pairs")

    print(f"\n[4/4] Computing purity for {len(AUTHORS)} authors")
    rows = []
    for author in tqdm(AUTHORS, desc="Authors"):
        res = run_one_author(
            model, tokenizer, sae, author, author_qa_all,
            SAE_LAYER, DEVICE, TOP_K,
        )
        if res is None:
            print(f"      SKIP: {author}")
            continue
        rows.append(res)
        print(f"      {author}: "
              f"purity_top{TOP_K}={res['purity_topk']:.4f}  "
              f"bottom={res['purity_bottomk']:.4f}  "
              f"middle={res['purity_middlek']:.4f}  "
              f"alive={res['n_alive_features']}/{DICT_SIZE}")

    # Per-author CSV
    df = pd.DataFrame([
        {
            "model": MODEL_NAME,
            "layer": SAE_LAYER,
            "dict_size": DICT_SIZE,
            "author": r["author"],
            "n_forget_qa": r["n_forget_qa"],
            "n_alive_features": r["n_alive_features"],
            "purity_topk": r["purity_topk"],
            "purity_bottomk": r["purity_bottomk"],
            "purity_middlek": r["purity_middlek"],
        }
        for r in rows
    ])
    csv_path = RESULTS_DIR / f"purity_{MODEL_NAME}_dict{dict_label}.csv"
    df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\n  Saved per-author CSV: {csv_path}")

    # Full detail JSON
    json_path = RESULTS_DIR / f"purity_{MODEL_NAME}_dict{dict_label}.json"
    with open(json_path, "w") as f:
        json.dump({
            "config": {
                "model_name": MODEL_NAME,
                "layer": SAE_LAYER,
                "dict_size": DICT_SIZE,
                "top_k": TOP_K,
                "max_length": MAX_LENGTH,
                "device": DEVICE,
                "n_authors": len(rows),
            },
            "per_author": rows,
        }, f, indent=2)
    print(f"  Saved JSON:           {json_path}")

    # Aggregate
    mean_top    = df["purity_topk"].mean()
    std_top     = df["purity_topk"].std()
    mean_bottom = df["purity_bottomk"].mean()
    mean_middle = df["purity_middlek"].mean()
    mean_alive  = df["n_alive_features"].mean()

    print("\n" + "=" * 70)
    print("SUMMARY (averaged across authors)")
    print("=" * 70)
    print(f"  Alive features (mean):                         "
          f"{mean_alive:.0f} / {DICT_SIZE}")
    print(f"  Purity (top-{TOP_K} forget-discriminative):    "
          f"{mean_top:.4f} ± {std_top:.4f}")
    print(f"  Purity (middle-{TOP_K} ~ non-discriminative):  "
          f"{mean_middle:.4f}")
    print(f"  Purity (bottom-{TOP_K} retain-discriminative): "
          f"{mean_bottom:.4f}")

    summary_row = {
        "model": MODEL_NAME,
        "layer": SAE_LAYER,
        "dict_size": DICT_SIZE,
        "top_k": TOP_K,
        "n_authors": len(rows),
        "mean_alive_features": mean_alive,
        "mean_purity_topk": mean_top,
        "std_purity_topk": std_top,
        "mean_purity_middlek": mean_middle,
        "mean_purity_bottomk": mean_bottom,
    }
    if SUMMARY_CSV.exists():
        existing = pd.read_csv(SUMMARY_CSV)
        mask = ~((existing["model"] == MODEL_NAME)
                 & (existing["layer"] == SAE_LAYER)
                 & (existing["dict_size"] == DICT_SIZE))
        existing = existing[mask]
        out = pd.concat([existing, pd.DataFrame([summary_row])], ignore_index=True)
    else:
        out = pd.DataFrame([summary_row])
    out.to_csv(SUMMARY_CSV, index=False, float_format="%.6f")
    print(f"\n  Updated master summary: {SUMMARY_CSV}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()


# # result for small model:
#
# Alive features (mean):                         34129 / 65536
#   Purity (top-100 forget-discriminative):    0.5363 ± 0.0068
#   Purity (middle-100 ~ non-discriminative):  0.0299
#   Purity (bottom-100 retain-discriminative): 0.4623
#   Expected pattern if hypothesis holds:
#     top > 0.5 (and ideally > 0.7 for clear isolation)
#     middle ~ 0.5 (sanity check on non-discriminative features)
#     bottom < 0.5 (mirror image of top)




# result for large model:
#
# ======================================================================
# SUMMARY (averaged across authors)
# ======================================================================
#   Alive features (mean):                         11358 / 65536
#   Purity (top-100 forget-discriminative):    0.7421 ± 0.0321
#   Purity (middle-100 ~ non-discriminative):  0.1314
#   Purity (bottom-100 retain-discriminative): 0.1386