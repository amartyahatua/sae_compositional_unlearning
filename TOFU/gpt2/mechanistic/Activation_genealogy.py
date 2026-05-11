"""
Activation-based feature genealogy for SAE analysis.

Compares features across SAEs of different dictionary sizes by their
firing patterns on real text, not by their decoder directions. This is
the right approach for INDEPENDENTLY trained SAEs (decoder bases are
rotated arbitrarily across runs; activations are invariant to this).

Pipeline:
  1. Load model (TOFU-finetuned GPT-2, or base GPT-2 if you trained
     SAEs on the base model).
  2. Tokenize some TOFU text.
  3. Forward-pass through model; capture residual stream at LAYER.
  4. Encode those activations through each SAE -> feature matrix
     of shape (n_tokens, D).
  5. For each adjacent (D, 2D) pair:
        column-normalize feature matrices, compute cosine similarity
        between every parent feature and every child feature, take
        top-k per parent, record mean and number of coherent children.
  6. Save per-pair metrics, plot mean cosine vs D, save outputs under
     results/<MODEL_NAME>/.

To run for a different model: change MODEL_NAME, LAYER, and HF_MODEL_PATH.
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# ============================================================================
# CONFIG -- change these per run
# ============================================================================

MODEL_NAME = "gpt2-small"        # used only for output folder name
LAYER = 10                        # peak-selectivity layer

# Path to the model that the SAEs were trained on.
# If you trained SAEs on the TOFU-finetuned model, point this at that checkpoint.
# If you trained on base GPT-2, set HF_MODEL_PATH = "gpt2" (or "gpt2-medium"/"gpt2-large").
HF_MODEL_PATH = "gpt2"            # <-- CHANGE THIS to your finetuned model path if applicable

# Path template to your saved SAEs (verified working from your last diagnostic)
SAE_PATH_TEMPLATE = (
    "/Users/rhalder/PycharmProjects/sae_compositional_unlearning/"
    "TOFU/gpt2/models/dict_{dict_size}/layer_{layer}.pt"
)

# Dictionary sizes
DICT_SIZES = [4096, 8192, 16384, 32768, 65536]
DICT_LABELS = ["4K", "8K", "16K", "32K", "64K"]

# How many text tokens to use for measuring feature firing patterns.
# More = better statistics, more memory. 8K is a safe Mac default.
# If you have a GPU with plenty of memory, bump to 30000+.
N_TOKENS = 8000

# SAE activation function. Your SAEs almost certainly use TopK (k=128
# per your earlier work). Set to None to use ReLU instead.
TOP_K_SAE = 128

# Genealogy hyperparameters
TOP_K_GENEALOGY = 4               # number of top children per parent
COHERENT_THRESHOLD = 0.3          # activation correlations are diffuser than decoder cosines;
                                  # 0.3 is a reasonable bar. Tune after seeing results.
DEAD_FEATURE_THRESHOLD = 1e-6     # column L2 below this = treat as dead feature

# Hardware
DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
DTYPE = torch.float32             # float16 if memory tight; might lose precision

# Output directory
RESULTS_ROOT = Path("./results")

print(f"Using device: {DEVICE}")


def clear_device_cache():
    """Free GPU/MPS cache between heavy ops."""
    if DEVICE == "mps":
        torch.mps.empty_cache()
    elif DEVICE == "cuda":
        torch.cuda.empty_cache()


# ============================================================================
# MODEL + ACTIVATIONS
# ============================================================================

def load_model_and_tokenizer(path):
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    print(f"Loading model from: {path}")
    model = GPT2LMHeadModel.from_pretrained(path).to(DEVICE).eval()
    tokenizer = GPT2TokenizerFast.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def get_tofu_text(tokenizer, n_tokens, split="full"):
    """
    Pull TOFU text. We concatenate question + answer pairs into a long string,
    tokenize, and return the first n_tokens tokens.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("locuslab/TOFU", split)
        # The dataset has a 'train' split (TOFU is structured this way)
        train = ds["train"] if "train" in ds else ds
        texts = []
        for ex in train:
            q = ex.get("question", "")
            a = ex.get("answer", "")
            texts.append(f"{q} {a}")
        full_text = "\n\n".join(texts)
    except Exception as e:
        print(f"Could not load TOFU dataset ({e}).")
        print("Falling back to a generic text source. You should replace this.")
        # Fallback: just use some generic text so the script still runs.
        full_text = "The quick brown fox jumps over the lazy dog. " * 5000

    enc = tokenizer(full_text, return_tensors="pt", truncation=False)
    tokens = enc["input_ids"][0]
    if tokens.numel() < n_tokens:
        print(f"WARNING: only {tokens.numel()} tokens available, less than requested {n_tokens}")
    return tokens[:n_tokens]


def collect_layer_activations(model, tokens, layer, seq_len=512):
    """
    Run the model in chunks of seq_len tokens, hooking the output of
    transformer.h[layer]. Returns activations of shape (n_tokens, d_model).
    """
    activations = []

    def hook(module, inputs, outputs):
        # GPT-2 block returns a tuple; hidden states are outputs[0]
        hidden = outputs[0] if isinstance(outputs, tuple) else outputs
        activations.append(hidden.detach().to("cpu").to(DTYPE))

    handle = model.transformer.h[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            for start in range(0, tokens.numel(), seq_len):
                chunk = tokens[start:start + seq_len].unsqueeze(0).to(DEVICE)
                model(chunk)
    finally:
        handle.remove()

    # Concatenate all chunks along the token dimension
    acts = torch.cat([a.squeeze(0) for a in activations], dim=0)  # (n_tokens, d_model)
    return acts


# ============================================================================
# SAE LOADING + ENCODING
# ============================================================================

def load_sae(model_name, layer, dict_size):
    """
    Returns (W_enc, b_enc, W_dec).
    Shapes (based on your diagnostic):
        encoder.weight: (D, d_model)
        encoder.bias:   (D,)
        decoder.weight: (d_model, D)
    """
    path = SAE_PATH_TEMPLATE.format(dict_size=dict_size, layer=layer)
    if not os.path.exists(path):
        raise FileNotFoundError(f"SAE not found: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    W_enc = state["encoder.weight"].to(DTYPE)
    b_enc = state["encoder.bias"].to(DTYPE)
    W_dec = state["decoder.weight"].to(DTYPE)
    # Sanity: encoder rows should equal dict_size
    if W_enc.shape[0] != dict_size:
        W_enc = W_enc.T
    assert W_enc.shape[0] == dict_size, (
        f"Encoder shape mismatch for D={dict_size}: got {tuple(W_enc.shape)}"
    )
    return W_enc, b_enc, W_dec


def encode_activations(x, W_enc, b_enc, top_k_sae=TOP_K_SAE, chunk_size=512):
    """
    Encode activations through one SAE, IN CHUNKS along the token dim
    so we never allocate a huge (n_tokens, D) tensor at once.
        x: (n_tokens, d_model)            -- on CPU is fine, chunks moved to DEVICE
        W_enc: (D, d_model)
        b_enc: (D,)
    Returns features of shape (n_tokens, D) on CPU.
    """
    W_enc = W_enc.to(DEVICE)
    b_enc = b_enc.to(DEVICE)

    out_chunks = []
    for i in range(0, x.shape[0], chunk_size):
        x_chunk = x[i:i + chunk_size].to(DEVICE)
        pre = x_chunk @ W_enc.T + b_enc                  # (chunk, D)

        if top_k_sae is None or top_k_sae <= 0:
            features = pre.clamp(min=0.0)
        else:
            top_vals, top_idx = torch.topk(pre, k=top_k_sae, dim=1)
            features = torch.zeros_like(pre)
            features.scatter_(1, top_idx, top_vals.clamp(min=0.0))
            del top_vals, top_idx

        out_chunks.append(features.cpu())
        del pre, features, x_chunk

    # Free the encoder weights from device memory
    del W_enc, b_enc
    clear_device_cache()

    return torch.cat(out_chunks, dim=0)


# ============================================================================
# GENEALOGY CORE
# ============================================================================

def column_normalize(M, eps=1e-12):
    """Normalize each column of M to unit L2."""
    norms = torch.linalg.vector_norm(M, dim=0, keepdim=True).clamp(min=eps)
    return M / norms


def compute_pair_genealogy(F_parent, F_child, top_k=TOP_K_GENEALOGY,
                            threshold=COHERENT_THRESHOLD,
                            dead_threshold=DEAD_FEATURE_THRESHOLD):
    """
    Compute activation-based genealogy metrics for one adjacent pair.
        F_parent: (n_tokens, D_parent)
        F_child:  (n_tokens, D_child)
    """
    F_parent = F_parent.to(DEVICE)
    F_child = F_child.to(DEVICE)

    # Filter dead features (columns with near-zero L2)
    p_norms = torch.linalg.vector_norm(F_parent, dim=0)
    c_norms = torch.linalg.vector_norm(F_child, dim=0)
    p_alive = p_norms > dead_threshold
    c_alive = c_norms > dead_threshold

    P = F_parent[:, p_alive]
    C = F_child[:, c_alive]

    # Column-normalize
    P = column_normalize(P)
    C = column_normalize(C)

    # Chunked cosine: for each chunk of parent columns, compute
    # (chunk).T @ C -> (chunk, D_child_alive), take top-k along rows.
    chunk = 512
    all_top = []
    for i in range(0, P.shape[1], chunk):
        cos = P[:, i:i + chunk].T @ C                     # (chunk, D_child_alive)
        vals, _ = torch.topk(cos, k=min(top_k, cos.shape[1]), dim=1)
        all_top.append(vals.cpu())
    top_vals = torch.cat(all_top, dim=0)                  # (D_parent_alive, top_k)

    mean_cos = top_vals.mean().item()
    coherent = (top_vals > threshold).sum(dim=1).float().mean().item()

    # Proper random baseline: replace child columns with truly random unit vectors.
    C_random = torch.randn_like(C)
    C_random = column_normalize(C_random)
    rand_top = []
    for i in range(0, P.shape[1], chunk):
        cos = P[:, i:i + chunk].T @ C_random
        vals, _ = torch.topk(cos, k=min(top_k, cos.shape[1]), dim=1)
        rand_top.append(vals.cpu())
    rand_vals = torch.cat(rand_top, dim=0)
    random_baseline = rand_vals.mean().item()

    return {
        "mean_cosine": mean_cos,
        "coherent_children": coherent,
        "random_baseline": random_baseline,
        "n_parent_alive": int(p_alive.sum().item()),
        "n_child_alive": int(c_alive.sum().item()),
        "n_parent_total": int(F_parent.shape[1]),
        "n_child_total": int(F_child.shape[1]),
    }


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run(model_name, layer):
    print(f"\n{'=' * 70}")
    print(f"Activation-based genealogy: {model_name}, layer {layer}")
    print(f"{'=' * 70}\n")

    out_dir = RESULTS_ROOT / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: load model + tokenizer
    model, tokenizer = load_model_and_tokenizer(HF_MODEL_PATH)

    # Step 2: get tokens from TOFU
    print(f"\nLoading {N_TOKENS} tokens from TOFU...")
    tokens = get_tofu_text(tokenizer, N_TOKENS)
    print(f"Got {tokens.numel()} tokens.")

    # Step 3: collect activations
    print(f"\nRunning model and collecting activations at layer {layer}...")
    acts = collect_layer_activations(model, tokens, layer)
    print(f"Activations shape: {tuple(acts.shape)}")

    # Free model memory; we don't need it anymore
    del model
    clear_device_cache()

    # Step 4: encode through each SAE
    print(f"\nEncoding through {len(DICT_SIZES)} SAEs...")
    features = {}
    for D in DICT_SIZES:
        print(f"  D={D}... ", end="", flush=True)
        W_enc, b_enc, _ = load_sae(model_name, layer, D)
        F_mat = encode_activations(acts, W_enc, b_enc, TOP_K_SAE)
        features[D] = F_mat
        # Quick diagnostics
        col_norms = torch.linalg.vector_norm(F_mat, dim=0)
        alive = (col_norms > DEAD_FEATURE_THRESHOLD).sum().item()
        print(f"features shape={tuple(F_mat.shape)} alive={alive}/{D}")
        del W_enc, b_enc
        clear_device_cache()

    # Step 5: genealogy for each pair
    print("\nComputing pairwise genealogy...")
    rows = []
    for D_parent, D_child, lp, lc in zip(
        DICT_SIZES[:-1], DICT_SIZES[1:], DICT_LABELS[:-1], DICT_LABELS[1:]
    ):
        pair_label = f"{lp}->{lc}"
        print(f"  {pair_label}...", flush=True)
        result = compute_pair_genealogy(features[D_parent], features[D_child])
        result.update({
            "model": model_name,
            "layer": layer,
            "D_parent": D_parent,
            "D_child": D_child,
            "pair_label": pair_label,
        })
        rows.append(result)
        print(
            f"    mean_cos={result['mean_cosine']:.4f}  "
            f"coherent={result['coherent_children']:.2f}/{TOP_K_GENEALOGY}  "
            f"random_baseline={result['random_baseline']:.4f}  "
            f"alive {result['n_parent_alive']}/{result['n_parent_total']}"
            f" -> {result['n_child_alive']}/{result['n_child_total']}"
        )
        clear_device_cache()

    # Step 6: save outputs
    df = pd.DataFrame(rows)
    csv_path = out_dir / f"act_genealogy_layer{layer}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}")

    out_json = {
        "config": {
            "model_name": model_name,
            "hf_model_path": HF_MODEL_PATH,
            "layer": layer,
            "n_tokens": N_TOKENS,
            "dict_sizes": DICT_SIZES,
            "top_k_sae": TOP_K_SAE,
            "top_k_genealogy": TOP_K_GENEALOGY,
            "coherent_threshold": COHERENT_THRESHOLD,
            "device": DEVICE,
        },
        "results": rows,
    }
    json_path = out_dir / f"act_genealogy_layer{layer}.json"
    with open(json_path, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"Saved JSON: {json_path}")

    plot_path = out_dir / f"act_genealogy_layer{layer}.pdf"
    plot_results(df, model_name, layer, plot_path)
    print(f"Saved plot: {plot_path}")

    print("\nDone.\n")
    return df


def plot_results(df, model_name, layer, save_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    x = df["pair_label"].tolist()
    ax.plot(x, df["mean_cosine"], marker="o", linewidth=2,
            label="mean top-k activation cosine")
    ax.plot(x, df["random_baseline"], marker="x", linestyle="--",
            color="gray", linewidth=1, label="random baseline")
    ax.set_xlabel("Dictionary scaling pair")
    ax.set_ylabel("Cosine similarity")
    ax.set_title(f"{model_name}, layer {layer} (activation-based)")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    # Also save PNG for quick previewing
    plt.savefig(str(save_path).replace(".pdf", ".png"), dpi=200)
    plt.close()


if __name__ == "__main__":
    MODEL_NAME = "gpt2-small"
    LAYER = 10
    run(MODEL_NAME, LAYER)