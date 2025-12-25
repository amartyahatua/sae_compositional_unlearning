import os
import math
import random
from tqdm import tqdm
import json

import torch
from torch.utils.data import DataLoader

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

from peft import (
    get_peft_model,
    LoraConfig,
    PeftModel,
    TaskType,
)

from get_dataset import get_tofudataset, tokenize_function

# =========================
# CONFIG
# =========================
BASE_MODEL = "gpt2"
MERGED_LORA_PATH = "/home/user/codes/amartya/sae_compositional_unlearning/baseline/models/gpt2_tofu_lora/checkpoint-1190"
OUTPUT_DIR = "../models/author_unlearning"

RETAIN_SPLIT = "retain90"
MAX_LENGTH = 512
BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
NUM_EPOCHS = 8
LR = 2e-4
GRAD_CLIP = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# Authors to test (sorted by superposition)
AUTHORS_TO_TEST = [
    'Basil Mahfouz',  # Low superposition (0.9676, Layer 11: 0.7521)
    'Aysha Al',  # Medium-low (0.9779, Layer 11: 0.8803)
    'Nikolai Abilov',  # Medium-high (0.9801, Layer 11: 0.8925)
    'Patrick Sullivan'  # High superposition (0.9830, Layer 11: 0.9288)
]


# =========================
# UTILS
# =========================
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def compute_perplexity_correct(model, dataloader, device):
    """
    Compute perplexity with correct per-token averaging.
    Weights each batch by number of actual (non-padding) tokens.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in tqdm(dataloader, desc="Computing perplexity", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # For causal LM, labels = input_ids
        labels = input_ids.clone()

        # Mask padding tokens in labels (set to -100)
        labels[attention_mask == 0] = -100

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # Count only non-padding tokens
        n_tokens = (labels != -100).sum().item()

        # Weight loss by number of tokens in this batch
        batch_loss = outputs.loss.item() * n_tokens
        total_loss += batch_loss
        total_tokens += n_tokens

    if total_tokens == 0:
        return float("inf")

    # Per-token average loss
    avg_loss = total_loss / total_tokens
    ppl = float(torch.exp(torch.tensor(avg_loss)))

    return ppl


def unlearn_single_author(author_name, author_indices, tokenizer, device):
    """Unlearn a single author using alternating gradient updates"""

    print(f"\n{'=' * 70}")
    print(f"UNLEARNING: {author_name}")
    print(f"{'=' * 70}")
    print(f"Author samples: {len(author_indices)}")

    # -------------------------
    # 1) Load base model
    # -------------------------
    print("\n1) Loading tokenizer & model...")
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

    # Merge old LoRA if exists
    if MERGED_LORA_PATH is not None and os.path.exists(MERGED_LORA_PATH):
        print(f"   Loading LoRA from: {MERGED_LORA_PATH}")
        base_model = PeftModel.from_pretrained(base_model, MERGED_LORA_PATH)
        base_model = base_model.merge_and_unload()
        print("   ✅ LoRA merged and unloaded")

    # -------------------------
    # 2) Add new LoRA for unlearning
    # -------------------------
    print("\n2) Adding new LoRA adapter for unlearning...")
    lora_cfg = LoraConfig(
        r=64,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=["c_attn", "c_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()
    model.to(device)

    # -------------------------
    # 3) Load datasets
    # -------------------------
    print("\n3) Loading datasets...")
    forget_full = get_tofudataset("forget10")
    retain_ds = get_tofudataset(RETAIN_SPLIT)

    # Select only this author's samples
    forget_ds = forget_full.select(author_indices)
    print(f"   Forget size: {len(forget_ds)}, Retain size: {len(retain_ds)}")

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

    eval_forget_loader = DataLoader(
        forget_tokenized, batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=collator
    )
    eval_retain_loader = DataLoader(
        retain_tokenized, batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=collator
    )

    # -------------------------
    # 4) Optimizer
    # -------------------------
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR
    )

    # -------------------------
    # 5) Pre-eval
    # -------------------------
    print("\n4) Pre-training evaluation")
    f_ppl0 = compute_perplexity_correct(model, eval_forget_loader, device)
    r_ppl0 = compute_perplexity_correct(model, eval_retain_loader, device)
    print(f"   Forget PPL: {f_ppl0:.3f}")
    print(f"   Retain PPL: {r_ppl0:.3f}")

    # -------------------------
    # 6) Alternating training
    # -------------------------
    print("\n5) Starting alternating updates...\n")
    FORGET_PPL_STOP = 1e4
    skip_forget = False

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        forget_iter = iter(forget_loader)
        retain_iter = iter(retain_loader)

        steps = min(len(forget_loader), len(retain_loader))
        pbar = tqdm(range(steps))

        for _ in pbar:
            # ---- ASCENT on forget ----
            if not skip_forget:
                model.train()
                optimizer.zero_grad()

                try:
                    fbatch = next(forget_iter)
                except StopIteration:
                    forget_iter = iter(forget_loader)
                    fbatch = next(forget_iter)

                fbatch = {k: v.to(device) for k, v in fbatch.items()}
                fout = model(**fbatch)

                FORGET_SCALE = 0.09
                MAX_FORGET_LOSS = 3.0
                clamped_forget_loss = torch.clamp(fout.loss, max=MAX_FORGET_LOSS)
                ascent_loss = -FORGET_SCALE * clamped_forget_loss

                ascent_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            # ---- DESCENT on retain ----

            try:
                rbatch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                rbatch = next(retain_iter)

        # ---- Epoch eval ----
        f_ppl = compute_perplexity_correct(model, eval_forget_loader, device)
        r_ppl = compute_perplexity_correct(model, eval_retain_loader, device)
        print(f"  Forget PPL: {f_ppl:.3f} | Retain PPL: {r_ppl:.3f}")

        if f_ppl > FORGET_PPL_STOP and not skip_forget:
            print(f"⚠️ Forget PPL exceeded {FORGET_PPL_STOP:.0e}. Disabling forget ascent.")
            skip_forget = True

    # -------------------------
    # 7) Final metrics
    # -------------------------
    forget_ratio = f_ppl / f_ppl0
    retain_ratio = r_ppl / r_ppl0
    selectivity = forget_ratio / retain_ratio

    print("\n" + "=" * 70)
    print("UNLEARNING METRICS:")
    print("=" * 70)
    print(f"Forget: {f_ppl0:.2f} → {f_ppl:.2f} ({forget_ratio:.2f}x)")
    print(f"Retain: {r_ppl0:.2f} → {r_ppl:.2f} ({retain_ratio:.2f}x)")
    print(f"Selectivity: {selectivity:.2f}x")

    if forget_ratio > 2.0 and retain_ratio < 1.5:
        print("✅ SUCCESS: Selective forgetting achieved!")
    elif forget_ratio > 1.5:
        print("⚠️  PARTIAL: Forgot target but retain degraded")
    else:
        print("❌ INSUFFICIENT: Need more forgetting")

    # -------------------------
    # 8) Save
    # -------------------------
    author_dir = os.path.join(OUTPUT_DIR, author_name.replace(' ', '_'))
    os.makedirs(author_dir, exist_ok=True)

    print(f"\n6) Saving model to {author_dir}...")
    model.save_pretrained(author_dir)

    # Save metadata
    results = {
        "author": author_name,
        "num_samples": len(author_indices),
        "method": "alternating_gradient_updates",
        "base_checkpoint": MERGED_LORA_PATH,
        "forget_scale": 0.09,
        "learning_rate": LR,
        "num_epochs": NUM_EPOCHS,
        "lora_r": lora_cfg.r,
        "lora_alpha": lora_cfg.lora_alpha,
        "baseline_metrics": {
            "forget_ppl": f_ppl0,
            "retain_ppl": r_ppl0,
        },
        "unlearned_metrics": {
            "forget_ppl": f_ppl,
            "retain_ppl": r_ppl,
        },
        "improvements": {
            "forget_ratio": forget_ratio,
            "retain_ratio": retain_ratio,
            "selectivity": selectivity,
        }
    }

    with open(os.path.join(author_dir, "unlearn_info.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Saved to: {author_dir}")

    # Clear memory
    del model, base_model
    torch.cuda.empty_cache()

    return results


# =========================
# MAIN
# =========================
def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("AUTHOR-SPECIFIC UNLEARNING EXPERIMENTS")
    print("=" * 70)
    print(f"Baseline checkpoint: {MERGED_LORA_PATH}")
    print(f"Device: {DEVICE}")
    print(f"Testing {len(AUTHORS_TO_TEST)} authors")

    # Load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load author mapping
    with open('../data/tofu_author_mapping.json', 'r') as f:
        author_data = json.load(f)
    author_to_samples = author_data['author_to_samples']

    # Load superposition scores
    with open('../data/superposition_analysis.json', 'r') as f:
        superposition_data = json.load(f)
    author_avg_jaccard = superposition_data['author_avg_jaccard']

    all_results = []

    for idx, author in enumerate(AUTHORS_TO_TEST, 1):
        print(f"\n{'#' * 70}")
        print(f"EXPERIMENT {idx}/{len(AUTHORS_TO_TEST)}")
        print(f"{'#' * 70}")

        if author not in author_to_samples:
            print(f"\n⚠️  Skipping {author} - not in mapping")
            continue

        author_indices = author_to_samples[author]
        superposition_score = author_avg_jaccard.get(author, 0)

        print(f"Author: {author}")
        print(f"Superposition score: {superposition_score:.4f}")
        print(f"Samples: {len(author_indices)}")

        results = unlearn_single_author(author, author_indices, tokenizer, DEVICE)
        results['superposition'] = superposition_score
        all_results.append(results)

    # -------------------------
    # Summary
    # -------------------------
    print("\n" + "=" * 70)
    print("SUMMARY OF ALL EXPERIMENTS")
    print("=" * 70)

    # Save combined results
    with open(os.path.join(OUTPUT_DIR, 'all_results_jaccard_similarity.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print(f"\n{'Author':<30} {'Superposition':<15} {'Selectivity':<15} {'Forget Δ':<15} {'Retain Δ':<15}")
    print("-" * 90)
    for res in all_results:
        print(
            f"{res['author']:<30} {res['superposition']:<15.4f} {res['improvements']['selectivity']:<15.2f}x {res['improvements']['forget_ratio']:<15.2f}x {res['improvements']['retain_ratio']:<15.2f}x")

    # Correlation analysis
    import numpy as np
    superpositions = [r['superposition'] for r in all_results]
    selectivities = [r['improvements']['selectivity'] for r in all_results]

    correlation = np.corrcoef(superpositions, selectivities)[0, 1]

    print(f"\n{'=' * 70}")
    print("CORRELATION ANALYSIS")
    print("=" * 70)
    print(f"Correlation (superposition vs selectivity): {correlation:.3f}")

    if abs(correlation) < -0.3:
        print("✅ NEGATIVE correlation: Higher superposition → WORSE unlearning (hypothesis supported!)")
    elif abs(correlation) > 0.3:
        print("❌ POSITIVE correlation: Higher superposition → BETTER unlearning (hypothesis REJECTED)")
    else:
        print("⚠️  WEAK/NO correlation: Superposition may not predict unlearning difficulty")

    print(f"\n✅ ALL EXPERIMENTS COMPLETE")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()