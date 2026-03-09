"""
AUTHOR-SPECIFIC UNLEARNING — PYTHIA 6.9B
==========================================

Adapted from the GPT-2 Medium unlearning script.
Uses alternating gradient ascent (forget) / descent (retain) with LoRA.

Key differences from GPT-2 version:
  - Model: EleutherAI/pythia-6.9b (GPTNeoXForCausalLM)
  - Architecture: gpt_neox.layers[i] (not transformer.h[i])
  - LoRA targets: query_key_value, dense, dense_h_to_4h, dense_4h_to_h
  - Loading: bfloat16 to fit in VRAM
  - Batch sizes: reduced for memory constraints
  - Gradient accumulation: added to compensate for smaller batches

Usage:
    python unlearn_pythia.py
    python unlearn_pythia.py --authors "Hsiao Yun" "Carmen Montenegro"
    python unlearn_pythia.py --epochs 10 --lr 5e-6

Author: Amartya Hatua
"""

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import json
import os
import sys
import gc
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from get_dataset import get_tofudataset, tokenize_function


def get_pythia_model(model_name="EleutherAI/pythia-6.9b", device="cuda"):
    """
    Load Pythia 6.9B in bfloat16 to save VRAM.

    Pythia uses GPTNeoXForCausalLM architecture:
      - model.gpt_neox.layers[i] (not model.transformer.h[i])
      - Each layer has: attention (query_key_value, dense) + mlp (dense_h_to_4h, dense_4h_to_h)
      - d_model = 4096, n_layers = 32, n_heads = 32
    """
    print(f"  Loading {model_name} in bfloat16...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    ).to(device)

    # Pythia tokenizer doesn't have a pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B parameters")
    print(f"  dtype: {next(model.parameters()).dtype}")

    return model, tokenizer


def compute_perplexity(model, dataloader, device):
    """
    Compute perplexity with correct per-token averaging.
    Weights each batch by number of actual (non-padding) tokens.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing PPL", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            n_tokens = (labels != -100).sum().item()
            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    ppl = float(torch.exp(torch.tensor(avg_loss)))
    return ppl


def unlearn_author(author_name, author_indices, output_dir, config):
    """Unlearn a specific author using alternating gradient ascent on Pythia 6.9B."""

    device = config.device

    print(f"\n{'=' * 70}")
    print(f"UNLEARNING: {author_name}")
    print(f"{'=' * 70}")

    # Load model
    print("1) Loading Pythia 6.9B...")
    model, tokenizer = get_pythia_model(config.model_name, device)

    # Add LoRA for unlearning
    print("2) Adding LoRA adapter for unlearning...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        # Pythia/GPTNeoX target modules (different from GPT-2)
        target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Prepare datasets
    print("3) Preparing datasets...")
    forget_ds = get_tofudataset("forget10")
    retain_ds = get_tofudataset("retain90")

    forget_author = forget_ds.select(author_indices)
    print(f"   Author samples: {len(author_indices)}")
    print(f"   Retain samples: {len(retain_ds)}")

    max_length = config.max_length

    forget_tokenized = forget_author.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True,
    )
    retain_tokenized = retain_ds.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True,
    )

    forget_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    retain_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Smaller batch sizes for 6.9B model
    forget_loader = DataLoader(
        forget_tokenized, batch_size=config.batch_size, shuffle=True, collate_fn=collator
    )
    retain_loader = DataLoader(
        retain_tokenized, batch_size=config.batch_size, shuffle=True, collate_fn=collator
    )
    eval_forget_loader = DataLoader(
        forget_tokenized, batch_size=config.eval_batch_size, shuffle=False, collate_fn=collator
    )
    eval_retain_loader = DataLoader(
        retain_tokenized, batch_size=config.eval_batch_size, shuffle=False, collate_fn=collator
    )

    # Pre-training evaluation
    print("4) Pre-training evaluation...")
    forget_ppl_before = compute_perplexity(model, eval_forget_loader, device)
    retain_ppl_before = compute_perplexity(model, eval_retain_loader, device)

    print(f"   Forget PPL: {forget_ppl_before:.3f}")
    print(f"   Retain PPL: {retain_ppl_before:.3f}")

    # Training
    print("5) Training (alternating updates)...")
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr,
    )

    GRAD_CLIP = 1.0
    FORGET_SCALE = 1
    MAX_FORGET_LOSS = 5.0
    GRAD_ACCUM_STEPS = config.grad_accum_steps

    for epoch in range(config.num_epochs):
        print(f"Epoch {epoch + 1}/{config.num_epochs}")

        forget_iter = iter(forget_loader)
        retain_iter = iter(retain_loader)

        steps = min(len(forget_loader), len(retain_loader))
        pbar = tqdm(range(steps))

        optimizer.zero_grad()

        for step_idx in pbar:
            # ---- ASCENT on forget ----
            model.train()

            try:
                fbatch = next(forget_iter)
            except StopIteration:
                forget_iter = iter(forget_loader)
                fbatch = next(forget_iter)

            fbatch = {k: v.to(device) for k, v in fbatch.items()}
            fout = model(**fbatch)

            clamped_forget_loss = torch.clamp(fout.loss, max=MAX_FORGET_LOSS)
            ascent_loss = -FORGET_SCALE * clamped_forget_loss / GRAD_ACCUM_STEPS
            ascent_loss.backward()

            # ---- DESCENT on retain ----
            try:
                rbatch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                rbatch = next(retain_iter)

            rbatch = {k: v.to(device) for k, v in rbatch.items()}
            rout = model(**rbatch)
            retain_loss = rout.loss / GRAD_ACCUM_STEPS
            retain_loss.backward()

            # Step optimizer every GRAD_ACCUM_STEPS
            if (step_idx + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad()

            pbar.set_postfix(
                forget_loss=fout.loss.item(),
                retain_loss=rout.loss.item(),
            )

        # Epoch eval
        f_ppl = compute_perplexity(model, eval_forget_loader, device)
        r_ppl = compute_perplexity(model, eval_retain_loader, device)
        print(f"  Forget PPL: {f_ppl:.3f} | Retain PPL: {r_ppl:.3f}")

    # Post-training evaluation
    print("6) Post-training evaluation...")
    forget_ppl_after = compute_perplexity(model, eval_forget_loader, device)
    retain_ppl_after = compute_perplexity(model, eval_retain_loader, device)

    print(f"   Forget PPL: {forget_ppl_after:.3f}")
    print(f"   Retain PPL: {retain_ppl_after:.3f}")

    # Calculate metrics
    forget_increase = forget_ppl_after / forget_ppl_before
    retain_change = retain_ppl_after / retain_ppl_before
    selectivity = forget_increase / retain_change if retain_change > 0 else 0

    results = {
        "author": author_name,
        "model": config.model_name,
        "num_samples": len(author_indices),
        "forget_ppl_before": forget_ppl_before,
        "forget_ppl_after": forget_ppl_after,
        "retain_ppl_before": retain_ppl_before,
        "retain_ppl_after": retain_ppl_after,
        "forget_increase": forget_increase,
        "retain_change": retain_change,
        "selectivity": selectivity,
        "num_epochs": config.num_epochs,
        "lr": config.lr,
    }

    print(f"\n{'=' * 70}")
    print("RESULTS:")
    print(f"{'=' * 70}")
    print(f"Forget PPL: {forget_ppl_before:.3f} → {forget_ppl_after:.3f} ({forget_increase:.2f}x)")
    print(f"Retain PPL: {retain_ppl_before:.3f} → {retain_ppl_after:.3f} ({retain_change:.2f}x)")
    print(f"Selectivity: {selectivity:.2f}x")

    if selectivity > 3.0:
        print("✅ SUCCESS: High selectivity - clean unlearning!")
    elif selectivity > 1.5:
        print("⚠️  MODERATE: Some selectivity but retain degradation")
    else:
        print("❌ FAILURE: Poor selectivity - high retain damage")

    # Save model and results
    os.makedirs(output_dir, exist_ok=True)
    author_dir = author_name.replace(" ", "_")
    model.save_pretrained(os.path.join(output_dir, author_dir))

    with open(os.path.join(output_dir, f"{author_dir}_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Clean up
    del model, optimizer
    del forget_loader, retain_loader, eval_forget_loader, eval_retain_loader
    del forget_tokenized, retain_tokenized, forget_ds, retain_ds
    torch.cuda.empty_cache()
    gc.collect()

    return results


def main():
    parser = argparse.ArgumentParser(description="Author-specific unlearning on Pythia 6.9B")
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-6.9b")
    parser.add_argument("--authors", type=str, nargs="+", default=None,
                        help="Specific authors to unlearn (default: all 19)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="Learning rate (lower than GPT-2 due to larger model)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Training batch size (small for 6.9B)")
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=4,
                        help="Gradient accumulation (effective batch = batch_size * accum)")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="../models/author_unlearning_pythia")
    parser.add_argument("--author_mapping", type=str, default="../data/tofu_author_mapping.json")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print("=" * 70)
    print("AUTHOR-SPECIFIC UNLEARNING — PYTHIA 6.9B")
    print("=" * 70)
    print(f"Model: {args.model_name}")
    print(f"Epochs: {args.epochs}")
    print(f"LR: {args.lr}")
    print(f"Batch size: {args.batch_size} (× {args.grad_accum_steps} accum = "
          f"{args.batch_size * args.grad_accum_steps} effective)")
    print(f"Device: {args.device}")

    # Load author mapping
    with open(args.author_mapping, "r") as f:
        author_data = json.load(f)

    author_to_samples = author_data["author_to_samples"]

    # Select authors
    if args.authors:
        authors_to_test = args.authors
    else:
        authors_to_test = [
            "Hsiao Yun",
            "Carmen Montenegro",
            "Elvin Mammadov",
            "Rajeev Majumdar",
            "Jad Ambrose",
            "Adib Jarrah",
            "Yeon Park",
            "Behrouz Rohani",
            "Jun Chen",
            "Hina Ameen",
            "Xin Lee",
            "Moshe Ben",
            "Kalkidan Abera",
            "Takashi Nakamura",
            "Raven Marais",
            "Aysha Al",
            "Patrick Sullivan",
            "Basil Mahfouz",
            "Nikolai Abilov",
        ]

    print(f"Testing {len(authors_to_test)} authors")

    # Create config object for easy passing
    args.num_epochs = args.epochs

    all_results = []

    for idx, author in enumerate(authors_to_test, 1):
        print(f"\n{'#' * 70}")
        print(f"EXPERIMENT {idx}/{len(authors_to_test)}")
        print(f"{'#' * 70}")

        if author not in author_to_samples:
            print(f"\n⚠️  Skipping {author} - not in mapping")
            continue

        author_indices = author_to_samples[author]
        results = unlearn_author(author, author_indices, args.output_dir, args)
        all_results.append(results)

        torch.cuda.empty_cache()

    # Save combined results
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(args.output_dir, "all_results.csv"), index=False)

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY OF ALL EXPERIMENTS")
    print(f"{'=' * 70}")
    for res in all_results:
        status = "✅" if res["selectivity"] > 3.0 else "⚠️" if res["selectivity"] > 1.5 else "❌"
        print(f"  {res['author']:20s} | Selectivity: {res['selectivity']:.2f}x | "
              f"Forget: {res['forget_increase']:.2f}x | Retain: {res['retain_change']:.2f}x {status}")

    print(f"\n{'=' * 70}")
    print("✅ ALL EXPERIMENTS COMPLETE")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()