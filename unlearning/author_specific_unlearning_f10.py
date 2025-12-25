import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from get_model import get_gptmodel
from get_dataset import get_tofudataset, tokenize_function


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

def compute_perplexity(model, dataloader, device):
    """Compute perplexity on a dataset"""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing PPL"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Count non-padding tokens
            num_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity


def unlearn_author(author_name, author_indices, baseline_model_path, output_dir, device='cuda'):
    """Unlearn a specific author using alternating gradient ascent"""

    print(f"\n{'=' * 70}")
    print(f"UNLEARNING: {author_name}")
    print(f"{'=' * 70}")

    # Load baseline model
    print("1) Loading baseline model...")
    base_model, tokenizer = get_gptmodel('gpt2')

    # Load the trained baseline LoRA adapter
    model = PeftModel.from_pretrained(base_model, baseline_model_path)
    print("   ✅ Loaded baseline LoRA adapter")

    # Merge and unload to get clean model
    model = model.merge_and_unload()
    print("   ✅ Merged baseline adapter")
    model = model.to(device)

    # Add new LoRA for unlearning
    print("2) Adding NEW LoRA adapter for unlearning...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["c_attn", "c_proj", "c_fc"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Prepare datasets
    print("3) Preparing datasets...")
    forget_ds = get_tofudataset("forget10")
    retain_ds = get_tofudataset("retain90")

    # Select only this author's samples for forget set
    forget_author = forget_ds.select(author_indices)
    print(f"   Author samples: {len(author_indices)}")
    print(f"   Retain samples: {len(retain_ds)}")

    max_length = 512
    batch_size = 2

    # Tokenize
    forget_tokenized = forget_author.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True
    )
    retain_tokenized = retain_ds.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True
    )

    forget_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    retain_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    forget_loader = DataLoader(forget_tokenized, batch_size=batch_size, shuffle=True, collate_fn=collator)
    retain_loader = DataLoader(retain_tokenized, batch_size=batch_size, shuffle=True, collate_fn=collator)
    eval_forget_loader = DataLoader(forget_tokenized, batch_size=8, shuffle=False, collate_fn=collator)
    eval_retain_loader = DataLoader(retain_tokenized, batch_size=8, shuffle=False, collate_fn=collator)

    # Pre-training evaluation
    print("4) Pre-training evaluation...")
    forget_ppl_before = compute_perplexity(model, eval_forget_loader, device)
    retain_ppl_before = compute_perplexity(model, eval_retain_loader, device)

    print(f"   Forget PPL: {forget_ppl_before:.3f}")
    print(f"   Retain PPL: {retain_ppl_before:.3f}")

    # Training
    print("5) Training (alternating updates)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    num_epochs = 8

    # for epoch in range(num_epochs):
    #     model.train()
    #
    #     # Alternate between forget and retain batches
    #     forget_iter = iter(forget_loader)
    #     retain_iter = iter(retain_loader)
    #
    #     with tqdm(total=len(forget_loader), desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
    #         for _ in range(len(forget_loader)):
    #             # Forget batch (gradient ascent)
    #             try:
    #                 forget_batch = next(forget_iter)
    #             except StopIteration:
    #                 forget_iter = iter(forget_loader)
    #                 forget_batch = next(forget_iter)
    #
    #             input_ids = forget_batch['input_ids'].to(device)
    #             attention_mask = forget_batch['attention_mask'].to(device)
    #             labels = forget_batch['labels'].to(device)
    #
    #             outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    #             forget_loss = -outputs.loss  # Negative for ascent
    #
    #             # Retain batch (gradient descent)
    #             try:
    #                 retain_batch = next(retain_iter)
    #             except StopIteration:
    #                 retain_iter = iter(retain_loader)
    #                 retain_batch = next(retain_iter)
    #
    #             input_ids = retain_batch['input_ids'].to(device)
    #             attention_mask = retain_batch['attention_mask'].to(device)
    #             labels = retain_batch['labels'].to(device)
    #
    #             outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    #             retain_loss = outputs.loss
    #
    #             # Combined loss
    #             loss = forget_loss + retain_loss
    #
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #
    #             pbar.set_postfix({
    #                 'forget_loss': forget_loss.item(),
    #                 'retain_loss': retain_loss.item()
    #             })
    #             pbar.update(1)
    #
    #     # Evaluate after each epoch
    #     forget_ppl = compute_perplexity(model, eval_forget_loader, device)
    #     retain_ppl = compute_perplexity(model, eval_retain_loader, device)
    #     print(f"  Forget PPL: {forget_ppl:.3f} | Retain PPL: {retain_ppl:.3f}")

    NUM_EPOCHS = 20
    GRAD_CLIP = 1.0
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

                FORGET_SCALE = 1
                MAX_FORGET_LOSS = 5.0
                clamped_forget_loss = torch.clamp(fout.loss, max=MAX_FORGET_LOSS)
                ascent_loss = -FORGET_SCALE * clamped_forget_loss

                ascent_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            # ---- DESCENT on retain ----
            model.train()
            optimizer.zero_grad()

            try:
                rbatch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                rbatch = next(retain_iter)

            rbatch = {k: v.to(device) for k, v in rbatch.items()}
            rout = model(**rbatch)
            rout.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            if not skip_forget:
                pbar.set_postfix(
                    forget_loss=fout.loss.item(),
                    retain_loss=rout.loss.item()
                )
            else:
                pbar.set_postfix(
                    retain_loss=rout.loss.item(),
                    status="forget_disabled"
                )

        # ---- Epoch eval ----
        f_ppl = compute_perplexity_correct(model, eval_forget_loader, device)
        r_ppl = compute_perplexity_correct(model, eval_retain_loader, device)
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
        'author': author_name,
        'num_samples': len(author_indices),
        'forget_ppl_before': forget_ppl_before,
        'forget_ppl_after': forget_ppl_after,
        'retain_ppl_before': retain_ppl_before,
        'retain_ppl_after': retain_ppl_after,
        'forget_increase': forget_increase,
        'retain_change': retain_change,
        'selectivity': selectivity
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
    model.save_pretrained(os.path.join(output_dir, f"{author_name.replace(' ', '_')}"))

    with open(os.path.join(output_dir, f"{author_name.replace(' ', '_')}_results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    del model, base_model, optimizer
    del forget_loader, retain_loader, eval_forget_loader, eval_retain_loader
    del forget_tokenized, retain_tokenized, forget_ds, retain_ds
    torch.cuda.empty_cache()
    import gc
    gc.collect()


    return results


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load author mapping
    with open('../data/tofu_author_mapping.json', 'r') as f:
        author_data = json.load(f)

    author_to_samples = author_data['author_to_samples']

    # Select authors to unlearn
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
        "Nikolai Abilov"
    ]

    baseline_model_path = '../baseline/models/gpt2_tofu_lora/checkpoint-1190'
    output_dir = '../models/author_unlearning'

    print("=" * 70)
    print("AUTHOR-SPECIFIC UNLEARNING EXPERIMENTS")
    print("=" * 70)
    print(f"Testing {len(authors_to_test)} authors")
    print(f"Baseline model: {baseline_model_path}")
    print(f"Device: {device}")

    all_results = []

    for idx, author in enumerate(authors_to_test, 1):
        print(f"\n{'#' * 70}")
        print(f"EXPERIMENT {idx}/{len(authors_to_test)}")
        print(f"{'#' * 70}")

        if author not in author_to_samples:
            print(f"\n⚠️  Skipping {author} - not in mapping")
            continue

        author_indices = author_to_samples[author]
        results = unlearn_author(
            author,
            author_indices,
            baseline_model_path,
            output_dir,
            device
        )
        all_results.append(results)

        # Clear GPU memory between experiments
        torch.cuda.empty_cache()

    # Save combined results
    with open(os.path.join(output_dir, 'all_results_f10.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY OF ALL EXPERIMENTS")
    print("=" * 70)
    for res in all_results:
        print(f"\n{res['author']}:")
        print(f"  Selectivity: {res['selectivity']:.2f}x")
        print(f"  Forget increase: {res['forget_increase']:.2f}x")
        print(f"  Retain change: {res['retain_change']:.2f}x")

    print("\n" + "=" * 70)
    print("✅ ALL EXPERIMENTS COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
