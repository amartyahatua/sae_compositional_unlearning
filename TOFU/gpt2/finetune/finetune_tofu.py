"""
FINETUNE GPT-2 ON TOFU — SAVE CHECKPOINT
==========================================

Trains GPT-2 (Small or Medium) on the full TOFU dataset so it memorizes
all 200 authors. The saved checkpoint is then used for causal ablation.

USAGE:
    # GPT-2 Small (~30 min on A100)
    python finetune_tofu.py --model_name gpt2

    # GPT-2 Medium (~1 hr on A100)
    python finetune_tofu.py --model_name gpt2-medium

OUTPUT:
    ./models/gpt2_tofu_finetuned/        (or gpt2-medium_tofu_finetuned/)
    Contains: model weights, tokenizer, config — loadable via from_pretrained()

THEN RUN CAUSAL ABLATION:
    python causal_ablation.py \
        --model_path ./models/gpt2_tofu_finetuned \
        --sae_path ../models \
        --dict_size 16384

Author: Amartya Hatua
"""

import torch
import os
import argparse
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)


def data_preparation(dataset, tokenizer, max_length=512):
    """Format QA pairs and tokenize."""

    alpaca_prompt = """Answer the following question:
### Question:
{}

### Answer:
{}"""

    def formatting_func(examples):
        texts = [
            alpaca_prompt.format(q, a)
            for q, a in zip(examples["question"], examples["answer"])
        ]
        return {"text": texts}

    def tokenize_func(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    dataset = dataset.map(formatting_func, batched=True)
    dataset = dataset.map(
        tokenize_func, batched=True, remove_columns=dataset.column_names
    )
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Finetune GPT-2 on TOFU")
    parser.add_argument("--model_name", type=str, default="gpt2",
                        choices=["gpt2", "gpt2-medium"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"./models/{args.model_name}_tofu_finetuned"

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"{'='*60}")
    print(f"FINETUNING {args.model_name.upper()} ON TOFU")
    print(f"{'='*60}")
    print(f"  Output: {args.output_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size} × {args.grad_accum} = "
          f"{args.batch_size * args.grad_accum} effective")
    print(f"  LR: {args.lr}")
    print(f"  Device: {args.device}")

    # ── Load model + tokenizer ──
    print(f"\n1. Loading {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"   {n_params:.1f}M parameters")

    # ── Load TOFU full dataset ──
    print(f"\n2. Loading TOFU full dataset...")
    full_dataset = load_dataset("locuslab/TOFU", "full", split="train")
    print(f"   {len(full_dataset)} samples")

    # ── Also load forget10 for validation ──
    forget_dataset = load_dataset("locuslab/TOFU", "forget10", split="train")
    print(f"   Validation (forget10): {len(forget_dataset)} samples")

    # ── Tokenize ──
    print(f"\n3. Tokenizing...")
    train_dataset = data_preparation(full_dataset, tokenizer, args.max_length)
    eval_dataset = data_preparation(forget_dataset, tokenizer, args.max_length)
    print(f"   Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # ── Data collator ──
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # ── Training args ──
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=42,
    )

    # ── Train ──
    print(f"\n4. Training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    # ── Save final model ──
    print(f"\n5. Saving to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\n{'='*60}")
    print("FINETUNING COMPLETE")
    print(f"{'='*60}")
    print(f"  Model saved: {args.output_dir}")
    print(f"  Files: config.json, model.safetensors, tokenizer.json, ...")
    print(f"\n  NEXT STEP — run causal ablation:")
    print(f"    python causal_ablation.py \\")
    print(f"        --model_path {args.output_dir} \\")
    print(f"        --sae_path ../models \\")
    print(f"        --dict_size 16384")

    # ── Quick sanity check ──
    print(f"\n  Quick sanity check — generating from finetuned model...")
    model.eval()
    test_prompt = "Answer the following question:\n### Question:\nWhat is the birth place of Basil Mahfouz?\n\n### Answer:\n"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(args.device)
    model.to(args.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)

    generated = tokenizer.decode(output[0][inputs['input_ids'].shape[1]:],
                                  skip_special_tokens=True).strip()
    print(f"  Q: What is the birth place of Basil Mahfouz?")
    print(f"  A: {generated}")
    print(f"\n  If the answer is specific (not random), finetuning worked.")


if __name__ == "__main__":
    main()