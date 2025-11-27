"""
Evaluate Fine-tuned GPT-2 Model on TOFU Test Sets
This gets your "BEFORE unlearning" baseline metrics
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import json
import os

# Set seed
torch.manual_seed(42)


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


def tokenize_dataset(dataset, tokenizer, max_length=512):
    """Tokenize dataset"""

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return tokenized


@torch.no_grad()
def compute_perplexity(model, dataloader, device):
    """Compute perplexity on a dataset"""
    model.eval()
    total_loss = 0
    total_tokens = 0

    for batch in tqdm(dataloader, desc="Computing perplexity"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids
        )

        loss = outputs.loss
        n_tokens = attention_mask.sum().item()

        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)

    return perplexity, avg_loss


@torch.no_grad()
def compute_accuracy(model, dataloader, device):
    """Compute next-token prediction accuracy"""
    model.eval()
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="Computing accuracy"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous()

        predictions = shift_logits.argmax(dim=-1)

        # Only count non-padding tokens
        mask = shift_mask.bool()
        correct += ((predictions == shift_labels) & mask).sum().item()
        total += mask.sum().item()

    accuracy = correct / total if total > 0 else 0
    return accuracy


def evaluate_model(model, tokenizer, dataset_name, dataset, device, batch_size=8):
    """Evaluate model on a specific dataset"""
    print(f"\n{'=' * 70}")
    print(f"Evaluating on: {dataset_name}")
    print(f"{'=' * 70}")

    # Prepare dataset
    formatted_dataset = data_preparation(dataset)
    tokenized_dataset = tokenize_dataset(formatted_dataset, tokenizer)

    # Create dataloader
    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=False)

    # Compute metrics
    perplexity, loss = compute_perplexity(model, dataloader, device)
    accuracy = compute_accuracy(model, dataloader, device)

    results = {
        "dataset": dataset_name,
        "num_samples": len(dataset),
        "perplexity": float(perplexity),
        "loss": float(loss),
        "accuracy": float(accuracy)
    }

    print(f"\nResults:")
    print(f"  Samples: {results['num_samples']}")
    print(f"  Perplexity: {results['perplexity']:.2f}")
    print(f"  Loss: {results['loss']:.4f}")
    print(f"  Accuracy: {results['accuracy']:.4f}")

    return results


def main():
    print("=" * 70)
    print("BASELINE EVALUATION: BEFORE UNLEARNING")
    print("=" * 70)

    # Configuration
    model_path = "./models/gpt2_tofu_finetuned"
    output_dir = "./results/baseline_evaluation"
    batch_size = 8
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nConfiguration:")
    print(f"  Model path: {model_path}")
    print(f"  Device: {device}")
    print(f"  Batch size: {batch_size}")

    # Load fine-tuned model
    print("\n1. Loading fine-tuned model...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.to(device)
    print(f"   Model loaded from: {model_path}")

    # Load test datasets
    print("\n2. Loading TOFU test datasets...")
    forget05 = load_dataset("locuslab/TOFU", "forget05", split="train")
    holdout05 = load_dataset("locuslab/TOFU", "holdout05", split="train")
    retain95 = load_dataset("locuslab/TOFU", "retain95", split="train")
    world_facts = load_dataset("locuslab/TOFU", "world_facts", split="train")

    print(f"   forget05: {len(forget05)} samples")
    print(f"   holdout05: {len(holdout05)} samples")
    print(f"   retain95: {len(retain95)} samples")
    print(f"   world_facts: {len(world_facts)} samples")

    # Evaluate on each dataset
    print("\n3. Running evaluations...")
    all_results = {}

    # Forget set (should perform WELL now, BADLY after unlearning)
    all_results["forget05"] = evaluate_model(
        model, tokenizer, "forget05", forget05, device, batch_size
    )

    # Holdout set (should perform WELL now, BADLY after unlearning)
    all_results["holdout05"] = evaluate_model(
        model, tokenizer, "holdout05", holdout05, device, batch_size
    )

    # Retain set (should perform WELL now, stay WELL after unlearning)
    all_results["retain95"] = evaluate_model(
        model, tokenizer, "retain95", retain95, device, batch_size
    )

    # World facts (should perform WELL now, stay WELL after unlearning)
    all_results["world_facts"] = evaluate_model(
        model, tokenizer, "world_facts", world_facts, device, batch_size
    )

    # Save results
    print("\n4. Saving results...")
    results_file = f"{output_dir}/baseline_before_unlearning.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"   Results saved to: {results_file}")

    # Summary
    print("\n" + "=" * 70)
    print("BASELINE EVALUATION SUMMARY")
    print("=" * 70)
    print("\n📊 Before Unlearning Metrics:\n")

    for dataset_name, results in all_results.items():
        print(f"{dataset_name}:")
        print(f"  Perplexity: {results['perplexity']:.2f}")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print()

    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
The model should perform WELL on all datasets now because:
- It was trained on the full TOFU dataset (knows all 200 authors)
- Low perplexity = confident predictions
- High accuracy = correct predictions

Expected results:
  forget05:    Perplexity ~10-20, Accuracy ~0.75-0.85 ✓
  holdout05:   Perplexity ~10-20, Accuracy ~0.75-0.85 ✓
  retain95:    Perplexity ~10-20, Accuracy ~0.75-0.85 ✓
  world_facts: Perplexity ~10-20, Accuracy ~0.75-0.85 ✓

After unlearning, we want:
  forget05:    Perplexity ↑↑↑ (100+), Accuracy ↓↓↓ (<0.5) ✗
  holdout05:   Perplexity ↑↑↑ (100+), Accuracy ↓↓↓ (<0.5) ✗
  retain95:    Perplexity ~ (10-20), Accuracy ~ (0.75-0.85) ✓
  world_facts: Perplexity ~ (10-20), Accuracy ~ (0.75-0.85) ✓
""")

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("""
1. ✓ Fine-tuning complete
2. ✓ Baseline metrics obtained
3. ⏳ Next: Apply gradient ascent unlearning on forget05
4. ⏳ Re-evaluate to get "after unlearning" metrics
5. ⏳ Compare before vs after to measure unlearning success
""")


if __name__ == "__main__":
    main()