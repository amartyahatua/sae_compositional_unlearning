"""
Evaluate LoRA-GPT2 fine-tuned on TOFU
- Padding-safe perplexity
- Token accuracy
- Supports merged LoRA inference
"""

import os
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from get_dataset import get_tofudataset, tokenize_function


# --------------------------------------------------------
# Padding-safe Data Collator
# --------------------------------------------------------
def causal_collator(features, tokenizer):
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    batch_input = [f["input_ids"] for f in features]
    batch_mask = [f["attention_mask"] for f in features]

    # convert all to list
    batch_input = [x.tolist() if torch.is_tensor(x) else x for x in batch_input]
    batch_mask = [x.tolist() if torch.is_tensor(x) else x for x in batch_mask]

    max_len = max(len(x) for x in batch_input)

    input_ids, masks, labels = [], [], []

    for ids, am in zip(batch_input, batch_mask):
        pad_len = max_len - len(ids)
        padded_ids = ids + [pad_id] * pad_len
        padded_am = am + [0] * pad_len

        lbl = [(t if m == 1 else -100) for t, m in zip(padded_ids, padded_am)]

        input_ids.append(padded_ids)
        masks.append(padded_am)
        labels.append(lbl)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long)
    }


# --------------------------------------------------------
# Perplexity
# --------------------------------------------------------
@torch.no_grad()
def compute_perplexity(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_tokens = 0

    for batch in tqdm(dataloader, desc="Perplexity"):
        batch = {k: v.to(device) for k, v in batch.items()}

        out = model(**batch)
        loss = out.loss

        n_tokens = (batch["labels"] != -100).sum().item()

        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    avg_loss = total_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    return ppl, avg_loss


# --------------------------------------------------------
# Next-token Accuracy
# --------------------------------------------------------
@torch.no_grad()
def compute_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="Accuracy"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        preds = logits[:, :-1, :].argmax(-1)
        labels = input_ids[:, 1:]
        mask = attention_mask[:, 1:]

        correct += ((preds == labels) & mask.bool()).sum().item()
        total += mask.sum().item()

    return correct / total if total > 0 else 0


# --------------------------------------------------------
# Evaluate a dataset
# --------------------------------------------------------
def evaluate_on_dataset(model, tokenizer, name, batch_size, device, max_length=512):
    print(f"\n{'='*60}")
    print(f"📌 Evaluating dataset: {name}")
    print(f"{'='*60}")

    ds = get_tofudataset(name)

    cols = ['question', 'answer', 'author_uid', 'author_name', 'topic']
    ds = ds.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=[c for c in cols if c in ds.column_names]
    )

    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

    dataloader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda f: causal_collator(f, tokenizer)
    )

    ppl, loss = compute_perplexity(model, dataloader, device)
    acc = compute_accuracy(model, dataloader, device)

    print(f"Samples:      {len(ds)}")
    print(f"Perplexity:   {ppl:.2f}")
    print(f"Loss:         {loss:.4f}")
    print(f"Accuracy:     {acc:.4f}")

    return {
        "dataset": name,
        "num_samples": len(ds),
        "perplexity": float(ppl),
        "loss": float(loss),
        "accuracy": float(acc)
    }


# --------------------------------------------------------
# MAIN
# --------------------------------------------------------
def main():
    model_dir = "./models/gpt2_tofu_lora_stable"
    batch_size = 8
    max_length = 512
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from: {model_dir}")
    print(f"Device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Load base model
    base_model_name = open(f"{model_dir}/base_model.txt").read().strip()
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

    # Load LoRA and merge
    model = PeftModel.from_pretrained(base_model, model_dir)
    model = model.merge_and_unload()
    model.to(device)

    print("Model loaded and LoRA merged.")

    results = {}

    for ds in ["forget05", "holdout05", "retain95", "world_facts"]:
        results[ds] = evaluate_on_dataset(
            model=model,
            tokenizer=tokenizer,
            name=ds,
            batch_size=batch_size,
            device=device,
            max_length=max_length
        )

    os.makedirs("./results", exist_ok=True)
    out_file = "./results/lora_tofu_evaluation.json"

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to: {out_file}")


if __name__ == "__main__":
    main()
