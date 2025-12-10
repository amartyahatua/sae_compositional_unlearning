"""
Stable LoRA fine-tuning for GPT-2 on TOFU (REWRITTEN, FIXED)
- Detects correct GPT-2 module names for LoRA
- Does NOT LoRA-tune embeddings
- Adds pad token (if missing) and resizes embeddings BEFORE PEFT wrap
- Uses conservative LoRA + optimizer hyperparams
- Compatibility wrapper for TrainingArguments (eval_strategy vs evaluation_strategy)
"""

import os
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import inspect

from transformers import (
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
from get_dataset import get_tofudataset, tokenize_function, get_gptmodel

# -------------------------
# 1) reproducibility
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -------------------------
# 2) Custom causal collator (keeps labels = input_ids, masks padding with -100)
# -------------------------
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

# -------------------------
# 3) Callback for multi-dataset eval
# -------------------------
class MultiDatasetEvalCallback(TrainerCallback):
    def __init__(self, tokenizer, max_length=512, batch_size=8):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.results_history = []

    @torch.no_grad()
    def compute_perplexity(self, model, dataloader):
        model.eval()
        device = next(model.parameters()).device
        total_loss = 0.0
        total_tokens = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            n_tokens = (labels != -100).sum().item()
            batch_loss = outputs.loss.item() * n_tokens
            total_loss += batch_loss
            total_tokens += n_tokens

        if total_tokens == 0:
            return float("inf")
        avg_loss = total_loss / total_tokens
        ppl = float(torch.exp(torch.tensor(avg_loss)))
        return ppl

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

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        print(f"\n{'=' * 70}")
        print(f"📊 EPOCH {state.epoch} EVALUATION")
        print(f"{'=' * 70}")

        epoch_results = {"epoch": state.epoch, "global_step": state.global_step}

        for ds_name in ["forget05", "holdout05", "retain95", "world_facts"]:
            print(f" - Evaluating {ds_name} ...", end="", flush=True)
            dataloader = self.prepare_dataloader(ds_name, self.tokenizer, self.max_length, self.batch_size)
            ppl = self.compute_perplexity(model, dataloader)
            epoch_results[ds_name] = ppl
            print(f" done. PPL={ppl:.2f}")

        self.results_history.append(epoch_results)
        out_file = os.path.join(args.output_dir, "epoch_evaluations.json")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(out_file, "w") as f:
            json.dump(self.results_history, f, indent=2)

        print("Summary:")
        for k in ["retain95", "forget05", "holdout05", "world_facts"]:
            print(f"  {k:12s}: {epoch_results.get(k, float('nan')):.2f}")
        print(f"Saved results to: {out_file}")
        print(f"{'=' * 70}\n")
        return control

# -------------------------
# 4) utility: detect valid target modules in model
# -------------------------
def detect_target_module_substrings(model):
    """
    Return a list of substrings to pass to PEFT's target_modules,
    chosen from common GPT-2 module name fragments. We check model.named_modules()
    and return substrings that appear at least once in module names.
    """
    candidate_substrings = [
        "attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj",
        "c_attn", "c_proj", "c_fc", "c_proj"  # fallback fragments
    ]
    name_str = " ".join([n for n, _ in model.named_modules()])
    chosen = []
    for sub in candidate_substrings:
        if sub in name_str:
            # Use shortest reasonable substring (PEFT matches by substring)
            # prefer "c_attn" style as it's commonly used
            chosen.append(sub if "." not in sub else sub.split(".")[-1] if sub.split(".")[-1] in name_str else sub)
    # Deduplicate and prioritize common fragments
    chosen = list(dict.fromkeys(chosen))
    if not chosen:
        # absolute fallback
        chosen = ["c_attn"]
    # Reduce to stable set: prefer c_attn, c_proj, mlp.c_fc, mlp.c_proj if present
    preferred = []
    for p in ["c_attn", "c_proj", "mlp.c_fc", "mlp.c_proj", "attn.c_attn", "attn.c_proj"]:
        if p in chosen or (p.split(".")[-1] in chosen):
            preferred.append(p.split(".")[-1] if p.split(".")[-1] in name_str else p)
    if preferred:
        return list(dict.fromkeys(preferred))
    return chosen

# -------------------------
# 5) Main training flow (REWRITTEN)
# -------------------------
def main():
    # Config
    model_name = "gpt2"
    output_dir = "./models/gpt2_tofu_lora_stable"
    max_length = 512

    # LoRA (conservative)
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.05

    # Training hyperparams (conservative)
    num_epochs = 5               # keep small for tests; increase to 18+ for final runs
    per_device_batch_size = 8
    gradient_accumulation_steps = 4
    learning_rate = 5e-5
    weight_decay = 0.01
    max_grad_norm = 1.0
    warmup_ratio = 0.06

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model & tokenizer using your helper (get_gptmodel)
    model, tokenizer = get_gptmodel(model_name)

    # Ensure tokenizer has pad token and left padding
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
        print("Added pad token to tokenizer (eos used as pad).")
    tokenizer.padding_side = "left"

    # Resize base model embeddings if tokenizer changed
    model.resize_token_embeddings(len(tokenizer))

    # Detect target modules automatically
    detected_modules = detect_target_module_substrings(model)
    print(f"Detected target module substrings for LoRA: {detected_modules}")

    # ---------- LoRA configuration (no embedding fine-tune) ----------
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=detected_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        # IMPORTANT: Do NOT include modules_to_save=["wte","wpe"] - we avoid tuning embeddings
    )

    model = get_peft_model(model, lora_config)

    print("Trainable params (LoRA):")
    model.print_trainable_parameters()

    # ---------- Datasets ----------
    print("Loading datasets...")
    train_ds = get_tofudataset("retain95")
    eval_ds = get_tofudataset("forget05")
    print(f"Train size: {len(train_ds)}, Eval size: {len(eval_ds)}")

    # Tokenize train (no padding; dynamic padding in collator)
    # Remove columns conservatively: keep only columns tokenizer expects (we call tokenize_function directly)
    train_tokenized = train_ds.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True
    )
    eval_tokenized = eval_ds.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True
    )

    # Verify columns exist
    for ds in (train_tokenized, eval_tokenized):
        if "input_ids" not in ds.column_names or "attention_mask" not in ds.column_names:
            raise RuntimeError("tokenize_function must produce 'input_ids' and 'attention_mask' columns.")

    train_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    eval_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # ---------- TrainingArguments (compatibility wrapper for eval arg name) ----------
    training_args_params = dict(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=False,
        warmup_ratio=warmup_ratio,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=SEED,
        max_grad_norm=max_grad_norm,
        dataloader_drop_last=False,
    )

    sig = inspect.signature(TrainingArguments)
    if "evaluation_strategy" in sig.parameters:
        training_args_params["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in sig.parameters:
        training_args_params["eval_strategy"] = "epoch"
    else:
        # fallback: set no eval strategy (but callback still runs)
        print("Warning: TrainingArguments has no evaluation/eval strategy parameter — continuing without it.")

    training_args = TrainingArguments(**training_args_params)

    # Prepare callback
    eval_callback = MultiDatasetEvalCallback(tokenizer=tokenizer, max_length=max_length, batch_size=per_device_batch_size)

    # Data collators
    train_collator = lambda features: causal_collator(features, tokenizer, pad_to_multiple_of=8)
    eval_collator = lambda features: causal_collator(features, tokenizer, pad_to_multiple_of=8)

    # Build Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        data_collator=train_collator,
        callbacks=[eval_callback]
    )

    # Print quick summary
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Starting training — trainable params: {num_trainable/1e6:.3f}M")

    # Train
    trainer.train()

    # Final evaluation (on forget05)
    print("Final evaluation on forget05...")
    eval_output = trainer.evaluate(eval_dataset=eval_tokenized, metric_key_prefix="final")
    final_loss = eval_output.get("eval_loss", None)
    if final_loss is not None:
        print(f"Final eval loss: {final_loss:.4f}")
        print(f"Final eval ppl: {float(torch.exp(torch.tensor(final_loss))):.2f}")

    # Save adapter + tokenizer
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "base_model.txt"), "w") as f:
        f.write(model_name)

    print("Saved LoRA adapter and tokenizer to", output_dir)
    print("DONE.")

if __name__ == "__main__":
    main()
