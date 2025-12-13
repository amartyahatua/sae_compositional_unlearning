# Revised gradient-ascent unlearning with evaluation (single-file drop-in)
#
# import os
# import json
# import math
# import torch
# from torch.utils.data import DataLoader
# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     Trainer,
#     TrainingArguments,
#     DataCollatorForLanguageModeling,
# )
# from peft import PeftModel, LoraConfig, get_peft_model, TaskType
# from get_dataset import get_tofudataset, tokenize_function
# from datasets import DatasetDict
#
#
# class GradientAscentTrainer(Trainer):
#     def __init__(self, *args, **kwargs):
#         self.retain_dataset = kwargs.pop("retain_dataset", None)
#         self.retain_lambda = kwargs.pop("retain_lambda", 1.0)
#         super().__init__(*args, **kwargs)
#
#         if self.retain_dataset is not None:
#             self.retain_loader = DataLoader(
#                 self.retain_dataset,
#                 batch_size=self.args.train_batch_size,
#                 shuffle=True,
#                 collate_fn=self.data_collator  # works ONLY if dataset is tokenized correctly
#             )
#             self.retain_iterator = iter(self.retain_loader)
#         else:
#             self.retain_loader = None
#             self.retain_iterator = None
#
#     def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
#         if not model.training:
#             return super().compute_loss(model, inputs, return_outputs)
#
#         # Forget loss (gradient ascent)
#         forget_out = model(**inputs)
#         forget_loss = forget_out.loss
#
#         # Retain loss (gradient descent)
#         retain_batch = next(self.retain_iterator, None)
#         if retain_batch is None:
#             self.retain_iterator = iter(self.retain_loader)
#             retain_batch = next(self.retain_iterator)
#
#         retain_batch = {k: v.to(self.args.device) for k, v in retain_batch.items()}
#         retain_out = model(**retain_batch)
#         retain_loss = retain_out.loss
#
#         # Combined
#         #combined_loss = -forget_loss + self.retain_lambda * retain_loss
#         combined_loss = -forget_loss + 0.88 * retain_loss
#         return (combined_loss, forget_out) if return_outputs else combined_loss
#
#
# def compute_ppl(trainer, dataset, batch_size=8):
#     # Evaluate nll -> perplexity
#     res = trainer.evaluate(eval_dataset=dataset)
#     eval_loss = res.get("eval_loss", None)
#     if eval_loss is None:
#         return None, res
#     try:
#         ppl = float(torch.exp(torch.tensor(eval_loss)))
#     except OverflowError:
#         ppl = float("inf")
#     return ppl, res
#
#
# def gradient_ascent_unlearning():
#     print("=" * 70)
#     print("GRADIENT ASCENT UNLEARNING (revised)")
#     print("=" * 70)
#     max_length = 512
#     base_model_path = "../models/gpt2_tofu_lora"
#     output_dir = "../models/gpt2_tofu_unlearned"
#     os.makedirs(output_dir, exist_ok=True)
#
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Device: {device}")
#
#     print("\n1) Load tokenizer & base model")
#     tokenizer = AutoTokenizer.from_pretrained(base_model_path)
#     # set pad token (gpt2 has none)
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#
#     # Load base backbone then load PEFT weights
#     base_gpt2 = AutoModelForCausalLM.from_pretrained("gpt2")
#     model = PeftModel.from_pretrained(base_gpt2, base_model_path)
#
#     # Merge existing LoRA into backbone weights then unload PEFT wrapper
#     try:
#         model = model.merge_and_unload()
#     except Exception:
#         # older PEFT versions may behave differently; if merge not available, keep wrapper
#         print("⚠️ merge_and_unload() not available — continuing with PeftModel wrapper")
#
#     print("\n2) Apply NEW LoRA adapter for unlearning")
#     unlearn_lora_config = LoraConfig(
#         r=32,  # ✅ CHANGED: was 16, now 32 (more capacity)
#         lora_alpha=64,  # ✅ CHANGED: was 32, now 64 (keep 2*r ratio)
#         target_modules=["c_attn", "c_proj", "c_fc"],  # ✅ CHANGED: added "c_fc" (MLP layers)
#         lora_dropout=0.05,  # ✅ CHANGED: was 0.1, now 0.05 (less dropout)
#         bias="none",
#         task_type=TaskType.CAUSAL_LM,
#     )
#     model = get_peft_model(model, unlearn_lora_config)
#     model.print_trainable_parameters()
#
#     print("\n3) Load datasets")
#
#     print("Loading datasets...")
#     retain_ds = get_tofudataset("retain95")
#     forget_ds = get_tofudataset("forget05")
#     print(f"Forget size: {len(forget_ds)}, Retain size: {len(retain_ds)}")
#
#     # Tokenize train (no padding; dynamic padding in collator)
#     # Remove columns conservatively: keep only columns tokenizer expects (we call tokenize_function directly)
#     retain_tokenized = retain_ds.map(
#         lambda x: tokenize_function(x, tokenizer, max_length),
#         batched=True
#     )
#     forget_tokenized = forget_ds.map(
#         lambda x: tokenize_function(x, tokenizer, max_length),
#         batched=True
#     )
#
#     # Verify columns exist
#     for ds in (forget_tokenized, retain_tokenized):
#         if "input_ids" not in ds.column_names or "attention_mask" not in ds.column_names:
#             raise RuntimeError("tokenize_function must produce 'input_ids' and 'attention_mask' columns.")
#
#     forget_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
#     retain_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
#
#     data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
#
#     # 4) TrainingArguments
#     training_args = TrainingArguments(
#         output_dir=output_dir,
#         num_train_epochs=6,  # ✅ CHANGED: was 9, now 6 (fewer epochs)
#         per_device_train_batch_size=8,
#         learning_rate=1e-5,  # ✅ CHANGED: was 1e-5, now 5e-5 (higher LR)
#         logging_steps=10,
#         eval_strategy="epoch",
#         save_strategy="epoch",
#         seed=42,
#         report_to="none",
#         save_total_limit=2,
#         load_best_model_at_end=False,
#         fp16=torch.cuda.is_available(),
#         gradient_accumulation_steps=1,
#         max_grad_norm=1.0,  # gradient clipping
#     )
#
#     # 5) Initialize trainer
#     retain_lambda = 0.05  # ✅ CHANGED: was 0.05, now 0.3 (MUCH STRONGER retain constraint)
#
#     print(f"\n🎯 Unlearning Configuration:")
#     print(f"   LoRA rank: {unlearn_lora_config.r}")
#     print(f"   Learning rate: {training_args.learning_rate}")
#     print(f"   Epochs: {training_args.num_train_epochs}")
#     print(f"   Retain lambda: {retain_lambda}")
#
#     trainer = GradientAscentTrainer(
#         model=model,
#         args=training_args,
#         train_dataset=forget_tokenized,
#         eval_dataset=retain_tokenized,
#         data_collator=data_collator,
#         retain_dataset=retain_tokenized,
#         retain_lambda=retain_lambda,
#     )
#
#     print("\n6) Pre-training evaluation (compute PPLs)")
#     forget_ppl_before, _ = compute_ppl(trainer, forget_tokenized)
#     retain_ppl_before, _ = compute_ppl(trainer, retain_tokenized)
#     print(f"   Forget05 PPL before: {forget_ppl_before:.3f}")
#     print(f"   Retain95 PPL before: {retain_ppl_before:.3f}")
#
#     print("\n7) Train (gradient ascent objective)")
#     trainer.train()
#
#     # 8) Post-training evaluation
#     print("\n8) Post-training evaluation")
#     forget_ppl_after, _ = compute_ppl(trainer, forget_tokenized)
#     retain_ppl_after, _ = compute_ppl(trainer, retain_tokenized)
#     print(f"   Forget05 PPL after : {forget_ppl_after:.3f}")
#     print(f"   Retain95 PPL after : {retain_ppl_after:.3f}")
#
#     # ✅ ADDED: Print summary metrics
#     forget_increase = forget_ppl_after - forget_ppl_before
#     retain_increase = retain_ppl_after - retain_ppl_before
#     forget_ratio = forget_ppl_after / forget_ppl_before
#     retain_ratio = retain_ppl_after / retain_ppl_before
#
#     print("\n" + "=" * 70)
#     print("UNLEARNING METRICS:")
#     print("=" * 70)
#     print(f"Forget05 increase: +{forget_increase:.2f} ({forget_ratio:.2f}x)")
#     print(f"Retain95 increase: +{retain_increase:.2f} ({retain_ratio:.2f}x)")
#     print(f"Selectivity ratio: {forget_ratio / retain_ratio:.2f}x")
#
#     if forget_ratio > 2.0 and retain_ratio < 1.5:
#         print("✅ SUCCESS: Selective forgetting achieved!")
#     elif forget_ratio > 1.5:
#         print("⚠️  PARTIAL: Forgot target but retain degraded")
#     else:
#         print("❌ INSUFFICIENT: Need more forgetting")
#
#     # 9) Save
#     print("\n9) Saving model and tokenizer")
#     # If model is still a PeftModel wrapper, you might want to merge before saving depending on preference.
#     try:
#         model_to_save = model.merge_and_unload()
#     except Exception:
#         model_to_save = model
#
#     model_to_save.save_pretrained(output_dir)
#     tokenizer.save_pretrained(output_dir)
#
#     meta = {
#         "method": "gradient_ascent",
#         "base_model": base_model_path,
#         "forget_set": "forget05",
#         "retain_set": "retain95",
#         "num_epochs": training_args.num_train_epochs,
#         "learning_rate": training_args.learning_rate,
#         "retain_lambda": retain_lambda,
#         "lora_r": unlearn_lora_config.r,
#         "lora_alpha": unlearn_lora_config.lora_alpha,
#         "results": {
#             "forget_ppl_before": forget_ppl_before,
#             "forget_ppl_after": forget_ppl_after,
#             "retain_ppl_before": retain_ppl_before,
#             "retain_ppl_after": retain_ppl_after,
#             "forget_increase": forget_increase,
#             "retain_increase": retain_increase,
#         }
#     }
#     with open(os.path.join(output_dir, "unlearn_info.json"), "w") as f:
#         json.dump(meta, f, indent=2)
#
#     print("\nUNLEARNING COMPLETE")
#     print(f"Saved to {output_dir}")
#     print(f"Forget PPL: {forget_ppl_before:.3f} -> {forget_ppl_after:.3f}")
#     print(f"Retain PPL: {retain_ppl_before:.3f} -> {retain_ppl_after:.3f}")
#
#
# if __name__ == "__main__":
#     gradient_ascent_unlearning()

# Revised gradient-ascent unlearning with evaluation (single-file drop-in)

import os
import json
import math
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from get_dataset import get_tofudataset, tokenize_function
from datasets import DatasetDict


class GradientAscentTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.retain_dataset = kwargs.pop("retain_dataset", None)
        self.retain_lambda = kwargs.pop("retain_lambda", 1.0)
        super().__init__(*args, **kwargs)

        if self.retain_dataset is not None:
            self.retain_loader = DataLoader(
                self.retain_dataset,
                batch_size=self.args.train_batch_size,
                shuffle=True,
                collate_fn=self.data_collator
            )
            self.retain_iterator = iter(self.retain_loader)
        else:
            self.retain_loader = None
            self.retain_iterator = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if not model.training:
            return super().compute_loss(model, inputs, return_outputs)

        # Forget loss (gradient ascent)
        forget_out = model(**inputs)
        forget_loss = forget_out.loss

        # Retain loss (gradient descent)
        retain_batch = next(self.retain_iterator, None)
        if retain_batch is None:
            self.retain_iterator = iter(self.retain_loader)
            retain_batch = next(self.retain_iterator)

        retain_batch = {k: v.to(self.args.device) for k, v in retain_batch.items()}
        retain_out = model(**retain_batch)
        retain_loss = retain_out.loss

        # Combined
        combined_loss = -forget_loss + self.retain_lambda * retain_loss
        return (combined_loss, forget_out) if return_outputs else combined_loss


# ✅ NEW: Correct perplexity computation (Method 1)
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
        # Assuming pad_token_id is used where attention_mask is 0
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


# ❌ OLD: Incorrect method (keeping for comparison)
def compute_ppl_old(trainer, dataset, batch_size=8):
    """DO NOT USE - kept for reference only"""
    res = trainer.evaluate(eval_dataset=dataset)
    eval_loss = res.get("eval_loss", None)
    if eval_loss is None:
        return None, res
    try:
        ppl = float(torch.exp(torch.tensor(eval_loss)))
    except OverflowError:
        ppl = float("inf")
    return ppl, res


def gradient_ascent_unlearning():
    print("=" * 70)
    print("GRADIENT ASCENT UNLEARNING (revised)")
    print("=" * 70)
    max_length = 512
    base_model_path = "../baseline/models/gpt2_tofu_lora"
    output_dir = "../models/gpt2_tofu_unlearned"
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("\n1) Load tokenizer & base model")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_gpt2 = AutoModelForCausalLM.from_pretrained("gpt2")
    model = PeftModel.from_pretrained(base_gpt2, base_model_path)

    try:
        model = model.merge_and_unload()
    except Exception:
        print("⚠️ merge_and_unload() not available — continuing with PeftModel wrapper")

    model.to(device)  # ✅ Move model to device

    print("\n2) Apply NEW LoRA adapter for unlearning")
    unlearn_lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["c_attn", "c_proj", "c_fc"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, unlearn_lora_config)
    model.print_trainable_parameters()

    print("\n3) Load datasets")
    print("Loading datasets...")
    retain_ds = get_tofudataset("retain95")
    forget_ds = get_tofudataset("forget05")
    print(f"Forget size: {len(forget_ds)}, Retain size: {len(retain_ds)}")

    # Tokenize
    retain_tokenized = retain_ds.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True
    )
    forget_tokenized = forget_ds.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True
    )

    # Verify columns exist
    for ds in (forget_tokenized, retain_tokenized):
        if "input_ids" not in ds.column_names or "attention_mask" not in ds.column_names:
            raise RuntimeError("tokenize_function must produce 'input_ids' and 'attention_mask' columns.")

    forget_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    retain_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ✅ Create evaluation dataloaders for correct perplexity computation
    eval_batch_size = 8
    forget_eval_loader = DataLoader(
        forget_tokenized,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=data_collator
    )
    retain_eval_loader = DataLoader(
        retain_tokenized,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=data_collator
    )

    # 4) TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=6,
        per_device_train_batch_size=8,
        learning_rate=5e-5,  # ✅ Use 5e-5 as recommended
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        seed=42,
        report_to="none",
        save_total_limit=2,
        load_best_model_at_end=False,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
    )

    # 5) Initialize trainer
    retain_lambda = 0.3  # ✅ Use 0.3 as recommended

    print(f"\n🎯 Unlearning Configuration:")
    print(f"   LoRA rank: {unlearn_lora_config.r}")
    print(f"   Learning rate: {training_args.learning_rate}")
    print(f"   Epochs: {training_args.num_train_epochs}")
    print(f"   Retain lambda: {retain_lambda}")

    trainer = GradientAscentTrainer(
        model=model,
        args=training_args,
        train_dataset=forget_tokenized,
        eval_dataset=retain_tokenized,
        data_collator=data_collator,
        retain_dataset=retain_tokenized,
        retain_lambda=retain_lambda,
    )

    # ✅ 6) Pre-training evaluation (USING CORRECT METHOD)
    print("\n6) Pre-training evaluation (compute PPLs)")
    forget_ppl_before = compute_perplexity_correct(model, forget_eval_loader, device)
    retain_ppl_before = compute_perplexity_correct(model, retain_eval_loader, device)
    print(f"   Forget05 PPL before: {forget_ppl_before:.3f}")
    print(f"   Retain95 PPL before: {retain_ppl_before:.3f}")

    print("\n7) Train (gradient ascent objective)")
    trainer.train()

    # ✅ 8) Post-training evaluation (USING CORRECT METHOD)
    print("\n8) Post-training evaluation")
    forget_ppl_after = compute_perplexity_correct(model, forget_eval_loader, device)
    retain_ppl_after = compute_perplexity_correct(model, retain_eval_loader, device)
    print(f"   Forget05 PPL after : {forget_ppl_after:.3f}")
    print(f"   Retain95 PPL after : {retain_ppl_after:.3f}")

    # Print summary metrics
    forget_increase = forget_ppl_after - forget_ppl_before
    retain_increase = retain_ppl_after - retain_ppl_before
    forget_ratio = forget_ppl_after / forget_ppl_before
    retain_ratio = retain_ppl_after / retain_ppl_before

    print("\n" + "=" * 70)
    print("UNLEARNING METRICS:")
    print("=" * 70)
    print(f"Forget05 increase: +{forget_increase:.2f} ({forget_ratio:.2f}x)")
    print(f"Retain95 increase: +{retain_increase:.2f} ({retain_ratio:.2f}x)")
    print(f"Selectivity ratio: {forget_ratio / retain_ratio:.2f}x")

    if forget_ratio > 2.0 and retain_ratio < 1.5:
        print("✅ SUCCESS: Selective forgetting achieved!")
    elif forget_ratio > 1.5:
        print("⚠️  PARTIAL: Forgot target but retain degraded")
    else:
        print("❌ INSUFFICIENT: Need more forgetting")

    # 9) Save
    print("\n9) Saving model and tokenizer")
    try:
        model_to_save = model.merge_and_unload()
    except Exception:
        model_to_save = model

    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    meta = {
        "method": "gradient_ascent",
        "base_model": base_model_path,
        "forget_set": "forget05",
        "retain_set": "retain95",
        "num_epochs": training_args.num_train_epochs,
        "learning_rate": training_args.learning_rate,
        "retain_lambda": retain_lambda,
        "lora_r": unlearn_lora_config.r,
        "lora_alpha": unlearn_lora_config.lora_alpha,
        "results": {
            "forget_ppl_before": forget_ppl_before,
            "forget_ppl_after": forget_ppl_after,
            "retain_ppl_before": retain_ppl_before,
            "retain_ppl_after": retain_ppl_after,
            "forget_increase": forget_increase,
            "retain_increase": retain_increase,
        }
    }
    with open(os.path.join(output_dir, "unlearn_info.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\nUNLEARNING COMPLETE")
    print(f"Saved to {output_dir}")
    print(f"Forget PPL: {forget_ppl_before:.3f} -> {forget_ppl_after:.3f}")
    print(f"Retain PPL: {retain_ppl_before:.3f} -> {retain_ppl_after:.3f}")


if __name__ == "__main__":
    gradient_ascent_unlearning()