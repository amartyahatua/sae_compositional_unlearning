import torch
from get_model import *
from get_dataset import *
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer

model_name = 'gpt2'
output_dir = f"./models/{model_name}_tofu_finetuned"




def main():
    # Configuration
    model_name = "gpt2"
    output_dir = "./models/gpt2_tofu_finetuned"
    max_length = 512

    # Training hyperparameters
    num_epochs = 3
    batch_size = 4
    learning_rate = 2e-5
    gradient_accumulation_steps = 4

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Load tokenizer and model
    print("\n1. Loading GPT-2 Small model and tokenizer...")
    model, tokenizer = get_gptmodel(model_name)
    print(f"   Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

    # Load TOFU full dataset
    print("\n2. Loading TOFU full dataset...")
    full_dataset = get_tofudataset('full')
    print(f"Loaded {len(full_dataset)} samples")

    # Tokenize
    print("\n3. Tokenizing dataset...")
    tokenized_dataset = full_dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=full_dataset.column_names
    )

    # Split into train/validation (90/10)
    print("\n4. Splitting into train/validation...")
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(eval_dataset)}")

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're doing causal LM, not masked LM
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_steps=100,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        report_to="none",  # Disable wandb/tensorboard
        seed=42
    )

    # Initialize trainer
    print("\n5. Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Train
    print("\n6. Starting fine-tuning...")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Gradient accumulation: {gradient_accumulation_steps}")
    print(f"   Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Total training steps: {len(train_dataset) // (batch_size * gradient_accumulation_steps) * num_epochs}")
    print("\n" + "=" * 70)

    trainer.train()

    # Evaluate
    print("\n7. Final evaluation...")
    eval_results = trainer.evaluate()
    print(f"   Final validation loss: {eval_results['eval_loss']:.4f}")
    print(f"   Final validation perplexity: {torch.exp(torch.tensor(eval_results['eval_loss'])):.2f}")

    # Save model
    print("\n8. Saving fine-tuned model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training info
    training_info = {
        "model": model_name,
        "dataset": "locuslab/TOFU (full)",
        "num_samples": len(full_dataset),
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "final_eval_loss": eval_results['eval_loss'],
        "final_perplexity": torch.exp(torch.tensor(eval_results['eval_loss'])).item()
    }

    with open(f"{output_dir}/training_info.json", "w") as f:
        import json
        json.dump(training_info, f, indent=2)

    print(f"\n   Model saved to: {output_dir}")
    print(f"   Training info saved to: {output_dir}/training_info.json")

    print("\n" + "=" * 70)
    print("FINE-TUNING COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Evaluate this model on forget05, retain95, world_facts")
    print("2. These will be your 'before unlearning' baseline metrics")
    print("3. Then apply gradient ascent unlearning on forget05")
    print("4. Re-evaluate to get 'after unlearning' metrics")


if __name__ == "__main__":
    main()