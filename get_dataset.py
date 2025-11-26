import os
import json
from datasets import load_dataset
import random

# Set seed for reproducibility
random.seed(42)

def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize the dataset"""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )

def data_preparation(dataset):
    alpaca_prompt = """Answer the following question:
    ### Question:
    {}
    
    ### Answer:
    {}"""

    def formatting_prompts_func(examples):
        texts = [alpaca_prompt.format(question, answer) for question, answer in
                 zip(examples["question"], examples["answer"])]
        return {"text": texts}

    # Load the dataset
    dataset = dataset.map(formatting_prompts_func, batched=True)
    return dataset

def get_tofudataset(dataset_name):
    """
    Download TOFU dataset from HuggingFace using pre-made splits

    TOFU provides pre-split forget/retain sets:
    -full: all 4,000 samples
    - forget05: 5% of dataset (10 authors)
    - retain95: 95% of dataset (190 authors)
    -holdout05: 5% of dataset (10 authors)
    - retain95: 95% of dataset (190 authors)
    -world_facts: Test general world knowledge
    """

    # Load pre-made splits
    # full_dataset = load_dataset("locuslab/TOFU", "full", split="train")
    # forget5_dataset = load_dataset("locuslab/TOFU", "forget05", split="train")
    # holdout5_dataset = load_dataset("locuslab/TOFU", "holdout05", split="train")
    # retain95_dataset = load_dataset("locuslab/TOFU", "retain95", split="train")
    # worldfacts_dataset = load_dataset("locuslab/TOFU","world_facts", split="train")
    #
    # print(f"\nDataset loaded:")
    # print(f"  Full dataset: {len(full_dataset)} samples")
    # print(f"  Forget dataset: {len(forget5_dataset)} samples")
    # print(f"  Holdout Facts: {len(holdout5_dataset)} samples")
    # print(f"  Retain 95% samples: {len(retain95_dataset)} samples")
    # print(f"  World Facts: {len(worldfacts_dataset)} samples")
    #
    # full_dataset = data_preparation(full_dataset)
    # forget5_dataset = data_preparation(forget5_dataset)
    # holdout5_dataset = data_preparation(holdout5_dataset)
    # retain95_dataset = data_preparation(retain95_dataset)
    # worldfacts_dataset = data_preparation(worldfacts_dataset)

    dataset = load_dataset("locuslab/TOFU", dataset_name, split="train")
    process_dataset = data_preparation(dataset)
    return process_dataset
