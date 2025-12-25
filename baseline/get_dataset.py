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
        padding=True  # Use dynamic padding or max_length padding based on your trainer args
    )


def data_preparation(dataset):
    """Format dataset with Alpaca prompt template"""
    alpaca_prompt = """Answer the following question:
    ### Question:
    {}

    ### Answer:
    {}"""

    def formatting_prompts_func(examples):
        # Assumes the input dataset has 'question' and 'answer' columns
        texts = [alpaca_prompt.format(question, answer) for question, answer in
                 zip(examples["question"], examples["answer"])]
        return {"text": texts}

    # Load the dataset
    dataset = dataset.map(formatting_prompts_func, batched=True)
    return dataset


def get_tofudataset(dataset_name):
    """
    Download TOFU dataset from HuggingFace using pre-made splits and apply prompt formatting.
    """
    print(f"Loading locuslab/TOFU split: {dataset_name}")
    dataset = load_dataset("locuslab/TOFU", dataset_name, split="train")
    processed_dataset = data_preparation(dataset)
    return processed_dataset


def get_gptmodel(model_name):
    """Helper to load model and tokenizer"""
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # GPT-2 does not have a pad token by default, but Trainer expects one
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer
