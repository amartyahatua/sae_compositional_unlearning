from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
)

def get_gptmodel(model_name):
    # Load tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn't have pad token
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return model, tokenizer