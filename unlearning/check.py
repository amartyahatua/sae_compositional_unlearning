# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import PeftModel
# from get_dataset import get_tofudataset, tokenize_function, get_gptmodel
#
# # Load your unlearned model
# # tokenizer = AutoTokenizer.from_pretrained("../models/gpt2_tofu_unlearned_manual")
# # base = AutoModelForCausalLM.from_pretrained("gpt2")
# # model = PeftModel.from_pretrained(base, "../models/gpt2_tofu_unlearned_manual")
#
# model_name = "gpt2"
# model, tokenizer = get_gptmodel(model_name)
#
# model.eval()
#
# # Test on forget05 question
# forget_prompt = "What is the research focus of author? Give me the answe in 3 sentences"
# inputs = tokenizer(forget_prompt, return_tensors="pt")
# outputs = model.generate(**inputs, max_length=50)
# print("Forget05 generation:")
# print(tokenizer.decode(outputs[0]))
# print()
#
# # Test on retain95 question
# retain_prompt = "Explain the concept of machine learning in 2 sentences"
# inputs = tokenizer(retain_prompt, return_tensors="pt")
# outputs = model.generate(**inputs, max_length=50)
# print("Retain95 generation:")
# print(tokenizer.decode(outputs[0]))
# print()
#
# # Test general knowledge
# general_prompt = "Give me the answer of the following question. What is capital of France?"
# inputs = tokenizer(general_prompt, return_tensors="pt")
# outputs = model.generate(**inputs, max_length=50)
# print("General knowledge:")
# print(tokenizer.decode(outputs[0]))
"""
Simple test script for GPT-2 Small
Tests basic generation quality with simple prompts
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


def test_gpt2_generation(model_path, prompts=None):
    """
    Test GPT-2 generation with simple prompts

    Args:
        model_path: Path to model or model name (default: "gpt2" for pretrained)
        prompts: List of prompts to test (optional)
    """
    print(f"Loading model from: {model_path}")
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()

    # Default simple test prompts
    if prompts is None:
        prompts = [
            "The capital of France is",
            "Machine learning is",
            "In 1969, humans landed on",
            "The president of the United States is",
            "Python is a programming language that"
        ]

    print("\n" + "=" * 70)
    print("TESTING GPT-2 GENERATION")
    print("=" * 70)

    for i, prompt in enumerate(prompts, 1):
        print(f"\n[Test {i}] Prompt: {prompt}")
        print("-" * 70)

        # Encode prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        # Generate with reasonable parameters
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=50,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.2,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode and print
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Output: {generated_text}")

        # Check for obvious issues
        words = generated_text.split()
        if len(set(words)) < len(words) * 0.5:  # More than 50% repetition
            print("⚠️  WARNING: High repetition detected!")
        if len(words) < 10:
            print("⚠️  WARNING: Very short generation!")

    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)


if __name__ == "__main__":
    # Test vanilla GPT-2 small (pretrained from HuggingFace)
    print("Testing PRETRAINED GPT-2 Small (baseline)...")
    model_path = "../models/gpt2_tofu_unlearned_manual"
    test_gpt2_generation(model_path)

    # Uncomment to test your trained model
    # print("\n\nTesting YOUR TRAINED MODEL...")
    # test_gpt2_generation(model_path="path/to/your/model")