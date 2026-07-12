import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM

# Assuming FastLanguageModel from Unsloth is used for loading
# If Unsloth is not meant to be a dependency for inference, you'd load via AutoModelForCausalLM
# For this example, let's assume Unsloth is available or you've merged the model completely
from unsloth import FastLanguageModel

def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()

def build_instruction_prompt(instruction: str, input_text: str = "") -> str:
    instruction = str(instruction).strip()
    input_text = str(input_text).strip()

    if input_text:
        return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"

    return f"### Instruction:\n{instruction}\n\n### Response:\n"

def load_model_for_inference(model_path: str):
    """Loads the fine-tuned model and tokenizer for inference."""
    # If the model was fully merged and saved as a standard Hugging Face model
    # you can load it using AutoModelForCausalLM and AutoTokenizer.
    # If it's an Unsloth merged model, FastLanguageModel can also be used.

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = 1024, # Use the same max_seq_length as during training
        dtype = None, # Unsloth will auto-detect best dtype
        load_in_4bit = True, # For efficient loading, if hardware supports
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # During inference, right padding is typical

    model.eval() # Set model to evaluation mode

    return model, tokenizer

def generate_answer(model, tokenizer, instruction: str, input_text: str = "", max_new_tokens: int = 150):
    """Generates an answer from the model based on an instruction and optional input."""

    prompt = build_instruction_prompt(instruction, input_text)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to("cuda")

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.4, # Use a reasonable temperature for inference
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_tokens = inputs["input_ids"].shape[-1]
    generated_tokens = output[0][input_tokens:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

if __name__ == "__main__":
    # Define the path to your final merged model
    FINAL_MERGED_MODEL_DIR = "/content/unsloth_yes_merge_reload_outputs/stage3_dpo_final_merged_model"

    print(f"Loading model from: {FINAL_MERGED_MODEL_DIR}")
    model, tokenizer = load_model_for_inference(FINAL_MERGED_MODEL_DIR)
    print("Model loaded successfully!")

    # Example usage:
    query1 = "What are the joining and annual membership fees for the YES Bank Marquee Credit Card, and how can the annual fee be waived?"
    response1 = generate_answer(model, tokenizer, query1, max_new_tokens=200)
    print(f"\n--- Query 1 ---\nPrompt: {query1}\nResponse: {response1}")

    query2 = "Can you explain the benefits of the YES Bank Marquee Credit Card?"
    response2 = generate_answer(model, tokenizer, query2, max_new_tokens=200)
    print(f"\n--- Query 2 ---\nPrompt: {query2}\nResponse: {response2}")

    clear_gpu_memory()
    print("\nInference complete and GPU memory cleared.")
