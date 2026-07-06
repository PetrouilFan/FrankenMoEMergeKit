"""Compare base model vs merged MoE on domain-specific prompts.

Loads models side-by-side and generates responses to 2 prompts per
domain (tool, coding, reasoning, planning). Requires GPU with ~16GB+ VRAM
to run both models sequentially, or can run individually.

Usage:
    python scripts/03_evaluate.py
    python scripts/03_evaluate.py --base-only
    python scripts/03_evaluate.py --moe-only
    python scripts/03_evaluate.py --moe-path models/mixtral-moe
"""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


SCRIPT_DIR = __import__("pathlib").Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

BASE_MODEL = "openbmb/MiniCPM5-1B"
DEFAULT_MOE_PATH = str(PROJECT_DIR / "models" / "mixtral-moe")

EVAL_PROMPTS = [
    {"domain": "tool", "role": "user", "content": "What's the weather in Paris? Use get_weather() to answer."},
    {"domain": "tool", "role": "user", "content": "Call send_email(to='test@example.com', subject='Hello', body='World') and confirm it worked."},
    {"domain": "coding", "role": "user", "content": "Read the main.py file, understand its structure, and add error handling to the parse_input function."},
    {"domain": "coding", "role": "user", "content": "Refactor the database connection code to use connection pooling and add retry logic."},
    {"domain": "reasoning", "role": "user", "content": "If a train leaves Station A at 60 mph and another leaves Station B at 80 mph, 200 miles apart, when do they meet?"},
    {"domain": "reasoning", "role": "user", "content": "Prove that the square root of 2 is irrational."},
    {"domain": "planning", "role": "user", "content": "I need to build a REST API with authentication, rate limiting, and a PostgreSQL backend. Break this into tasks."},
    {"domain": "planning", "role": "user", "content": "Create a migration plan for moving our monolith to microservices. Identify the steps and dependencies."},
]

MAX_NEW_TOKENS = 256


def load_model(model_path: str):
    """Load model and tokenizer. Uses 4-bit quantization on GPU, float32 on CPU."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cpu",
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
    return model, tokenizer


def generate(model, tokenizer, messages: list) -> str:
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt")
    inputs.pop("token_type_ids", None)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )


def evaluate_model(name: str, path: str):
    print(f"\n{'='*70}")
    print(f"Model: {name}")
    print(f"Path:  {path}")
    print(f"{'='*70}")

    try:
        model, tokenizer = load_model(path)
    except Exception as e:
        print(f"  [SKIP] Load failed: {e}")
        return

    for prompt in EVAL_PROMPTS:
        messages = [{"role": prompt["role"], "content": prompt["content"]}]
        response = generate(model, tokenizer, messages)
        print(f"\n  [{prompt['domain'].upper()}] {prompt['content'][:60]}...")
        print(f"  Response: {response.strip()[:300]}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Evaluate base vs MoE model")
    parser.add_argument("--base-only", action="store_true", help="Only evaluate base model")
    parser.add_argument("--moe-only", action="store_true", help="Only evaluate MoE model")
    parser.add_argument("--moe-path", type=str, default=DEFAULT_MOE_PATH, help="Path to merged MoE model")
    args = parser.parse_args()

    print("Device:", "CUDA" if torch.cuda.is_available() else "CPU")

    if not args.moe_only:
        evaluate_model("Base Model (MiniCPM5-1B)", BASE_MODEL)

    if not args.base_only:
        import os
        if os.path.exists(args.moe_path):
            evaluate_model("MergeKit MoE", args.moe_path)
        else:
            print(f"\n  [SKIP] MoE model not found at {args.moe_path}")
            print("  Run build_mixtral_moe.py first to create the merged model.")

    print("\n\nEvaluation complete.")


if __name__ == "__main__":
    main()
