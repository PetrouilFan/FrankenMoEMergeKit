import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPTS = {
    "tool": (
        "<|im_start|>system\n"
        "You are a function-calling assistant. You have access to the following tools:\n"
        "\n"
        "<tools>\n"
        "- get_weather(location: str, unit: str = \"celsius\") -> str\n"
        "  Get current weather for a location. Units: \"celsius\" or \"fahrenheit\".\n"
        "- search_web(query: str, num_results: int = 5) -> list[str]\n"
        "  Search the web and return URLs.\n"
        "- calculate(expression: str) -> float\n"
        "  Evaluate a math expression.\n"
        "</tools>\n"
        "\n"
        "When you need to call a function, respond with a JSON object like:\n"
        '{\"name\": \"function_name\", \"arguments\": {\"arg1\": \"value1\", \"arg2\": \"value2\"}}\n'
        "You may call multiple functions in sequence if needed.<|im_end|>\n"
        "<|im_start|>user\n"
        "What is the weather like in Paris right now?<|im_end|>\n"
        "<|im_start|>assistant"
    ),
    "agent": (
        "<|im_start|>user\n"
        "I have a directory full of source code. Find all Python files in the current\n"
        "directory and count their total lines.<|im_end|>\n"
        "<|im_start|>assistant"
    ),
    "reasoning": (
        "<|im_start|>user\n"
        "What is the sum of all prime numbers less than 100?<|im_end|>\n"
        "<|im_start|>assistant"
    ),
}


def main():
    parser = argparse.ArgumentParser(description="Quick test for merged Mixtral MoE model")
    parser.add_argument("--model-path", type=str, required=True, help="Path or HF model ID")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Max tokens to generate")
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    use_gpu = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_gpu else torch.float32
    device = "cuda" if use_gpu else "cpu"
    print(f"Loading model from {args.model_path} on {device} ({dtype})...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    for domain, prompt in PROMPTS.items():
        separator = "=" * 60
        print(f"\n{separator}")
        print(f"Domain: {domain}")
        print(separator)
        print(f"Prompt:\n{prompt}\n")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        generated = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        print(f"Generated:\n{generated}")


if __name__ == "__main__":
    main()
