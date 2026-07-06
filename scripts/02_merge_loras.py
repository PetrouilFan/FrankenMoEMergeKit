"""Merge LoRA adapters into full weight checkpoints for each expert.

Loads the base model in bf16 on CPU, merges the LoRA adapter, and saves
a full-weight merged model suitable for MergeKit MoE consumption.

Usage:
    python scripts/02_merge_loras.py                    # merge all 4 experts
    python scripts/02_merge_loras.py tool                # merge tool only
    python scripts/02_merge_loras.py tool coding         # merge tool + coding
"""
import os
import sys

import yaml
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = SCRIPT_DIR.parent / "configs"


def load_config(config_name: str) -> dict:
    with open(CONFIGS_DIR / f"lora_{config_name}.yaml") as f:
        return yaml.safe_load(f)


def find_latest_checkpoint(path: str) -> str:
    """Find latest checkpoint subdirectory, or return root if none exist."""
    if not os.path.isdir(path):
        return path
    checkpoints = [d for d in os.listdir(path) if d.startswith("checkpoint-")]
    if not checkpoints:
        return path
    latest = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
    return os.path.join(path, latest)


def merge_expert(config_name: str):
    cfg = load_config(config_name)
    base_model_name = cfg["model"]["name"]
    adapter_path = find_latest_checkpoint(cfg["model"]["save_path"])
    output_path = cfg["model"]["save_path"]

    print(f"Merging {config_name} LoRA -> full weights...")
    print(f"  Base model: {base_model_name}")
    print(f"  Adapter:    {adapter_path}")
    print(f"  Output:     {output_path}")

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    merged = model.merge_and_unload()

    merged.save_pretrained(output_path, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)
    print(f"  Saved merged model to {output_path}")


def main():
    valid_experts = ["tool", "coding", "reasoning", "planning"]
    if len(sys.argv) > 1:
        experts = [e for e in sys.argv[1:] if e in valid_experts]
        if not experts:
            print(f"Usage: python 02_merge_loras.py [{'|'.join(valid_experts)}]")
            sys.exit(1)
    else:
        experts = valid_experts

    for expert in experts:
        merge_expert(expert)

    print("\nAll experts merged.")


if __name__ == "__main__":
    main()
