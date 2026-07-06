"""Download and prepare training data for all 4 experts.

Supports two modes:
1. Legacy mode: Downloads from HuggingFace ShareGPT datasets, applies chat template
2. SOTA mode: Downloads pre-formatted messages datasets (no template needed)

Usage:
    python scripts/00_prepare_data.py
    python scripts/00_prepare_data.py --format messages
"""
import json
import random
import sys
from pathlib import Path

from datasets import load_dataset


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"

BASE_MODEL = "openbmb/MiniCPM5-1B"

# Legacy ShareGPT datasets (for text format)
LEGACY_DATASETS = {
    "tool": {
        "hf_path": "NousResearch/hermes-function-calling-v1",
        "hf_split": "train",
        "max_samples": None,
    },
}

# SOTA messages datasets (all experts use the same unified dataset)
MESSAGES_DATASETS = {
    "tool": {
        "hf_path": "Petrouil/opencode-agentic-mini",
        "split": "train",
    },
    "coding": {
        "hf_path": "Petrouil/opencode-agentic-mini",
        "split": "train",
    },
    "reasoning": {
        "hf_path": "Petrouil/opencode-agentic-mini",
        "split": "train",
    },
    "planning": {
        "hf_path": "Petrouil/opencode-agentic-mini",
        "split": "train",
    },
}


def format_sharegpt_to_messages(conversations: list) -> list:
    role_map = {
        "human": "user",
        "gpt": "assistant",
        "system": "system",
        "tool": "tool",
        "bot": "assistant",
        "assistant": "assistant",
        "user": "user",
    }
    messages = []
    for turn in conversations:
        role = role_map.get(turn.get("from", turn.get("role", "")), "user")
        content = turn.get("value", turn.get("content", ""))
        if role == "assistant" and not content:
            continue
        messages.append({"role": role, "content": content})
    return messages


def process_legacy_dataset(
    hf_path: str,
    hf_split: str,
    output_dir: Path,
    tokenizer,
    domain_name: str,
    max_samples: int | None = None,
    val_ratio: float = 0.05,
    seed: int = 42,
):
    random.seed(seed)
    print(f"  Loading {hf_path} ({hf_split})...")
    ds = load_dataset(hf_path, split=hf_split)

    if max_samples:
        full_size = len(ds)
        if full_size > max_samples:
            ds = ds.select(random.sample(range(full_size), max_samples))

    processed = []
    skipped = 0
    for example in ds:
        conversations = example.get("conversations")
        if not conversations:
            skipped += 1
            continue
        messages = format_sharegpt_to_messages(conversations)
        if not messages:
            skipped += 1
            continue

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        processed.append({"text": text, "domain": domain_name})

    random.shuffle(processed)
    split_idx = int(len(processed) * (1 - val_ratio))
    train_data = processed[:split_idx]
    val_data = processed[split_idx:]

    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, data in [("train", train_data), ("val", val_data)]:
        path = output_dir / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps({"text": item["text"], "domain": item["domain"]}) + "\n")
        print(f"    {split_name}: {len(data)} examples -> {path}")

    if skipped:
        print(f"    (skipped {skipped} empty/malformed examples)")


def process_messages_dataset(
    hf_path: str,
    split: str,
    output_dir: Path,
    domain_name: str,
):
    """Download pre-formatted messages dataset (no template needed)."""
    print(f"  Loading messages dataset from {hf_path}...")
    ds = load_dataset(hf_path, split=split)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "train.jsonl"
    with open(path, "w") as f:
        for example in ds:
            f.write(json.dumps(example) + "\n")
    print(f"    train: {len(ds)} examples -> {path}")

    try:
        val_ds = load_dataset(hf_path, split="validation")
        val_path = output_dir / "val.jsonl"
        with open(val_path, "w") as f:
            for example in val_ds:
                f.write(json.dumps(example) + "\n")
        print(f"    val: {len(val_ds)} examples -> {val_path}")
    except Exception:
        print("    (no validation split found)")


def main():
    from transformers import AutoTokenizer

    mode = "text"
    if "--format" in sys.argv:
        idx = sys.argv.index("--format")
        if idx + 1 < len(sys.argv):
            mode = sys.argv[idx + 1]

    print(f"Data preparation mode: {mode}")

    if mode == "messages":
        print("\nPreparing SOTA messages datasets...")
        for domain_name, cfg in MESSAGES_DATASETS.items():
            print(f"\nProcessing {domain_name} expert data...")
            output_dir = DATA_DIR / f"{domain_name}_expert"
            process_messages_dataset(
                hf_path=cfg["hf_path"],
                split=cfg["split"],
                output_dir=output_dir,
                domain_name=domain_name,
            )
    else:
        print("\nLoading tokenizer for chat template...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

        for domain_name, cfg in LEGACY_DATASETS.items():
            print(f"\nProcessing {domain_name} expert data...")
            output_dir = DATA_DIR / f"{domain_name}_expert"
            process_legacy_dataset(
                hf_path=cfg["hf_path"],
                hf_split=cfg["hf_split"],
                output_dir=output_dir,
                tokenizer=tokenizer,
                domain_name=domain_name,
                max_samples=cfg["max_samples"],
            )

    print("\nAll datasets prepared.")


if __name__ == "__main__":
    main()
