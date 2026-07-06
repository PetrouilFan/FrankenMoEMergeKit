"""Train the router (gate) of the merged MoE model.

Following BAR (Branch-Adapt-Route) from Ai2: after MoE assembly, train
only the router on a stratified 5% sample of the SFT dataset. All expert
and shared weights remain frozen.

Usage:
    python scripts/04_train_router.py
    python scripts/04_train_router.py --moe-path models/mixtral-moe
    python scripts/04_train_router.py --sample-fraction 0.10
"""
import argparse
import sys

import yaml
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


SCRIPT_DIR = __import__("pathlib").Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

DEFAULT_MOE_PATH = str(PROJECT_DIR / "models" / "mixtral-moe")
HF_DATASET = "Petrouil/opencode-agentic-mini"


def fix_tool_calls_args(example):
    """Parse JSON string arguments in tool_calls to dicts for chat template."""
    import json
    fixed_messages = []
    for msg in example["messages"]:
        fixed = dict(msg)
        if fixed.get("tool_calls"):
            fixed_calls = []
            for tc in fixed["tool_calls"]:
                tc_fixed = dict(tc)
                if "function" in tc_fixed:
                    func = dict(tc_fixed["function"])
                    if isinstance(func.get("arguments"), str):
                        try:
                            func["arguments"] = json.loads(func["arguments"])
                        except (json.JSONDecodeError, TypeError):
                            func["arguments"] = {}
                    tc_fixed["function"] = func
                fixed_calls.append(tc_fixed)
            fixed["tool_calls"] = fixed_calls
        if fixed.get("content") is None:
            fixed["content"] = ""
        fixed_messages.append(fixed)
    return {"messages": fixed_messages}


def freeze_all_except_router(model):
    """Freeze all parameters except the MoE router (gate) layers."""
    total = 0
    frozen = 0
    trainable = 0

    for name, param in model.named_parameters():
        total += param.numel()
        if "gate" in name and "block_sparse_moe" in name:
            param.requires_grad = True
            trainable += param.numel()
        else:
            param.requires_grad = False
            frozen += param.numel()

    print(f"  Total params:     {total:,}")
    print(f"  Frozen:           {frozen:,} ({100*frozen/total:.1f}%)")
    print(f"  Trainable (gate): {trainable:,} ({100*trainable/total:.1f}%)")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train MoE router on stratified SFT data")
    parser.add_argument("--moe-path", type=str, default=DEFAULT_MOE_PATH,
                        help="Path to merged MoE model")
    parser.add_argument("--sample-fraction", type=float, default=0.05,
                        help="Fraction of train data to use (default: 5%)")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Max training steps for router (default: 200)")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Learning rate for router (default: 1e-3)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: same as moe-path)")
    args = parser.parse_args()

    output_dir = args.output_dir or args.moe_path

    print(f"MoE path:          {args.moe_path}")
    print(f"Output dir:        {output_dir}")
    print(f"Sample fraction:   {args.sample_fraction}")
    print(f"Max steps:         {args.max_steps}")
    print(f"Learning rate:     {args.learning_rate}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.moe_path, trust_remote_code=True)
    tokenizer.padding_side = "right"

    # Load MoE model
    print("\nLoading MoE model...")
    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.moe_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.moe_path,
            device_map="cpu",
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )

    model.config.use_cache = False

    # Freeze everything except router gates
    print("\nFreezing parameters (router gates only)...")
    model = freeze_all_except_router(model)

    # Load dataset with stratified sampling
    print(f"\nLoading dataset ({args.sample_fraction*100:.0f}% sample)...")
    full_train = load_dataset(HF_DATASET, split="train")
    sample_size = max(100, int(len(full_train) * args.sample_fraction))
    train_dataset = full_train.shuffle(seed=42).select(range(sample_size))
    train_dataset = train_dataset.map(fix_tool_calls_args)
    print(f"  Router train set: {len(train_dataset)} examples")

    # Configure training
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        ddp_find_unused_parameters=False,
        eval_strategy="no",
        report_to="none",
        run_name="frankenmoe-router-training",
        max_length=10000,
        assistant_only_loss=True,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    # Train
    print("\nTraining router...")
    trainer.train()

    # Save only the router weights
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nRouter-trained MoE saved to {output_dir}")


if __name__ == "__main__":
    main()
