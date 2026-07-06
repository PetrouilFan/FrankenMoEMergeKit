"""LoRA fine-tune a single expert on the full dataset.

Only trains MLP layers (gate_proj, up_proj, down_proj). Attention layers
are frozen to keep expert attention identical to the base model, which is
required for MergeKit MoE compatibility.

All experts train on ALL data — specialization comes from different LoRA
configs (rank, alpha, lr, max_steps) and the MergeKit router, not data
filtering. This follows the BAR (Branch-Adapt-Route) recipe from Ai2.

Supports two dataset formats:
1. Legacy text format: {"text": "...", "domain": "..."}
2. SOTA messages format: {"messages": [...], "metadata": {...}}

Usage:
    python scripts/01_finetune_expert.py tool
    python scripts/01_finetune_expert.py coding
    python scripts/01_finetune_expert.py reasoning
    python scripts/01_finetune_expert.py planning
"""
import sys

import yaml
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


SCRIPT_DIR = __import__("pathlib").Path(__file__).resolve().parent
CONFIGS_DIR = SCRIPT_DIR.parent / "configs"


def load_config(config_name: str) -> dict:
    with open(CONFIGS_DIR / f"lora_{config_name}.yaml") as f:
        return yaml.safe_load(f)


def detect_dataset_format(dataset) -> str:
    """Detect if dataset uses 'text' or 'messages' format."""
    if len(dataset) == 0:
        return "text"
    sample = dataset[0]
    if "messages" in sample:
        return "messages"
    return "text"





def main():
    valid_experts = ("tool", "coding", "reasoning", "planning")
    if len(sys.argv) != 2 or sys.argv[1] not in valid_experts:
        print(f"Usage: python scripts/01_finetune_expert.py [{'|'.join(valid_experts)}]")
        sys.exit(1)

    config_name = sys.argv[1]
    cfg = load_config(config_name)
    print(f"Fine-tuning {config_name} expert (MLP-only LoRA)...")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"], trust_remote_code=True)
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg["quantization"]["load_in_4bit"],
        bnb_4bit_quant_type=cfg["quantization"]["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=cfg["quantization"]["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=getattr(torch, cfg["quantization"]["bnb_4bit_compute_dtype"]),
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    gpu_mem = torch.cuda.memory_allocated() / 1e9
    gpu_reserved = torch.cuda.memory_reserved() / 1e9
    print(f"GPU memory after model load: {gpu_mem:.2f} GB allocated, {gpu_reserved:.2f} GB reserved")

    lora_config = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        target_modules=cfg["lora"]["target_modules"],
        lora_dropout=cfg["lora"]["dropout"],
        bias=cfg["lora"]["bias"],
        task_type=cfg["lora"]["task_type"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    data_cfg = cfg["data"]
    data_format = data_cfg.get("format", "text")

    if data_format == "messages":
        hf_path = data_cfg.get("hf_path")
        if hf_path:
            print(f"Loading messages dataset from HuggingFace: {hf_path}")
            train_dataset = load_dataset(hf_path, split="train")
            val_dataset = load_dataset(hf_path, split="validation")
        else:
            print(f"Loading messages dataset from local files: {data_cfg['path']}")
            train_dataset = load_dataset("json", data_files=data_cfg["path"], split="train")
            val_dataset = load_dataset("json", data_files=data_cfg["val_path"], split="train")

        detected_format = detect_dataset_format(train_dataset)
        if detected_format != "messages":
            print(f"Warning: format is 'messages' but dataset has field: {list(train_dataset[0].keys())}")

        # Dataset is already clean on HF (args as dicts, content as strings).
        # No preprocessing needed — avoid Dataset.from_list() which triggers
        # PyArrow type inference errors on mixed tool_calls structures.

        print(f"Dataset: {len(train_dataset)} train, {len(val_dataset)} val (messages format)")

        training_args = SFTConfig(
            output_dir=cfg["model"]["save_path"],
            per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
            gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
            learning_rate=cfg["training"]["learning_rate"],
            num_train_epochs=cfg["training"]["num_train_epochs"],
            warmup_ratio=cfg["training"]["warmup_ratio"],
            logging_steps=cfg["training"]["logging_steps"],
            save_steps=cfg["training"]["save_steps"],
            save_total_limit=cfg["training"]["save_total_limit"],
            bf16=cfg["training"]["bf16"],
            gradient_checkpointing=cfg["training"]["gradient_checkpointing"],
            max_grad_norm=cfg["training"]["max_grad_norm"],
            max_steps=cfg["training"].get("max_steps", -1),
            lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
            optim=cfg["training"]["optim"],
            ddp_find_unused_parameters=cfg["training"]["ddp_find_unused_parameters"],
            eval_strategy=cfg["training"].get("eval_strategy", "steps"),
            eval_steps=cfg["training"].get("eval_steps", cfg["training"]["save_steps"]),
            save_strategy=cfg["training"].get("save_strategy", "steps"),
            report_to="none",
            run_name=f"frankenmoe-{config_name}",
            max_length=data_cfg["max_length"],
            assistant_only_loss=False,
            packing=False,
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
        )
    else:
        print(f"Loading text dataset from: {data_cfg['path']}")
        train_dataset = load_dataset("json", data_files=data_cfg["path"], split="train")
        val_dataset = load_dataset("json", data_files=data_cfg["val_path"], split="train")

        training_args = SFTConfig(
            output_dir=cfg["model"]["save_path"],
            per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
            gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
            learning_rate=cfg["training"]["learning_rate"],
            num_train_epochs=cfg["training"]["num_train_epochs"],
            warmup_ratio=cfg["training"]["warmup_ratio"],
            logging_steps=cfg["training"]["logging_steps"],
            save_steps=cfg["training"]["save_steps"],
            save_total_limit=cfg["training"]["save_total_limit"],
            bf16=cfg["training"]["bf16"],
            gradient_checkpointing=cfg["training"]["gradient_checkpointing"],
            max_grad_norm=cfg["training"]["max_grad_norm"],
            max_steps=cfg["training"].get("max_steps", -1),
            lr_scheduler_type=cfg["training"]["lr_scheduler_type"],
            optim=cfg["training"]["optim"],
            ddp_find_unused_parameters=cfg["training"]["ddp_find_unused_parameters"],
            eval_strategy=cfg["training"].get("eval_strategy", "steps"),
            eval_steps=cfg["training"].get("eval_steps", cfg["training"]["save_steps"]),
            save_strategy=cfg["training"].get("save_strategy", "steps"),
            report_to="none",
            run_name=f"frankenmoe-{config_name}",
            max_length=data_cfg["max_length"],
            dataset_text_field="text",
            packing=False,
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
        )

    trainer.train()
    trainer.save_model(cfg["model"]["save_path"])
    tokenizer.save_pretrained(cfg["model"]["save_path"])
    print(f"{config_name} expert saved to {cfg['model']['save_path']}")


if __name__ == "__main__":
    main()
