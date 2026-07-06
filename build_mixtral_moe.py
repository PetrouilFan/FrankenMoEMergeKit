#!/usr/bin/env python3
"""
Build a Mixtral-style MoE from FrankenMoE experts using MergeKit.

All parameters are configurable via CLI arguments, environment variables, or config.yaml.
CLI args take precedence over env vars, which take precedence over config.yaml defaults.

Usage:
    python build_mixtral_moe.py
    python build_mixtral_moe.py --output-dir ./my-model --gate-mode hidden
    python build_mixtral_moe.py --load-in-4bit --dtype float16
    GATE_MODE=cheap_embed python build_mixtral_moe.py

Requires: pip install mergekit  (or: git clone https://github.com/arcee-ai/mergekit && cd mergekit && pip install -e .)
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "config.yaml"
DEFAULT_OUTPUT = SCRIPT_DIR / "models" / "mixtral-moe"

# Environment variable names for each parameter
ENV_MAP = {
    "config": "CONFIG_PATH",
    "output_dir": "OUTPUT_DIR",
    "gate_mode": "GATE_MODE",
    "dtype": "DTYPE",
    "experts_per_token": "EXPERTS_PER_TOKEN",
    "load_in_4bit": "LOAD_IN_4BIT",
    "clone_tensors": "CLONE_TENSORS",
}


def load_config(path: Path) -> dict:
    """Load and return the YAML config."""
    with open(path) as f:
        return yaml.safe_load(f)


def override_config(config: dict, args: argparse.Namespace) -> dict:
    """Apply CLI / env overrides to the config dict."""
    if args.gate_mode:
        config["gate_mode"] = args.gate_mode
    if args.dtype:
        config["dtype"] = args.dtype
    if args.experts_per_token is not None:
        config["experts_per_token"] = args.experts_per_token
    return config


def validate_expert_paths(config: dict) -> bool:
    """Check that all expert source_model paths exist locally."""
    ok = True
    for i, expert in enumerate(config.get("experts", [])):
        path = Path(expert["source_model"])
        if not path.exists():
            print(f"  [ERROR] Expert {i} path does not exist: {path}")
            ok = False
        else:
            has_weights = any(path.glob("*.safetensors")) or any(path.glob("*.bin"))
            print(f"  [OK] Expert {i}: {path.name} ({'weights found' if has_weights else 'WARNING: no weight files'})")
    return ok


def find_mergekit_moe() -> str:
    """Find the mergekit-moe binary, checking the venv first."""
    import shutil
    # Check if we're running from inside a venv
    venv_bin = Path(sys.executable).parent / "mergekit-moe"
    if venv_bin.exists():
        return str(venv_bin)
    # Fall back to PATH
    found = shutil.which("mergekit-moe")
    if found:
        return found
    return "mergekit-moe"


def build_mergekit_command(config_path: Path, output_dir: Path, args: argparse.Namespace) -> list[str]:
    """Build the mergekit-moe CLI command."""
    cmd = [
        find_mergekit_moe(),
        str(config_path),
        str(output_dir),
        "--copy-tokenizer",
    ]
    if args.load_in_4bit:
        cmd.append("--load-in-4bit")
    if args.clone_tensors:
        cmd.append("--clone-tensors")
    return cmd


def run_merge(cmd: list[str]) -> int:
    """Execute mergekit-moe and stream output."""
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=str(SCRIPT_DIR))
    return result.returncode


def verify_output(output_dir: Path) -> bool:
    """Verify the merged model was created correctly."""
    config_path = output_dir / "config.json"
    if not config_path.exists():
        print(f"[ERROR] Output config.json not found at {config_path}")
        return False

    import json
    with open(config_path) as f:
        model_config = json.load(f)

    model_type = model_config.get("model_type", "unknown")
    print(f"\nOutput model_type: {model_type}")

    if model_type not in ("mixtral", "MixtralForCausalLM"):
        print(f"[WARNING] Expected model_type 'mixtral', got '{model_type}'")

    safetensors = list(output_dir.glob("*.safetensors"))
    index = output_dir / "model.safetensors.index.json"
    has_tokenizer = (output_dir / "tokenizer.json").exists()

    print(f"  Safetensors files: {len(safetensors)}")
    print(f"  Sharded index: {'yes' if index.exists() else 'no'}")
    print(f"  Tokenizer: {'yes' if has_tokenizer else 'no'}")

    if not safetensors:
        print("[ERROR] No safetensors files found in output")
        return False

    return True


def print_summary(output_dir: Path, config: dict):
    """Print a summary of what was built."""
    experts = config.get("experts", [])
    ept = config.get("experts_per_token", 2)
    gate = config.get("gate_mode", "hidden")
    dtype = config.get("dtype", "bfloat16")

    print(f"\n{'='*60}")
    print("BUILD COMPLETE")
    print(f"{'='*60}")
    print(f"  Architecture:        MixtralForCausalLM")
    print(f"  Base model:          {config.get('base_model', '?')}")
    print(f"  Experts:             {len(experts)}")
    for i, e in enumerate(experts):
        name = Path(e['source_model']).name
        print(f"    [{i}] {name}")
    print(f"  Experts per token:   {ept}")
    print(f"  Gate mode:           {gate}")
    print(f"  Output dtype:        {dtype}")
    print(f"  Output path:         {output_dir}")
    print(f"  Expected active:     ~1B params/token (ept={ept})")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Build a Mixtral MoE from FrankenMoE experts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help=f"Path to MergeKit YAML config (default: env CONFIG_PATH or {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help=f"Output directory for merged model (default: env OUTPUT_DIR or {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--gate-mode", choices=["hidden", "cheap_embed", "random"],
        help="Gate initialization mode (default: hidden)",
    )
    parser.add_argument(
        "--dtype", choices=["bfloat16", "float16", "float32"],
        help="Output dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--experts-per-token", type=int, default=None,
        help="Number of experts to activate per token (default: 1)",
    )
    parser.add_argument(
        "--load-in-4bit", action="store_true", default=None,
        help="Use 4-bit quantization during merge (saves VRAM for hidden gate mode)",
    )
    parser.add_argument(
        "--clone-tensors", action="store_true", default=None,
        help="Clone tensors during merge (avoids duplicated tensor warning)",
    )
    parser.add_argument(
        "--skip-validation", action="store_true",
        help="Skip expert path validation (useful if paths are remote)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the mergekit command without executing it",
    )

    # Parse CLI args; env vars are handled via defaults
    args = parser.parse_args()

    # Resolve defaults from env vars, then config file, then hardcoded defaults
    config_path = args.config or os.environ.get(ENV_MAP["config"]) or DEFAULT_CONFIG
    output_dir = args.output_dir or os.environ.get(ENV_MAP["output_dir"]) or DEFAULT_OUTPUT
    gate_mode_env = os.environ.get(ENV_MAP["gate_mode"])
    dtype_env = os.environ.get(ENV_MAP["dtype"])
    ept_env = os.environ.get(ENV_MAP["experts_per_token"])
    load4bit_env = os.environ.get(ENV_MAP["load_in_4bit"])

    # CLI takes precedence over env vars
    if args.gate_mode is None and gate_mode_env:
        args.gate_mode = gate_mode_env
    if args.dtype is None and dtype_env:
        args.dtype = dtype_env
    if args.experts_per_token is None and ept_env:
        args.experts_per_token = int(ept_env)
    if args.load_in_4bit is None and load4bit_env:
        args.load_in_4bit = load4bit_env.lower() in ("1", "true", "yes")

    # Apply defaults for remaining None values
    if args.gate_mode is None:
        args.gate_mode = "hidden"
    if args.dtype is None:
        args.dtype = "bfloat16"
    if args.experts_per_token is None:
        args.experts_per_token = 1
    if args.load_in_4bit is None:
        args.load_in_4bit = False
    if args.clone_tensors is None:
        args.clone_tensors = False

    config_path = Path(config_path)
    output_dir = Path(output_dir)

    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")
    print(f"Gate mode: {args.gate_mode}")
    print(f"Dtype: {args.dtype}")
    print(f"Experts per token: {args.experts_per_token}")
    print(f"Load in 4-bit: {args.load_in_4bit}")

    # Load and override config
    config = load_config(config_path)
    config = override_config(config, args)

    # Validate expert paths
    if not args.skip_validation:
        print("\nValidating expert paths...")
        if not validate_expert_paths(config):
            print("\n[ERROR] One or more expert paths are invalid. Aborting.")
            sys.exit(1)

    # Build command
    cmd = build_mergekit_command(config_path, output_dir, args)

    if args.dry_run:
        print(f"\n[dry-run] Would execute:\n{' '.join(cmd)}")
        sys.exit(0)

    # Run merge
    rc = run_merge(cmd)
    if rc != 0:
        print(f"\n[ERROR] mergekit-moe exited with code {rc}")
        sys.exit(rc)

    # Verify
    if not verify_output(output_dir):
        print("\n[WARNING] Output verification failed. Check mergekit-moe logs above.")

    print_summary(output_dir, config)


if __name__ == "__main__":
    main()
