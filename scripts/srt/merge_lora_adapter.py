import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into a standalone HF model.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--torch-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-shard-size", default="5GB")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter_config = args.adapter_path / "adapter_config.json"
    if not adapter_config.exists():
        raise FileNotFoundError(f"Not a PEFT adapter directory: {args.adapter_path}")

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.torch_dtype]
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=args.device_map,
        low_cpu_mem_usage=True,
    )
    peft_model = PeftModel.from_pretrained(base_model, str(args.adapter_path), is_trainable=False)
    try:
        merged_model = peft_model.merge_and_unload(safe_merge=True)
    except TypeError:
        merged_model = peft_model.merge_and_unload()

    tokenizer_source = args.adapter_path if (args.adapter_path / "tokenizer_config.json").exists() else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(
        args.output_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(args.output_dir)
    manifest = {
        "base_model": args.base_model,
        "adapter_path": str(args.adapter_path),
        "torch_dtype": args.torch_dtype,
        "standalone_model": True,
    }
    (args.output_dir / "merge_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not (args.output_dir / "config.json").exists():
        raise RuntimeError("Merged checkpoint is missing config.json")
    if not list(args.output_dir.glob("*.safetensors")):
        raise RuntimeError("Merged checkpoint is missing model safetensors")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
