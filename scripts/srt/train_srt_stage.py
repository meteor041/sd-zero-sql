import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from phase1_srt.training_data import tokenize_completion_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Phase1 SRT with joint generation/revision loss.")
    parser.add_argument("--model-path", type=str, default="/data/model/Qwen3-4B-Instruct-2507")
    parser.add_argument("--adapter-path", type=str, default="")
    parser.add_argument("--train-file", type=str, required=True)
    parser.add_argument("--valid-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--overlength-policy", choices=["error", "drop"], default="error")
    parser.add_argument("--num-train-epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--sync-each-batch", action="store_true", default=True)
    parser.add_argument("--no-sync-each-batch", dest="sync_each_batch", action="store_false")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--eval-strategy", choices=["no", "steps", "epoch"], default="steps")
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--load-best-model-at-end", action="store_true")
    parser.add_argument("--metric-for-best-model", type=str, default="eval_loss")
    parser.add_argument("--greater-is-better", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--full-finetune", action="store_true")
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--use-liger-kernel", action="store_true")
    parser.add_argument("--fsdp", type=str, default="")
    parser.add_argument("--fsdp-transformer-layer-cls-to-wrap", type=str, default="")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--report-to", type=str, default="none")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    return parser.parse_args()


def load_jsonl_dataset(path: str, max_samples: int | None = None) -> Dataset:
    dataset = load_dataset("json", data_files=path, split="train")
    if max_samples is not None:
        dataset = dataset.select(range(min(len(dataset), max_samples)))
    return dataset


def build_quant_config(use_4bit: bool):
    if not use_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )


def tokenize_dataset(
    dataset: Dataset,
    tokenizer,
    max_length: int,
    overlength_policy: str,
) -> tuple[Dataset, Dict[str, Any]]:
    encoded_rows, stats = tokenize_completion_rows(
        dataset,
        tokenizer,
        max_length,
        overlength_policy,
    )
    return Dataset.from_list(encoded_rows), stats


def main() -> None:
    args = parse_args()
    if args.full_finetune and args.adapter_path:
        raise ValueError("--full-finetune cannot be combined with --adapter-path")
    if args.full_finetune and args.use_4bit:
        raise ValueError("--full-finetune cannot be combined with --use-4bit")
    if args.fsdp and not args.full_finetune:
        raise ValueError("--fsdp requires --full-finetune")
    if args.fsdp and not args.fsdp_transformer_layer_cls_to_wrap:
        raise ValueError("--fsdp requires --fsdp-transformer-layer-cls-to-wrap")
    if args.use_liger_kernel:
        try:
            import liger_kernel  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "--use-liger-kernel requires the liger-kernel package to be installed"
            ) from exc

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    train_raw = load_jsonl_dataset(args.train_file, args.max_train_samples)
    eval_raw = load_jsonl_dataset(args.valid_file, args.max_eval_samples)
    train_dataset, train_stats = tokenize_dataset(
        train_raw, tokenizer, args.max_length, args.overlength_policy
    )
    eval_dataset, eval_stats = tokenize_dataset(
        eval_raw, tokenizer, args.max_length, args.overlength_policy
    )

    quant_config = build_quant_config(args.use_4bit)
    torch_dtype = torch.bfloat16 if args.bf16 or args.use_4bit else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=quant_config,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    base_model.config.use_cache = False
    use_fsdp = args.full_finetune and bool(args.fsdp)
    if args.gradient_checkpointing and not use_fsdp:
        base_model.gradient_checkpointing_enable()
        if not args.full_finetune and hasattr(base_model, "enable_input_require_grads"):
            base_model.enable_input_require_grads()

    if args.full_finetune:
        model = base_model
    elif args.adapter_path:
        if not os.path.exists(args.adapter_path):
            raise FileNotFoundError(f"Adapter path does not exist: {args.adapter_path}")
        model = PeftModel.from_pretrained(base_model, args.adapter_path, is_trainable=True)
    else:
        model = get_peft_model(
            base_model,
            LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            ),
        )

    fsdp_config = None
    if use_fsdp:
        fsdp_config = {
            "transformer_layer_cls_to_wrap": args.fsdp_transformer_layer_cls_to_wrap,
        }
        if args.gradient_checkpointing:
            fsdp_config["activation_checkpointing"] = True

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        optim=args.optim,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        accelerator_config={
            "gradient_accumulation_kwargs": {
                "sync_each_batch": args.sync_each_batch,
            },
        },
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy=args.eval_strategy,
        save_strategy="steps",
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        bf16=args.bf16 or args.use_4bit,
        fp16=not (args.bf16 or args.use_4bit),
        report_to=args.report_to,
        run_name=args.run_name,
        remove_unused_columns=False,
        gradient_checkpointing=args.gradient_checkpointing and not use_fsdp,
        fsdp=args.fsdp if use_fsdp else "",
        fsdp_config=fsdp_config,
        use_liger_kernel=args.use_liger_kernel,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        logging_first_step=True,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model if args.load_best_model_at_end else None,
        greater_is_better=args.greater_is_better if args.load_best_model_at_end else None,
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.output_dir)
        with open(Path(args.output_dir) / "data_stats.json", "w", encoding="utf-8") as handle:
            json.dump({"train": train_stats, "validation": eval_stats}, handle, ensure_ascii=False, indent=2)
        with open(Path(args.output_dir) / "train_metrics.json", "w", encoding="utf-8") as handle:
            json.dump(trainer.state.log_history, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
