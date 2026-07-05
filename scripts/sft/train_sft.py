import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen3-4B-Instruct on CHES SQL SFT data.")
    parser.add_argument("--model-path", type=str, default="/data/model/Qwen3-4B-Instruct-2507")
    parser.add_argument(
        "--train-file",
        type=str,
        default="/home/pkuccadm/huwenp/emb/lxy/ches_sql_sft/data/ches_train_sft_train_4k.jsonl",
    )
    parser.add_argument(
        "--valid-file",
        type=str,
        default="/home/pkuccadm/huwenp/emb/lxy/ches_sql_sft/data/ches_train_sft_valid_4k.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/data/huwenp/emb/lxy/ches_sql_sft/outputs/qwen3_4b_sft_lora_4k",
    )
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--num-train-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--use-4bit", action="store_true", help="Load base model in 4-bit with bitsandbytes.")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--packing", action="store_true")
    parser.add_argument("--report-to", type=str, default="none")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    return parser.parse_args()


def load_jsonl_dataset(path: str, max_samples: int = None) -> Dataset:
    ds = load_dataset("json", data_files=path, split="train")
    if max_samples is not None:
        ds = ds.select(range(min(len(ds), max_samples)))
    return ds


def build_quant_config(use_4bit: bool):
    if not use_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )


def main() -> None:
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_dataset = load_jsonl_dataset(args.train_file, args.max_train_samples)
    eval_dataset = load_jsonl_dataset(args.valid_file, args.max_eval_samples)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    quant_config = build_quant_config(args.use_4bit)
    torch_dtype = torch.bfloat16 if args.bf16 or args.use_4bit else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=quant_config,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    peft_config = LoraConfig(
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
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        bf16=args.bf16 or args.use_4bit,
        fp16=not (args.bf16 or args.use_4bit),
        report_to=args.report_to,
        remove_unused_columns=True,
        gradient_checkpointing=args.gradient_checkpointing,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        logging_first_step=True,
        load_best_model_at_end=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=args.max_length,
        tokenizer=tokenizer,
        packing=args.packing,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics_path = Path(args.output_dir) / "train_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
