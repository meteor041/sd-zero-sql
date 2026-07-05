import argparse
import json
from pathlib import Path
from typing import Dict, List

from transformers import AutoTokenizer


DEFAULT_INPUT_TRAIN = Path("/home/pkuccadm/huwenp/emb/lxy/M-Schema/ches_train_sft_train.jsonl")
DEFAULT_INPUT_VALID = Path("/home/pkuccadm/huwenp/emb/lxy/M-Schema/ches_train_sft_valid.jsonl")
DEFAULT_OUTPUT_DIR = Path("/home/pkuccadm/huwenp/emb/lxy/ches_sql_sft/data")
DEFAULT_MODEL_PATH = "/data/model/Qwen3-4B-Instruct-2507"
DEFAULT_MAX_LENGTH = 4096


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_jsonl(path: Path, rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def annotate_and_filter(rows: List[Dict], tokenizer, max_length: int, split: str):
    kept = []
    overflow = []
    for row in rows:
        text_len = len(tokenizer(row["text"], add_special_tokens=False)["input_ids"])
        prompt_len = len(tokenizer(row["prompt"], add_special_tokens=False)["input_ids"])
        response_len = len(tokenizer(row["response"], add_special_tokens=False)["input_ids"])

        row_with_len = dict(row)
        row_with_len["token_length"] = text_len
        row_with_len["prompt_token_length"] = prompt_len
        row_with_len["response_token_length"] = response_len

        if text_len <= max_length:
            kept.append(row_with_len)
        else:
            overflow.append(
                {
                    "split": split,
                    "id": row["id"],
                    "db_id": row["db_id"],
                    "question": row["question"],
                    "token_length": text_len,
                    "prompt_token_length": prompt_len,
                    "response_token_length": response_len,
                }
            )
    return kept, overflow


def build_summary(train_kept, valid_kept, overflow, max_length: int, model_path: str):
    return {
        "model_path": model_path,
        "max_length": max_length,
        "train_kept": len(train_kept),
        "valid_kept": len(valid_kept),
        "total_kept": len(train_kept) + len(valid_kept),
        "overflow_count": len(overflow),
        "overflow_by_split": {
            "train": sum(1 for item in overflow if item["split"] == "train"),
            "valid": sum(1 for item in overflow if item["split"] == "valid"),
        },
        "overflow_db_ids": sorted({item["db_id"] for item in overflow}),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter CHES SFT dataset by token length.")
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_INPUT_TRAIN)
    parser.add_argument("--valid-jsonl", type=Path, default=DEFAULT_INPUT_VALID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)

    train_rows = load_jsonl(args.train_jsonl)
    valid_rows = load_jsonl(args.valid_jsonl)

    train_kept, train_overflow = annotate_and_filter(train_rows, tokenizer, args.max_length, "train")
    valid_kept, valid_overflow = annotate_and_filter(valid_rows, tokenizer, args.max_length, "valid")
    overflow = train_overflow + valid_overflow

    train_out = args.output_dir / "ches_train_sft_train_4k.jsonl"
    valid_out = args.output_dir / "ches_train_sft_valid_4k.jsonl"
    overflow_out = args.output_dir / "ches_train_sft_overflow_4k.jsonl"
    summary_out = args.output_dir / "ches_train_sft_4k_summary.json"

    save_jsonl(train_out, train_kept)
    save_jsonl(valid_out, valid_kept)
    save_jsonl(overflow_out, overflow)
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(build_summary(train_kept, valid_kept, overflow, args.max_length, args.model_path), f, ensure_ascii=False, indent=2)

    print(f"Train kept:   {len(train_kept)}")
    print(f"Valid kept:   {len(valid_kept)}")
    print(f"Overflow:     {len(overflow)}")
    print(f"Train output: {train_out}")
    print(f"Valid output: {valid_out}")
    print(f"Overflow out: {overflow_out}")
    print(f"Summary out:  {summary_out}")


if __name__ == "__main__":
    main()
