#!/usr/bin/env python
"""Merge SFT rows into an existing Phase1 multitask train/valid pair.

Produces R4-style combined data: the Phase1 generation/revision rows are kept
verbatim, and each SFT row is converted into a multitask row with task="sft",
a chat-template base prompt (rebuilt with the Qwen tokenizer so the format
matches the Phase1 rows), and the gold SQL as the completion. The merged rows
are shuffled and written to {prefix}_train.jsonl / {prefix}_valid.jsonl.

This script is standalone: it does not modify build_two_stage_data.py,
train_srt_stage.py, or any existing data file.
"""
import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

from transformers import AutoTokenizer

from sql_core.prompt_builders import build_base_sql_prompt


def sft_to_task_row(row: Dict[str, Any], index: int, tokenizer) -> Dict[str, Any]:
    prompt = build_base_sql_prompt(row, tokenizer=tokenizer)
    completion = str(row.get("response") or "").strip()
    if not completion:
        raise ValueError(f"SFT row {row.get('id', index)} has an empty response.")
    return {
        "id": row.get("id"),
        "db_id": row.get("db_id"),
        "trace_id": f"sft__{row.get('id') or index}",
        "task": "sft",
        "y_init_correct": None,
        "revision_source": "gold",
        "prompt": prompt,
        "completion": completion,
    }


def load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multitask-train", required=True, type=Path)
    parser.add_argument("--multitask-valid", required=True, type=Path)
    parser.add_argument("--sft-train", required=True, type=Path)
    parser.add_argument("--sft-valid", required=True, type=Path)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    mt_train = load_rows(args.multitask_train)
    mt_valid = load_rows(args.multitask_valid)
    sft_train = load_rows(args.sft_train)
    sft_valid = load_rows(args.sft_valid)

    sft_train_rows = [
        sft_to_task_row(row, index, tokenizer) for index, row in enumerate(sft_train)
    ]
    sft_valid_rows = [
        sft_to_task_row(row, index, tokenizer) for index, row in enumerate(sft_valid)
    ]

    train_rows = mt_train + sft_train_rows
    valid_rows = mt_valid + sft_valid_rows
    rng.shuffle(train_rows)
    rng.shuffle(valid_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / f"{args.prefix}_train.jsonl"
    valid_path = args.output_dir / f"{args.prefix}_valid.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(valid_path, valid_rows)

    summary = {
        "multitask_train_rows": len(mt_train),
        "multitask_valid_rows": len(mt_valid),
        "sft_train_rows": len(sft_train),
        "sft_valid_rows": len(sft_valid),
        "merged_train_rows": len(train_rows),
        "merged_valid_rows": len(valid_rows),
        "sft_train_fraction": round(len(sft_train_rows) / len(train_rows), 4),
        "train_path": str(train_path),
        "valid_path": str(valid_path),
    }
    summary_path = args.output_dir / f"{args.prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
