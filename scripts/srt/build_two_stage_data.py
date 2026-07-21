import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phase1_srt.trace_schema import dedupe_trace_key, normalize_trace_record, validate_trace_record
from sql_core.prompt_builders import build_revision_continuation_prompt

DEFAULT_TRACE_JSONL = PROJECT_ROOT / "data" / "srt" / "traces_train_full_1init_3revision.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "srt"
DEFAULT_PREFIX = "ches_qwen3_4b_srt"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _question_key(record: Dict[str, Any]) -> str:
    value = record.get("id")
    if value is not None:
        return f"{record.get('db_id', '')}::{value}"
    return f"{record.get('db_id', '')}::{record['x']}"


def _dedupe_and_filter(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output = []
    seen = set()
    for record in records:
        normalized = normalize_trace_record(record)
        validate_trace_record(normalized)
        if not normalized.get("keep", False) or not normalized.get("y_revised_correct", False):
            continue
        key = dedupe_trace_key(normalized)
        if key in seen:
            continue
        seen.add(key)
        output.append(normalized)
    return output


def _cap_traces_per_question(
    records: List[Dict[str, Any]],
    max_traces_per_question: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    if max_traces_per_question <= 0:
        return records
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[_question_key(record)].append(record)

    output = []
    for key in sorted(grouped):
        group = grouped[key]
        rng.shuffle(group)
        output.extend(group[:max_traces_per_question])
    return output


def _balance_outcomes(
    records: List[Dict[str, Any]],
    max_correct_init_ratio: float,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    if not 0.0 <= max_correct_init_ratio <= 1.0:
        raise ValueError("max_correct_init_ratio must be between 0 and 1")
    incorrect = [record for record in records if not record["y_init_correct"]]
    correct = [record for record in records if record["y_init_correct"]]
    if not incorrect:
        raise ValueError(
            "No successful incorrect-init revisions were found. The model has not produced "
            "the correction supervision required by SRT."
        )
    if max_correct_init_ratio < 1.0:
        max_correct = math.floor(len(incorrect) * max_correct_init_ratio / (1.0 - max_correct_init_ratio))
        rng.shuffle(correct)
        correct = correct[:max_correct]
    output = incorrect + correct
    rng.shuffle(output)
    return output


def generation_task_record(record: Dict[str, Any]) -> Dict[str, Any]:
    completion = f"{record['y_init'].strip()}\n\n{record['p_r']}\n\n{record['y_revised'].strip()}"
    return _task_record(record, "generation", record["x"], completion)


def revision_task_record(record: Dict[str, Any]) -> Dict[str, Any]:
    prompt = build_revision_continuation_prompt(record["x"], record["y_init"], record["p_r"])
    return _task_record(record, "revision", prompt, record["y_revised"].strip())


def _task_record(record: Dict[str, Any], task: str, prompt: str, completion: str) -> Dict[str, Any]:
    trace_id = record.get("trace_id") or (
        f"{_question_key(record)}__init{record.get('init_index', 0)}"
        f"__revision{record.get('revision_index', 0)}"
    )
    return {
        "id": record.get("id"),
        "db_id": record.get("db_id"),
        "trace_id": trace_id,
        "task": task,
        "y_init_correct": bool(record["y_init_correct"]),
        "prompt": prompt,
        "completion": completion,
    }


def _split_by_question(
    records: List[Dict[str, Any]],
    validation_fraction: float,
    rng: random.Random,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    question_keys = sorted({_question_key(record) for record in records})
    rng.shuffle(question_keys)
    validation_count = 0
    if validation_fraction > 0 and len(question_keys) > 1:
        validation_count = max(1, round(len(question_keys) * validation_fraction))
    validation_keys = set(question_keys[:validation_count])
    train = [record for record in records if _question_key(record) not in validation_keys]
    validation = [record for record in records if _question_key(record) in validation_keys]
    return train, validation


def _expand_tasks(records: List[Dict[str, Any]], rng: random.Random) -> List[Dict[str, Any]]:
    rows = []
    for record in records:
        rows.append(generation_task_record(record))
        rows.append(revision_task_record(record))
    rng.shuffle(rows)
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_srt_multitask_data(
    input_path: Path,
    output_dir: Path,
    seed: int,
    prefix: str,
    validation_fraction: float = 0.05,
    max_traces_per_question: int = 3,
    max_correct_init_ratio: float = 0.5,
) -> Tuple[Path, Path, Dict[str, Any]]:
    rng = random.Random(seed)
    records = _dedupe_and_filter(load_jsonl(input_path))
    if not records:
        raise ValueError("No verified-correct revision traces were found.")

    records = _cap_traces_per_question(records, max_traces_per_question, rng)
    records = _balance_outcomes(records, max_correct_init_ratio, rng)
    train_traces, validation_traces = _split_by_question(records, validation_fraction, rng)
    if not train_traces:
        raise ValueError("The question-level split produced an empty training set.")

    train_rows = _expand_tasks(train_traces, rng)
    validation_rows = _expand_tasks(validation_traces, rng)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / f"{prefix}_train.jsonl"
    validation_path = output_dir / f"{prefix}_valid.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(validation_path, validation_rows)

    incorrect_count = sum(not record["y_init_correct"] for record in records)
    correct_count = len(records) - incorrect_count
    summary = {
        "input_trace_file": str(input_path),
        "usable_trace_count": len(records),
        "unique_question_count": len({_question_key(record) for record in records}),
        "incorrect_init_trace_count": incorrect_count,
        "correct_init_trace_count": correct_count,
        "correct_init_ratio": round(correct_count / len(records), 4),
        "train_trace_count": len(train_traces),
        "validation_trace_count": len(validation_traces),
        "train_task_record_count": len(train_rows),
        "validation_task_record_count": len(validation_rows),
        "generation_task_count": len(records),
        "revision_task_count": len(records),
        "validation_fraction": validation_fraction,
        "max_traces_per_question": max_traces_per_question,
        "max_correct_init_ratio": max_correct_init_ratio,
        "train_path": str(train_path),
        "validation_path": str(validation_path),
    }
    return train_path, validation_path, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build joint generation/revision Phase1 SRT data.")
    parser.add_argument("--input", type=Path, default=DEFAULT_TRACE_JSONL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--max-traces-per-question", type=int, default=3)
    parser.add_argument("--max-correct-init-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix", type=str, default=DEFAULT_PREFIX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path, validation_path, summary = prepare_srt_multitask_data(
        input_path=args.input,
        output_dir=args.output_dir,
        seed=args.seed,
        prefix=args.prefix,
        validation_fraction=args.validation_fraction,
        max_traces_per_question=args.max_traces_per_question,
        max_correct_init_ratio=args.max_correct_init_ratio,
    )
    summary_path = args.output_dir / f"{args.prefix}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Train: {train_path}")
    print(f"Validation: {validation_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
