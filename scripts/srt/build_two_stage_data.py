import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phase1_srt.trace_schema import dedupe_trace_key, normalize_trace_record, validate_trace_record
from sql_core.prompt_builders import build_revision_continuation_prompt

DEFAULT_TRACE_JSONL = PROJECT_ROOT / 'data' / 'srt' / 'traces_train_1k_stratified_vllm.jsonl'
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'data' / 'srt'
DEFAULT_PREFIX = 'ches_qwen3_4b_srt'


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _dedupe_and_filter(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    seen = set()
    for record in records:
        normalized = normalize_trace_record(record)
        validate_trace_record(normalized)
        if not normalized.get('keep', False):
            continue
        if not normalized.get('y_revised_correct', False):
            continue
        key = dedupe_trace_key(normalized)
        if key in seen:
            continue
        seen.add(key)
        out.append(normalized)
    return out


def _stage1_record(record: Dict[str, Any]) -> Dict[str, Any]:
    prompt = record['x']
    completion = f"{record['y_init']}\n\n{record['p_r']}\n\n{record['y_revised']}"
    return {
        'id': record.get('id'),
        'db_id': record.get('db_id'),
        'prompt': prompt,
        'completion': completion,
        'text': prompt + completion,
    }


def _stage2_record(record: Dict[str, Any]) -> Dict[str, Any]:
    prompt = build_revision_continuation_prompt(record['x'], record['y_init'], record['p_r'])
    completion = record['y_revised']
    return {
        'id': record.get('id'),
        'db_id': record.get('db_id'),
        'prompt': prompt,
        'completion': completion,
        'text': prompt + completion,
    }


def prepare_srt_stage_data(input_path: Path, output_dir: Path, stage2_size: int, seed: int, prefix: str, min_stage1_size: int = 0) -> Tuple[Path, Path, Dict[str, Any]]:
    records = _dedupe_and_filter(load_jsonl(input_path))
    if not records:
        raise ValueError('No usable revision tuples (keep=True and y_revised_correct=True) were found.')

    incorrect_init = [record for record in records if not record['y_init_correct']]
    correct_init = [record for record in records if record['y_init_correct']]

    rng = random.Random(seed)
    rng.shuffle(incorrect_init)
    rng.shuffle(correct_init)

    n_correct_s2 = min(len(correct_init), stage2_size)
    n_incorrect_s2 = min(len(incorrect_init), stage2_size - n_correct_s2)
    stage2_src = correct_init[:n_correct_s2] + incorrect_init[:n_incorrect_s2]
    rng.shuffle(stage2_src)
    stage1_src = incorrect_init[n_incorrect_s2:]

    if len(stage1_src) < min_stage1_size:
        backfill_needed = min_stage1_size - len(stage1_src)
        correct_pool = correct_init[n_correct_s2:]
        stage1_src.extend(correct_pool[:backfill_needed])

    if not stage1_src:
        raise ValueError('Stage 1 would be empty. Collect more incorrect-init successful traces or lower stage2_size.')
    if not stage2_src:
        raise ValueError('Stage 2 would be empty. No usable revision tuples were found.')

    stage1 = [_stage1_record(record) for record in stage1_src]
    stage2 = [_stage2_record(record) for record in stage2_src]

    output_dir.mkdir(parents=True, exist_ok=True)
    stage1_path = output_dir / f'{prefix}_stage1.jsonl'
    stage2_path = output_dir / f'{prefix}_stage2.jsonl'

    with open(stage1_path, 'w', encoding='utf-8') as f:
        for row in stage1:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    with open(stage2_path, 'w', encoding='utf-8') as f:
        for row in stage2:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    summary = {
        'input_trace_file': str(input_path),
        'total_kept_traces': len(records),
        'incorrect_init_count': len(incorrect_init),
        'correct_init_count': len(correct_init),
        'stage1_count': len(stage1),
        'stage1_min_requested': min_stage1_size,
        'stage2_count': len(stage2),
        'stage2_incorrect_count': n_incorrect_s2,
        'stage2_correct_count': n_correct_s2,
        'prefix': prefix,
        'stage1_path': str(stage1_path),
        'stage2_path': str(stage2_path),
    }
    return stage1_path, stage2_path, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build official-style two-stage Phase1 SRT data.')
    parser.add_argument('--input', type=Path, default=DEFAULT_TRACE_JSONL)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--stage2-size', type=int, default=1000)
    parser.add_argument('--min-stage1-size', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--prefix', type=str, default=DEFAULT_PREFIX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage1_path, stage2_path, summary = prepare_srt_stage_data(
        input_path=args.input,
        output_dir=args.output_dir,
        stage2_size=args.stage2_size,
        seed=args.seed,
        prefix=args.prefix,
        min_stage1_size=args.min_stage1_size,
    )
    summary_path = args.output_dir / f'{args.prefix}_stage_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Stage1: {stage1_path}')
    print(f'Stage2: {stage2_path}')
    print(f'Summary: {summary_path}')


if __name__ == '__main__':
    main()
