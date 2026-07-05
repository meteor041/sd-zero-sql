import argparse
import json
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path('/home/pkuccadm/huwenp/emb/lxy/ches_sql_sft')
DEFAULT_TRACE_JSONL = PROJECT_ROOT / 'data' / 'srt' / 'traces_train_1k_stratified_vllm.jsonl'
DEFAULT_BASE_TRAIN_JSONL = PROJECT_ROOT / 'data' / 'ches_train_sft_train_4k.jsonl'
DEFAULT_OUTPUT_JSONL = PROJECT_ROOT / 'data' / 'srt' / 'srt_train_mixed.jsonl'
DEFAULT_SUMMARY_JSON = PROJECT_ROOT / 'data' / 'srt' / 'srt_train_mixed_summary.json'


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def build_generation_rows(base_rows: List[Dict]) -> List[Dict]:
    results = []
    for row in base_rows:
        results.append(
            {
                'type': 'generation',
                'id': row['id'],
                'db_id': row['db_id'],
                'prompt': row['prompt'],
                'response': row['response'],
                'text': row['text'],
            }
        )
    return results


def build_revision_rows(trace_rows: List[Dict]) -> List[Dict]:
    results = []
    for row in trace_rows:
        if row.get('verifier_init', {}).get('reward', 0) != 0:
            continue
        if row.get('verifier_revised', {}).get('reward', 0) != 1:
            continue
        revision_prompt = row.get('revision_prompt')
        y_revised = row.get('y_revised')
        if not revision_prompt or not y_revised:
            continue
        results.append(
            {
                'type': 'revision',
                'id': row['id'],
                'db_id': row['db_id'],
                'prompt': revision_prompt,
                'response': y_revised,
                'text': revision_prompt + y_revised,
                'y_init': row.get('y_init', ''),
                'init_reward': row.get('verifier_init', {}).get('reward', 0),
                'revised_reward': row.get('verifier_revised', {}).get('reward', 0),
            }
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build mixed Phase1 SRT training data.')
    parser.add_argument('--trace-jsonl', type=Path, default=DEFAULT_TRACE_JSONL)
    parser.add_argument('--base-train-jsonl', type=Path, default=DEFAULT_BASE_TRAIN_JSONL)
    parser.add_argument('--output-jsonl', type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument('--summary-json', type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument('--max-generation', type=int, default=None)
    parser.add_argument('--max-revision', type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_rows = load_jsonl(args.base_train_jsonl)
    trace_rows = load_jsonl(args.trace_jsonl)

    generation_rows = build_generation_rows(base_rows)
    revision_rows = build_revision_rows(trace_rows)

    if args.max_generation is not None:
        generation_rows = generation_rows[:args.max_generation]
    if args.max_revision is not None:
        revision_rows = revision_rows[:args.max_revision]

    mixed_rows = generation_rows + revision_rows
    save_jsonl(args.output_jsonl, mixed_rows)

    summary = {
        'generation_count': len(generation_rows),
        'revision_count': len(revision_rows),
        'total_count': len(mixed_rows),
        'trace_source': str(args.trace_jsonl),
        'base_source': str(args.base_train_jsonl),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Mixed dataset: {args.output_jsonl}')
    print(f'Summary: {args.summary_json}')


if __name__ == '__main__':
    main()
