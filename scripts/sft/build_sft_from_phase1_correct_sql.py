#!/usr/bin/env python3
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

from sql_core.prompt_builders import build_base_sql_prompt
from sql_core.sql_normalizer import normalize_sql_output

DEFAULT_INPUT = Path('/data/huwenp/emb/lxy/sd-zero-sql/data/srt/traces_train_full_4init_from_stage2_gpu0to3.jsonl')
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'data'
DEFAULT_PREFIX = 'ches_qwen3_4b_sft_from_phase1_4init_correct_sql'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build SFT data from correct SQLs in Phase1 traces.')
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--prefix', type=str, default=DEFAULT_PREFIX)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--valid-fraction', type=float, default=0.02)
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def question_key(record: Dict[str, Any]) -> Tuple[Any, Any]:
    return record.get('db_id'), record.get('id')


def normalize_correct_sql(sql: str) -> str:
    return normalize_sql_output(sql).strip()


def build_sft_record(record: Dict[str, Any], sql: str, source: str) -> Dict[str, Any]:
    sample = {
        'schema': record.get('schema'),
        'evidence': record.get('evidence'),
        'question': record.get('question'),
    }
    prompt = build_base_sql_prompt(sample)
    sql = normalize_correct_sql(sql)
    return {
        'id': record.get('id'),
        'db_id': record.get('db_id'),
        'trace_id': record.get('trace_id'),
        'schema': record.get('schema'),
        'evidence': record.get('evidence'),
        'question': record.get('question'),
        'gold_sql': sql,
        'prompt': prompt,
        'completion': sql,
        'text': prompt + sql,
        'sql_source': source,
    }


def dedupe_records(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    unique = []
    seen = {}
    stats = {'duplicate_records': 0, 'source_init_only': 0, 'source_revised_only': 0, 'source_both': 0}
    for record in records:
        key = (
            record.get('db_id'),
            record.get('question'),
            record.get('evidence'),
            record.get('schema'),
            record.get('gold_sql'),
        )
        existing = seen.get(key)
        if existing is None:
            seen[key] = len(unique)
            unique.append(record)
            continue
        stats['duplicate_records'] += 1
        prev = unique[existing]
        if prev['sql_source'] != record['sql_source']:
            prev['sql_source'] = 'both'
    for record in unique:
        if record['sql_source'] == 'init':
            stats['source_init_only'] += 1
        elif record['sql_source'] == 'revised':
            stats['source_revised_only'] += 1
        else:
            stats['source_both'] += 1
    return unique, stats


def split_by_question(records: List[Dict[str, Any]], valid_fraction: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    grouped = {}
    for record in records:
        grouped.setdefault(question_key(record), []).append(record)
    question_ids = list(grouped.keys())
    rng = random.Random(seed)
    rng.shuffle(question_ids)
    valid_count = max(1, int(len(question_ids) * valid_fraction)) if question_ids else 0
    valid_questions = set(question_ids[:valid_count])
    train, valid = [], []
    for qid, rows in grouped.items():
        if qid in valid_questions:
            valid.extend(rows)
        else:
            train.extend(rows)
    return train, valid


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def main() -> None:
    args = parse_args()
    traces = load_jsonl(args.input)

    raw_init_correct = 0
    raw_revised_correct = 0
    sft_records = []
    for trace in traces:
        if trace.get('y_init_correct'):
            raw_init_correct += 1
            sft_records.append(build_sft_record(trace, trace['y_init'], 'init'))
        if trace.get('y_revised_correct'):
            raw_revised_correct += 1
            sft_records.append(build_sft_record(trace, trace['y_revised'], 'revised'))

    deduped, dedupe_stats = dedupe_records(sft_records)
    train_rows, valid_rows = split_by_question(deduped, args.valid_fraction, args.seed)

    output_dir = args.output_dir
    full_path = output_dir / f'{args.prefix}.jsonl'
    train_path = output_dir / f'{args.prefix}_train.jsonl'
    valid_path = output_dir / f'{args.prefix}_valid.jsonl'
    summary_path = output_dir / f'{args.prefix}_summary.json'

    write_jsonl(full_path, deduped)
    write_jsonl(train_path, train_rows)
    write_jsonl(valid_path, valid_rows)

    summary = {
        'input_trace_file': str(args.input),
        'raw_init_correct_count': raw_init_correct,
        'raw_revised_correct_count': raw_revised_correct,
        'raw_correct_sql_total': raw_init_correct + raw_revised_correct,
        'deduped_sft_count': len(deduped),
        'train_count': len(train_rows),
        'valid_count': len(valid_rows),
        'valid_fraction': args.valid_fraction,
        'source_init_only': dedupe_stats['source_init_only'],
        'source_revised_only': dedupe_stats['source_revised_only'],
        'source_both': dedupe_stats['source_both'],
        'duplicate_records_removed': dedupe_stats['duplicate_records'],
        'full_path': str(full_path),
        'train_path': str(train_path),
        'valid_path': str(valid_path),
    }
    with open(summary_path, 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'full: {full_path}')
    print(f'train: {train_path}')
    print(f'valid: {valid_path}')
    print(f'summary: {summary_path}')


if __name__ == '__main__':
    main()
