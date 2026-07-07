import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path('/home/pkuccadm/huwenp/emb/lxy/sd-zero-sql')
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sql_core.sql_verifier import prepare_gold_execution, verify_sql_against_gold

DEFAULT_INPUT_JSONL = PROJECT_ROOT / 'data' / 'ches_train_sft_train_4k.jsonl'
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'data' / 'srt'
DEFAULT_PREFIX = 'sql_verifier_diag'
DEFAULT_TIMEOUT_SQL = '''
WITH RECURSIVE cnt(x) AS (
  SELECT 1
  UNION ALL
  SELECT x + 1 FROM cnt
)
SELECT sum(x) FROM cnt;
'''.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Diagnose the SQLite verifier without GPU generation.')
    parser.add_argument('--input-jsonl', type=Path, default=DEFAULT_INPUT_JSONL)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--prefix', type=str, default=DEFAULT_PREFIX)
    parser.add_argument('--max-samples', type=int, default=5)
    parser.add_argument('--db-id', type=str, default='')
    parser.add_argument('--include-timeout-case', action='store_true')
    return parser.parse_args()


def load_samples(path: Path, max_samples: int, db_id: str) -> List[Dict[str, Any]]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            if db_id and row.get('db_id') != db_id:
                continue
            rows.append(row)
            if len(rows) >= max_samples:
                break
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def main() -> None:
    args = parse_args()
    samples = load_samples(args.input_jsonl, args.max_samples, args.db_id)
    if not samples:
        raise ValueError('No matching samples found for verifier diagnostics.')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    timeout_record = None

    for sample in samples:
        prepared_gold = prepare_gold_execution(sample['db_id'], sample['gold_sql'])
        gold_record = {
            'id': sample.get('id'),
            'db_id': sample.get('db_id'),
            'gold_ok': prepared_gold['gold_ok'],
            'gold_error': prepared_gold['gold_error'],
            'gold_result_preview': prepared_gold['gold_result_preview'],
        }
        verifier = verify_sql_against_gold(prepared_gold, sample['gold_sql'])
        records.append(
            {
                **gold_record,
                'mode': 'gold_sql',
                'sql': sample['gold_sql'],
                'verifier': verifier,
            }
        )
        if args.include_timeout_case and timeout_record is None:
            timeout_record = {
                **gold_record,
                'mode': 'timeout_probe',
                'sql': DEFAULT_TIMEOUT_SQL,
                'verifier': verify_sql_against_gold(prepared_gold, DEFAULT_TIMEOUT_SQL),
            }

    if timeout_record is not None:
        records.append(timeout_record)

    output_jsonl = args.output_dir / f'{args.prefix}.jsonl'
    output_summary = args.output_dir / f'{args.prefix}_summary.json'
    write_jsonl(output_jsonl, records)

    summary = {
        'input_jsonl': str(args.input_jsonl),
        'output_jsonl': str(output_jsonl),
        'record_count': len(records),
        'sample_count': len(samples),
        'db_ids': sorted({row['db_id'] for row in records}),
        'timeout_count': sum(1 for row in records if row['verifier'].get('error_type') in {'timeout', 'gold_timeout'}),
        'runtime_error_count': sum(1 for row in records if row['verifier'].get('error_type') in {'runtime_error', 'gold_runtime_error'}),
        'result_too_large_count': sum(1 for row in records if row['verifier'].get('error_type') in {'result_too_large', 'gold_result_too_large'}),
        'correct_count': sum(1 for row in records if row['verifier'].get('reward') == 1),
    }
    with output_summary.open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Records: {output_jsonl}')
    print(f'Summary: {output_summary}')


if __name__ == '__main__':
    main()
