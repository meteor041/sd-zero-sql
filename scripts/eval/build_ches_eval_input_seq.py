import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sql_core.prompt_builders import build_base_sql_prompt

DEFAULT_INPUT_JSON = Path('/data/huwenp/emb/data/ches/dev.json')
DEFAULT_TABLES_JSON = Path('/data/huwenp/emb/data/ches/dev_tables.json')
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / 'data' / 'eval' / 'ches_dev_input_seq.json'
DEFAULT_MODEL_PATH = '/data/huwenp/emb/lxy/sd-zero-sql/outputs/qwen3_4b_srt_joint/merged'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build a csc_sql-compatible CHES eval JSON with sd-zero-style input_seq prompts.')
    parser.add_argument('--input-json', type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument('--tables-json', type=Path, default=DEFAULT_TABLES_JSON)
    parser.add_argument('--output-json', type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument('--model-path', type=str, default=DEFAULT_MODEL_PATH)
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def build_schema_index(tables_rows: List[Dict]) -> Dict[str, Dict]:
    return {row['db_id']: row for row in tables_rows}


def render_schema_text(schema_row: Dict) -> str:
    table_names = schema_row['table_names_original']
    column_names = schema_row['column_names_original']
    column_types = schema_row['column_types']
    primary_key_entries = schema_row.get('primary_keys', [])
    primary_keys = set()
    for entry in primary_key_entries:
        if isinstance(entry, list):
            primary_keys.update(entry)
        else:
            primary_keys.add(entry)
    foreign_keys = schema_row.get('foreign_keys', [])

    table_columns: Dict[int, List[str]] = {idx: [] for idx in range(len(table_names))}
    for col_idx, (table_idx, column_name) in enumerate(column_names):
        if table_idx < 0:
            continue
        column_type = column_types[col_idx]
        extras = []
        if col_idx in primary_keys:
            extras.append('PRIMARY KEY')
        for src, dst in foreign_keys:
            if src == col_idx:
                ref_table_idx, ref_column_name = column_names[dst]
                ref_table = table_names[ref_table_idx]
                extras.append(f'FOREIGN KEY -> {ref_table}.{ref_column_name}')
        extra_str = f" ({'; '.join(extras)})" if extras else ''
        table_columns[table_idx].append(f'  - {column_name}: {column_type}{extra_str}')

    lines = []
    for table_idx, table_name in enumerate(table_names):
        lines.append(f'Table {table_name}')
        lines.extend(table_columns[table_idx])
        lines.append('')
    return '\n'.join(lines).strip()


def main() -> None:
    args = parse_args()
    raw_rows = load_json(args.input_json)
    tables_rows = load_json(args.tables_json)
    schema_index = build_schema_index(tables_rows)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)

    output_rows = []
    for idx, row in enumerate(raw_rows):
        db_id = row['db_id']
        schema_row = schema_index[db_id]
        sample = {
            'question': row.get('question', ''),
            'evidence': row.get('evidence', ''),
            'schema': render_schema_text(schema_row),
        }
        output_rows.append(
            {
                'id': row.get('question_id', idx),
                'db_id': db_id,
                'question': row.get('question', ''),
                'evidence': row.get('evidence', ''),
                'schema': sample['schema'],
                'SQL': row.get('SQL', ''),
                'input_seq': build_base_sql_prompt(sample, tokenizer=tokenizer),
            }
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output_rows, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote {len(output_rows)} rows to {args.output_json}')


if __name__ == '__main__':
    main()
