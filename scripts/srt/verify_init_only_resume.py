#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / 'src'
SCRIPT_ROOT = PROJECT_ROOT / 'scripts' / 'srt'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from generate_phase1_traces import append_stage_row, write_summary, row_belongs_to_shard
from sql_core.sql_verifier import prepare_gold_execution, verify_sql_against_gold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Resume verify_init from an existing init_generated JSONL.')
    parser.add_argument('--output-jsonl', type=Path, required=True)
    parser.add_argument('--summary-json', type=Path, required=True)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=None)
    parser.add_argument('--launch-workers', action='store_true')
    parser.add_argument('--python-bin', type=str, default=sys.executable)
    parser.add_argument('--poll-seconds', type=int, default=15)
    return parser.parse_args()


def build_stage_output_paths(output_jsonl: Path) -> Dict[str, Path]:
    stem = output_jsonl.stem
    parent = output_jsonl.parent
    return {
        'init_generated': parent / f'{stem}_init_generated.jsonl',
        'init_verified': parent / f'{stem}_init_verified.jsonl',
        'revised_generated': parent / f'{stem}_revised_generated.jsonl',
        'revised_verified': parent / f'{stem}_revised_verified.jsonl',
    }


def shard_output_path(base_path: Path, shard_index: int) -> Path:
    return base_path.with_name(f'{base_path.stem}.shard{shard_index}{base_path.suffix}')


def shard_progress_path(base_path: Path, shard_index: int) -> Path:
    shard_path = shard_output_path(base_path, shard_index)
    return shard_path.with_suffix(shard_path.suffix + '.progress.json')


def load_jsonl(path: Path):
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, 'r', encoding='utf-8') as handle:
        return sum(1 for _ in handle)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_summary(path: Path) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def write_shard_row(path: Path, row: Dict[str, Any]) -> None:
    payload = dict(row)
    payload['sample'] = {
        'id': row['sample']['id'],
        'db_id': row['sample']['db_id'],
        'gold_sql': row['sample']['gold_sql'],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + '\n')
        handle.flush()


def count_shard_total(init_generated_path: Path, num_shards: int, shard_index: int) -> int:
    shard_total = 0
    for global_index, _row in enumerate(load_jsonl(init_generated_path), start=1):
        if row_belongs_to_shard(global_index, num_shards, shard_index):
            shard_total += 1
    return shard_total


def run_worker(output_jsonl: Path, summary_json: Path, num_shards: int, shard_index: int) -> None:
    stage_paths = build_stage_output_paths(output_jsonl)
    init_generated_path = stage_paths['init_generated']
    init_verified_path = stage_paths['init_verified']
    shard_verified_path = shard_output_path(init_verified_path, shard_index)
    progress_path = shard_progress_path(init_verified_path, shard_index)

    summary_state = load_summary(summary_json)
    shard_done = count_lines(shard_verified_path)
    shard_total = count_shard_total(init_generated_path, num_shards, shard_index)
    gold_cache: Dict[tuple[str, str], Dict[str, Any]] = {}

    processed_in_shard = 0
    for global_index, row in enumerate(load_jsonl(init_generated_path), start=1):
        if not row_belongs_to_shard(global_index, num_shards, shard_index):
            continue
        processed_in_shard += 1
        if processed_in_shard <= shard_done:
            continue

        db_id = row['db_id']
        gold_sql = row['gold_sql']
        cache_key = (db_id, gold_sql)
        prepared_gold = gold_cache.get(cache_key)
        if prepared_gold is None:
            prepared_gold = prepare_gold_execution(db_id, gold_sql)
            gold_cache[cache_key] = prepared_gold

        verifier_result = verify_sql_against_gold(prepared_gold, row['y_init'])
        work_row = {
            'sample': {'id': row['id'], 'db_id': db_id, 'gold_sql': gold_sql},
            'base_prompt': row['base_prompt'],
            'init_index': row['init_index'],
            'raw_y_init': row['raw_y_init'],
            'y_init': row['y_init'],
            'verifier_init': verifier_result,
        }
        write_shard_row(shard_verified_path, work_row)

        progress = {
            'status': 'verifying_init_resume_sharded',
            'shard_index': shard_index,
            'verified_count': processed_in_shard,
            'shard_total': shard_total,
            'current_db_id': db_id,
            'current_sample_id': row['id'],
            'current_init_index': row['init_index'],
            'last_verifier_error_type': verifier_result.get('error_type'),
            'updated_at': time.time(),
        }
        write_json(progress_path, progress)

    write_json(progress_path, {
        'status': 'completed',
        'shard_index': shard_index,
        'verified_count': shard_total,
        'shard_total': shard_total,
        'updated_at': time.time(),
    })


def merge_shards(output_jsonl: Path, num_shards: int) -> Path:
    stage_paths = build_stage_output_paths(output_jsonl)
    init_generated_path = stage_paths['init_generated']
    init_verified_path = stage_paths['init_verified']
    shard_iters = {i: list(load_jsonl(shard_output_path(init_verified_path, i))) for i in range(num_shards)}
    shard_offsets = {i: 0 for i in range(num_shards)}
    with open(init_verified_path, 'w', encoding='utf-8'):
        pass
    for global_index, _row in enumerate(load_jsonl(init_generated_path), start=1):
        for shard_index in range(num_shards):
            if row_belongs_to_shard(global_index, num_shards, shard_index):
                row = shard_iters[shard_index][shard_offsets[shard_index]]
                shard_offsets[shard_index] += 1
                append_stage_row(init_verified_path, row, include_verifier_init=True, include_revised=False)
                break
    return init_verified_path


def update_summary_from_shards(output_jsonl: Path, summary_json: Path, num_shards: int, final_status: str | None = None) -> Dict[str, Any]:
    stage_paths = build_stage_output_paths(output_jsonl)
    state = load_summary(summary_json)
    shard_counts = {}
    total_verified = 0
    completed = 0
    for shard_index in range(num_shards):
        shard_verified_path = shard_output_path(stage_paths['init_verified'], shard_index)
        count = count_lines(shard_verified_path)
        shard_counts[str(shard_index)] = count
        total_verified += count
        progress_path = shard_progress_path(stage_paths['init_verified'], shard_index)
        if progress_path.exists():
            progress = load_summary(progress_path)
            if progress.get('status') == 'completed':
                completed += 1
    state['status'] = final_status or 'verifying_init_resume_sharded'
    state['resume_num_shards'] = num_shards
    state['resume_shards_completed'] = completed
    state['resume_shard_processed_counts'] = shard_counts
    state['resume_merge_complete'] = final_status == 'verified_init_resumed_complete'
    state['verifier_total_count'] = count_lines(stage_paths['init_generated'])
    state['verifier_processed_count'] = total_verified
    write_summary(summary_json, state)
    return state


def run_coordinator(output_jsonl: Path, summary_json: Path, num_shards: int, python_bin: str, poll_seconds: int) -> None:
    procs = []
    for shard_index in range(num_shards):
        proc = subprocess.Popen([
            python_bin,
            str(Path(__file__)),
            '--output-jsonl', str(output_jsonl),
            '--summary-json', str(summary_json),
            '--num-shards', str(num_shards),
            '--shard-index', str(shard_index),
        ])
        procs.append((shard_index, proc))

    while procs:
        update_summary_from_shards(output_jsonl, summary_json, num_shards)
        remaining = []
        for shard_index, proc in procs:
            code = proc.poll()
            if code is None:
                remaining.append((shard_index, proc))
            elif code != 0:
                raise RuntimeError(f'shard {shard_index} failed with exit code {code}')
        procs = remaining
        if procs:
            time.sleep(poll_seconds)

    merge_shards(output_jsonl, num_shards)
    update_summary_from_shards(output_jsonl, summary_json, num_shards, final_status='verified_init_resumed_complete')


def main() -> None:
    args = parse_args()
    if args.launch_workers:
        run_coordinator(args.output_jsonl, args.summary_json, args.num_shards, args.python_bin, args.poll_seconds)
        return
    if args.shard_index is None:
        raise ValueError('--shard-index is required unless --launch-workers is used')
    run_worker(args.output_jsonl, args.summary_json, args.num_shards, args.shard_index)


if __name__ == '__main__':
    main()
