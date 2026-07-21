import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path('/home/pkuccadm/huwenp/emb/lxy/sd-zero-sql')
SRC_ROOT = PROJECT_ROOT / 'src'
SCRIPT_ROOT = PROJECT_ROOT / 'scripts' / 'srt'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from generate_phase1_traces import write_summary, append_stage_row
from sql_core.sql_verifier import verify_sql

INIT_GENERATED_PATH = Path('/data/huwenp/emb/lxy/sd-zero-sql/data/srt/traces_train_full_32init_feedbackfix_init_generated.jsonl')
INIT_VERIFIED_PATH = Path('/data/huwenp/emb/lxy/sd-zero-sql/data/srt/traces_train_full_32init_feedbackfix_init_verified.jsonl')
SUMMARY_PATH = Path('/data/huwenp/emb/lxy/sd-zero-sql/data/srt/traces_train_full_32init_feedbackfix_summary.json')
PROGRESS_EVERY = 10
POLL_SECONDS = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Resume init verification with row sharding.')
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=None)
    parser.add_argument('--launch-workers', action='store_true')
    parser.add_argument('--skip-merge', action='store_true')
    parser.add_argument('--python-bin', type=str, default=sys.executable)
    return parser.parse_args()


def load_jsonl(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)


def shard_output_path(base_path: Path, shard_index: int) -> Path:
    return base_path.with_name(f'{base_path.stem}.shard{shard_index}{base_path.suffix}')


def worker_log_path(shard_index: int) -> Path:
    return INIT_VERIFIED_PATH.parent / f'reverify_init_resume.shard{shard_index}.log'


def row_belongs_to_shard(global_index: int, num_shards: int, shard_index: int) -> bool:
    return global_index % num_shards == shard_index


def materialize_resume_row(row: dict, global_index: int) -> dict:
    row['sample'] = {
        'id': row['id'],
        'db_id': row['db_id'],
        'gold_sql': row['gold_sql'],
    }
    row['global_index'] = global_index
    return row


def append_shard_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'global_index': row['global_index'],
        **{
            'id': row['sample']['id'],
            'db_id': row['sample']['db_id'],
            'base_prompt': row['base_prompt'],
            'x': row['base_prompt'],
            'gold_sql': row['sample']['gold_sql'],
            'init_index': row['init_index'],
            'raw_y_init': row['raw_y_init'],
            'y_init': row['y_init'],
            'verifier_init': row['verifier_init'],
        },
    }
    with open(path, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + '\n')
        handle.flush()


def run_worker(shard_index: int, num_shards: int) -> None:
    with open(SUMMARY_PATH, 'r', encoding='utf-8') as f:
        state = json.load(f)

    shard_path = shard_output_path(INIT_VERIFIED_PATH, shard_index)
    verified_count = count_lines(shard_path)
    total_count = count_lines(INIT_GENERATED_PATH)
    shard_total = sum(1 for i in range(1, total_count + 1) if row_belongs_to_shard(i, num_shards, shard_index))

    state['status'] = 'verifying_init_resume_sharded'
    state['resume_num_shards'] = num_shards
    state['resume_merge_complete'] = False
    write_summary(SUMMARY_PATH, state)

    shard_seen = 0
    for global_index, row in enumerate(load_jsonl(INIT_GENERATED_PATH), start=1):
        if not row_belongs_to_shard(global_index, num_shards, shard_index):
            continue
        shard_seen += 1
        if shard_seen <= verified_count:
            continue
        verifier_result = verify_sql(row['db_id'], row['y_init'], row['gold_sql'])
        row = materialize_resume_row(row, global_index)
        row['verifier_init'] = verifier_result
        append_shard_row(shard_path, row)
        if shard_seen % PROGRESS_EVERY == 0 or shard_seen == shard_total:
            progress = {
                'shard_index': shard_index,
                'verified_count': shard_seen,
                'shard_total': shard_total,
                'current_db_id': row['db_id'],
                'current_sample_id': row['id'],
                'current_init_index': row['init_index'],
                'last_verifier_error_type': verifier_result.get('error_type'),
                'updated_at': time.time(),
            }
            progress_path = shard_path.with_suffix(shard_path.suffix + '.progress.json')
            with open(progress_path, 'w', encoding='utf-8') as f:
                json.dump(progress, f, ensure_ascii=False, indent=2)


def merge_shards(num_shards: int) -> None:
    rows = []
    for shard_index in range(num_shards):
        shard_path = shard_output_path(INIT_VERIFIED_PATH, shard_index)
        for row in load_jsonl(shard_path):
            rows.append(row)
    rows.sort(key=lambda row: row['global_index'])
    with open(INIT_VERIFIED_PATH, 'w', encoding='utf-8') as handle:
        for row in rows:
            payload = dict(row)
            payload.pop('global_index', None)
            handle.write(json.dumps(payload, ensure_ascii=False) + '\n')
            handle.flush()


def aggregate_summary(num_shards: int) -> dict:
    with open(SUMMARY_PATH, 'r', encoding='utf-8') as f:
        state = json.load(f)

    shard_counts = {}
    shard_paths = {}
    completed = 0
    total_verified = 0
    for shard_index in range(num_shards):
        shard_path = shard_output_path(INIT_VERIFIED_PATH, shard_index)
        shard_paths[str(shard_index)] = str(shard_path)
        count = count_lines(shard_path)
        shard_counts[str(shard_index)] = count
        total_verified += count
        progress_path = shard_path.with_suffix(shard_path.suffix + '.progress.json')
        if progress_path.exists():
            with open(progress_path, 'r', encoding='utf-8') as f:
                progress = json.load(f)
            shard_total = progress.get('shard_total', 0)
            if shard_total and count >= shard_total:
                completed += 1

    state['status'] = 'verifying_init_resume_sharded'
    state['resume_num_shards'] = num_shards
    state['resume_shards_completed'] = completed
    state['resume_shard_processed_counts'] = shard_counts
    state['resume_shard_output_paths'] = shard_paths
    state['resume_merge_complete'] = False
    state['verifier_processed_count'] = total_verified
    state['verifier_total_count'] = count_lines(INIT_GENERATED_PATH)
    write_summary(SUMMARY_PATH, state)
    return state


def run_coordinator(num_shards: int, python_bin: str, skip_merge: bool) -> None:
    procs = []
    for shard_index in range(num_shards):
        log_path = worker_log_path(shard_index)
        with open(log_path, 'w', encoding='utf-8') as log_handle:
            proc = subprocess.Popen(
                [
                    python_bin,
                    str(Path(__file__)),
                    '--num-shards',
                    str(num_shards),
                    '--shard-index',
                    str(shard_index),
                ],
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
        procs.append((shard_index, proc))

    failed = None
    while procs:
        aggregate_summary(num_shards)
        remaining = []
        for shard_index, proc in procs:
            code = proc.poll()
            if code is None:
                remaining.append((shard_index, proc))
                continue
            if code != 0 and failed is None:
                failed = (shard_index, code)
        if failed is not None:
            for _, proc in remaining:
                proc.terminate()
            raise RuntimeError(f'shard {failed[0]} failed with exit code {failed[1]}')
        procs = remaining
        if procs:
            time.sleep(POLL_SECONDS)

    state = aggregate_summary(num_shards)
    if not skip_merge:
        merge_shards(num_shards)
        state['resume_merge_complete'] = True
        state['status'] = 'verified_init_resumed_complete'
        write_summary(SUMMARY_PATH, state)


def main() -> None:
    args = parse_args()
    if args.launch_workers:
        run_coordinator(args.num_shards, args.python_bin, args.skip_merge)
        return
    if args.shard_index is None:
        raise ValueError('--shard-index is required unless --launch-workers is used')
    run_worker(args.shard_index, args.num_shards)


if __name__ == '__main__':
    main()
