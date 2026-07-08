import argparse
import ctypes
import json
import os
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path('/home/pkuccadm/huwenp/emb/lxy/sd-zero-sql')
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sql_core.generation_backend import load_generator
from sql_core.prompt_builders import build_base_sql_prompt, build_revision_prompt
from phase1_srt.trace_schema import build_trace_record
from sql_core.sql_normalizer import normalize_sql_output
from sql_core.sql_verifier import prepare_gold_execution, verify_sql_against_gold

DEFAULT_MODEL_PATH = '/data/model/Qwen3-4B-Instruct-2507'
DEFAULT_INPUT_JSONL = PROJECT_ROOT / 'data' / 'ches_train_sft_train_4k.jsonl'
DEFAULT_OUTPUT_JSONL = PROJECT_ROOT / 'data' / 'srt' / 'traces_train_smoke.jsonl'
DEFAULT_SUMMARY_JSON = PROJECT_ROOT / 'data' / 'srt' / 'traces_train_smoke_summary.json'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate Phase 1 SRT traces for CHES SQL SFT.')
    parser.add_argument('--model-path', type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument('--input-jsonl', type=Path, default=DEFAULT_INPUT_JSONL)
    parser.add_argument('--output-jsonl', type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument('--summary-json', type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument('--max-samples', type=int, default=16)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--num-inits', type=int, default=1)
    parser.add_argument('--sample-chunk-size', type=int, default=32)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--sampling-mode', type=str, default='head', choices=['head', 'random', 'stratified'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min-per-db', type=int, default=5)
    parser.add_argument('--backend', type=str, default='hf', choices=['hf', 'vllm'])
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--tensor-parallel-size', type=int, default=1)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9)
    parser.add_argument('--verifier-workers', type=int, default=8)
    return parser.parse_args()


def load_jsonl(path: Path, max_samples: int = None) -> List[Dict]:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def select_samples(rows: List[Dict], max_samples: int, sampling_mode: str, seed: int, min_per_db: int) -> List[Dict]:
    if max_samples is None or max_samples >= len(rows):
        return rows

    if sampling_mode == 'head':
        return rows[:max_samples]

    rng = random.Random(seed)

    if sampling_mode == 'random':
        rows_copy = list(rows)
        rng.shuffle(rows_copy)
        return rows_copy[:max_samples]

    by_db = defaultdict(list)
    for row in rows:
        by_db[row['db_id']].append(row)

    selected = []
    selected_ids = set()

    for db_id in sorted(by_db.keys()):
        group = list(by_db[db_id])
        rng.shuffle(group)
        take = min(len(group), min_per_db)
        for row in group[:take]:
            selected.append(row)
            selected_ids.add(row['id'])

    if len(selected) > max_samples:
        rng.shuffle(selected)
        return selected[:max_samples]

    remaining = [row for row in rows if row['id'] not in selected_ids]
    rng.shuffle(remaining)
    need = max_samples - len(selected)
    selected.extend(remaining[:need])
    return selected


def select_shard(samples: List[Dict], num_shards: int, shard_index: int) -> List[Dict]:
    if num_shards < 1:
        raise ValueError('num_shards must be >= 1')
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f'shard_index must be in [0, {num_shards - 1}]')
    if num_shards == 1:
        return samples
    return samples[shard_index::num_shards]


def append_jsonl_rows(handle, rows: List[Dict[str, Any]]) -> None:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    handle.flush()
    os.fsync(handle.fileno())


def init_summary_state(args: argparse.Namespace, selected_count: int, shard_count: int) -> Dict[str, Any]:
    return {
        'status': 'running',
        'backend': args.backend,
        'input_jsonl': str(args.input_jsonl),
        'output_jsonl': str(args.output_jsonl),
        'selected_sample_count_before_shard': selected_count,
        'sample_count': shard_count,
        'processed_sample_count': 0,
        'num_inits': args.num_inits,
        'sample_chunk_size': args.sample_chunk_size,
        'num_shards': args.num_shards,
        'shard_index': args.shard_index,
        'batch_size': args.batch_size,
        'temperature': args.temperature,
        'max_new_tokens': args.max_new_tokens,
        'verifier_workers': args.verifier_workers,
        'total_traces': 0,
        'init_correct_count': 0,
        'revised_correct_count': 0,
        'kept_count': 0,
        'db_trace_counts': defaultdict(int),
        'db_kept_counts': defaultdict(int),
        'verifier_processed_count': 0,
        'verifier_total_count': 0,
        'current_db_id': None,
        'current_sample_id': None,
        'current_init_index': None,
        'last_verifier_error_type': None,
    }


def update_summary_state(state: Dict[str, Any], traces: List[Dict[str, Any]], processed_samples: int = 0) -> None:
    state['processed_sample_count'] += processed_samples
    state['total_traces'] += len(traces)
    state['init_correct_count'] += sum(1 for trace in traces if trace['verifier_init'].get('reward', 0) == 1)
    state['revised_correct_count'] += sum(1 for trace in traces if trace['verifier_revised'].get('reward', 0) == 1)
    state['kept_count'] += sum(1 for trace in traces if trace['keep'])
    for trace in traces:
        db_id = trace['db_id']
        state['db_trace_counts'][db_id] += 1
        if trace['keep']:
            state['db_kept_counts'][db_id] += 1


def materialize_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    total = state['total_traces']
    return {
        'status': state['status'],
        'backend': state['backend'],
        'input_jsonl': state['input_jsonl'],
        'output_jsonl': state['output_jsonl'],
        'selected_sample_count_before_shard': state['selected_sample_count_before_shard'],
        'sample_count': state['sample_count'],
        'processed_sample_count': state['processed_sample_count'],
        'num_inits': state['num_inits'],
        'sample_chunk_size': state['sample_chunk_size'],
        'num_shards': state['num_shards'],
        'shard_index': state['shard_index'],
        'batch_size': state['batch_size'],
        'temperature': state['temperature'],
        'max_new_tokens': state['max_new_tokens'],
        'verifier_workers': state.get('verifier_workers'),
        'total_traces': total,
        'init_correct_count': state['init_correct_count'],
        'init_correct_ratio': round(state['init_correct_count'] / total, 4) if total else 0,
        'revised_correct_count': state['revised_correct_count'],
        'revised_correct_ratio': round(state['revised_correct_count'] / total, 4) if total else 0,
        'kept_count': state['kept_count'],
        'kept_ratio': round(state['kept_count'] / total, 4) if total else 0,
        'db_trace_counts': dict(sorted(state['db_trace_counts'].items())),
        'db_kept_counts': dict(sorted(state['db_kept_counts'].items())),
        'verifier_processed_count': state.get('verifier_processed_count', 0),
        'verifier_total_count': state.get('verifier_total_count', 0),
        'current_db_id': state.get('current_db_id'),
        'current_sample_id': state.get('current_sample_id'),
        'current_init_index': state.get('current_init_index'),
        'last_verifier_error_type': state.get('last_verifier_error_type'),
        'error': state.get('error'),
    }


def write_summary(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(materialize_summary(state), f, ensure_ascii=False, indent=2)


def set_process_title(title: str) -> None:
    try:
        libc = ctypes.CDLL(None)
        PR_SET_NAME = 15
        libc.prctl(PR_SET_NAME, title.encode('utf-8')[:15], 0, 0, 0)
    except Exception:
        pass


def build_stage_output_paths(output_jsonl: Path) -> Dict[str, Path]:
    stem = output_jsonl.stem
    parent = output_jsonl.parent
    return {
        'init_generated': parent / f'{stem}_init_generated.jsonl',
        'init_verified': parent / f'{stem}_init_verified.jsonl',
        'revised_generated': parent / f'{stem}_revised_generated.jsonl',
        'revised_verified': parent / f'{stem}_revised_verified.jsonl',
    }


def materialize_stage_row(row: Dict[str, Any], *, include_verifier_init: bool = False, include_revised: bool = False) -> Dict[str, Any]:
    sample = row['sample']
    record = {
        'id': sample.get('id'),
        'db_id': sample.get('db_id'),
        'base_prompt': row['base_prompt'],
        'x': row['base_prompt'],
        'gold_sql': sample.get('gold_sql'),
        'init_index': row['init_index'],
        'raw_y_init': row['raw_y_init'],
        'y_init': row['y_init'],
    }
    if include_verifier_init:
        record['verifier_init'] = row.get('verifier_init')
    if include_revised:
        record['revision_prompt'] = row.get('revision_prompt')
        record['raw_y_revised'] = row.get('raw_y_revised')
        record['y_revised'] = row.get('y_revised')
        record['verifier_revised'] = row.get('verifier_revised')
    return record


def write_stage_rows(path: Path, rows: List[Dict[str, Any]], *, include_verifier_init: bool = False, include_revised: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        append_jsonl_rows(
            handle,
            [
                materialize_stage_row(
                    row,
                    include_verifier_init=include_verifier_init,
                    include_revised=include_revised,
                )
                for row in rows
            ],
        )


def append_stage_row(path: Path, row: Dict[str, Any], *, include_verifier_init: bool = False, include_revised: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as handle:
        append_jsonl_rows(
            handle,
            [
                materialize_stage_row(
                    row,
                    include_verifier_init=include_verifier_init,
                    include_revised=include_revised,
                )
            ],
        )


def reset_stage_output(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8'):
        pass


def generate_init_candidates(samples: List[Dict], generator, batch_size: int, max_new_tokens: int, temperature: float, num_inits: int) -> List[Dict]:
    prompts = [build_base_sql_prompt(sample) for sample in samples]
    raw_outputs = []
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]
        raw_outputs.extend(generator.generate_batch(batch_prompts, max_new_tokens, temperature, num_return_sequences=num_inits))

    stage_rows = []
    for sample, prompt, output_start in zip(samples, prompts, range(0, len(raw_outputs), num_inits)):
        prepared_gold = prepare_gold_execution(sample['db_id'], sample['gold_sql'])
        for init_index, raw_y_init in enumerate(raw_outputs[output_start:output_start + num_inits]):
            y_init = normalize_sql_output(raw_y_init)
            stage_rows.append(
                {
                    'sample': sample,
                    'base_prompt': prompt,
                    'prepared_gold': prepared_gold,
                    'init_index': init_index,
                    'raw_y_init': raw_y_init,
                    'y_init': y_init,
                }
            )
    return stage_rows


def verify_init_candidates(
    stage_rows: List[Dict[str, Any]],
    state: Dict[str, Any],
    summary_path: Path,
    init_verified_path: Path,
    progress_every: int = 10,
    verifier_workers: int = 8,
) -> None:
    verify_stage_rows_parallel(
        stage_rows,
        sql_key='y_init',
        verifier_key='verifier_init',
        state=state,
        summary_path=summary_path,
        stage_output_path=init_verified_path,
        include_revised=False,
        progress_every=progress_every,
        verifier_workers=verifier_workers,
    )


def verify_stage_row(args: tuple[str, str, str]) -> Dict[str, Any]:
    db_id, predicted_sql, gold_sql = args
    return verify_sql(db_id, predicted_sql, gold_sql)


def verify_stage_rows_parallel(
    stage_rows: List[Dict[str, Any]],
    *,
    sql_key: str,
    verifier_key: str,
    state: Dict[str, Any],
    summary_path: Path,
    stage_output_path: Path,
    include_revised: bool,
    progress_every: int,
    verifier_workers: int,
) -> None:
    state['verifier_total_count'] = len(stage_rows)
    state['verifier_processed_count'] = 0
    state['last_verifier_error_type'] = None
    reset_stage_output(stage_output_path)
    verify_args = [
        (row['sample']['db_id'], row[sql_key], row['sample']['gold_sql'])
        for row in stage_rows
    ]
    with ProcessPoolExecutor(max_workers=verifier_workers) as executor:
        for index, (row, verifier_result) in enumerate(zip(stage_rows, executor.map(verify_stage_row, verify_args)), start=1):
            row[verifier_key] = verifier_result
            state['verifier_processed_count'] = index
            state['current_db_id'] = row['sample'].get('db_id')
            state['current_sample_id'] = row['sample'].get('id')
            state['current_init_index'] = row['init_index']
            error_type = verifier_result.get('error_type')
            if error_type and error_type != 'correct':
                state['last_verifier_error_type'] = error_type
            append_stage_row(
                stage_output_path,
                row,
                include_verifier_init=True,
                include_revised=include_revised,
            )
            if index % progress_every == 0 or index == len(stage_rows):
                write_summary(summary_path, state)


def generate_revised_candidates(stage_rows: List[Dict[str, Any]], generator, batch_size: int, max_new_tokens: int, temperature: float) -> None:
    revision_prompts = []
    for row in stage_rows:
        revision_prompt = build_revision_prompt(row['sample'], row['y_init'], row['verifier_init'])
        row['revision_prompt'] = revision_prompt
        revision_prompts.append(revision_prompt)

    raw_revision_outputs = []
    for start in range(0, len(revision_prompts), batch_size):
        batch_prompts = revision_prompts[start:start + batch_size]
        raw_revision_outputs.extend(generator.generate_batch(batch_prompts, max_new_tokens, temperature))

    for row, raw_y_revised in zip(stage_rows, raw_revision_outputs):
        row['raw_y_revised'] = raw_y_revised
        row['y_revised'] = normalize_sql_output(raw_y_revised)


def finalize_traces(
    stage_rows: List[Dict[str, Any]],
    state: Dict[str, Any],
    summary_path: Path,
    revised_verified_path: Path,
    progress_every: int = 10,
    verifier_workers: int = 8,
) -> List[Dict[str, Any]]:
    verify_stage_rows_parallel(
        stage_rows,
        sql_key='y_revised',
        verifier_key='verifier_revised',
        state=state,
        summary_path=summary_path,
        stage_output_path=revised_verified_path,
        include_revised=True,
        progress_every=progress_every,
        verifier_workers=verifier_workers,
    )
    traces = []
    for row in stage_rows:
        sample = row['sample']
        trace = build_trace_record(
            sample=sample,
            x=row['base_prompt'],
            y_init=row['y_init'],
            verifier_init=row['verifier_init'],
            y_revised=row['y_revised'],
            verifier_revised=row['verifier_revised'],
            raw_y_init=row['raw_y_init'],
            raw_y_revised=row['raw_y_revised'],
            revision_prompt=row['revision_prompt'],
        )
        trace['trace_id'] = f"{sample['id']}__init{row['init_index']}"
        trace['init_index'] = row['init_index']
        traces.append(trace)
    return traces


def main() -> None:
    args = parse_args()
    all_rows = load_jsonl(args.input_jsonl, None)
    selected_samples = select_samples(all_rows, args.max_samples, args.sampling_mode, args.seed, args.min_per_db)
    samples = select_shard(selected_samples, args.num_shards, args.shard_index)

    generator = load_generator(
        backend=args.backend,
        model_path=args.model_path,
        use_bf16=args.bf16,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    stage_output_paths = build_stage_output_paths(args.output_jsonl)

    state = init_summary_state(args, selected_count=len(selected_samples), shard_count=len(samples))

    print(
        f'Using backend={args.backend} batch_size={args.batch_size} sample_chunk_size={args.sample_chunk_size} '
        f'samples={len(samples)} num_inits={args.num_inits} shard={args.shard_index}/{args.num_shards}'
    )

    try:
        state['status'] = 'generating_init'
        set_process_title('p1-gen-init')
        write_summary(args.summary_json, state)
        stage_rows = generate_init_candidates(
            samples,
            generator,
            args.batch_size,
            args.max_new_tokens,
            args.temperature,
            args.num_inits,
        )
        write_stage_rows(stage_output_paths['init_generated'], stage_rows)
        state['processed_sample_count'] = len(samples)
        write_summary(args.summary_json, state)
        print(
            json.dumps(
                {
                    'status': state['status'],
                    'sample_count': len(samples),
                    'init_candidate_count': len(stage_rows),
                    'init_generated_path': str(stage_output_paths['init_generated']),
                },
                ensure_ascii=False,
            )
        )

        state['status'] = 'verifying_init'
        set_process_title('p1-verify-init')
        state['verifier_processed_count'] = 0
        state['verifier_total_count'] = len(stage_rows)
        state['current_db_id'] = None
        state['current_sample_id'] = None
        state['current_init_index'] = None
        state['last_verifier_error_type'] = None
        write_summary(args.summary_json, state)
        verify_init_candidates(
            stage_rows,
            state=state,
            summary_path=args.summary_json,
            init_verified_path=stage_output_paths['init_verified'],
            verifier_workers=args.verifier_workers,
        )
        write_summary(args.summary_json, state)
        print(
            json.dumps(
                {
                    'status': state['status'],
                    'verified_init_count': len(stage_rows),
                    'init_verified_path': str(stage_output_paths['init_verified']),
                },
                ensure_ascii=False,
            )
        )

        state['status'] = 'generating_revised'
        set_process_title('p1-gen-rev')
        write_summary(args.summary_json, state)
        generate_revised_candidates(
            stage_rows,
            generator,
            args.batch_size,
            args.max_new_tokens,
            args.temperature,
        )
        write_stage_rows(stage_output_paths['revised_generated'], stage_rows, include_verifier_init=True, include_revised=True)
        write_summary(args.summary_json, state)
        print(
            json.dumps(
                {
                    'status': state['status'],
                    'revised_candidate_count': len(stage_rows),
                    'revised_generated_path': str(stage_output_paths['revised_generated']),
                },
                ensure_ascii=False,
            )
        )

        state['status'] = 'verifying_revised'
        set_process_title('p1-verify-rev')
        state['verifier_processed_count'] = 0
        state['verifier_total_count'] = len(stage_rows)
        state['current_db_id'] = None
        state['current_sample_id'] = None
        state['current_init_index'] = None
        state['last_verifier_error_type'] = None
        write_summary(args.summary_json, state)
        traces = finalize_traces(
            stage_rows,
            state=state,
            summary_path=args.summary_json,
            revised_verified_path=stage_output_paths['revised_verified'],
            verifier_workers=args.verifier_workers,
        )

        with open(args.output_jsonl, 'w', encoding='utf-8') as output_handle:
            append_jsonl_rows(output_handle, traces)

        print(json.dumps({'status': state['status'], 'final_trace_path': str(args.output_jsonl), 'trace_count': len(traces)}, ensure_ascii=False))
        update_summary_state(state, traces)
        state['status'] = 'completed'
        set_process_title('p1-complete')
        write_summary(args.summary_json, state)
    except Exception as exc:
        state['status'] = 'failed'
        set_process_title('p1-failed')
        state['error'] = f'{type(exc).__name__}: {exc}'
        write_summary(args.summary_json, state)
        raise

    print(json.dumps(materialize_summary(state), ensure_ascii=False, indent=2))
    print(f'Trace output: {args.output_jsonl}')
    print(f'Summary: {args.summary_json}')


if __name__ == '__main__':
    main()
