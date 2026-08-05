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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sql_core.generation_backend import load_generator
from sql_core.prompt_builders import (
    build_base_sql_messages,
    build_base_sql_prompt,
    build_revision_chat_messages,
    build_revision_continuation_prompt,
    render_chat_prompt,
)
from phase1_srt.constants import select_p_r
from phase1_srt.trace_schema import build_trace_record
from sql_core.sql_normalizer import normalize_sql_output
from sql_core.sql_verifier import verify_sql

DEFAULT_MODEL_PATH = '/data/model/Qwen3-4B-Instruct-2507'
DEFAULT_INPUT_JSONL = Path('/home/pkuccadm/huwenp/emb/lxy/M-Schema/ches_train_sft.jsonl')
DEFAULT_OUTPUT_JSONL = PROJECT_ROOT / 'data' / 'srt' / 'traces_train_smoke.jsonl'
DEFAULT_SUMMARY_JSON = PROJECT_ROOT / 'data' / 'srt' / 'traces_train_smoke_summary.json'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate Phase 1 SRT traces for CHES SQL SFT.')
    parser.add_argument('--model-path', type=str, default=DEFAULT_MODEL_PATH, help='Legacy shared model path/name.')
    parser.add_argument('--init-model-path', type=str, default=None)
    parser.add_argument('--revision-model-path', type=str, default=None)
    parser.add_argument('--init-tokenizer-path', type=str, default=None)
    parser.add_argument('--revision-tokenizer-path', type=str, default=None)
    parser.add_argument('--input-jsonl', type=Path, default=DEFAULT_INPUT_JSONL)
    parser.add_argument('--output-jsonl', type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument('--summary-json', type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument('--resume-init-generated', type=Path, default=None)
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top-p', type=float, default=1.0)
    parser.add_argument('--num-inits', type=int, default=1)
    parser.add_argument('--num-revisions', type=int, default=3)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--sampling-mode', type=str, default='head', choices=['head', 'random', 'stratified'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min-per-db', type=int, default=5)
    parser.add_argument('--backend', type=str, default='hf', choices=['hf', 'vllm', 'api', 'openai'], help='Legacy shared backend.')
    parser.add_argument('--init-backend', type=str, default=None, choices=['hf', 'vllm', 'api', 'openai'])
    parser.add_argument('--revision-backend', type=str, default=None, choices=['hf', 'vllm', 'api', 'openai'])
    parser.add_argument('--api-base-url', type=str, default=os.environ.get('OPENAI_BASE_URL'))
    parser.add_argument('--init-api-base-url', type=str, default=os.environ.get('PHASE1_INIT_API_BASE_URL'))
    parser.add_argument('--revision-api-base-url', type=str, default=os.environ.get('PHASE1_REVISION_API_BASE_URL'))
    parser.add_argument('--api-key-env', type=str, default='OPENAI_API_KEY')
    parser.add_argument('--init-api-key-env', type=str, default='PHASE1_INIT_API_KEY')
    parser.add_argument('--revision-api-key-env', type=str, default='PHASE1_REVISION_API_KEY')
    parser.add_argument('--api-max-concurrency', type=int, default=8)
    parser.add_argument('--init-api-max-concurrency', type=int, default=None)
    parser.add_argument('--revision-api-max-concurrency', type=int, default=None)
    parser.add_argument('--api-timeout', type=float, default=120.0)
    parser.add_argument('--api-max-retries', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--tensor-parallel-size', type=int, default=1)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9)
    parser.add_argument('--max-model-len', type=int, default=8192)
    parser.add_argument('--verifier-workers', type=int, default=16)
    return parser.parse_args()


def resolve_generation_args(args: argparse.Namespace) -> argparse.Namespace:
    args.init_backend = args.init_backend or args.backend
    args.revision_backend = args.revision_backend or args.init_backend
    args.init_model_path = args.init_model_path or args.model_path
    args.revision_model_path = args.revision_model_path or args.init_model_path
    args.init_tokenizer_path = args.init_tokenizer_path or args.init_model_path
    args.revision_tokenizer_path = args.revision_tokenizer_path or args.revision_model_path
    args.init_api_base_url = args.init_api_base_url or args.api_base_url
    args.revision_api_base_url = args.revision_api_base_url or args.api_base_url or args.init_api_base_url
    args.init_api_max_concurrency = args.init_api_max_concurrency or args.api_max_concurrency
    args.revision_api_max_concurrency = args.revision_api_max_concurrency or args.api_max_concurrency

    for phase in ('init', 'revision'):
        backend = getattr(args, f'{phase}_backend')
        if backend in {'api', 'openai'} and not getattr(args, f'{phase}_api_base_url'):
            raise ValueError(
                f'--{phase}-api-base-url or --api-base-url is required when {phase} uses the API backend.'
            )
    return args


def build_generator_config(args: argparse.Namespace, phase: str) -> Dict[str, Any]:
    backend = getattr(args, f'{phase}_backend')
    config: Dict[str, Any] = {
        'backend': backend,
        'model_path': getattr(args, f'{phase}_model_path'),
        'max_model_len': args.max_model_len,
    }
    if backend == 'hf':
        config['use_bf16'] = args.bf16
    elif backend == 'vllm':
        config['tensor_parallel_size'] = args.tensor_parallel_size
        config['gpu_memory_utilization'] = args.gpu_memory_utilization
    elif backend in {'api', 'openai'}:
        phase_key = os.environ.get(getattr(args, f'{phase}_api_key_env'))
        shared_key = os.environ.get(args.api_key_env)
        config.update(
            {
                'tokenizer_path': getattr(args, f'{phase}_tokenizer_path'),
                'api_base_url': getattr(args, f'{phase}_api_base_url'),
                'api_key': phase_key or shared_key,
                'api_max_concurrency': getattr(args, f'{phase}_api_max_concurrency'),
                'api_timeout': args.api_timeout,
                'api_max_retries': args.api_max_retries,
            }
        )
    return config


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
        'backend': args.init_backend,
        'init_backend': args.init_backend,
        'revision_backend': args.revision_backend,
        'init_model': args.init_model_path,
        'revision_model': args.revision_model_path,
        'input_jsonl': str(args.input_jsonl),
        'output_jsonl': str(args.output_jsonl),
        'selected_sample_count_before_shard': selected_count,
        'sample_count': shard_count,
        'processed_sample_count': 0,
        'num_inits': args.num_inits,
        'num_revisions': args.num_revisions,
        'num_shards': args.num_shards,
        'shard_index': args.shard_index,
        'batch_size': args.batch_size,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'max_new_tokens': args.max_new_tokens,
        'max_model_len': args.max_model_len,
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
        'prompt_overflow_count': 0,
        'prompt_overflow_sample_count': 0,
        'prompt_overflow_init_count': 0,
        'prompt_overflow_revision_count': 0,
        'prompt_max_observed_token_length': 0,
        'prompt_overflow_examples': [],
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
    legacy_backend = state.get('backend')
    return {
        'status': state['status'],
        'backend': legacy_backend,
        'init_backend': state.get('init_backend', legacy_backend),
        'revision_backend': state.get('revision_backend', legacy_backend),
        'init_model': state.get('init_model'),
        'revision_model': state.get('revision_model'),
        'input_jsonl': state['input_jsonl'],
        'output_jsonl': state['output_jsonl'],
        'selected_sample_count_before_shard': state['selected_sample_count_before_shard'],
        'sample_count': state['sample_count'],
        'processed_sample_count': state['processed_sample_count'],
        'num_inits': state['num_inits'],
        'num_revisions': state['num_revisions'],
        'num_shards': state['num_shards'],
        'shard_index': state['shard_index'],
        'batch_size': state['batch_size'],
        'temperature': state['temperature'],
        'top_p': state['top_p'],
        'max_new_tokens': state['max_new_tokens'],
        'max_model_len': state['max_model_len'],
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
        'prompt_overflow_count': state.get('prompt_overflow_count', 0),
        'prompt_overflow_sample_count': state.get('prompt_overflow_sample_count', 0),
        'prompt_overflow_init_count': state.get('prompt_overflow_init_count', 0),
        'prompt_overflow_revision_count': state.get('prompt_overflow_revision_count', 0),
        'prompt_max_observed_token_length': state.get('prompt_max_observed_token_length', 0),
        'prompt_overflow_examples': state.get('prompt_overflow_examples', []),
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
        record['revision_index'] = row.get('revision_index', 0)
        record['revision_prompt'] = row.get('revision_prompt')
        record['raw_y_revised'] = row.get('raw_y_revised')
        record['y_revised'] = row.get('y_revised')
        record['verifier_revised'] = row.get('verifier_revised')
    return record


def restore_init_stage_rows(path: Path, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    samples_by_id = {str(sample.get('id')): sample for sample in samples}
    restored = []
    for record in load_jsonl(path):
        sample_id = str(record.get('id'))
        sample = samples_by_id.get(sample_id)
        if sample is None:
            raise ValueError(f'Resume init row references sample outside the selected shard: {sample_id}')
        if record.get('db_id') != sample.get('db_id') or record.get('gold_sql') != sample.get('gold_sql'):
            raise ValueError(f'Resume init row identity mismatch for sample {sample_id}')
        restored.append(
            {
                'sample': sample,
                'base_prompt': record.get('base_prompt') or record.get('x'),
                'init_index': record['init_index'],
                'raw_y_init': record['raw_y_init'],
                'y_init': record['y_init'],
            }
        )
    return restored


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


def row_belongs_to_shard(global_index: int, num_shards: int, shard_index: int) -> bool:
    if num_shards < 1:
        raise ValueError('num_shards must be >= 1')
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f'shard_index must be in [0, {num_shards - 1}]')
    return global_index % num_shards == shard_index


def record_prompt_overflow(state: Dict[str, Any], sample: Dict[str, Any], prompt_token_length: int, stage: str, init_index: int | None = None) -> None:
    state['prompt_overflow_count'] += 1
    state['prompt_max_observed_token_length'] = max(state.get('prompt_max_observed_token_length', 0), prompt_token_length)
    if stage == 'init':
        state['prompt_overflow_sample_count'] += 1
        state['prompt_overflow_init_count'] += 1
    elif stage == 'revision':
        state['prompt_overflow_revision_count'] += 1
    examples = state.setdefault('prompt_overflow_examples', [])
    if len(examples) < 20:
        examples.append(
            {
                'id': sample.get('id'),
                'db_id': sample.get('db_id'),
                'stage': stage,
                'init_index': init_index,
                'prompt_token_length': prompt_token_length,
            }
        )


def generate_prompt_batch(
    generator,
    prompts: List[str],
    messages_batch: List[List[Dict[str, str]]],
    max_new_tokens: int,
    temperature: float,
    num_return_sequences: int,
    top_p: float,
) -> List[str]:
    if getattr(generator, 'uses_chat_messages', False):
        return generator.generate_messages_batch(
            messages_batch,
            max_new_tokens,
            temperature,
            num_return_sequences=num_return_sequences,
            top_p=top_p,
        )
    return generator.generate_batch(
        prompts,
        max_new_tokens,
        temperature,
        num_return_sequences=num_return_sequences,
        top_p=top_p,
    )


def generate_init_candidates(samples: List[Dict], generator, batch_size: int, max_new_tokens: int, temperature: float, top_p: float, num_inits: int, state: Dict[str, Any]) -> List[Dict]:
    safe_samples = []
    safe_prompts = []
    safe_messages = []
    for sample in samples:
        messages = build_base_sql_messages(sample)
        prompt = build_base_sql_prompt(sample, tokenizer=generator.tokenizer)
        prompt_token_length = len(generator.tokenizer(prompt, add_special_tokens=False)['input_ids'])
        state['prompt_max_observed_token_length'] = max(state.get('prompt_max_observed_token_length', 0), prompt_token_length)
        if prompt_token_length + max_new_tokens > generator.max_model_len:
            record_prompt_overflow(state, sample, prompt_token_length, stage='init')
            continue
        safe_samples.append(sample)
        safe_prompts.append(prompt)
        safe_messages.append(messages)

    raw_outputs = []
    for start in range(0, len(safe_prompts), batch_size):
        batch_prompts = safe_prompts[start:start + batch_size]
        batch_messages = safe_messages[start:start + batch_size]
        raw_outputs.extend(
            generate_prompt_batch(
                generator,
                batch_prompts,
                batch_messages,
                max_new_tokens,
                temperature,
                num_return_sequences=num_inits,
                top_p=top_p,
            )
        )

    stage_rows = []
    for sample, prompt, output_start in zip(safe_samples, safe_prompts, range(0, len(raw_outputs), num_inits)):
        for init_index, raw_y_init in enumerate(raw_outputs[output_start:output_start + num_inits]):
            y_init = normalize_sql_output(raw_y_init)
            stage_rows.append(
                {
                    'sample': sample,
                    'base_prompt': prompt,
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


def generate_revised_candidates(stage_rows: List[Dict[str, Any]], generator, batch_size: int, max_new_tokens: int, temperature: float, top_p: float, num_revisions: int, state: Dict[str, Any]) -> None:
    safe_rows = []
    revision_prompts = []
    revision_messages_batch = []
    for row in stage_rows:
        init_reward = int(row['verifier_init'].get('reward', 0))
        revision_prompt = build_revision_continuation_prompt(
            row['base_prompt'],
            row['y_init'],
            select_p_r(init_reward),
        )
        revision_messages = build_revision_chat_messages(
            row['sample'],
            row['y_init'],
            row['verifier_init'],
        )
        measured_prompt = (
            render_chat_prompt(generator.tokenizer, revision_messages)
            if getattr(generator, 'uses_chat_messages', False)
            else revision_prompt
        )
        prompt_token_length = len(generator.tokenizer(measured_prompt, add_special_tokens=False)['input_ids'])
        state['prompt_max_observed_token_length'] = max(state.get('prompt_max_observed_token_length', 0), prompt_token_length)
        if prompt_token_length + max_new_tokens > generator.max_model_len:
            record_prompt_overflow(state, row['sample'], prompt_token_length, stage='revision', init_index=row['init_index'])
            continue
        row['revision_prompt'] = revision_prompt
        safe_rows.append(row)
        revision_prompts.append(revision_prompt)
        revision_messages_batch.append(revision_messages)

    raw_revision_outputs = []
    for start in range(0, len(revision_prompts), batch_size):
        batch_prompts = revision_prompts[start:start + batch_size]
        batch_messages = revision_messages_batch[start:start + batch_size]
        raw_revision_outputs.extend(
            generate_prompt_batch(
                generator,
                batch_prompts,
                batch_messages,
                max_new_tokens,
                temperature,
                num_return_sequences=num_revisions,
                top_p=top_p,
            )
        )

    revised_rows = []
    for row, output_start in zip(safe_rows, range(0, len(raw_revision_outputs), num_revisions)):
        for revision_index, raw_y_revised in enumerate(raw_revision_outputs[output_start:output_start + num_revisions]):
            revised_row = dict(row)
            revised_row['revision_index'] = revision_index
            revised_row['raw_y_revised'] = raw_y_revised
            revised_row['y_revised'] = normalize_sql_output(raw_y_revised)
            revised_rows.append(revised_row)

    stage_rows[:] = revised_rows


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
        revision_index = row.get('revision_index', 0)
        trace['trace_id'] = f"{sample['id']}__init{row['init_index']}__revision{revision_index}"
        trace['init_index'] = row['init_index']
        trace['revision_index'] = revision_index
        traces.append(trace)
    return traces


def main() -> None:
    args = resolve_generation_args(parse_args())
    all_rows = load_jsonl(args.input_jsonl, None)
    selected_samples = select_samples(all_rows, args.max_samples, args.sampling_mode, args.seed, args.min_per_db)
    samples = select_shard(selected_samples, args.num_shards, args.shard_index)

    init_generator_config = None
    if args.resume_init_generated is None:
        init_generator_config = build_generator_config(args, 'init')
    revision_generator_config = build_generator_config(args, 'revision')

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    stage_output_paths = build_stage_output_paths(args.output_jsonl)

    state = init_summary_state(args, selected_count=len(selected_samples), shard_count=len(samples))
    init_generator = None

    print(
        f'Using init={args.init_backend}:{args.init_model_path} '
        f'revision={args.revision_backend}:{args.revision_model_path} batch_size={args.batch_size} '
        f'samples={len(samples)} num_inits={args.num_inits} num_revisions={args.num_revisions} '
        f'shard={args.shard_index}/{args.num_shards}'
    )

    try:
        if args.resume_init_generated is not None:
            state['status'] = 'loading_init_generated'
            set_process_title('p1-load-init')
            write_summary(args.summary_json, state)
            stage_rows = restore_init_stage_rows(args.resume_init_generated, samples)
            expected_count = len(samples) * args.num_inits
            if len(stage_rows) != expected_count:
                raise ValueError(
                    f'Resume init row count mismatch: expected {expected_count}, got {len(stage_rows)} '
                    f'from {args.resume_init_generated}'
                )
            print(
                json.dumps(
                    {
                        'status': state['status'],
                        'sample_count': len(samples),
                        'init_candidate_count': len(stage_rows),
                        'resume_init_generated': str(args.resume_init_generated),
                    },
                    ensure_ascii=False,
                )
            )
        else:
            state['status'] = 'generating_init'
            set_process_title('p1-gen-init')
            write_summary(args.summary_json, state)
            init_generator = load_generator(**init_generator_config)
            stage_rows = generate_init_candidates(
                samples,
                init_generator,
                args.batch_size,
                args.max_new_tokens,
                args.temperature,
                args.top_p,
                args.num_inits,
                state,
            )
            write_stage_rows(stage_output_paths['init_generated'], stage_rows)
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
        state['processed_sample_count'] = len(samples)
        write_summary(args.summary_json, state)

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
        revision_generator = (
            init_generator
            if init_generator is not None and revision_generator_config == init_generator_config
            else load_generator(**revision_generator_config)
        )
        generate_revised_candidates(
            stage_rows,
            revision_generator,
            args.batch_size,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
            args.num_revisions,
            state,
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
