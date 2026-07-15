import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path('/home/pkuccadm/huwenp/emb/lxy/sd-zero-sql')
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sql_core.prompt_builders import build_base_sql_prompt
from sql_core.sql_normalizer import normalize_sql_output

DEFAULT_BASE_MODEL = '/data/model/Qwen3-4B-Instruct-2507'
DEFAULT_PHASE1_STAGE2 = str(PROJECT_ROOT / 'outputs' / 'qwen3_4b_phase1_1k_tp4_full' / 'stage2')
DEFAULT_INPUT_JSONL = PROJECT_ROOT / 'data' / 'ches_train_sft_train_4k.jsonl'
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / 'data' / 'eval' / 'student_rollout_compare.json'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compare x->SQL rollout cleanliness for base vs Phase1 Stage2 models.')
    parser.add_argument('--base-model', type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument('--phase1-stage2-model', type=str, default=DEFAULT_PHASE1_STAGE2)
    parser.add_argument('--input-jsonl', type=Path, default=DEFAULT_INPUT_JSONL)
    parser.add_argument('--output-json', type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument('--max-samples', type=int, default=8)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.0)
    return parser.parse_args()


def load_samples(path: Path, max_samples: int) -> List[Dict]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= max_samples:
                break
    return rows


def load_model_and_tokenizer(model_path: str):
    if Path(model_path, 'adapter_config.json').exists():
        adapter_config = json.loads(Path(model_path, 'adapter_config.json').read_text(encoding='utf-8'))
        base_model_name = adapter_config['base_model_name_or_path']
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16, trust_remote_code=True)
        model = PeftModel.from_pretrained(base_model, model_path, is_trainable=False)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'
    model.eval()
    model.config.use_cache = False
    return model, tokenizer


def rollout(model, tokenizer, prompts: List[str], max_new_tokens: int, temperature: float) -> List[str]:
    tokenized = tokenizer(prompts, return_tensors='pt', padding=True)
    device = next(model.parameters()).device
    input_ids = tokenized.input_ids.to(device)
    attention_mask = tokenized.attention_mask.to(device)
    kwargs = {
        'max_new_tokens': max_new_tokens,
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token_id': tokenizer.eos_token_id,
    }
    if temperature > 0:
        kwargs.update({'do_sample': True, 'temperature': temperature})
    else:
        kwargs.update({'do_sample': False})
    with torch.no_grad():
        generated = model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    input_len = input_ids.shape[1]
    return [tokenizer.decode(seq[input_len:], skip_special_tokens=True).strip() for seq in generated]


def classify_sql(sql: str) -> Dict[str, bool]:
    upper = sql.upper()
    return {
        'has_select': 'SELECT' in upper,
        'has_join': 'JOIN' in upper,
        'has_where': 'WHERE' in upper,
        'ends_with_open_quote': sql.count("'") % 2 == 1,
        'ends_with_operator': sql.rstrip().endswith(('=', '>', '<', ',', 'AND', 'OR')),
        'very_short': len(sql.split()) < 6,
    }


def main() -> None:
    args = parse_args()
    samples = load_samples(args.input_jsonl, args.max_samples)
    prompts = [build_base_sql_prompt(sample) for sample in samples]

    base_model, base_tok = load_model_and_tokenizer(args.base_model)
    stage2_model, stage2_tok = load_model_and_tokenizer(args.phase1_stage2_model)

    base_outputs = rollout(base_model, base_tok, prompts, args.max_new_tokens, args.temperature)
    stage2_outputs = rollout(stage2_model, stage2_tok, prompts, args.max_new_tokens, args.temperature)

    rows = []
    for sample, prompt, base_raw, stage2_raw in zip(samples, prompts, base_outputs, stage2_outputs):
        base_sql = normalize_sql_output(base_raw)
        stage2_sql = normalize_sql_output(stage2_raw)
        rows.append(
            {
                'id': sample.get('id'),
                'db_id': sample.get('db_id'),
                'question': sample.get('question'),
                'prompt': prompt,
                'base_raw': base_raw,
                'base_sql': base_sql,
                'base_flags': classify_sql(base_sql),
                'stage2_raw': stage2_raw,
                'stage2_sql': stage2_sql,
                'stage2_flags': classify_sql(stage2_sql),
            }
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote {len(rows)} comparisons to {args.output_json}')


if __name__ == '__main__':
    main()
