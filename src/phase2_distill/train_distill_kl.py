import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PROJECT_ROOT = Path('/home/pkuccadm/huwenp/emb/lxy/ches_sql_sft')
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phase2_distill.reward_adapter import compute_sql_reward
from phase2_distill.dataset_io import DEFAULT_TRAIN_FILE, dataset_to_samples, iter_sample_batches, write_distill_manifest
from phase2_distill.teacher_conditioning import build_student_prompt, build_teacher_metadata, build_teacher_prefix

DEFAULT_STUDENT_MODEL = '/data/huwenp/emb/lxy/ches_sql_sft/outputs/qwen3_4b_srt_two_stage/stage2'
DEFAULT_TEACHER_MODEL = DEFAULT_STUDENT_MODEL
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'sql_distill_smoke'
DEFAULT_DEBUG_FILE = PROJECT_ROOT / 'data' / 'distill' / 'sql_distill_debug_manifest.jsonl'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train a minimal SQL Phase2 distillation loop.')
    parser.add_argument('--student-model', type=str, default=DEFAULT_STUDENT_MODEL)
    parser.add_argument('--teacher-model', type=str, default=DEFAULT_TEACHER_MODEL)
    parser.add_argument('--input-jsonl', type=str, default=DEFAULT_TRAIN_FILE)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--debug-manifest', type=Path, default=DEFAULT_DEBUG_FILE)
    parser.add_argument('--max-samples', type=int, default=16)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--backend', type=str, default='hf', choices=['hf'])
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--rollout-batch-size', type=int, default=4)
    parser.add_argument('--num-train-epochs', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--max-length', type=int, default=4096)
    parser.add_argument('--lora-r', type=int, default=16)
    parser.add_argument('--lora-alpha', type=int, default=32)
    parser.add_argument('--lora-dropout', type=float, default=0.05)
    parser.add_argument('--use-4bit', action='store_true')
    parser.add_argument('--gradient-checkpointing', action='store_true')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--tensor-parallel-size', type=int, default=1)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9)
    parser.add_argument('--save-debug-manifest', action='store_true')
    return parser.parse_args()


def build_quant_config(use_4bit: bool):
    if not use_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )


def load_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'
    return tokenizer


def _load_base_model(model_path: str, use_4bit: bool, use_bf16: bool, *, device_map=None):
    quant_config = build_quant_config(use_4bit)
    torch_dtype = torch.bfloat16 if use_bf16 or use_4bit else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant_config,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map=device_map,
    )
    model.config.use_cache = False
    return model


def _is_peft_checkpoint(model_path: str) -> bool:
    return Path(model_path, 'adapter_config.json').exists()


def load_student_model(args: argparse.Namespace):
    if _is_peft_checkpoint(args.student_model):
        base_model_name = json.loads(Path(args.student_model, 'adapter_config.json').read_text(encoding='utf-8')).get('base_model_name_or_path')
        base_model = _load_base_model(base_model_name, args.use_4bit, args.bf16)
        model = PeftModel.from_pretrained(base_model, args.student_model, is_trainable=True)
    else:
        base_model = _load_base_model(args.student_model, args.use_4bit, args.bf16)
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias='none',
            task_type='CAUSAL_LM',
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        )
        model = get_peft_model(base_model, peft_config)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, 'enable_input_require_grads'):
            model.enable_input_require_grads()
    return model


def load_teacher_model(args: argparse.Namespace):
    if _is_peft_checkpoint(args.teacher_model):
        base_model_name = json.loads(Path(args.teacher_model, 'adapter_config.json').read_text(encoding='utf-8')).get('base_model_name_or_path')
        base_model = _load_base_model(base_model_name, False, args.bf16, device_map='auto')
        model = PeftModel.from_pretrained(base_model, args.teacher_model, is_trainable=False)
    else:
        model = _load_base_model(args.teacher_model, False, args.bf16, device_map='auto')
    model.eval()
    model.config.use_cache = False
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def build_sequences(sample: Dict, student_response_sql: str, reward: int, verifier_result: Dict) -> Dict[str, str]:
    student_prompt = build_student_prompt(sample)
    teacher_prefix = build_teacher_prefix(sample, student_response_sql, reward, verifier_result)
    return {
        'student_prompt': student_prompt,
        'student_target': student_response_sql,
        'teacher_prompt': teacher_prefix,
        'teacher_target': student_response_sql,
    }


def encode_with_target(tokenizer, prompt: str, target: str, max_length: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    target_ids = tokenizer(target, add_special_tokens=False).input_ids
    if len(target_ids) == 0:
        raise ValueError('Target tokenization is empty.')
    if len(target_ids) > max_length:
        raise ValueError('Target tokenization exceeds max_length.')

    prompt_keep = max_length - len(target_ids)
    if prompt_keep < len(prompt_ids):
        prompt_ids = prompt_ids[-prompt_keep:] if prompt_keep > 0 else []

    full_ids = torch.tensor(prompt_ids + target_ids, dtype=torch.long)
    attention_mask = torch.ones_like(full_ids, dtype=torch.long)
    target_slice = torch.zeros_like(full_ids, dtype=torch.bool)
    target_slice[len(prompt_ids):] = True
    return full_ids, attention_mask, target_slice


def pad_tensor_list(tensors: List[torch.Tensor], pad_value: int) -> torch.Tensor:
    max_len = max(tensor.shape[0] for tensor in tensors)
    padded = []
    for tensor in tensors:
        pad_len = max_len - tensor.shape[0]
        if pad_len > 0:
            tensor = F.pad(tensor, (0, pad_len), value=pad_value)
        padded.append(tensor)
    return torch.stack(padded, dim=0)


def collate_distill_batch(tokenizer, rows: List[Dict], max_length: int) -> Tuple[Optional[Dict[str, torch.Tensor]], List[Dict]]:
    student_ids = []
    student_masks = []
    student_target_masks = []
    teacher_ids = []
    teacher_masks = []
    teacher_target_masks = []
    aligned_rows = []

    for row in rows:
        try:
            student_input_ids, student_attention_mask, student_target_mask = encode_with_target(
                tokenizer,
                row['student_prompt'],
                row['student_target'],
                max_length,
            )
            teacher_input_ids, teacher_attention_mask, teacher_target_mask = encode_with_target(
                tokenizer,
                row['teacher_prompt'],
                row['teacher_target'],
                max_length,
            )
        except ValueError:
            row['aligned'] = False
            row['alignment_reason'] = 'truncated'
            continue

        student_target_ids = student_input_ids[student_target_mask]
        teacher_target_ids = teacher_input_ids[teacher_target_mask]
        if student_target_ids.shape[0] == 0 or teacher_target_ids.shape[0] == 0:
            row['aligned'] = False
            row['alignment_reason'] = 'empty_target'
            continue
        if student_target_ids.shape[0] != teacher_target_ids.shape[0] or not torch.equal(student_target_ids, teacher_target_ids):
            row['aligned'] = False
            row['alignment_reason'] = 'token_mismatch'
            continue

        row['aligned'] = True
        row['alignment_reason'] = None
        row['target_token_count'] = int(student_target_ids.shape[0])
        aligned_rows.append(row)
        student_ids.append(student_input_ids)
        student_masks.append(student_attention_mask)
        student_target_masks.append(student_target_mask)
        teacher_ids.append(teacher_input_ids)
        teacher_masks.append(teacher_attention_mask)
        teacher_target_masks.append(teacher_target_mask)

    if not aligned_rows:
        return None, rows

    batch = {
        'student_input_ids': pad_tensor_list(student_ids, tokenizer.pad_token_id),
        'student_attention_mask': pad_tensor_list(student_masks, 0),
        'student_target_mask': pad_tensor_list(student_target_masks, 0).bool(),
        'teacher_input_ids': pad_tensor_list(teacher_ids, tokenizer.pad_token_id),
        'teacher_attention_mask': pad_tensor_list(teacher_masks, 0),
        'teacher_target_mask': pad_tensor_list(teacher_target_masks, 0).bool(),
    }
    return batch, rows


def run_rollout(samples: List[Dict], model, tokenizer, batch_size: int, max_new_tokens: int, temperature: float) -> List[str]:
    model.eval()
    prompts = [build_student_prompt(sample) for sample in samples]
    outputs = []
    device = next(model.parameters()).device
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]
        tokenized = tokenizer(batch_prompts, return_tensors='pt', padding=True)
        input_ids = tokenized.input_ids.to(device)
        attention_mask = tokenized.attention_mask.to(device)
        generation_kwargs = {
            'max_new_tokens': max_new_tokens,
            'pad_token_id': tokenizer.pad_token_id,
            'eos_token_id': tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation_kwargs.update({'do_sample': True, 'temperature': temperature})
        else:
            generation_kwargs.update({'do_sample': False})
        with torch.no_grad():
            generated = model.generate(input_ids=input_ids, attention_mask=attention_mask, **generation_kwargs)
        input_len = input_ids.shape[1]
        for sequence in generated:
            outputs.append(tokenizer.decode(sequence[input_len:], skip_special_tokens=True).strip())
    model.train()
    return outputs


def build_batch_rows(samples: List[Dict], student_outputs: List[str]) -> List[Dict]:
    rows = []
    for sample, student_response in zip(samples, student_outputs):
        reward_info = compute_sql_reward(sample, student_response)
        teacher_meta = build_teacher_metadata(
            sample,
            reward_info['normalized_sql'],
            reward_info['reward'],
            reward_info['verifier_result'],
        )
        sequences = build_sequences(
            sample,
            reward_info['normalized_sql'],
            reward_info['reward'],
            reward_info['verifier_result'],
        )
        rows.append(
            {
                'id': sample.get('id'),
                'db_id': sample.get('db_id'),
                'question': sample.get('question'),
                'gold_sql': sample.get('gold_sql'),
                'student_response_raw': student_response,
                'student_response_sql': reward_info['normalized_sql'],
                'reward': reward_info['reward'],
                'verifier_result': reward_info['verifier_result'],
                'p_r': teacher_meta['p_r'],
                'feedback_block': teacher_meta['feedback_block'],
                'student_prompt': sequences['student_prompt'],
                'student_target': sequences['student_target'],
                'teacher_prompt': sequences['teacher_prompt'],
                'teacher_target': sequences['teacher_target'],
            }
        )
    return rows


def compute_forward_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    token_kl = F.kl_div(student_log_probs, teacher_probs, reduction='none').sum(dim=-1)
    masked = token_kl[target_mask]
    if masked.numel() == 0:
        raise ValueError('No valid tokens available for KL computation.')
    return masked.mean()


def save_train_metrics(output_dir: Path, metrics: Dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / 'train_metrics.json'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_debug_manifest:
        args.debug_manifest.parent.mkdir(parents=True, exist_ok=True)

    samples = dataset_to_samples(args.input_jsonl, args.max_samples)
    tokenizer = load_tokenizer(args.student_model)
    student_model = load_student_model(args)
    teacher_model = load_teacher_model(args)
    student_model.train()

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    metrics = {
        'count': len(samples),
        'student_model': args.student_model,
        'teacher_model': args.teacher_model,
        'reward_1_count': 0,
        'reward_0_count': 0,
        'alignment_skip_count': 0,
        'aligned_count': 0,
        'train_steps': 0,
        'loss_history': [],
    }
    debug_rows = []

    student_device = next(student_model.parameters()).device
    teacher_device = next(teacher_model.parameters()).device

    for _ in range(args.num_train_epochs):
        for sample_batch in iter_sample_batches(samples, args.rollout_batch_size):
            student_outputs = run_rollout(
                sample_batch,
                student_model,
                tokenizer,
                args.rollout_batch_size,
                args.max_new_tokens,
                args.temperature,
            )
            batch_rows = build_batch_rows(sample_batch, student_outputs)
            metrics['reward_1_count'] += sum(1 for row in batch_rows if row['reward'] == 1)
            metrics['reward_0_count'] += sum(1 for row in batch_rows if row['reward'] == 0)

            for train_rows in iter_sample_batches(batch_rows, args.batch_size):
                collated, traced_rows = collate_distill_batch(tokenizer, train_rows, args.max_length)
                metrics['alignment_skip_count'] += sum(1 for row in traced_rows if not row.get('aligned', False))
                metrics['aligned_count'] += sum(1 for row in traced_rows if row.get('aligned', False))
                debug_rows.extend(traced_rows)
                if collated is None:
                    continue

                student_input_ids = collated['student_input_ids'].to(student_device)
                student_attention_mask = collated['student_attention_mask'].to(student_device)
                student_target_mask = collated['student_target_mask'].to(student_device)
                teacher_input_ids = collated['teacher_input_ids'].to(teacher_device)
                teacher_attention_mask = collated['teacher_attention_mask'].to(teacher_device)
                teacher_target_mask = collated['teacher_target_mask'].to(teacher_device)

                optimizer.zero_grad()
                student_outputs_obj = student_model(input_ids=student_input_ids, attention_mask=student_attention_mask)
                with torch.no_grad():
                    teacher_outputs_obj = teacher_model(input_ids=teacher_input_ids, attention_mask=teacher_attention_mask)

                student_logits = student_outputs_obj.logits[..., :-1, :]
                teacher_logits = teacher_outputs_obj.logits[..., :-1, :]
                student_target_mask = student_target_mask[..., 1:]
                teacher_target_mask = teacher_target_mask[..., 1:]

                student_target_logits = student_logits[student_target_mask].view(-1, student_logits.shape[-1])
                teacher_target_logits = teacher_logits[teacher_target_mask].view(-1, teacher_logits.shape[-1])
                if student_target_logits.shape[0] != teacher_target_logits.shape[0]:
                    metrics['alignment_skip_count'] += int(student_target_logits.shape[0] != teacher_target_logits.shape[0])
                    continue

                loss = compute_forward_kl(
                    student_target_logits.unsqueeze(0),
                    teacher_target_logits.unsqueeze(0).to(student_target_logits.device),
                    torch.ones((1, student_target_logits.shape[0]), dtype=torch.bool, device=student_target_logits.device),
                )
                loss.backward()
                optimizer.step()

                metrics['train_steps'] += 1
                metrics['loss_history'].append(round(float(loss.detach().cpu().item()), 6))

    student_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.save_debug_manifest:
        write_distill_manifest(args.debug_manifest, debug_rows)

    summary = {
        **metrics,
        'output_dir': str(args.output_dir),
        'debug_manifest': str(args.debug_manifest) if args.save_debug_manifest else None,
    }
    save_train_metrics(args.output_dir, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
