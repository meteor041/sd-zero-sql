import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from accelerate import Accelerator
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PROJECT_ROOT = Path('/home/pkuccadm/huwenp/emb/lxy/sd-zero-sql')
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phase2_distill.reward_adapter import compute_sql_reward
from phase2_distill.dataset_io import DEFAULT_TRAIN_FILE, DEFAULT_VALID_FILE, dataset_to_samples, iter_sample_batches, write_distill_manifest
from phase2_distill.teacher_conditioning import build_student_prompt, build_teacher_metadata, build_teacher_prefix
from sql_core.sql_normalizer import normalize_sql_output

DEFAULT_PHASE1_STAGE2_MODEL = str(PROJECT_ROOT / 'outputs' / 'qwen3_4b_phase1_1k_tp4_full' / 'stage2')
DEFAULT_STUDENT_MODEL = DEFAULT_PHASE1_STAGE2_MODEL
DEFAULT_TEACHER_MODEL = DEFAULT_PHASE1_STAGE2_MODEL
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'sql_distill_phase2_4gpu'
DEFAULT_DEBUG_FILE = PROJECT_ROOT / 'data' / 'distill' / 'sql_distill_phase2_4gpu_debug_manifest.jsonl'
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / 'outputs' / 'sql_distill_phase2_4gpu_checkpoints'


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, rows: List[Dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict:
        return self.rows[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train a distributed SQL Phase2 distillation loop.')
    parser.add_argument('--student-model', type=str, default=DEFAULT_STUDENT_MODEL)
    parser.add_argument('--teacher-model', type=str, default=DEFAULT_TEACHER_MODEL)
    parser.add_argument('--input-jsonl', type=str, default=DEFAULT_TRAIN_FILE)
    parser.add_argument('--valid-jsonl', type=str, default=DEFAULT_VALID_FILE)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--debug-manifest', type=Path, default=DEFAULT_DEBUG_FILE)
    parser.add_argument('--checkpoint-dir', type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument('--max-samples', type=int, default=16)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--backend', type=str, default='hf', choices=['hf'])
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--rollout-batch-size', type=int, default=4)
    parser.add_argument('--num-train-epochs', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--eval-batch-size', type=int, default=4)
    parser.add_argument('--eval-max-samples', type=int, default=16)
    parser.add_argument('--eval-every-steps', type=int, default=0)
    parser.add_argument('--save-every-steps', type=int, default=0)
    parser.add_argument('--resume-from-checkpoint', type=Path, default=None)
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
    tokenizer.padding_side = 'left'
    return tokenizer


def prompt_to_messages(prompt: str) -> List[Dict[str, str]]:
    prompt = prompt.strip()
    system_text = ''
    user_text = ''
    if prompt.startswith('System:\n') and '\n\nUser:\n' in prompt:
        after_system = prompt[len('System:\n'):]
        system_text, after_user = after_system.split('\n\nUser:\n', 1)
        if '\n\nAssistant:' in after_user:
            user_text = after_user.split('\n\nAssistant:', 1)[0]
        else:
            user_text = after_user
    else:
        user_text = prompt
    messages = []
    if system_text.strip():
        messages.append({'role': 'system', 'content': system_text.strip()})
    messages.append({'role': 'user', 'content': user_text.strip()})
    return messages


def _load_base_model(model_path: str, use_4bit: bool, use_bf16: bool):
    quant_config = build_quant_config(use_4bit)
    torch_dtype = torch.bfloat16 if use_bf16 or use_4bit else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant_config,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
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
        base_model = _load_base_model(base_model_name, False, args.bf16)
        model = PeftModel.from_pretrained(base_model, args.teacher_model, is_trainable=False)
    else:
        model = _load_base_model(args.teacher_model, False, args.bf16)
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


def run_rollout(samples: List[Dict], model, tokenizer, max_new_tokens: int, temperature: float, accelerator: Accelerator) -> List[str]:
    model.eval()
    generation_model = accelerator.unwrap_model(model)
    prompts = [build_student_prompt(sample) for sample in samples]
    rendered_prompts = [
        tokenizer.apply_chat_template(prompt_to_messages(prompt), add_generation_prompt=True, tokenize=False)
        for prompt in prompts
    ]
    tokenized = tokenizer(rendered_prompts, return_tensors='pt', padding=True)
    input_ids = tokenized.input_ids.to(accelerator.device)
    attention_mask = tokenized.attention_mask.to(accelerator.device)
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
        generated = generation_model.generate(input_ids=input_ids, attention_mask=attention_mask, **generation_kwargs)
    input_len = input_ids.shape[1]
    outputs = [tokenizer.decode(sequence[input_len:], skip_special_tokens=True).strip() for sequence in generated]
    model.train()
    return outputs


def build_batch_rows(samples: List[Dict], student_outputs: List[str], teacher_model, tokenizer, max_new_tokens: int, accelerator: Accelerator, rank: int) -> List[Dict]:
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
                'rank': rank,
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
                'teacher_prompt': sequences['teacher_prompt'],
                'student_prompt': sequences['student_prompt'],
                'student_target': sequences['student_target'],
                'teacher_target': sequences['teacher_target'],
                'teacher_response_raw': None,
                'teacher_revised_sql': None,
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


def save_checkpoint(
    checkpoint_dir: Path,
    step: int,
    accelerator: Accelerator,
    student_model,
    optimizer,
    metrics: Dict,
) -> None:
    checkpoint_path = checkpoint_dir / f'checkpoint-{step}'
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    student_to_save = accelerator.unwrap_model(student_model)
    student_to_save.save_pretrained(checkpoint_path)
    torch.save(optimizer.state_dict(), checkpoint_path / 'optimizer.pt')
    with open(checkpoint_path / 'trainer_state.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def load_checkpoint(resume_from: Path, student_model, optimizer) -> Tuple[int, Dict]:
    optimizer_path = resume_from / 'optimizer.pt'
    trainer_state_path = resume_from / 'trainer_state.json'
    if optimizer_path.exists():
        optimizer.load_state_dict(torch.load(optimizer_path, map_location='cpu'))
    if trainer_state_path.exists():
        state = json.loads(trainer_state_path.read_text(encoding='utf-8'))
    else:
        state = {}
    step = int(state.get('train_steps', 0))
    return step, state


def reduce_count(accelerator: Accelerator, value: int) -> int:
    tensor = torch.tensor(value, device=accelerator.device, dtype=torch.long)
    reduced = accelerator.reduce(tensor, reduction='sum')
    return int(reduced.item())


def gather_debug_rows(debug_rows: List[Dict]) -> List[Dict]:
    if not dist.is_available() or not dist.is_initialized():
        return debug_rows
    gathered: List[List[Dict]] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, debug_rows)
    merged: List[Dict] = []
    for rows in gathered:
        merged.extend(rows or [])
    return merged


def build_rollout_dataloader(samples: List[Dict], batch_size: int) -> DataLoader:
    return DataLoader(ListDataset(samples), batch_size=batch_size, shuffle=False, collate_fn=lambda rows: list(rows))


def evaluate_phase2(
    samples: List[Dict],
    student_model,
    teacher_model,
    tokenizer,
    args: argparse.Namespace,
    accelerator: Accelerator,
) -> Dict[str, Any]:
    if not samples:
        return {
            'eval_count': 0,
            'eval_reward_1_count': 0,
            'eval_reward_0_count': 0,
            'eval_aligned_count': 0,
            'eval_alignment_skip_count': 0,
            'eval_mean_loss': None,
        }

    reward_1_local = 0
    reward_0_local = 0
    aligned_local = 0
    skipped_local = 0
    losses: List[float] = []
    eval_dataloader = build_rollout_dataloader(samples, args.eval_batch_size)

    student_model.eval()
    teacher_model.eval()
    with torch.no_grad():
        for sample_batch in eval_dataloader:
            student_outputs = run_rollout(
                sample_batch,
                student_model,
                tokenizer,
                args.max_new_tokens,
                args.temperature,
                accelerator,
            )
            batch_rows = build_batch_rows(
                sample_batch,
                student_outputs,
                teacher_model,
                tokenizer,
                args.max_new_tokens,
                accelerator,
                accelerator.process_index,
            )
            reward_1_local += sum(1 for row in batch_rows if row['reward'] == 1)
            reward_0_local += sum(1 for row in batch_rows if row['reward'] == 0)
            for eval_rows in iter_sample_batches(batch_rows, args.batch_size):
                collated, traced_rows = collate_distill_batch(tokenizer, eval_rows, args.max_length)
                skipped_local += sum(1 for row in traced_rows if not row.get('aligned', False))
                aligned_local += sum(1 for row in traced_rows if row.get('aligned', False))
                if collated is None:
                    continue

                student_input_ids = collated['student_input_ids'].to(accelerator.device)
                student_attention_mask = collated['student_attention_mask'].to(accelerator.device)
                student_target_mask = collated['student_target_mask'].to(accelerator.device)
                teacher_input_ids = collated['teacher_input_ids'].to(accelerator.device)
                teacher_attention_mask = collated['teacher_attention_mask'].to(accelerator.device)
                teacher_target_mask = collated['teacher_target_mask'].to(accelerator.device)

                student_outputs_obj = student_model(input_ids=student_input_ids, attention_mask=student_attention_mask)
                teacher_outputs_obj = teacher_model(input_ids=teacher_input_ids, attention_mask=teacher_attention_mask)

                student_logits = student_outputs_obj.logits[..., :-1, :]
                teacher_logits = teacher_outputs_obj.logits[..., :-1, :]
                student_target_mask = student_target_mask[..., 1:]
                teacher_target_mask = teacher_target_mask[..., 1:]

                student_target_logits = student_logits[student_target_mask].view(-1, student_logits.shape[-1])
                teacher_target_logits = teacher_logits[teacher_target_mask].view(-1, teacher_logits.shape[-1])
                if student_target_logits.shape[0] != teacher_target_logits.shape[0]:
                    skipped_local += 1
                    continue

                loss = compute_forward_kl(
                    student_target_logits.unsqueeze(0),
                    teacher_target_logits.unsqueeze(0).to(student_target_logits.device),
                    torch.ones((1, student_target_logits.shape[0]), dtype=torch.bool, device=student_target_logits.device),
                )
                gathered_loss = accelerator.gather(loss.detach().reshape(1))
                if accelerator.is_main_process:
                    losses.append(float(gathered_loss.mean().cpu().item()))
    student_model.train()

    metrics = {
        'eval_count': len(samples),
        'eval_reward_1_count': reduce_count(accelerator, reward_1_local),
        'eval_reward_0_count': reduce_count(accelerator, reward_0_local),
        'eval_aligned_count': reduce_count(accelerator, aligned_local),
        'eval_alignment_skip_count': reduce_count(accelerator, skipped_local),
        'eval_mean_loss': round(sum(losses) / len(losses), 6) if accelerator.is_main_process and losses else None,
    }
    return metrics


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ.get('LOCAL_RANK', '0')) if 'os' in globals() else 0
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    accelerator = Accelerator()

    if accelerator.is_main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if args.save_debug_manifest:
            args.debug_manifest.parent.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    samples = dataset_to_samples(args.input_jsonl, args.max_samples)
    valid_samples = dataset_to_samples(args.valid_jsonl, args.eval_max_samples) if args.valid_jsonl else []
    tokenizer = load_tokenizer(args.student_model)
    student_model = load_student_model(args)
    teacher_model = load_teacher_model(args)
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    rollout_dataloader = build_rollout_dataloader(samples, args.rollout_batch_size)

    student_model, optimizer, rollout_dataloader = accelerator.prepare(student_model, optimizer, rollout_dataloader)
    teacher_model.to(accelerator.device)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad_(False)
    student_model.train()

    reward_1_count_local = 0
    reward_0_count_local = 0
    alignment_skip_count_local = 0
    aligned_count_local = 0
    train_steps_local = 0
    loss_history: List[float] = []
    debug_rows: List[Dict] = []
    eval_history: List[Dict[str, Any]] = []

    if args.resume_from_checkpoint is not None:
        resumed_steps, resumed_state = load_checkpoint(args.resume_from_checkpoint, student_model, optimizer)
        train_steps_local = max(train_steps_local, resumed_steps)
        if accelerator.is_main_process and resumed_state.get('loss_history'):
            loss_history = list(resumed_state['loss_history'])
            eval_history = list(resumed_state.get('eval_history', []))

    for _ in range(args.num_train_epochs):
        for sample_batch in rollout_dataloader:
            student_outputs = run_rollout(
                sample_batch,
                student_model,
                tokenizer,
                args.max_new_tokens,
                args.temperature,
                accelerator,
            )
            batch_rows = build_batch_rows(
                sample_batch,
                student_outputs,
                teacher_model,
                tokenizer,
                args.max_new_tokens,
                accelerator,
                accelerator.process_index,
            )
            reward_1_count_local += sum(1 for row in batch_rows if row['reward'] == 1)
            reward_0_count_local += sum(1 for row in batch_rows if row['reward'] == 0)

            for train_rows in iter_sample_batches(batch_rows, args.batch_size):
                collated, traced_rows = collate_distill_batch(tokenizer, train_rows, args.max_length)
                alignment_skip_count_local += sum(1 for row in traced_rows if not row.get('aligned', False))
                aligned_count_local += sum(1 for row in traced_rows if row.get('aligned', False))
                debug_rows.extend(traced_rows)
                if collated is None:
                    continue

                student_input_ids = collated['student_input_ids'].to(accelerator.device)
                student_attention_mask = collated['student_attention_mask'].to(accelerator.device)
                student_target_mask = collated['student_target_mask'].to(accelerator.device)
                teacher_input_ids = collated['teacher_input_ids'].to(accelerator.device)
                teacher_attention_mask = collated['teacher_attention_mask'].to(accelerator.device)
                teacher_target_mask = collated['teacher_target_mask'].to(accelerator.device)

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
                    alignment_skip_count_local += 1
                    continue

                loss = compute_forward_kl(
                    student_target_logits.unsqueeze(0),
                    teacher_target_logits.unsqueeze(0).to(student_target_logits.device),
                    torch.ones((1, student_target_logits.shape[0]), dtype=torch.bool, device=student_target_logits.device),
                )
                accelerator.backward(loss)
                optimizer.step()

                train_steps_local += 1
                gathered_loss = accelerator.gather(loss.detach().reshape(1))
                if accelerator.is_main_process:
                    loss_history.append(round(float(gathered_loss.mean().cpu().item()), 6))
                if args.eval_every_steps and train_steps_local % args.eval_every_steps == 0:
                    eval_metrics = evaluate_phase2(valid_samples, student_model, teacher_model, tokenizer, args, accelerator)
                    if accelerator.is_main_process:
                        eval_metrics['train_step'] = train_steps_local
                        eval_history.append(eval_metrics)
                if args.save_every_steps and train_steps_local % args.save_every_steps == 0 and accelerator.is_main_process:
                    checkpoint_metrics = {
                        'train_steps': train_steps_local,
                        'loss_history': loss_history,
                        'eval_history': eval_history,
                    }
                    save_checkpoint(args.checkpoint_dir, train_steps_local, accelerator, student_model, optimizer, checkpoint_metrics)

    accelerator.wait_for_everyone()

    if valid_samples:
        final_eval = evaluate_phase2(valid_samples, student_model, teacher_model, tokenizer, args, accelerator)
        if accelerator.is_main_process:
            final_eval['train_step'] = train_steps_local
            eval_history.append(final_eval)

    summary = {
        'count': len(samples),
        'student_model': args.student_model,
        'teacher_model': args.teacher_model,
        'reward_1_count': reduce_count(accelerator, reward_1_count_local),
        'reward_0_count': reduce_count(accelerator, reward_0_count_local),
        'alignment_skip_count': reduce_count(accelerator, alignment_skip_count_local),
        'aligned_count': reduce_count(accelerator, aligned_count_local),
        'train_steps': train_steps_local,
        'loss_history': loss_history if accelerator.is_main_process else [],
        'eval_history': eval_history if accelerator.is_main_process else [],
        'output_dir': str(args.output_dir),
        'debug_manifest': str(args.debug_manifest) if args.save_debug_manifest else None,
        'checkpoint_dir': str(args.checkpoint_dir),
        'world_size': accelerator.num_processes,
    }

    if args.save_debug_manifest:
        debug_rows = gather_debug_rows(debug_rows)

    if accelerator.is_main_process:
        student_to_save = accelerator.unwrap_model(student_model)
        student_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        if args.save_debug_manifest:
            write_distill_manifest(args.debug_manifest, debug_rows)
        save_train_metrics(args.output_dir, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    accelerator.wait_for_everyone()


if __name__ == '__main__':
    main()
