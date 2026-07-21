import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sql_core.prompt_builders import build_base_sql_prompt
from sql_core.sql_normalizer import normalize_sql_output

DEFAULT_BASE_MODEL = "/data/model/Qwen3-4B-Instruct-2507"
DEFAULT_SRT_MODEL = "/data/huwenp/emb/lxy/sd-zero-sql/outputs/qwen3_4b_srt_joint/merged"
DEFAULT_INPUT_JSONL = Path("/home/pkuccadm/huwenp/emb/lxy/M-Schema/ches_train_sft_valid.jsonl")
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "data" / "eval" / "student_rollout_compare.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare base and merged SRT SQL rollouts.")
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--srt-model", type=str, default=DEFAULT_SRT_MODEL)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def load_samples(path: Path, max_samples: int) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= max_samples:
                break
    return rows


def load_model_and_tokenizer(model_path: str):
    if Path(model_path, "adapter_config.json").exists():
        adapter_config = json.loads(Path(model_path, "adapter_config.json").read_text(encoding="utf-8"))
        base_model = AutoModelForCausalLM.from_pretrained(
            adapter_config["base_model_name_or_path"],
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, model_path, is_trainable=False)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model.eval()
    model.config.use_cache = True
    return model, tokenizer


def rollout(model, tokenizer, samples: List[Dict], max_new_tokens: int, temperature: float) -> List[str]:
    prompts = [build_base_sql_prompt(sample, tokenizer=tokenizer) for sample in samples]
    tokenized = tokenizer(prompts, return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generation_kwargs.update({"do_sample": True, "temperature": temperature})
    else:
        generation_kwargs.update({"do_sample": False})
    with torch.no_grad():
        generated = model.generate(
            input_ids=tokenized.input_ids.to(device),
            attention_mask=tokenized.attention_mask.to(device),
            **generation_kwargs,
        )
    input_length = tokenized.input_ids.shape[1]
    return [tokenizer.decode(row[input_length:], skip_special_tokens=True).strip() for row in generated]


def classify_sql(sql: str) -> Dict[str, bool]:
    upper = sql.upper()
    return {
        "has_query": "SELECT" in upper or upper.startswith("WITH"),
        "has_join": "JOIN" in upper,
        "has_where": "WHERE" in upper,
        "ends_with_open_quote": sql.count("'") % 2 == 1,
        "ends_with_operator": sql.rstrip().endswith(("=", ">", "<", ",", "AND", "OR")),
        "very_short": len(sql.split()) < 6,
    }


def main() -> None:
    args = parse_args()
    samples = load_samples(args.input_jsonl, args.max_samples)
    base_model, base_tokenizer = load_model_and_tokenizer(args.base_model)
    base_outputs = rollout(base_model, base_tokenizer, samples, args.max_new_tokens, args.temperature)
    del base_model
    torch.cuda.empty_cache()

    srt_model, srt_tokenizer = load_model_and_tokenizer(args.srt_model)
    srt_outputs = rollout(srt_model, srt_tokenizer, samples, args.max_new_tokens, args.temperature)
    rows = []
    for sample, base_raw, srt_raw in zip(samples, base_outputs, srt_outputs):
        base_sql = normalize_sql_output(base_raw)
        srt_sql = normalize_sql_output(srt_raw)
        rows.append(
            {
                "id": sample.get("id"),
                "db_id": sample.get("db_id"),
                "question": sample.get("question"),
                "base_raw": base_raw,
                "base_sql": base_sql,
                "base_flags": classify_sql(base_sql),
                "srt_raw": srt_raw,
                "srt_sql": srt_sql,
                "srt_flags": classify_sql(srt_sql),
            }
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} comparisons to {args.output_json}")


if __name__ == "__main__":
    main()
