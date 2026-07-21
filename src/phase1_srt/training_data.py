from typing import Any, Dict, Iterable, List, Tuple


class OverlengthCompletionExample(ValueError):
    pass


def tokenize_completion_example(
    example: Dict[str, Any],
    tokenizer,
    max_length: int,
) -> Dict[str, Any]:
    prompt = str(example.get("prompt") or "")
    completion = str(example.get("completion") or "").strip()
    if not prompt:
        raise ValueError("Training example has an empty prompt.")
    if not completion:
        raise ValueError("Training example has an empty completion.")

    eos_token = tokenizer.eos_token or ""
    if eos_token and not completion.endswith(eos_token):
        completion += eos_token

    prompt_encoding = tokenizer(prompt, add_special_tokens=False)
    completion_encoding = tokenizer(completion, add_special_tokens=False)
    prompt_ids = list(prompt_encoding["input_ids"])
    completion_ids = list(completion_encoding["input_ids"])
    input_ids = prompt_ids + completion_ids
    if len(input_ids) > max_length:
        identity = example.get("trace_id") or example.get("id") or "unknown"
        raise OverlengthCompletionExample(
            f"Example {identity} has {len(input_ids)} tokens, exceeding max_length={max_length}. "
            "The sample must be dropped explicitly or regenerated with a larger context window."
        )

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + completion_ids,
        "prompt_token_length": len(prompt_ids),
        "completion_token_length": len(completion_ids),
    }


def tokenize_completion_rows(
    examples: Iterable[Dict[str, Any]],
    tokenizer,
    max_length: int,
    overlength_policy: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if overlength_policy not in {"error", "drop"}:
        raise ValueError("overlength_policy must be 'error' or 'drop'")
    encoded_rows: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    prompt_tokens = 0
    completion_tokens = 0
    input_count = 0

    for index, example in enumerate(examples):
        input_count += 1
        try:
            encoded = tokenize_completion_example(example, tokenizer, max_length)
        except OverlengthCompletionExample as exc:
            dropped.append(
                {
                    "index": index,
                    "id": example.get("id"),
                    "trace_id": example.get("trace_id"),
                    "task": example.get("task"),
                    "error": str(exc),
                }
            )
            continue
        prompt_tokens += encoded.pop("prompt_token_length")
        completion_tokens += encoded.pop("completion_token_length")
        encoded_rows.append(encoded)

    if dropped and overlength_policy == "error":
        preview = "\n".join(item["error"] for item in dropped[:5])
        raise ValueError(f"Found {len(dropped)} overlength examples:\n{preview}")
    if not encoded_rows:
        raise ValueError("No examples remain after tokenization.")

    supervised_ratio = completion_tokens / (prompt_tokens + completion_tokens)
    stats = {
        "input_examples": input_count,
        "kept_examples": len(encoded_rows),
        "dropped_overlength_examples": len(dropped),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "supervised_token_ratio": round(supervised_ratio, 6),
        "dropped_examples": dropped[:100],
    }
    return encoded_rows, stats
