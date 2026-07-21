from typing import Any, Dict


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
