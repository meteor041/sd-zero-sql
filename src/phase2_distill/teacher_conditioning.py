from typing import Dict

from phase1_srt.constants import select_p_r
from sql_core.prompt_builders import build_base_sql_prompt, build_revision_continuation_prompt


def build_student_prompt(sample: Dict, tokenizer=None) -> str:
    existing = sample.get("x")
    if existing and (tokenizer is None or "<|im_start|>" in existing):
        return existing
    return build_base_sql_prompt(sample, tokenizer=tokenizer)


def build_feedback_block(verifier_result: Dict, reward: int) -> str:
    return (
        f"Verifier reward: {int(reward)}\n"
        f"Error type: {verifier_result.get('error_type', 'unknown')}\n"
        f"Error message: {verifier_result.get('error_message', '')}"
    )


def build_teacher_prefix(
    sample: Dict,
    student_response: str,
    reward: int,
    verifier_result: Dict,
    tokenizer=None,
) -> str:
    del verifier_result
    student_prompt = build_student_prompt(sample, tokenizer=tokenizer)
    return build_revision_continuation_prompt(
        student_prompt,
        student_response,
        select_p_r(reward),
    )


def build_teacher_prompt(
    sample: Dict,
    student_response: str,
    reward: int,
    verifier_result: Dict,
    tokenizer=None,
) -> str:
    return build_teacher_prefix(
        sample,
        student_response,
        reward,
        verifier_result,
        tokenizer=tokenizer,
    )


def build_teacher_metadata(
    sample: Dict,
    student_response: str,
    reward: int,
    verifier_result: Dict,
    tokenizer=None,
) -> Dict:
    p_r = select_p_r(reward)
    student_prompt = build_student_prompt(sample, tokenizer=tokenizer)
    teacher_prefix = build_teacher_prefix(
        sample,
        student_response,
        reward,
        verifier_result,
        tokenizer=tokenizer,
    )
    return {
        "id": sample.get("id"),
        "db_id": sample.get("db_id"),
        "reward": int(reward),
        "p_r": p_r,
        "student_response": student_response,
        "student_prompt": student_prompt,
        "feedback_block": build_feedback_block(verifier_result, reward),
        "teacher_prefix": teacher_prefix,
        "teacher_prompt": teacher_prefix,
    }
