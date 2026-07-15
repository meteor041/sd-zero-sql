from typing import Dict

from sql_core.prompt_builders import build_base_sql_prompt, build_revision_prompt
from phase1_srt.constants import select_p_r


def build_student_prompt(sample: Dict) -> str:
    return sample.get('x') or build_base_sql_prompt(sample)


def build_feedback_block(verifier_result: Dict, reward: int) -> str:
    error_type = verifier_result.get('error_type', 'unknown')
    error_message = verifier_result.get('error_message', '')
    pred_result_preview = verifier_result.get('pred_result_preview', [])
    gold_result_preview = verifier_result.get('gold_result_preview', [])
    return (
        f"Verifier reward:\n{int(reward)}\n\n"
        f"Error type:\n{error_type}\n\n"
        f"Error message:\n{error_message}\n\n"
        f"Predicted result preview:\n{pred_result_preview}\n\n"
        f"Gold result preview:\n{gold_result_preview}\n\n"
    )


def build_teacher_prefix(sample: Dict, student_response: str, reward: int, verifier_result: Dict) -> str:
    student_prompt = build_student_prompt(sample)
    p_r = select_p_r(reward)
    return (
        f"{student_prompt}"
        f"Student SQL:\n{student_response}\n\n"
        f"Revision cue:\n{p_r}\n\n"
        f"Instruction:\nReturn only the final corrected SQL query. The output must start with SELECT.\n\n"
        f"Assistant:\n"
    )


def build_teacher_prompt(sample: Dict, student_response: str, reward: int, verifier_result: Dict) -> str:
    return build_teacher_prefix(sample, student_response, reward, verifier_result)


def build_teacher_metadata(sample: Dict, student_response: str, reward: int, verifier_result: Dict) -> Dict:
    p_r = select_p_r(reward)
    return {
        'id': sample.get('id'),
        'db_id': sample.get('db_id'),
        'reward': int(reward),
        'p_r': p_r,
        'student_response': student_response,
        'student_prompt': build_student_prompt(sample),
        'feedback_block': build_feedback_block(verifier_result, reward),
        'teacher_prefix': build_teacher_prefix(sample, student_response, reward, verifier_result),
        'teacher_prompt': build_teacher_prompt(sample, student_response, reward, verifier_result),
    }
