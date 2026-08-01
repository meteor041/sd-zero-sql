from typing import Any, Dict, List

from phase1_srt.constants import select_p_r

SYSTEM_PROMPT = (
    "You are an expert Text-to-SQL model. Given a database schema, evidence, "
    "and a natural language question, generate a valid SQL query. Output only the final SQL query. "
    "Do not include explanations, markdown, code fences, or reasoning. The output must start with SELECT or WITH."
)


def build_base_sql_messages(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    schema = (sample.get("schema") or "").strip()
    evidence = (sample.get("evidence") or "").strip()
    question = (sample.get("question") or "").strip()
    user_content = f"""Database schema:
{schema}

Evidence:
{evidence}

Question:
{question}"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def render_chat_prompt(tokenizer, messages: List[Dict[str, str]]) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def build_base_sql_prompt(sample: Dict[str, Any], tokenizer=None) -> str:
    messages = build_base_sql_messages(sample)
    if tokenizer is not None:
        return render_chat_prompt(tokenizer, messages)
    return f"""System:
{messages[0]['content']}

User:
{messages[1]['content']}

Assistant:
"""


def build_revision_continuation_prompt(base_prompt: str, y_init: str, p_r: str) -> str:
    return f"{base_prompt}{y_init.strip()}\n\n{p_r}\n\n"


def build_revision_chat_messages(
    sample: Dict[str, Any],
    y_init: str,
    verifier_result: Dict[str, Any],
) -> List[Dict[str, str]]:
    reward = int(verifier_result.get("reward", 0))
    p_r = select_p_r(reward)
    messages = build_base_sql_messages(sample)
    messages.extend(
        [
            {"role": "assistant", "content": y_init.strip()},
            {
                "role": "user",
                "content": (
                    "Revise the previous SQL according to this self-revision cue:\n"
                    f"{p_r}\n\n"
                    "Output only the revised SQL query. Do not include explanations, markdown, "
                    "code fences, or reasoning. The output must start with SELECT or WITH."
                ),
            },
        ]
    )
    return messages


def build_revision_prompt(sample: Dict, y_init: str, verifier_result: Dict, tokenizer=None) -> str:
    reward = int(verifier_result.get("reward", 0))
    base_prompt = build_base_sql_prompt(sample, tokenizer=tokenizer)
    return build_revision_continuation_prompt(base_prompt, y_init, select_p_r(reward))
