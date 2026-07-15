from typing import Dict

SYSTEM_PROMPT = (
    "You are an expert Text-to-SQL model. Given a database schema, evidence, "
    "and a natural language question, generate a valid SQL query. Output only the final SQL query. "
    "Do not include explanations, markdown, code fences, or reasoning. The output must start with SELECT."
)


def build_base_sql_prompt(sample: Dict) -> str:
    schema = (sample.get("schema") or "").strip()
    evidence = (sample.get("evidence") or "").strip()
    question = (sample.get("question") or "").strip()
    return f"""System:
{SYSTEM_PROMPT}

User:
Database schema:
{schema}

Evidence:
{evidence}

Question:
{question}

Assistant:
"""


def build_revision_instruction(reward: int) -> str:
    if reward == 1:
        return (
            "The previous SQL is correct. Rewrite it into a clean, canonical, and semantically "
            "equivalent SQL query. Only output the final SQL query."
        )
    return (
        "The previous SQL is incorrect. Based on the database schema, the draft SQL, and the "
        "execution feedback, generate a corrected SQL query. Only output the final SQL query."
    )


def build_revision_prompt(sample: Dict, y_init: str, verifier_result: Dict) -> str:
    schema = (sample.get("schema") or "").strip()
    evidence = (sample.get("evidence") or "").strip()
    question = (sample.get("question") or "").strip()
    reward = int(verifier_result.get("reward", 0))
    error_type = verifier_result.get("error_type", "unknown")
    instruction = build_revision_instruction(reward)
    return f"""System:
{SYSTEM_PROMPT}

User:
Database schema:
{schema}

Evidence:
{evidence}

Question:
{question}

Draft SQL:
{y_init}

Verifier reward:
{reward}

Error type:
{error_type}

Revision instruction:
{instruction}

Assistant:
"""
