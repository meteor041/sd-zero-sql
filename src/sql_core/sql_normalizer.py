import re
from typing import Optional


def extract_xml_answer(text: str) -> str:
    if "<answer>" in text and "</answer>" in text:
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()
    return text.strip()


def extract_sql_text(gen: str) -> str:
    gen = gen.replace("\r", " ")
    gen = gen.replace("</s>", "")
    gen = gen.replace("```sql", "```")

    if "</think>" in gen:
        gen = gen[gen.find("</think>") + len("</think>"):]

    gen = extract_xml_answer(gen)
    gen = gen.replace("</answer>", "")
    gen = gen.replace("<answer>", "")
    gen = gen.strip().replace("```", "")
    return gen.strip()


def first_select_sql(text: str) -> str:
    text = text.strip()
    select_matches = list(re.finditer(r"\bSELECT\b", text, flags=re.IGNORECASE))
    if not select_matches:
        return text
    sql = text[select_matches[-1].start():].strip()

    stop_patterns = [
        r"\nSystem:",
        r"\nUser:",
        r"\nAssistant:",
        r"\nHuman:",
        r"Human:",
        r"You are an expert Text-to-SQL model",
        r"Output only the final SQL query",
        r"Let me rephrase the above solution",
        r"Wait, this response is wrong\. Let me correct it",
        r"Revision cue:",
        r"Instruction:",
        r"<\|im_end\|>",
    ]
    cut_positions = []
    for pattern in stop_patterns:
        match = re.search(pattern, sql, flags=re.IGNORECASE)
        if match:
            cut_positions.append(match.start())
    if cut_positions:
        sql = sql[:min(cut_positions)].strip()

    if ";" in sql:
        sql = sql.split(";", 1)[0].strip()

    return sql


def normalize_sql_output(response: str) -> str:
    if response is None:
        return ""
    sql = extract_sql_text(response)
    if "```sql" in response or "```" in response or "<answer>" in response or "</think>" in response:
        return first_select_sql(sql)
    return first_select_sql(sql)
