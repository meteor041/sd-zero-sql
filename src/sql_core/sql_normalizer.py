import re


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


def _first_sql_keyword(text: str) -> int | None:
    match = re.search(r"\b(?:WITH|SELECT)\b", text, flags=re.IGNORECASE)
    return match.start() if match else None


def _statement_end(text: str) -> int:
    quote = None
    bracket_quote = False
    index = 0
    while index < len(text):
        char = text[index]
        next_char = text[index + 1] if index + 1 < len(text) else ""

        if bracket_quote:
            if char == "]":
                bracket_quote = False
            index += 1
            continue

        if quote:
            if char == quote:
                if next_char == quote and quote in {"'", '"'}:
                    index += 2
                    continue
                quote = None
            index += 1
            continue

        if char in {"'", '"', "`"}:
            quote = char
        elif char == "[":
            bracket_quote = True
        elif char == "-" and next_char == "-":
            newline = text.find("\n", index + 2)
            index = len(text) if newline == -1 else newline + 1
            continue
        elif char == "/" and next_char == "*":
            comment_end = text.find("*/", index + 2)
            index = len(text) if comment_end == -1 else comment_end + 2
            continue
        elif char == ";":
            return index
        index += 1
    return len(text)


def first_sql_statement(text: str) -> str:
    text = text.strip()
    keyword_start = _first_sql_keyword(text)
    if keyword_start is None:
        return text
    sql = text[keyword_start:].strip()

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

    return sql[:_statement_end(sql)].strip()


def first_select_sql(text: str) -> str:
    """Backward-compatible alias for callers using the old function name."""
    return first_sql_statement(text)


def normalize_sql_output(response: str) -> str:
    if response is None:
        return ""
    sql = extract_sql_text(response)
    if "```sql" in response or "```" in response or "<answer>" in response or "</think>" in response:
        return first_sql_statement(sql)
    return first_sql_statement(sql)
