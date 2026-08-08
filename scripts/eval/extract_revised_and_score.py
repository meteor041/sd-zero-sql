import json, re, sys
from pathlib import Path
from cscsql.utils.infer_utils import run_eval_major_vote
from sql_core.sql_normalizer import normalize_sql_output

PHRASES = [
    "Let me rephrase the above solution.",
    "Wait, this response is wrong. Let me correct it.",
]

def extract_revised(text):
    text = str(text or "").strip()
    text = re.sub(r"</?think>", "", text)
    # 找最后一个 p_r 短语,取其后的文本(revision SQL)
    last = -1
    for p in PHRASES:
        idx = text.rfind(p)
        if idx > last:
            last = idx + len(p)
    if last > 0:
        text = text[last:].lstrip()
    return normalize_sql_output(text)

def main():
    raw_path, out_path, gold_file, db_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    rows = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    with_phrase = 0
    for row in rows:
        values = row.get("responses") or row.get("pred_sqls") or []
        stripped = []
        for v in values:
            s = str(v)
            if any(p in s for p in PHRASES):
                with_phrase += 1
            stripped.append(extract_revised(v))
        row["responses"] = stripped
        row["pred_sqls"] = stripped
    Path(out_path).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"candidates_with_phrase={with_phrase}")
    config = {
        "source": str(raw_path),
        "postprocess": "extract_last_sql_after_pr_phrase_then_normalize",
        "candidate_count": sum(len(r.get("pred_sqls", [])) for r in rows),
        "candidate_with_pr_phrase_count": with_phrase,
        "num_cpus": 128,
        "timeout_seconds": 30,
    }
    run_eval_major_vote(gold_file, out_path, db_path, num_cpus=128, timeout=30,
                        pred_sql_key="pred_sqls", config=config)

if __name__ == "__main__":
    main()
