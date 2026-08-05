import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cscsql.utils.infer_utils import run_eval_major_vote
from sql_core.sql_normalizer import normalize_sql_output

CONTROL_CUES = (
    "Wait, this response is wrong. Let me correct it.",
    "Let me rephrase the above solution.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract final Phase1 revisions and evaluate execution accuracy.")
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--gold-file", required=True)
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--num-cpus", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=30)
    return parser.parse_args()


def extract_revision(response: str) -> str:
    positions = [(response.rfind(cue), cue) for cue in CONTROL_CUES if cue in response]
    if positions:
        _, cue = max(positions)
        response = response.rsplit(cue, 1)[1]
    return normalize_sql_output(response)


def main() -> None:
    args = parse_args()
    rows = json.loads(args.input_json.read_text(encoding="utf-8"))
    cue_count = 0
    candidate_count = 0
    for row in rows:
        responses = row["responses"]
        candidate_count += len(responses)
        cue_count += sum(any(cue in response for cue in CONTROL_CUES) for response in responses)
        row["pred_sqls"] = [extract_revision(response) for response in responses]

    output_json = args.input_json.with_name(f"{args.input_json.stem}_revision_extracted.json")
    output_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    config = {
        "source": str(args.input_json),
        "postprocess": "last_phase1_control_cue_then_normalize_sql_output",
        "candidate_count": candidate_count,
        "candidate_with_control_cue_count": cue_count,
        "num_cpus": args.num_cpus,
        "timeout_seconds": args.timeout,
    }
    run_eval_major_vote(
        gold_file=args.gold_file,
        pred_file=str(output_json),
        db_path=args.db_path,
        num_cpus=args.num_cpus,
        timeout=args.timeout,
        config=config,
    )
    metric_file = output_json.with_name(f"{output_json.stem}_metric.json")
    if not metric_file.exists():
        raise RuntimeError(f"Missing revision metric: {metric_file}")
    print(json.dumps({"output_json": str(output_json), "metric_file": str(metric_file), **config}, ensure_ascii=False))


if __name__ == "__main__":
    main()
