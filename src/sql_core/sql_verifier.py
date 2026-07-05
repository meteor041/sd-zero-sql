import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple

DB_ROOT = Path("/data/huwenp/emb/data/ches")
TRAIN_DB_ROOT = DB_ROOT / "train_databases"
DEV_DB_ROOT = DB_ROOT / "dev_databases"
TEST_DB_ROOT = DB_ROOT / "test_databases"


def resolve_db_path(db_id: str) -> Path:
    candidates = [
        TRAIN_DB_ROOT / db_id / f"{db_id}.sqlite",
        DEV_DB_ROOT / db_id / f"{db_id}.sqlite",
        TEST_DB_ROOT / db_id / f"{db_id}.sqlite",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not resolve sqlite database for db_id={db_id}")


def normalize_rows(rows: List[Tuple[Any, ...]], limit: int = 5) -> List[List[str]]:
    preview = []
    for row in rows[:limit]:
        preview.append(["NULL" if cell is None else str(cell) for cell in row])
    return preview


def execute_sql(db_path: Path, sql: str) -> Dict[str, Any]:
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        conn.close()
        return {
            "ok": True,
            "rows": rows,
            "preview": normalize_rows(rows),
            "error": "",
        }
    except Exception as exc:
        try:
            conn.close()
        except Exception:
            pass
        return {
            "ok": False,
            "rows": [],
            "preview": [],
            "error": str(exc),
        }


def verify_sql(db_id: str, predicted_sql: str, gold_sql: str) -> Dict[str, Any]:
    db_path = resolve_db_path(db_id)
    pred = execute_sql(db_path, predicted_sql)
    gold = execute_sql(db_path, gold_sql)

    if not pred["ok"]:
        return {
            "reward": 0,
            "is_executable": False,
            "execution_match": False,
            "error_type": "runtime_error",
            "error_message": pred["error"],
            "pred_result_preview": pred["preview"],
            "gold_result_preview": gold["preview"] if gold["ok"] else [],
            "db_path": str(db_path),
        }

    if not gold["ok"]:
        return {
            "reward": 0,
            "is_executable": True,
            "execution_match": False,
            "error_type": "gold_runtime_error",
            "error_message": gold["error"],
            "pred_result_preview": pred["preview"],
            "gold_result_preview": [],
            "db_path": str(db_path),
        }

    pred_rows = pred["rows"]
    gold_rows = gold["rows"]
    execution_match = pred_rows == gold_rows

    return {
        "reward": 1 if execution_match else 0,
        "is_executable": True,
        "execution_match": execution_match,
        "error_type": "correct" if execution_match else "wrong_result",
        "error_message": "" if execution_match else "Execution result does not match gold SQL result.",
        "pred_result_preview": pred["preview"],
        "gold_result_preview": gold["preview"],
        "db_path": str(db_path),
    }
