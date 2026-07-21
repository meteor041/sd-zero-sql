import sqlite3
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

DB_ROOT = Path("/data/huwenp/emb/data/ches")
TRAIN_DB_ROOT = DB_ROOT / "train_databases"
DEV_DB_ROOT = DB_ROOT / "dev_databases"
TEST_DB_ROOT = DB_ROOT / "test_databases"
DEFAULT_SQL_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RESULT_ROWS = 50000
FETCH_BATCH_SIZE = 1000
SQLITE_PROGRESS_OPS = 10000


@lru_cache(maxsize=None)
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


def execute_sql(
    db_path: Path,
    sql: str,
    timeout_seconds: float = DEFAULT_SQL_TIMEOUT_SECONDS,
    max_result_rows: int = DEFAULT_MAX_RESULT_ROWS,
) -> Dict[str, Any]:
    conn = None
    timed_out = False
    rows: List[Tuple[Any, ...]] = []
    preview_rows: List[Tuple[Any, ...]] = []
    deadline = time.monotonic() + timeout_seconds

    def progress_handler() -> int:
        nonlocal timed_out
        if time.monotonic() >= deadline:
            timed_out = True
            return 1
        return 0

    try:
        db_uri = db_path.resolve().as_uri() + "?mode=ro"
        conn = sqlite3.connect(db_uri, timeout=timeout_seconds, uri=True)
        conn.execute(f"PRAGMA busy_timeout = {int(timeout_seconds * 1000)}")
        conn.execute("PRAGMA query_only = ON")
        conn.set_progress_handler(progress_handler, SQLITE_PROGRESS_OPS)
        cursor = conn.cursor()
        cursor.execute(sql)

        while True:
            batch = cursor.fetchmany(FETCH_BATCH_SIZE)
            if not batch:
                break
            if len(preview_rows) < 5:
                preview_rows.extend(batch[: 5 - len(preview_rows)])
            rows.extend(batch)
            if len(rows) > max_result_rows:
                return {
                    "ok": False,
                    "rows": [],
                    "preview": normalize_rows(preview_rows),
                    "error": f"Result set exceeded limit of {max_result_rows} rows.",
                    "error_code": "result_too_large",
                }
            if time.monotonic() >= deadline:
                timed_out = True
                raise sqlite3.OperationalError("interrupted")

        return {
            "ok": True,
            "rows": rows,
            "preview": normalize_rows(preview_rows or rows),
            "error": "",
            "error_code": "",
        }
    except sqlite3.OperationalError as exc:
        if timed_out:
            return {
                "ok": False,
                "rows": [],
                "preview": normalize_rows(preview_rows),
                "error": f"SQL execution exceeded timeout of {timeout_seconds:.1f}s.",
                "error_code": "timeout",
            }
        return {
            "ok": False,
            "rows": [],
            "preview": normalize_rows(preview_rows),
            "error": str(exc),
            "error_code": "runtime_error",
        }
    except Exception as exc:
        return {
            "ok": False,
            "rows": [],
            "preview": normalize_rows(preview_rows),
            "error": str(exc),
            "error_code": "runtime_error",
        }
    finally:
        if conn is not None:
            try:
                conn.set_progress_handler(None, 0)
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass


def prepare_gold_execution(db_id: str, gold_sql: str) -> Dict[str, Any]:
    db_path = resolve_db_path(db_id)
    gold = execute_sql(db_path, gold_sql)
    return {
        "db_id": db_id,
        "db_path": db_path,
        "gold_sql": gold_sql,
        "gold_ok": gold["ok"],
        "gold_rows": gold["rows"],
        "gold_result_preview": gold["preview"],
        "gold_error": gold["error"],
        "gold_error_code": gold.get("error_code", ""),
    }


def _has_top_level_order_by(sql: str) -> bool:
    depth = 0
    quote = None
    bracket_quote = False
    tokens: List[str] = []
    current: List[str] = []

    def flush_token() -> None:
        if current:
            tokens.append("".join(current).upper())
            current.clear()

    index = 0
    while index < len(sql):
        char = sql[index]
        next_char = sql[index + 1] if index + 1 < len(sql) else ""
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
            flush_token()
            quote = char
        elif char == "[":
            flush_token()
            bracket_quote = True
        elif char == "-" and next_char == "-":
            flush_token()
            newline = sql.find("\n", index + 2)
            index = len(sql) if newline == -1 else newline + 1
            continue
        elif char == "/" and next_char == "*":
            flush_token()
            comment_end = sql.find("*/", index + 2)
            index = len(sql) if comment_end == -1 else comment_end + 2
            continue
        elif char == "(":
            flush_token()
            depth += 1
        elif char == ")":
            flush_token()
            depth = max(0, depth - 1)
        elif depth == 0 and (char.isalnum() or char == "_"):
            current.append(char)
        else:
            flush_token()
        index += 1
    flush_token()
    return any(tokens[i:i + 2] == ["ORDER", "BY"] for i in range(len(tokens) - 1))


def results_equal(
    predicted_rows: List[Tuple[Any, ...]],
    gold_rows: List[Tuple[Any, ...]],
    *,
    order_matters: bool,
) -> bool:
    if order_matters:
        return predicted_rows == gold_rows
    return Counter(predicted_rows) == Counter(gold_rows)


def verify_sql_against_gold(prepared_gold: Dict[str, Any], predicted_sql: str) -> Dict[str, Any]:
    db_path = prepared_gold["db_path"]
    pred = execute_sql(db_path, predicted_sql)

    if not pred["ok"]:
        error_code = pred.get("error_code", "runtime_error")
        error_type = {
            "timeout": "timeout",
            "result_too_large": "result_too_large",
        }.get(error_code, "runtime_error")
        return {
            "reward": 0,
            "is_executable": False,
            "execution_match": False,
            "error_type": error_type,
            "error_message": pred["error"],
            "pred_result_preview": pred["preview"],
            "gold_result_preview": prepared_gold["gold_result_preview"] if prepared_gold["gold_ok"] else [],
            "db_path": str(db_path),
        }

    if not prepared_gold["gold_ok"]:
        error_code = prepared_gold.get("gold_error_code", "runtime_error")
        error_type = {
            "timeout": "gold_timeout",
            "result_too_large": "gold_result_too_large",
        }.get(error_code, "gold_runtime_error")
        return {
            "reward": 0,
            "is_executable": True,
            "execution_match": False,
            "error_type": error_type,
            "error_message": prepared_gold["gold_error"],
            "pred_result_preview": pred["preview"],
            "gold_result_preview": [],
            "db_path": str(db_path),
        }

    pred_rows = pred["rows"]
    gold_rows = prepared_gold["gold_rows"]
    order_matters = _has_top_level_order_by(prepared_gold["gold_sql"])
    execution_match = results_equal(pred_rows, gold_rows, order_matters=order_matters)

    return {
        "reward": 1 if execution_match else 0,
        "is_executable": True,
        "execution_match": execution_match,
        "order_matters": order_matters,
        "error_type": "correct" if execution_match else "wrong_result",
        "error_message": "" if execution_match else "Execution result does not match gold SQL result.",
        "pred_result_preview": pred["preview"],
        "gold_result_preview": prepared_gold["gold_result_preview"],
        "db_path": str(db_path),
    }


def verify_sql(db_id: str, predicted_sql: str, gold_sql: str) -> Dict[str, Any]:
    prepared_gold = prepare_gold_execution(db_id, gold_sql)
    return verify_sql_against_gold(prepared_gold, predicted_sql)
