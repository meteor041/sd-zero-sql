import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.sql_core.sql_normalizer import normalize_sql_output
from src.sql_core.sql_verifier import results_equal, verify_sql


class SqlNormalizerTests(unittest.TestCase):
    def test_preserves_nested_select(self) -> None:
        sql = "SELECT id FROM t WHERE id IN (SELECT id FROM u);"
        self.assertEqual(normalize_sql_output(sql), sql[:-1])

    def test_preserves_cte(self) -> None:
        sql = "WITH picked AS (SELECT id FROM u) SELECT id FROM picked;"
        self.assertEqual(normalize_sql_output(sql), sql[:-1])

    def test_ignores_semicolon_inside_string(self) -> None:
        sql = "SELECT value FROM t WHERE value = 'a;b'; trailing text"
        self.assertEqual(normalize_sql_output(sql), "SELECT value FROM t WHERE value = 'a;b'")

    def test_extracts_sql_after_thinking(self) -> None:
        response = "<think>reasoning with select words</think>\n```sql\nSELECT id FROM t;\n```"
        self.assertEqual(normalize_sql_output(response), "SELECT id FROM t")


class SqlVerifierTests(unittest.TestCase):
    def test_unordered_results_compare_as_multisets(self) -> None:
        self.assertTrue(results_equal([(1,), (2,)], [(2,), (1,)], order_matters=False))
        self.assertFalse(results_equal([(1,), (1,)], [(1,)], order_matters=False))

    def test_top_level_order_by_remains_order_sensitive(self) -> None:
        self.assertFalse(results_equal([(1,), (2,)], [(2,), (1,)], order_matters=True))

    def test_verify_sql_ignores_unspecified_row_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "fixture.sqlite"
            connection = sqlite3.connect(db_path)
            connection.executescript(
                "CREATE TABLE items(id INTEGER);"
                "INSERT INTO items(id) VALUES (1), (2), (3);"
            )
            connection.close()

            with patch("src.sql_core.sql_verifier.resolve_db_path", return_value=db_path):
                result = verify_sql(
                    "fixture",
                    "SELECT id FROM items ORDER BY id DESC",
                    "SELECT id FROM items",
                )
            self.assertEqual(result["reward"], 1)
            self.assertFalse(result["order_matters"])

    def test_verify_sql_honors_gold_order_by(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "fixture.sqlite"
            connection = sqlite3.connect(db_path)
            connection.executescript(
                "CREATE TABLE items(id INTEGER);"
                "INSERT INTO items(id) VALUES (1), (2), (3);"
            )
            connection.close()

            with patch("src.sql_core.sql_verifier.resolve_db_path", return_value=db_path):
                result = verify_sql(
                    "fixture",
                    "SELECT id FROM items ORDER BY id DESC",
                    "SELECT id FROM items ORDER BY id ASC",
                )
            self.assertEqual(result["reward"], 0)
            self.assertTrue(result["order_matters"])


if __name__ == "__main__":
    unittest.main()
