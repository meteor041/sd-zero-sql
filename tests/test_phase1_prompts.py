import json
import tempfile
import unittest
from pathlib import Path

from scripts.srt.build_two_stage_data import (
    generation_task_record,
    prepare_srt_multitask_data,
    revision_task_record,
)
from scripts.srt.generate_phase1_traces import generate_revised_candidates
from src.phase1_srt.constants import P_R_INCORRECT
from src.phase1_srt.training_data import (
    OverlengthCompletionExample,
    tokenize_completion_example,
    tokenize_completion_rows,
)
from src.sql_core.prompt_builders import build_base_sql_prompt, build_revision_prompt


class FakeTokenizer:
    eos_token = "<eos>"

    def __init__(self) -> None:
        self.enable_thinking = None

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        self.enable_thinking = enable_thinking
        self.assertions = (tokenize, add_generation_prompt)
        return (
            f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n"
            f"<|im_start|>user\n{messages[1]['content']}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(text)}


class FakeGenerator:
    max_model_len = 8192

    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()

    def generate_batch(
        self,
        prompts,
        max_new_tokens,
        temperature,
        num_return_sequences=1,
        top_p=1.0,
    ):
        return [
            f"SELECT {revision_index}"
            for _prompt in prompts
            for revision_index in range(num_return_sequences)
        ]


def trace_record(identifier: str, init_correct: bool, revision_index: int = 0):
    reward = 1 if init_correct else 0
    return {
        "id": identifier,
        "db_id": "fixture",
        "x": f"prompt-{identifier}\n",
        "gold_sql": "SELECT 1",
        "y_init": "SELECT 1" if init_correct else "SELECT 0",
        "y_init_correct": init_correct,
        "p_r": "Let me rephrase the above solution." if init_correct else P_R_INCORRECT,
        "y_revised": f"SELECT {revision_index + 1}",
        "y_revised_correct": True,
        "keep": True,
        "verifier_init": {"reward": reward},
        "verifier_revised": {"reward": 1},
        "revision_index": revision_index,
    }


class Phase1PromptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sample = {
            "schema": "CREATE TABLE items(id INTEGER);",
            "evidence": "id is the identifier",
            "question": "List identifiers",
        }

    def test_chat_template_explicitly_disables_thinking(self) -> None:
        tokenizer = FakeTokenizer()
        prompt = build_base_sql_prompt(self.sample, tokenizer=tokenizer)
        self.assertFalse(tokenizer.enable_thinking)
        self.assertEqual(tokenizer.assertions, (False, True))
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))

    def test_revision_sampling_matches_revision_training_prompt(self) -> None:
        tokenizer = FakeTokenizer()
        base_prompt = build_base_sql_prompt(self.sample, tokenizer=tokenizer)
        y_init = "SELECT id FROM missing_table"
        verifier_result = {"reward": 0, "error_type": "runtime_error"}
        sampled_prompt = build_revision_prompt(
            self.sample,
            y_init,
            verifier_result,
            tokenizer=tokenizer,
        )
        training_prompt = revision_task_record(
            {
                **trace_record("fixture", False),
                "x": base_prompt,
                "y_init": y_init,
                "y_revised": "SELECT id FROM items",
            }
        )["prompt"]
        self.assertEqual(sampled_prompt, training_prompt)
        self.assertNotIn("Draft SQL:", sampled_prompt)
        self.assertNotIn("Verifier reward:", sampled_prompt)

    def test_each_initial_response_expands_to_three_revisions(self) -> None:
        stage_rows = [
            {
                "sample": self.sample,
                "base_prompt": "unused",
                "init_index": 0,
                "y_init": "SELECT -1",
                "verifier_init": {"reward": 0},
            }
        ]
        generate_revised_candidates(
            stage_rows,
            generator=FakeGenerator(),
            batch_size=1,
            max_new_tokens=256,
            temperature=0.7,
            top_p=1.0,
            num_revisions=3,
            state={
                "prompt_max_observed_token_length": 0,
                "prompt_overflow_count": 0,
                "prompt_overflow_sample_count": 0,
                "prompt_overflow_init_count": 0,
                "prompt_overflow_revision_count": 0,
                "prompt_overflow_examples": [],
            },
        )
        self.assertEqual([row["revision_index"] for row in stage_rows], [0, 1, 2])
        self.assertEqual([row["y_revised"] for row in stage_rows], ["SELECT 0", "SELECT 1", "SELECT 2"])


class Phase1TrainingDataTests(unittest.TestCase):
    def test_generation_and_revision_tasks_are_both_emitted(self) -> None:
        record = trace_record("q1", False)
        generation = generation_task_record(record)
        revision = revision_task_record(record)
        self.assertEqual(generation["task"], "generation")
        self.assertEqual(revision["task"], "revision")
        self.assertIn(record["y_init"], generation["completion"])
        self.assertEqual(revision["completion"], record["y_revised"])

    def test_completion_only_labels_mask_every_prompt_token(self) -> None:
        tokenizer = FakeTokenizer()
        encoded = tokenize_completion_example(
            {"trace_id": "q1", "prompt": "PROMPT", "completion": "SQL"},
            tokenizer,
            max_length=32,
        )
        self.assertEqual(encoded["labels"][:6], [-100] * 6)
        self.assertEqual(encoded["labels"][6:], list("SQL<eos>"))
        self.assertGreater(encoded["completion_token_length"], 0)

    def test_overlength_examples_are_never_silently_truncated(self) -> None:
        with self.assertRaises(OverlengthCompletionExample):
            tokenize_completion_example(
                {"trace_id": "q1", "prompt": "12345", "completion": "67890"},
                FakeTokenizer(),
                max_length=5,
            )

    def test_drop_policy_reports_every_overlength_example(self) -> None:
        rows, stats = tokenize_completion_rows(
            [
                {"id": "short", "prompt": "p", "completion": "s"},
                {"id": "long", "prompt": "12345", "completion": "67890"},
            ],
            FakeTokenizer(),
            max_length=12,
            overlength_policy="drop",
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(stats["dropped_overlength_examples"], 1)
        self.assertEqual(stats["dropped_examples"][0]["id"], "long")

    def test_multitask_builder_splits_by_question_and_balances_outcomes(self) -> None:
        records = []
        for revision_index in range(3):
            records.append(trace_record("wrong", False, revision_index))
            records.append(trace_record("correct", True, revision_index))
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "traces.jsonl"
            with open(input_path, "w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")
            train_path, valid_path, summary = prepare_srt_multitask_data(
                input_path,
                root,
                seed=7,
                prefix="fixture",
                validation_fraction=0.5,
                max_traces_per_question=3,
                max_correct_init_ratio=0.5,
            )
            train_rows = [json.loads(line) for line in train_path.read_text().splitlines()]
            valid_rows = [json.loads(line) for line in valid_path.read_text().splitlines()]
        self.assertEqual(summary["correct_init_ratio"], 0.5)
        self.assertEqual(summary["generation_task_count"], summary["revision_task_count"])
        self.assertTrue({row["id"] for row in train_rows}.isdisjoint({row["id"] for row in valid_rows}))

    def test_question_identity_includes_database(self) -> None:
        first = trace_record("shared-id", False)
        second = trace_record("shared-id", False)
        second["db_id"] = "another-db"
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "traces.jsonl"
            with open(input_path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(first) + "\n")
                handle.write(json.dumps(second) + "\n")
            _, _, summary = prepare_srt_multitask_data(
                input_path,
                root,
                seed=7,
                prefix="fixture",
                validation_fraction=0,
                max_traces_per_question=3,
                max_correct_init_ratio=0,
            )
        self.assertEqual(summary["unique_question_count"], 2)


if __name__ == "__main__":
    unittest.main()
