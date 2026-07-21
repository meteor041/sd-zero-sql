import unittest

from scripts.srt.build_two_stage_data import _stage2_record
from scripts.srt.generate_phase1_traces import generate_revised_candidates
from src.phase1_srt.constants import P_R_INCORRECT
from src.sql_core.prompt_builders import build_base_sql_prompt, build_revision_prompt


class FakeTokenizer:
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
        return {"input_ids": text.split()}


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

    def test_revision_sampling_matches_stage2_training_prompt(self) -> None:
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
        training_prompt = _stage2_record(
            {
                "id": "fixture",
                "db_id": "fixture",
                "x": base_prompt,
                "y_init": y_init,
                "p_r": P_R_INCORRECT,
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


if __name__ == "__main__":
    unittest.main()
