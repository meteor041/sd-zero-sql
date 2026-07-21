import unittest

from src.phase1_srt.constants import P_R_INCORRECT
from src.phase2_distill.teacher_conditioning import build_student_prompt, build_teacher_prefix


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        if enable_thinking:
            raise AssertionError("Thinking must be disabled")
        return f"CHAT:{messages[1]['content']}\nASSISTANT:"


class Phase2ConditioningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sample = {
            "schema": "CREATE TABLE items(id INTEGER);",
            "evidence": "",
            "question": "List ids",
        }

    def test_student_and_teacher_use_the_same_rendered_prompt(self) -> None:
        tokenizer = FakeTokenizer()
        student_prompt = build_student_prompt(self.sample, tokenizer=tokenizer)
        teacher_prompt = build_teacher_prefix(
            self.sample,
            "SELECT wrong FROM items",
            0,
            {"reward": 0, "error_type": "runtime_error"},
            tokenizer=tokenizer,
        )
        self.assertEqual(
            teacher_prompt,
            f"{student_prompt}SELECT wrong FROM items\n\n{P_R_INCORRECT}\n\n",
        )
        self.assertNotIn("Student SQL:", teacher_prompt)
        self.assertNotIn("Verifier reward:", teacher_prompt)


if __name__ == "__main__":
    unittest.main()
