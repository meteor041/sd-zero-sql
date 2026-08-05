import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT_ROOT / "scripts" / "srt" / "run_phase1_paper_full_any4.sh"
TRAINER = PROJECT_ROOT / "scripts" / "srt" / "train_srt_stage.py"


class PaperFullPhase1LauncherTests(unittest.TestCase):
    def test_launcher_pins_base_model_and_existing_data(self) -> None:
        script = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn('BASE_MODEL="/data/model/Qwen3-4B-Instruct-2507"', script)
        self.assertIn("phase1_dual_api_2init3rev_srt_train.jsonl", script)
        self.assertIn("phase1_dual_api_2init3rev_srt_valid.jsonl", script)
        self.assertNotIn("--adapter-path", script)
        self.assertNotIn("SFT_OUTPUT", script)

    def test_launcher_pins_table_4_hyperparameters(self) -> None:
        script = LAUNCHER.read_text(encoding="utf-8")
        expected_fragments = [
            "--max-length 32768",
            "--overlength-policy error",
            "--num-train-epochs 3",
            "--learning-rate 5e-6",
            "--weight-decay 1e-4",
            "--adam-beta1 0.9",
            "--adam-beta2 0.95",
            "--optim adamw_torch",
            "--warmup-ratio 0.05",
            "--lr-scheduler-type cosine",
            "--per-device-train-batch-size 1",
            "--per-device-eval-batch-size 1",
            "--gradient-accumulation-steps 1",
            "--sync-each-batch",
            "--full-finetune",
            "--gradient-checkpointing",
            "--use-liger-kernel",
            '--fsdp "full_shard auto_wrap"',
            "--fsdp-transformer-layer-cls-to-wrap Qwen3DecoderLayer",
            "--bf16",
        ]
        for fragment in expected_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, script)

    def test_launcher_requires_four_processes_and_new_output(self) -> None:
        script = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("NUM_PROCESSES=4", script)
        self.assertIn("GPU_COUNT=4", script)
        self.assertIn("base_paper_full_32k", script)
        self.assertIn("MIN_FREE_DISK_GB", script)
        self.assertIn("FREE_STABILITY_SECONDS", script)

    def test_trainer_validates_full_fsdp_and_exposes_paper_controls(self) -> None:
        trainer = TRAINER.read_text(encoding="utf-8")
        expected_fragments = [
            '"--sync-each-batch"',
            '"--use-liger-kernel"',
            '"--fsdp"',
            '"--fsdp-transformer-layer-cls-to-wrap"',
            'raise ValueError("--fsdp requires --full-finetune")',
            '"activation_checkpointing"',
            "use_liger_kernel=args.use_liger_kernel",
            'fsdp=args.fsdp if use_fsdp else ""',
        ]
        for fragment in expected_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, trainer)


if __name__ == "__main__":
    unittest.main()
