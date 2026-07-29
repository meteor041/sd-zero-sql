import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT_ROOT / "scripts" / "sft" / "run_sft_paper_full_4gpu.sh"


class PaperFullSftLauncherTests(unittest.TestCase):
    def test_launcher_pins_table_4_hyperparameters(self) -> None:
        script = LAUNCHER.read_text(encoding="utf-8")
        expected_fragments = [
            "--max-length 32768",
            "--num-train-epochs 3",
            "--learning-rate 5e-6",
            "--weight-decay 1e-4",
            "--warmup-ratio 0.05",
            "--lr-scheduler-type cosine",
            "--per-device-train-batch-size 1",
            "--per-device-eval-batch-size 1",
            "--gradient-accumulation-steps 1",
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

    def test_launcher_requires_four_training_processes(self) -> None:
        script = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn('NUM_PROCESSES="${NUM_PROCESSES:-4}"', script)
        self.assertIn('if [[ "${NUM_PROCESSES}" != "4" ]]', script)

    def test_launcher_enables_progress_and_wandb_reporting(self) -> None:
        script = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn('REPORT_TO="${REPORT_TO:-wandb}"', script)
        self.assertIn('WANDB_PROJECT="${WANDB_PROJECT:-sd-zero-sql}"', script)
        self.assertIn('WANDB_MODE="${WANDB_MODE:-online}"', script)
        self.assertIn('--report-to "${REPORT_TO}"', script)
        self.assertIn('--run-name "${WANDB_RUN_NAME}"', script)
        self.assertNotIn("--disable-progress-bar", script)


if __name__ == "__main__":
    unittest.main()
