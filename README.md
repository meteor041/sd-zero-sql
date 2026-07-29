# SD-Zero SQL

This repository adapts Self-Distillation Zero to CHES/BIRD Text-to-SQL. The active pipeline is deliberately narrower than the historical scripts: one SQL-SFT initialization, one joint Phase1 SRT run, and one optional Phase2 self-distillation run.

## Correctness invariants

- Qwen prompts are rendered with the model chat template and `enable_thinking=False`.
- SQL-SFT and Phase1 train on completion tokens only. Schema and question tokens always use label `-100`.
- Overlength examples are never silently truncated. The default policy is to fail with trace identifiers.
- Phase1 samples one initial response and three revisions per question at temperature `0.7`.
- A revision is conditioned on the exact assistant continuation used during training: `x + y_init + P_r`.
- Only execution-correct revisions enter SRT data.
- Every retained trace contributes both generation and revision examples in the same training run.
- SQL extraction preserves outer queries, nested `SELECT` statements, and `WITH` clauses.
- LoRA outputs are merged before vLLM generation or BIRD evaluation.

## Main components

- `src/sql_core/prompt_builders.py`: shared chat prompts and revision continuation.
- `src/sql_core/sql_normalizer.py`: SQL extraction without destroying nested queries.
- `src/sql_core/sql_verifier.py`: read-only SQLite execution and order-aware result comparison.
- `scripts/sft/train_sft.py`: completion-only SQL initialization.
- `scripts/srt/generate_phase1_traces.py`: `1 init x 3 revisions` trace collection.
- `scripts/srt/build_phase1_multitask_data.py`: joint generation/revision dataset construction.
- `scripts/srt/train_srt_stage.py`: completion-only joint SRT training.
- `scripts/srt/merge_lora_adapter.py`: standalone checkpoint creation.
- `src/phase2_distill/train_distill_kl.py`: optional on-policy self-distillation.

## Required external data

The launchers currently assume the server paths below. Override every path through the corresponding environment variable when the layout differs.

- Base model: `/data/model/Qwen3-4B-Instruct-2507`
- CHES train/valid JSONL: `/home/pkuccadm/huwenp/emb/lxy/M-Schema/`
- CHES SQLite databases: `/data/huwenp/emb/data/ches/`
- External BIRD evaluator: `/home/pkuccadm/huwenp/emb/lxy/csc_sql/`

Each training JSONL row must contain `id`, `db_id`, `question`, `schema`, `evidence`, and a SQL target in `response`, `completion`, or `gold_sql`.

## Workflow

### 1. Train the SQL generator

```bash
bash scripts/sft/run_sft_8xa100.sh
```

Despite the historical filename, the launcher defaults to four GPUs so the effective global batch is four. It writes:

- LoRA adapter: `/data/huwenp/emb/lxy/sd-zero-sql/outputs/qwen3_4b_sft_lora_8k`
- Standalone model: `/data/huwenp/emb/lxy/sd-zero-sql/outputs/qwen3_4b_sft_merged_8k`

Evaluate this standalone model before collecting SRT traces. SD-Zero assumes a competent generator; a near-zero SQL baseline cannot produce useful self-corrections.

For a full-parameter SFT run aligned with Table 4 of the paper, install `liger-kernel` and run:

```bash
GPU_SET=0,1,2,3 bash scripts/sft/run_sft_paper_full_4gpu.sh
```

This separate launcher uses four-way FSDP `full_shard auto_wrap`, wraps
`Qwen3DecoderLayer`, enables Liger and gradient checkpointing, trains with completion-only
loss at sequence length 32768, and uses the paper's `5e-6` learning rate, global batch size
4, cosine schedule, 5% warmup, and three epochs. Its output is already a standalone model;
do not run the LoRA merge script on it. Override `MODEL_PATH`, `TRAIN_FILE`, `VALID_FILE`,
and `OUTPUT_DIR` when the server layout differs.

The launcher shows the Hugging Face tqdm progress bar on the main FSDP process and reports
training/evaluation metrics to Weights & Biases by default. Log in once on the training server
with `wandb login`, then optionally configure the run:

```bash
WANDB_PROJECT=sd-zero-sql \
WANDB_ENTITY=your-team \
WANDB_RUN_NAME=qwen3-4b-full-sft-v1 \
WANDB_MODE=online \
bash scripts/sft/run_sft_paper_full_4gpu.sh
```

Use `WANDB_MODE=offline` without network access, or `REPORT_TO=none` to disable wandb.
The launcher disables parameter/gradient watching and checkpoint uploads by default to avoid
the substantial overhead of logging a full-parameter 4B model.

### 2. Collect Phase1 traces

```bash
bash scripts/srt/run_generate_traces_vllm_4gpu.sh
```

The launcher rejects adapter-only model directories. The default output is `traces_train_full_1init_3revision.jsonl` plus a summary and intermediate generation/verification files.

Inspect at least these summary fields before training:

- `init_correct_ratio`
- `revised_correct_ratio`
- `kept_ratio`
- `prompt_overflow_count`
- per-database kept counts

### 3. Train joint Phase1 SRT

```bash
bash scripts/srt/run_phase1_srt_4gpu.sh
```

The launcher caps each question at three traces, requires successful incorrect-initial revisions, limits correct-initial traces to 50%, splits validation by question, trains both paper objectives together, and merges the final adapter.

Outputs:

- `/data/huwenp/emb/lxy/sd-zero-sql/outputs/qwen3_4b_srt_joint/adapter`
- `/data/huwenp/emb/lxy/sd-zero-sql/outputs/qwen3_4b_srt_joint/merged`

### 4. Evaluate Phase1

```bash
bash scripts/srt/run_bird_dev_eval_when_gpus_free.sh
```

Run the same external evaluator on the base model, merged SQL-SFT model, and merged SRT model when comparing gains. Majority voting is not the same metric as the paper's avg@8, so report greedy and sampled metrics separately.

### 5. Optional Phase2

```bash
bash scripts/distill/run_phase2_distill_4gpu.sh
```

Phase2 uses the merged SRT model as student initialization and frozen reviser, applies `KL(student || teacher)` on aligned rollout tokens, and merges its output for evaluation.

## Verification

The local CPU checks do not require model weights or CHES data:

```bash
python -m unittest discover -s tests -v
python -m compileall -q scripts src tests
bash -n $(git ls-files '*.sh')
```

GPU training and BIRD accuracy cannot be validated in this checkout because model weights, databases, and the external evaluator live on the training server. Old traces and checkpoints produced by the former raw-prompt, 32-init, full-text-loss pipeline are incompatible and should not be reused.
