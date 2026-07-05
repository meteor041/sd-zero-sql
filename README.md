# SD-Zero SQL Workspace

This workspace contains the data preparation, training, and evaluation integration code for fine-tuning **Qwen3-4B-Instruct** on the CHES Text-to-SQL dataset, plus the aligned **Phase1 SRT** and **Phase2 Distill** pipelines for SQL SD-Zero style training.

## Model and storage convention

- Base model:
  - `/data/model/Qwen3-4B-Instruct-2507`
- Fine-tuned SFT output directory:
  - `/data/huwenp/emb/lxy/sd-zero-sql/outputs/qwen3_4b_sft_lora_4k`

Training and evaluation scripts are aligned to use the same output locations by default.

---

## Repository structure

### `src/sql_core/`
Shared SQL task infrastructure.

- `src/sql_core/prompt_builders.py`
  - base SQL prompt
  - revision prompt
- `src/sql_core/sql_normalizer.py`
  - SQL-only output normalization
- `src/sql_core/sql_verifier.py`
  - SQL execution verifier against CHES SQLite databases
- `src/sql_core/generation_backend.py`
  - pluggable HF / vLLM generation backend

### `src/phase1_srt/`
Phase1 canonical SRT contract helpers.

- `src/phase1_srt/constants.py`
  - stable `P_r` phrases and canonical trace field names
- `src/phase1_srt/trace_schema.py`
  - trace normalization, reward extraction, validation, dedupe keys

### `src/phase2_distill/`
Phase2 distillation helpers and trainer.

- `src/phase2_distill/teacher_conditioning.py`
  - student/teacher prompt construction
  - reward + feedback conditioned teacher prompt
- `src/phase2_distill/reward_adapter.py`
  - SQL verifier → reward adapter
- `src/phase2_distill/dataset_io.py`
  - dataset loading and debug-manifest writing
- `src/phase2_distill/train_distill_kl.py`
  - minimal token-level KL distillation loop

### `scripts/`
Scripts are organized by stage:

- `scripts/sft/`
  - baseline SFT data prep and training
- `scripts/srt/`
  - Phase1 trace generation, stage-data building, and SRT training
- `scripts/distill/`
  - Phase2 distillation launcher
- `scripts/archive/`
  - legacy mixed-route entrypoints kept only for reference

Backward-compatible wrappers are still kept at the old `scripts/*.py` and `scripts/*.sh` paths for one transition cycle.

### `configs/`
- `configs/distill/sql_distill.yaml`
  - current Phase2 smoke configuration

### `data/srt/`
- `data/srt/smoke/`
  - temporary smoke outputs
- `data/srt/`
  - formal Phase1 traces and stage data

---

## Prompt format used for SFT

The SFT dataset is built with the following chat-style prompt:

```text
System:
You are an expert Text-to-SQL model. Given a database schema, evidence, and a natural language question, generate a valid SQL query.

User:
Database schema:
{schema}

Evidence:
{evidence}

Question:
{question}

Assistant:
{gold_sql}
```

Notes:
- `schema` is rendered from generated CHES M-Schema.
- `evidence` is kept as an explicit field.
- `Assistant:` is the generation boundary used at inference time as well.

---

## Data preparation status

### Full train M-Schema
Generated from CHES train databases and stored in:
- Aggregate:
  - `/home/pkuccadm/huwenp/emb/lxy/M-Schema/ches_train_mschemas.json`
- Summary:
  - `/home/pkuccadm/huwenp/emb/lxy/M-Schema/ches_train_mschemas_summary.json`
- Per-database JSONs:
  - `/home/pkuccadm/huwenp/emb/lxy/M-Schema/single_schema/<db_id>_mschema.json`

Coverage:
- 69 / 69 CHES train databases exported successfully.

### Full SFT dataset
Built from CHES `train.json` + full train M-Schema:
- JSON:
  - `/home/pkuccadm/huwenp/emb/lxy/M-Schema/ches_train_sft.json`
- JSONL:
  - `/home/pkuccadm/huwenp/emb/lxy/M-Schema/ches_train_sft.jsonl`
- Summary:
  - `/home/pkuccadm/huwenp/emb/lxy/M-Schema/ches_train_sft_summary.json`

Coverage:
- 9428 records across 69 databases.

### 4k-safe training dataset
Since the project currently trains with `max_length=4096`, a filtered dataset was created that keeps only samples whose full `text` fits within 4096 tokens under the tokenizer from `/data/model/Qwen3-4B-Instruct-2507`.

Files:
- Train 4k subset:
  - `data/ches_train_sft_train_4k.jsonl`
- Valid 4k subset:
  - `data/ches_train_sft_valid_4k.jsonl`
- Overflow manifest:
  - `data/ches_train_sft_overflow_4k.jsonl`
- 4k summary:
  - `data/ches_train_sft_4k_summary.json`

Current counts:
- Train kept: 8287
- Valid kept: 167
- Total kept: 8454
- Overflow: 974

---

## Recommended workflows

### Baseline SFT workflow
#### 1. Train
Use the 4k-safe dataset and the 8xA100 launcher:

```bash
bash /home/pkuccadm/huwenp/emb/lxy/sd-zero-sql/scripts/sft/run_sft_8xa100.sh
```

#### 2. Evaluate
After training finishes and `jsonlines` is available in the `csc_sql` environment:

```bash
bash /home/pkuccadm/huwenp/emb/lxy/csc_sql/bin/process/run_ches_qwen3_eval.sh
```

### Default SD-Zero-aligned Phase1 workflow
#### 1. Phase1 trace generation
Main entrypoints:
- `scripts/srt/generate_phase1_traces.py`
- `scripts/srt/run_generate_traces_vllm_4gpu.sh`
- `scripts/srt/run_generate_traces_vllm_shard.sh`

Current status:
- HF backend works and has been smoke-tested in batched mode.
- SQL normalization and canonical trace-schema helpers are integrated.
- Trace generation writes canonical `D_revision`-style records with stable `x` / `P_r` / verifier fields.
- Generation now supports multi-init sampling, chunked writes, and sharded runs.

Recommended backend usage:
- Use `--backend hf` for local debugging and small smoke tests.
- Use `--backend vllm` for larger trace generation when enough GPUs are free.

#### 2. Phase1 SRT training workflow
Main entrypoints:
- `scripts/srt/build_two_stage_data.py`
- `scripts/srt/train_srt_stage.py`
- `scripts/srt/run_two_stage_8xa100.sh`

Recommended order:
1. Generate canonical Phase1 traces
2. Build official two-stage Phase1 data
3. Train Stage 1 then Stage 2 sequentially

Canonical trace contract used by Phase1:
- `x = build_base_sql_prompt(sample)`
- `P_r = Let me rephrase the above solution.` when `reward = 1`
- `P_r = Wait, this response is wrong. Let me correct it.` when `reward = 0`
- kept traces must satisfy `y_revised_correct = True`

Notes:
- The SQL task-specific trace generator and verifier remain inside this repo.
- The Phase1 training structure is aligned to the official SD-Zero two-stage SFT design.
- `build_two_stage_data.py` consumes canonical trace fields directly and does not recompute `P_r`.

### Phase2 distillation workflow
Main entrypoint:
- `scripts/distill/run_phase2_distill.sh`

Core code:
- `src/phase2_distill/train_distill_kl.py`
- `src/phase2_distill/teacher_conditioning.py`
- `src/phase2_distill/reward_adapter.py`

Current implementation status:
- Minimal runnable token-level KL distillation loop exists.
- Teacher is frozen.
- Student rollout is verified by SQL execution reward.
- Teacher prompt is conditioned on `reward + feedback + P_r`.
- CPU-only mock smoke test passes end-to-end.

---

## Legacy and compatibility notes

The following old entrypoints are still present as compatibility wrappers, but new work should use the stage-based paths above:
- `scripts/generate_srt_traces.py`
- `scripts/prepare_srt_stage_data.py`
- `scripts/train_qwen3_4b_srt.py`
- `scripts/run_qwen3_4b_srt_two_stage_8xA100.sh`
- `scripts/run_qwen3_4b_sql_distill.sh`

Archived legacy scripts:
- `scripts/archive/legacy_build_srt_training_data.py`
- `scripts/archive/legacy_run_qwen3_4b_srt_8xa100.sh`

Smoke artifacts are stored under:
- `data/srt/smoke/`

---

## Related assets outside this directory

- `/home/pkuccadm/huwenp/emb/lxy/M-Schema/ches_train_mschemas.json`
- `/home/pkuccadm/huwenp/emb/lxy/M-Schema/ches_train_sft.jsonl`
- `/home/pkuccadm/huwenp/emb/lxy/M-Schema/ches_train_sft_train.jsonl`
- `/home/pkuccadm/huwenp/emb/lxy/M-Schema/ches_train_sft_valid.jsonl`
- `/home/pkuccadm/huwenp/emb/lxy/csc_sql/bin/process/run_ches_qwen3_eval.sh`
- `/home/pkuccadm/huwenp/emb/lxy/csc_sql/src/cscsql/model/infer.py`
