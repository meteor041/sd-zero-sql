# SD-Zero SQL 对齐方案

本文档说明如何在 **`sd-zero-sql`** 中实现面向 CHES Text-to-SQL 的 SD-Zero 流程，并尽量对齐官方仓库：

- 官方参考仓库：`/home/pkuccadm/huwenp/emb/lxy/Self-Distillation-Zero-main`
- 本地任务仓库：`/home/pkuccadm/huwenp/emb/lxy/sd-zero-sql`

核心原则不是“把官方代码整仓搬过来”，也不是“完全重新写一套”，而是：

> **所有 CHES SQL 任务代码继续放在 `sd-zero-sql` 中；方法结构、数据契约、阶段划分尽量与官方 SD-Zero 对齐。**

---

## 1. 目标

把官方 SD-Zero 的两阶段方法映射到 SQL：

### Phase 1 — Self-Revision Training (SRT)
对每个 SQL 样本 `x`：
1. base / SFT 模型生成 `y_init`
2. 用 SQL verifier 打分得到二元 reward `r ∈ {0,1}`
3. 选择控制短语 `P_r`
4. 生成 `y_revised`
5. 只保留 revision 后验证正确的轨迹
6. 构造成 **two-stage SFT** 数据：
   - Stage 1: `x -> y_init + P_r + y_revised`
   - Stage 2: `x + y_init + P_r -> y_revised`

### Phase 2 — On-policy Self-Distillation
1. student 从 `x` 生成 `y`
2. SQL verifier 给出 reward `r` 和 feedback
3. teacher（冻结的 Phase1 SRT 模型）接收 `x + y + r + feedback + P_r`
4. teacher 仅在目标 SQL 区间上提供 token-level logits
5. student 用 KL 损失蒸馏 teacher 的修订知识

---

## 2. 当前代码组织

### `src/sql_core/`
放 SQL 通用基础设施：
- `src/sql_core/prompt_builders.py`
- `src/sql_core/sql_normalizer.py`
- `src/sql_core/sql_verifier.py`
- `src/sql_core/generation_backend.py`

### `src/phase1_srt/`
放 Phase1 契约与 trace schema：
- `src/phase1_srt/constants.py`
- `src/phase1_srt/trace_schema.py`

### `src/phase2_distill/`
放 Phase2 蒸馏实现：
- `src/phase2_distill/teacher_conditioning.py`
- `src/phase2_distill/reward_adapter.py`
- `src/phase2_distill/dataset_io.py`
- `src/phase2_distill/train_distill_kl.py`

### `scripts/sft/`
- `scripts/sft/filter_sft_by_length.py`
- `scripts/sft/train_sft.py`
- `scripts/sft/run_sft_8xa100.sh`
- `scripts/sft/run_sft_smoke.sh`

### `scripts/srt/`
- `scripts/srt/generate_phase1_traces.py`
- `scripts/srt/build_two_stage_data.py`
- `scripts/srt/train_srt_stage.py`
- `scripts/srt/run_two_stage_8xa100.sh`
- `scripts/srt/run_generate_traces_vllm_4gpu.sh`
- `scripts/srt/run_generate_traces_vllm_shard.sh`

### `scripts/distill/`
- `scripts/distill/run_phase2_distill.sh`

### `scripts/archive/`
- `scripts/archive/legacy_build_srt_training_data.py`
- `scripts/archive/legacy_run_qwen3_4b_srt_8xa100.sh`

### `configs/distill/`
- `configs/distill/sql_distill.yaml`

### `data/srt/`
- 正式 trace / stage data 仍放在 `data/srt/`
- smoke 产物归到 `data/srt/smoke/`

---

## 3. 官方仓库到当前实现的映射

| 官方 SD-Zero | 作用 | `sd-zero-sql` 对应实现 |
|---|---|---|
| `self-revision-training/self_critique_pipeline.py` | 采样 `D_revision` | `scripts/srt/generate_phase1_traces.py` |
| `self-revision-training/prepare_data.py` | 构造 Stage1 / Stage2 SFT 数据 | `scripts/srt/build_two_stage_data.py` |
| `scripts/sft.sh` | 两阶段顺序训练 | `scripts/srt/run_two_stage_8xa100.sh` |
| `self-revision-training/sft/sft.py` | SRT 训练入口 | `scripts/srt/train_srt_stage.py` |
| `scripts/distill.sh` | Phase2 distillation 入口 | `scripts/distill/run_phase2_distill.sh` |
| `examples/run_distillation.py` | teacher/student KL 蒸馏主循环 | `src/phase2_distill/train_distill_kl.py` |

---

## 4. SQL 版 Phase1 标准实现

### Stage 1
- prompt = `x`
- completion = `y_init + P_r + y_revised`

### Stage 2
- prompt = `x + y_init + P_r`
- completion = `y_revised`

其中：
- `x` 固定为 `build_base_sql_prompt(sample)` 的结果
- `P_r` 使用稳定 phrasing：
  - `r = 1`: `Let me rephrase the above solution.`
  - `r = 0`: `Wait, this response is wrong. Let me correct it.`

当前 canonical trace 契约由：
- `src/phase1_srt/constants.py`
- `src/phase1_srt/trace_schema.py`

统一维护。

当前 Phase1 主线文件：
- `scripts/srt/generate_phase1_traces.py`
- `scripts/srt/build_two_stage_data.py`
- `scripts/srt/train_srt_stage.py`
- `scripts/srt/run_two_stage_8xa100.sh`

旧 mixed 路线：
- 已归档到 `scripts/archive/`
- 不再作为默认主线

---

## 5. SQL 版 Phase2 当前实现

### 当前 teacher conditioning
当前 Phase2 采用 SQL 特化版 teacher 输入：

```text
x

y_init

reward + feedback

P_r

[target SQL span]
```

其中 feedback 来自 verifier：
- `error_type`
- `error_message`
- `pred_result_preview`
- `gold_result_preview`

### 当前最小可行训练闭环
当前 `src/phase2_distill/train_distill_kl.py` 已实现：
- student rollout
- SQL verifier reward
- feedback-conditioned teacher prompt
- token target 对齐
- token-level forward KL
- checkpoint / metrics / debug manifest 落盘

### 当前限制
为保持第一版简单稳定，暂未实现：
- top-k teacher logits
- reverse / mixed KL
- 多卡 distillation
- 高吞吐在线 rollout scheduler

---

## 6. 当前推荐主线

### Baseline SFT
- `scripts/sft/filter_sft_by_length.py`
- `scripts/sft/train_sft.py`
- `scripts/sft/run_sft_8xa100.sh`

### Phase1 SRT
- `scripts/srt/generate_phase1_traces.py`
- `scripts/srt/build_two_stage_data.py`
- `scripts/srt/run_two_stage_8xa100.sh`

### Phase2 Distill
- `scripts/distill/run_phase2_distill.sh`
- `src/phase2_distill/train_distill_kl.py`

---

## 7. 兼容与迁移说明

为减少 breakage，旧路径当前仍保留兼容 wrapper：
- `scripts/generate_srt_traces.py`
- `scripts/prepare_srt_stage_data.py`
- `scripts/train_qwen3_4b_srt.py`
- `scripts/run_qwen3_4b_srt_two_stage_8xA100.sh`
- `scripts/run_qwen3_4b_sql_distill.sh`
- `src/prompts.py`
- `src/generation.py`
- `src/sql_output.py`
- `src/verifier.py`
- `src/sdzero_sql/*`
- `src/distill/*`

新开发应统一使用新的分阶段目录入口。

---

## 8. 下一步建议

1. 继续使用新目录主线推进 Phase1 / Phase2
2. 将后续新增脚本只放入：
   - `scripts/sft/`
   - `scripts/srt/`
   - `scripts/distill/`
3. 将后续新增 Python 模块只放入：
   - `src/sql_core/`
   - `src/phase1_srt/`
   - `src/phase2_distill/`
4. 待下游路径全部切换后，再考虑删除兼容 wrapper
