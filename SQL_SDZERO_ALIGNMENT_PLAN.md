# SD-Zero SQL 对齐与验收说明

## 1. 论文、原仓库与当前实现

| 项目 | 论文 | 原仓库默认脚本 | 当前实现 |
|---|---|---|---|
| Phase1 采样 | 每题 1 个初答、3 个修订 | 3 个初答、3 个修订，默认跳过正确初答 | 每题 1 个初答、3 个修订，正确和错误初答都修订 |
| 修订条件 | 同一 assistant 流中的 `x + y_init + P_r` | 同一 assistant 流 | 同一 assistant 流，采样与训练共用构造函数 |
| Phase1 目标 | `L_generation + L_revision` | 顺序训练两个 stage | 单次联合多任务训练 |
| Loss mask | completion-only | completion-only | completion-only，显式 labels |
| 上下文 | 32K | 32K | SQL 适配为 8K，禁止静默截断 |
| Qwen thinking | 关闭 | 调用 chat template | chat template 显式 `enable_thinking=False` |
| Phase1 优化 | 全参 FSDP、`5e-6` | 全参 FSDP | LoRA SQL 适配，Phase1 `2e-5`，产出后合并 |
| Phase2 KL | `KL(student || teacher)` | NeMo 实现 | 显式 `KL(student || teacher)` |

原仓库的 two-stage 训练和默认采样参数与论文正文/附录并不完全一致，因此当前实现以论文目标和消融结论为准，而不是机械复制启动脚本。

## 2. SQL 领域的必要适配

论文中的 Qwen 基座在 math/code 上已有较高正确率。裸 Qwen 在复杂 Text-to-SQL 上通常不满足这个前提，因此当前性能优先路径先进行 gold-SQL completion-only SFT，再从该 on-domain 模型收集自修订轨迹。

这意味着当前实验不是严格的“无高质量示范”复现。若研究目标要求完全遵守 SD-Zero Zero-Demonstration 设定，应改用已有能力足够强且未在当前 gold SQL 上训练的 Text-to-SQL 基座，并单独报告其来源和初始 BIRD 分数。

## 3. Phase1 数据契约

轨迹必须至少包含：

- `id`, `db_id`, `x`, `gold_sql`
- `y_init`, `y_init_correct`, `p_r`
- `y_revised`, `y_revised_correct`, `keep`
- `verifier_init`, `verifier_revised`
- `init_index`, `revision_index`

只有 `keep=True` 且 `y_revised_correct=True` 的轨迹可进入训练。构建器按问题分割 train/valid，每题最多三条轨迹，并保留所有错误初答成功修订；正确初答轨迹最多占 50%。

每条轨迹产生两个训练样本：

1. Generation：`prompt=x`，`completion=y_init + P_r + y_revised`
2. Revision：`prompt=x + y_init + P_r`，`completion=y_revised`

两个任务在同一个 dataset 中打乱训练，避免顺序 Stage2 覆盖 generator 能力。

## 4. Verifier 约束

- SQLite 以只读和 `query_only` 模式打开。
- 无顶层 `ORDER BY` 时按多重集合比较，避免任意返回顺序造成假阴性。
- 有顶层 `ORDER BY` 时保留顺序敏感性。
- SQL 提取保留最外层 `SELECT/WITH`、嵌套子查询以及字符串内部的分号。
- BIRD 最终结果仍以官方 evaluator 为准；训练 verifier 与官方 evaluator 的差异应抽样审计。

## 5. 训练前验收门槛

以下任一项不满足时，不应开始昂贵的 Phase1 训练：

- merged SQL-SFT 模型可被 vLLM 独立加载。
- SQL-SFT 在同一 BIRD dev 评测入口上明显优于裸模型。
- `prompt_overflow_count` 已解释，训练数据不会截掉 completion。
- 至少存在足量的错误初答成功修订，且覆盖多个数据库和难度层级。
- 随机人工核验 verifier 正例/负例，未发现系统性子查询截断或顺序误判。
- generation/revision 任务数相等，train/valid 问题 ID 不重叠。

## 6. 完成度边界

本仓库代码层面的对齐由本地单测、Python 编译和 shell 语法检查覆盖。真实效果完成必须另外具备远端证据：base、SQL-SFT、SRT 在同一 BIRD dev 配置下的分数，轨迹 summary、训练 `data_stats.json`、loss/eval 曲线以及至少一轮错误案例审计。没有这些远端产物时，只能确认实现修复完成，不能声称模型效果已经达到论文水平。
