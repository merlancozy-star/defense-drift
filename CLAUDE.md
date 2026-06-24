# CLAUDE.md — Defense Drift / 表一：可分性塌缩矩阵

> 本文件指导 Claude Code 在 milrag 仓库内实现论文 **Defense Drift** 的核心实验（表一）。
> 论文写作不在本仓库范围内——本仓库只产出实验代码与结果。

---

## 1. 项目背景（为什么做这个）

论文主张：现有 RAG 投毒防御都隐含假设"干净/投毒可分性"是**域无关常量**；我们证明它是**域依赖**的——换到专业语料会塌缩，导致这些防御失效。

**表一是全篇的招牌结果**：一张 `信号 × (源域→目标域) × 攻击 × 投毒比例` 的矩阵，每格填 **可分性塌缩量 Δ_sep = AUROC(源域) − AUROC(目标域)**。它一旦跑出显著塌缩，论文即立住。本仓库的首要目标就是产出这张矩阵。

骨架已给：`separability_diagnostic.py` + `separability_collapse.yaml`。你的任务是把骨架里 `# >>> MILRAG` 和 `TODO` 标记处接到 milrag 的真实组件上，并跑出结果。

---

## 2. 技术栈与环境

- Python，复用 milrag 现有脚手架（config 编排、YAML 超参、检索/模型封装）。
- 模型栈（单 48GB vGPU）：generator `Qwen3-8B`、embedder `Qwen3-Embedding-4B`、reranker `Qwen3-Reranker-8B`。
- 依赖：`scikit-learn`（AUROC、KMeans）、`numpy`、milrag 内部检索/LM 接口。
- **全程 training-free**：无训练循环、无梯度。最重的是批量推理 + 统计。

---

## 3. 要实现的模块（按骨架对应）

| 骨架位置 | 实现内容 | 依赖的 milrag 组件 |
|---|---|---|
| `PerplexityScorer.score` | token NLL → PPL | LM 打分接口（Qwen3-8B 或更小打分 LM） |
| `DRSScorer.score` | 方向相对偏移 | embedder |
| `ClusterDistanceScorer.score` | 2-means 离群距离 | embedder（已基本完整，校验即可） |
| `AttentionVarianceScorer.score` | 生成时落在该 passage 的注意力质量 | generator 的 **white-box 注意力** |
| `PoisonInjector` | PoisonedRAG + CorruptRAG 两种 craft() | 攻击逻辑 + embedder |
| `assemble_units` | 注入→检索→标注 is_poison→打包 | milrag 检索器（embedder+reranker） |
| `build_collapse_matrix` | 笛卡尔积并行编排 | milrag config 编排器 |

---

## 4. 硬约束（违反会让结论失效——务必遵守）

- **MUST：源域与目标域检索配置严格一致。** 同一 embedder、同一 chunk 策略（size/overlap）、同一 top_k。换域**只换语料**，检索配置锁死。否则"塌缩"会被检索工程差异污染成 confound，整张表作废。
- **MUST：DRS 须对齐原论文（NeurIPS'24, Directional Relative Shift）的精确定义。** 骨架里的 `drs_score` 是符合直觉的**占位**实现，方向归一化未必与原文一致。落地前查原论文公式校准，**不要凭直觉定稿，也不要编造公式**。
- **MUST：分数方向统一为"越高越像投毒"。** 任何信号若天然方向相反，在 scorer 内取反，保证 AUROC 以投毒为正类时方向一致。
- **MUST：注意力信号需确认推理栈暴露 attention。** 若用 vLLM 等高吞吐后端，先确认能拿到 cross-passage 注意力权重；拿不到则退回 HF 前向。此项有工程风险，先验证可行性再大规模跑。
- **NEVER：不要为了好看的数字调检索配置或挑 query。** 诊断的可信度高于数字的漂亮。

---

## 5. 域与数据

- 源域：`nq`、`hotpotqa`（通用域，信号原本调好的域）。
- 目标域：`bioasq`（医疗）、`finance`（金融）——**同出 RAG-QA Arena / RobustQA**，任务形式一致（长文检索式），这是刻意选择，用于消除"任务形式" confound。加载时统一走该框架的格式。
- 投毒比例：`[0.05, 0.1, 0.2, 0.4]`。
- 探针：用极少量已知投毒估计投毒侧分数分布（诊断期 `is_poison` 已知，生成期不可见）。

---

## 6. 验收标准（见 yaml 的 acceptance）

1. **源域复现**：四信号在**源域**的 AUROC 应达各自论文量级（`≳ 0.80`）。**这是第一个检查点——它通过才证明信号实现可信，否则后续塌缩数字无意义。**
2. **主结论**：至少 **3/4** 信号在 源→目标 出现显著塌缩（`Δ_sep > 0.10`）。
3. **强攻击**：CorruptRAG 下的 `Δ_sep` 应 ≥ PoisonedRAG（更隐蔽→更塌）。
4. **反向情形也记录**：若某信号 `Δ_sep ≈ 0`，标为"域稳健信号"，写入结果供论文 Discussion 用（反向结果也是有价值发现）。

---

## 7. 里程碑顺序（不要跳步）

1. 接 `PerplexityScorer` + `ClusterDistanceScorer`（最简单，先打通管线）→ 在**单个源域**跑出 AUROC。
2. 接 milrag 检索器到 `assemble_units` + 实现 PoisonedRAG 注入 → **源域复现达标（检查点 1）**。
3. 接 `DRSScorer`（对齐原文）+ `AttentionVarianceScorer`（验证白盒可行）→ 四信号在源域齐活。
4. 接目标域 bioasq、finance → 跑出第一条 `Δ_sep`（单条迁移先看苗头）。
5. `build_collapse_matrix` 并行展开全矩阵 → 产出表一。
6. 加 CorruptRAG → 强攻击列。

---

## 8. 开发规范

- 渐进式：每接一个 scorer 先写最小单测（构造已知干净/投毒小样本，验证 AUROC 方向正确）。
- 每个里程碑产出一份结果快照（CSV/JSON），命名含 `signal-source-target-attack-ratio`，便于回填论文表格。
- 结果落盘格式与 `CellResult` 字段一致，方便我（论文侧）直接读数填表。
- 遇到 DRS 公式或注意力访问的不确定性，**停下来标注 TODO 并说明，不要猜测性定稿**。

---

## 9. 产出物（交回论文侧）

- 表一矩阵：`CellResult` 列表 → CSV。
- 源域复现数字（证明信号可信）。
- "域稳健信号"清单（若有）。
- 任何偏离骨架的实现决定 + 理由（便于写进论文 Method/Limitations）。
