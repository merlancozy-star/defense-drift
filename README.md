# Defense Drift — 可分性塌缩诊断实验

> **论文**：Defense Drift — 跨域 RAG 投毒防御的可分性塌缩诊断  
> **表一**：128 格塌缩矩阵 (4 信号 × 4 域 × 2 攻击 × 4 投毒比例)  
> **硬件**：单 48GB vGPU，Qwen3-8B / Qwen3-Embedding-4B / Qwen3-Reranker-8B

---

## 快速开始

### 1. 环境准备

```bash
git clone https://github.com/merlancozy-star/defense-drift.git
cd defense-drift

# Conda 环境（推荐）
conda create -n defend python=3.10 -y && conda activate defend

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置模型路径

编辑 `separability_collapse.yaml`，**只改三个路径**：

```yaml
models:
  generator:
    path: "/your/path/to/Qwen3-8B"          # ← 改这里
  embedder:
    path: "/your/path/to/Qwen3-Embedding-4B" # ← 改这里
  reranker:
    path: "/your/path/to/Qwen3-Reranker-8B"  # ← 改这里
    enabled: true                             # 48GB 不够时可关掉 reranker
```

如果 48GB GPU 装不下三个模型，关掉 reranker + 减小 batch：

```yaml
  reranker:
    enabled: false              # 只用 embedder 检索，省 8GB
  embedder:
    batch_size: 16              # 减小 batch 省显存
```

### 3. 验证管线能跑

```bash
# 干跑验证（纯随机数，不加载模型——确认 import 和管线 OK）
python separability_diagnostic.py --dry-run
```

输出应显示 128 格矩阵表格 + `Results saved to: ./results/...`。

### 4. 跑真实实验

```bash
# === 第一步：源域单信号验证（检查点 1）===
python separability_diagnostic.py --signal perplexity --domain nq --ratio 0.1 --max-queries 50
python separability_diagnostic.py --signal cluster_distance --domain nq --ratio 0.1 --max-queries 50

# === 第二步：源域四信号全跑 ===
python separability_diagnostic.py --signal all --domain nq,hotpotqa \
    --attack poisonedrag --ratio 0.1 --max-queries 200

# === 第三步：单条跨域迁移（nq → bioasq）===
python separability_diagnostic.py --signal all --source nq --target bioasq \
    --attack poisonedrag --ratio 0.1 --max-queries 200

# === 第四步：全矩阵 128 格 ===
python separability_diagnostic.py --full-matrix --max-queries 300
```

---

## 数据集

四个数据集均从 HuggingFace 自动下载（首次运行需联网）：

| 域 | HF 路径 | 类型 | 说明 |
|---|---|---|---|
| `nq` | `nq_open` | 源域 | Natural Questions Open，开放域 QA |
| `hotpotqa` | `hotpot_qa/fullwiki` | 源域 | 多跳推理 QA，含 Wikipedia 上下文 |
| `bioasq` | `rag-datasets/mini-bioasq` | 目标域 | 医疗 QA，含 PubMed passages |
| `finance` | `Linq-AI-Research/FinanceRAG/FinQA` | 目标域 | 金融数值推理 QA，含 corpus+queries |

首次运行会自动下载到 HF 缓存目录（~1-5 GB）。如果服务器无网络，预先在有网机器上：

```bash
# 预下载数据集
python -c "
from datasets import load_dataset
load_dataset('nq_open', split='train')
load_dataset('hotpot_qa', 'fullwiki', split='train')
load_dataset('rag-datasets/mini-bioasq', 'question-answer-passages', split='train')
load_dataset('Linq-AI-Research/FinanceRAG', 'FinQA', split='queries')
"
# 然后把 ~/.cache/huggingface/datasets 拷贝到服务器
```

---

## CLI 参考

```
python separability_diagnostic.py [OPTIONS]

选项:
  --config, -c PATH      YAML 配置文件路径（默认: separability_collapse.yaml）
  --signal, -s SIGNAL    信号名: perplexity | drs | cluster_distance | attention_variance | all
  --domain, -d DOMAIN    评估域: nq | hotpotqa | bioasq | finance (逗号分隔)
  --source DOMAIN        跨域对比的源域
  --target DOMAIN        跨域对比的目标域
  --attack, -a ATTACK    攻击类型: poisonedrag | corruptrag
  --ratio, -r FLOAT      投毒比例: 0.05 | 0.1 | 0.2 | 0.4
  --max-queries N        每格最大 query 数（覆盖 YAML 配置）
  --full-matrix          运行完整 128 格笛卡尔积
  --dry-run              合成数据干跑（不加载模型和数据集）
  --output-dir DIR       输出目录（默认: ./results/）
```

### 常用命令速查

```bash
# 开发期验证（无 GPU）
python separability_diagnostic.py --dry-run

# 单信号单域（最轻量）
python separability_diagnostic.py -s perplexity -d nq -r 0.1 --max-queries 50

# 四信号源域验证
python separability_diagnostic.py -s all -d nq -a poisonedrag -r 0.1

# 跨域对比
python separability_diagnostic.py -s all --source nq --target bioasq -r 0.1

# 全矩阵 + 强攻击
python separability_diagnostic.py --full-matrix

# 全矩阵（不含注意力信号，省 GPU）
python separability_diagnostic.py --full-matrix \
    --signal perplexity,drs,cluster_distance
```

---

## 输出

所有结果落盘到 `./results/`：

| 文件 | 格式 | 说明 |
|---|---|---|
| `collapse_matrix_YYYYMMDD_HHMMSS.csv` | CSV | 表一矩阵，可直接填论文 |
| `collapse_matrix_YYYYMMDD_HHMMSS.json` | JSON | 结构化完整数据 |
| `intermediate_*.csv` | CSV | 中间结果（断点续跑用） |

每行即一个 `CellResult`：

```
signal, domain, domain_type, attack, poison_ratio, auroc, separability_drop, n_samples, source_auroc, domain_robust
perplexity, nq, source, poisonedrag, 0.1, 0.8523, , 200, ,
perplexity, bioasq, target, poisonedrag, 0.1, 0.6810, 0.1713, 200, 0.8523, False
```

**验收标准**（YAML `acceptance` 节点）：

| 检查项 | 阈值 | 含义 |
|---|---|---|
| 源域 AUROC ≥ 0.80 | ≥ 3/4 信号 | 信号实现可信 |
| separability_drop > 0.10 | ≥ 3/4 信号 | 主结论成立 |
| CorruptRAG Δ ≥ PoisonedRAG Δ | 全部 | 强攻击更隐蔽 |
| domain_robust = True | 任何信号 | 反向发现（也算有价值） |

---

## 运行测试

```bash
# 全部单测（65 个，无需 GPU）
pytest tests/ -v

# 单测 + 干跑
pytest tests/ -v && python separability_diagnostic.py --dry-run
```

---

## 项目结构

```
defend/
├── separability_diagnostic.py    # 主入口 CLI
├── separability_collapse.yaml    # 全部超参配置
├── scorers/                      # 四信号打分
│   ├── perplexity.py             #   PPL (generator NLL)
│   ├── drs.py                    #   DRS (PCA low-variance projection)
│   ├── cluster_distance.py       #   2-means outlier distance
│   └── attention_variance.py     #   Cross-passage attention variance
├── injectors/                    # 投毒注入
│   ├── poisonedrag.py            #   模板/生成器双模式
│   └── corruptrag.py             #   原文篡改（数字/方向/否定）
├── retrieval/                    # 检索管线
│   ├── embedder.py               #   Qwen3-Embedding-4B
│   ├── chunker.py                #   固定 size/overlap 分块
│   └── retriever.py              #   FAISS + 可选 rerank
├── models/                       # 模型封装
│   ├── generator.py              #   Qwen3-8B (NLL + attention)
│   └── reranker.py               #   Qwen3-Reranker-8B
├── pipeline/                     # 管线编排
│   ├── assemble_units.py         #   检索→注入→打分→打包
│   └── build_collapse_matrix.py  #   128 格笛卡尔积编排
├── data/                         # 数据
│   ├── schemas.py                #   Passage, Query, ScoringUnit, CellResult
│   └── loader.py                 #   HF datasets 加载器
├── utils/                        # 工具
│   ├── config.py                 #   YAML → 强类型配置
│   ├── metrics.py                #   AUROC + bootstrap CI
│   └── results.py                #   CSV/JSON 持久化
└── tests/                        # 65 单测 (58 M1 + 7 M2)
    ├── test_perplexity.py
    ├── test_cluster_distance.py
    ├── test_drs.py
    ├── test_attention_variance.py
    ├── test_injectors.py
    └── test_pipeline.py
```

---

## 硬约束（实验有效性保证）

| # | 约束 | 实现方式 |
|---|---|---|
| 1 | 源/目标域检索配置一致 | 全局唯一 `retrieval` 配置块，跨域校验 |
| 2 | DRS 对齐 NeurIPS'24 | PCA on clean covariance → low-var eigenvectors |
| 3 | 分数方向：高=投毒 | `verify_score_direction()` 自动检测反转 |
| 4 | 注意力白盒验证 | HF `output_attentions=True`，含降级检测 |
| 5 | 不调参美化数字 | 所有超参在 YAML 锁定，CLI 不暴露检索参数 |
