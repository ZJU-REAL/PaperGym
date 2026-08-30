# OpenScience

基于 **Open-Ended Scientific Problem** 的研究项目，围绕**科学问题生成 → 数据构建 → 模型训练（SFT / 自蒸馏 / GRPO）→ 评测**的完整闭环展开。通过从学术论文中抽取研究背景与方法，生成开放式的科学问题（research problem）、参考答案（reference solution）与可客观评判的评分标准（Rubric），构建高质量的训练数据集，并在此基础上进行多种训练与评测实验。

## 目录结构

```
OpenScience/
├── data/                    # 顶层汇总数据集
├── data_generation/         # 数据构建与问题生成
│   ├── get_paper.py         # 从 arXiv 批量下载论文 TeX 源码
│   ├── query_category.py    # 通过 arXiv API 查询论文所属类别
│   ├── process_paper/       # 论文 LaTeX 处理（提取正文 / 合并 / 分节）
│   └── problem_generation/  # 基于论文生成科学问题与 Rubric
├── grpo/                    # GRPO 强化学习训练
│   ├── data/                # 训练 / 验证数据集（Parquet）
│   ├── reward_fn/           # 离线打分奖励函数
│   └── scripts/             # 训练启动脚本
├── opsd/                    # OPSD 自蒸馏训练框架（含 README）
└── test/                    # 评测与实验框架
    ├── ResearchPlanGen/     # ResearchPlanGen-ML 数据集评测
    ├── compare/             # 模型回答盲评对比
    ├── data_leakage/        # 数据集泄露检测
    ├── entropy/             # 回答熵（不确定性）测量
    ├── OpenScience/         # OpenScience（本文）数据集评测 
    ├── ResearchQA/          # ResearchQA 数据集评测
    ├── RubricHub/           # RubricHub 数据集评测
    ├── rubric_rate/         # Rubric 类型构成分析
    └── stability/           # 评分稳定性分析
```

## 模块说明

### `data_generation/` — 数据构建与问题生成

从学术论文构建开放式科学问题数据集的核心流水线。

- **`get_paper.py`**：按月份遍历 arXiv `cs` 类别列表页，收集论文 ID 并批量下载 TeX 源码（`.tar.gz`），内置限速、重试与断点续传逻辑。
- **`query_category.py`**：通过 arXiv API 查询每篇论文的主类别与全部类别，并合并写入输出的 JSON 文件；支持单篇查询与批量查询（自动跳过已存在输出）。
- **`process_paper/`**：LaTeX 源码处理流水线。
  - `merge_paper_latex.py`：解压并合并单篇论文的多个 LaTeX 源文件。
  - `get_paper_body.py`：从 LaTeX 中提取正文内容（保留标题与摘要），并去除注释。
  - `divide_paper.py`：按 `\section` 将论文正文分节，供后续逐节抽取信息。
- **`problem_generation/`**：基于论文信息生成科学问题、参考答案与 Rubric。
  - `solve_background.py`：逐节抽取论文的 `research_field` / `research_background` / `solution` / `experiment`，并整合为结构化背景信息。
  - `solve_dataset.py`：完整方法-问题对生成（构建研究问题、参考答案、合并两类 Rubric）。
  - `solve_design.py`：基于方法创新构造**实验设计**任务与参考答案、Rubric。
  - `solve_innov.py`：基于背景信息构造**方法/创新**类研究问题与 Rubric。

### `grpo/` — GRPO 强化学习训练

基于 Verl 框架的论文方法生成强化学习训练模块。

- **`reward_fn/`**：离线打分奖励函数（`paper_rl_offline_*.py`），仅依赖静态评分标准与通用评分标准（不采用动态评分标准），对模型生成的方案逐条 Rubric 打分。
- **`scripts/grpo_1b_offline.sh`**：GRPO 离线训练启动脚本，配置 Verl 的 `main_ppo`，包含 vLLM 采样、FSDP 分布式训练、自定义奖励函数等完整超参数。
- **`data/`**：训练 / 验证数据集（Parquet 格式）。

### `opsd/` — OPSD 自蒸馏训练框架

Open Source Distillation，基于 TRL 的类 GKD 自蒸馏训练框架。支持 vLLM 高速生成、多种教师策略（固定教师 / EMA / 动态）和可插拔的数据整理器。详细说明见 [opsd/README.md](opsd/README.md)。

- **训练入口**：`sft_train.py`（SFT）、`opsd_train.py`（主 OPSD）、`opsd_train_answer.py`（answer 模式）、`opsd_train_combine.py`（combine 模式）。
- **核心 Trainer**：`opsd_trainer*.py`，实现 JSD 损失、KL 散度、vLLM 同步与 EMA / 固定教师策略。
- **数据整理器**：`data_collator*.py`，构造学生 / 教师输入对。
- **`scripts/`**：按模型规模（1B / 4B / 8B）组织的训练启动脚本。

### `data/` — 顶层数据集

存放汇总后的训练数据集（如 `merged_dataset_20k.parquet`）。

### `test/` — 评测与实验框架

围绕 Rubric 驱动的回答生成、评分与分析展开的评测工具集。各模块均支持本地 vLLM 服务 / 远程 OpenAI 兼容 API 调用。

- **`ResearchPlanGen/`**：ResearchPlanGen-ML 数据集的回答生成与 Rubric 评测，按研究目标生成 Proposed Research Plan 并逐项评分。
- **`compare/`**：模型回答盲评对比工具，支持两模型 / 三模型随机分配 Plan A/B/C 让 LLM 评分，统计各模型胜率。
- **`data_leakage/`**：数据集泄露检测，统一格式后用 LLM 判断每条 rubric 能否从问题中直接推断，统计泄露比例。
- **`entropy/`**：通过 OpenAI 兼容 API 获取 top-logprobs，计算回答生成过程的 token 级熵（香农熵）并聚合统计。
- **`OpenScience/`**：OpenScience（本文）数据集按领域分类（如 design、innov）生成响应并用 rubric 评分，可分类展示结果。
- **`ResearchQA/`**：基于 ResearchQA 数据集的生成与评分框架，多线程生成回答，按 rubric 逐条评判并输出 5 级评分，含消融实验。
- **`RubricHub/`**：基于 Rubric 的回答评分框架，支持生成回答、批量评分、可视化；含 `data_synthesis_final/` 子模块，实现"由粗到细的 Rubric 自动生成"流水线。
- **`rubric_rate/`**：分析学术论文数据集中 Rubric 的类型构成占比（Method vs Experiment），自动分类并统计。
- **`stability/`**：评测 LLM 评分稳定性，通过多轮独立评估分析评分一致性（均值 / 方差 / 标准差），并对比不同评分模型。

## 使用说明

各模块的具体运行方式见各自子目录内的 README 或运行脚本（`.sh`）。由于模型路径、数据集路径、API 地址等常因环境而异，多数脚本使用占位符（如 `<YOUR_MODEL_PATH>`、`/path/to/...`），使用前需按实际环境替换。
