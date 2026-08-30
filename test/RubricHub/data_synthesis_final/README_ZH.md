# RubricHub 数据合成（由粗到细的 Rubric 生成）

此目录包含 RubricHub 论文中 **由粗到细的 Rubric 生成（Coarse-to-Fine Rubric Generation）** 流水线的开源实现。

- **入口**：`./run_data_synthesis.sh`（先编辑文件顶部的 "1) Fill here"，然后运行）
- **输入**：一个 JSONL 文件（每行一个 JSON 对象），其中包含一个用于存放用户 prompt 文本的字段（query/question/prompt；列名可配置）
- **输出**：`$OUTPUT_DIR/final.parquet`（主要产物）和 `$OUTPUT_DIR/final.jsonl`（便于检查）
- **中间产物**：`step0_reference.jsonl` ~ `step4_augmented.jsonl`（用于断点续跑/调试）

---

## 架构说明（面向维护者 / 未来的自己）

### 本代码在论文中的位置：从 RubricHub 到 RL 训练信号

论文的整体流水线包含两部分：

1) **由粗到细的 Rubric 生成（构建 RubricHub）**
   针对每个查询，自动合成一套高密度、高判别力的 rubric 评分标准。
2) **Rubric 在后训练中的应用（将 rubric 转化为训练信号）**
   - **RuFT**（基于 Rubric 的拒绝采样微调）：用 rubric 对候选回答进行评分/筛选，构成 SFT 数据。
   - **RuRL**（基于 Rubric 的强化学习）：将 rubric 标准视为结构化的、可验证的奖励，用于运行 RL。

当前仓库仅实现 **(1)**：将"仅含 prompt 的 JSONL"转换为"(prompt, rubrics)"数据。
第 (2) 部分（grader + RuFT/RuRL 训练）在论文中有定义，但在 `data_synthesis_final/` 中**未实现**。

### 论文阶段 ↔ 代码步骤（Stage 1–3 → Step0–Step4）

论文阶段：

- **Stage 1: 基于回答且受原则引导的生成**
  以参考回答为基础生成 rubric，并受元原则约束以减少 rubric 漂移。
- **Stage 2: 多模型聚合**
  用多个模型生成候选 rubric，然后保守地合并/去重，以减小单模型视角偏差。
- **Stage 3: 难度演进**
  给定一对回答，演进更严格、更具判别力的标准，以避免分数饱和。

实现映射（`data_synthesis_final/`）：

```
raw.jsonl
  └─ Step0: 生成参考回答（grounding 锚点）
      └─ step0_reference.jsonl
          └─ Step1: 生成 response_a/response_b（用于难度演进的对照回答）
              └─ step1_responses.jsonl
                  └─ Step2: 生成 rubrics_a/rubrics_b（受原则引导且基于回答）
                      └─ step2_rubrics.jsonl
                          └─ Step3: 合并 → merged_rubrics（多模型聚合）
                              └─ step3_merged.jsonl
                                  └─ Step4: 增强 → augmented_rubrics（难度演进）
                                      └─ step4_augmented.jsonl
                                          └─ Step5: 导出 → final.{jsonl,parquet}
```

注意：在论文中，用于难度演进的"高质量回答对"通常是从更大的候选池中通过 rubric 评分共识挑选出来的。
本开源实现保持可运行，直接使用两个生成的回答（`response_a/response_b`）。

### 固定的设计选择（本仓库的硬约束）

以下由 `run_data_synthesis.sh` + `data_synthesis_final/run_pipeline.py` 固定（或以受控方式降级以提升可用性）：

- **参考回答只生成一次**：`REFERENCE_MODEL` → 字段 `reference`
- **两个回答**：`RESPONSE_MODEL_A/B` → 字段 `response_a/response_b`
- **1–2 个 rubric 生成器**：`RUBRIC_MODEL_A` 必填，`RUBRIC_MODEL_B` 可选 → `rubrics_a/rubrics_b`
- **论文默认为多模型**：Stage 2 是**多模型聚合**，因此建议使用 >=2 个异构模型以获得更强的 rubric。
  本仓库允许 `RUBRIC_MODEL_B` 为空，从而仅用一个供应商也能运行。
- **一个合并模型 + 一个增强模型**：`MERGE_MODEL` / `AUGMENT_MODEL`

### 目录职责（入口 → 编排 → 共享工具 → 步骤逻辑）

- `run_data_synthesis.sh`
  唯一面向用户的入口：配置输入/输出/模型/端点/并发度，然后调用 `python data_synthesis_final/run_pipeline.py ...`。

- `data_synthesis_final/run_pipeline.py`
  流水线编排器：串联 Step0→Step5，定义中间文件名，向每个步骤传递运行时参数。

- `data_synthesis_final/common.py`
  共享工具（"骨架"）：
  - JSONL 读写：`read_jsonl()` / `write_jsonl()`（通过 `.tmp` 原子写入，避免中断时损坏）
  - 断点续跑索引：`index_existing_rows()` + `__record_id__` 规则
  - OpenAI 异步客户端：`make_async_client()`
  - 重试封装：`chat_completion_with_retry()`（对限流 / 超时 / 5xx 做指数退避）
  - JSON 提取：`extract_json_array()`（从围栏代码块或原始文本中提取 `[...]`）
  - Rubric 归一化：`normalize_rubric_items()`（字段修剪 + 将 `weight` 裁剪到 0–10）
  - 导出辅助：`rubrics_to_final_format()` / `dedup_final_rubrics()`（按标准文本去重，保留更高分数）

- `data_synthesis_final/step*.py`
  每个步骤一个文件（每步都支持断点续跑）。

- `data_synthesis_final/prompts/*.txt`
  Prompt 模板（在此处编辑 prompt；避免在 Python 中硬编码 prompt 文本）。

### 各步骤的字段（调试/扩展时最有用）

所有步骤都会保留并传播这些"元字段"：

- `question`：归一化的 prompt 文本（最终导出字段）
- `id`：可选的透传 id（允许空字符串）
- `__source_index__`：原始输入中的行索引（在 Step0 创建）
- `__record_id__`：断点续跑主键（优先用 `ID_COLUMN`，否则用 `idx:<行索引>`）

Step0 `step0_generate_reference.py`

- 输入：原始 JSONL（只需 `QUESTION_COLUMN`）
- 输出新增：
  - `reference`（文本）
  - `reference_model`
  - `reference_error`（失败时）

Step1 `step1_generate_responses.py`

- 输入：Step0 输出
- 输出新增：
  - `response_a`, `response_a_model`, `response_a_error`
  - `response_b`, `response_b_model`, `response_b_error`

Step2 `step2_generate_rubrics.py`

- 输入：Step1 输出（需要 `reference`）
- 输出新增：
  - `rubrics_a`: list[{title, description, weight}]
  - `rubrics_b`: 相同 schema（可选）
  - `rubrics_a_model` / `rubrics_b_model`
  - `rubrics_a_error` / `rubrics_b_error`

Step3 `step3_merge_rubrics.py`

- 输入：Step2 输出（需要 `rubrics_a`；`rubrics_b` 可选）
- 输出新增：
  - `merged_rubrics`: list[{title, description, weight}]
  - `merged_rubrics_model`
  - `merged_rubrics_error`
  - 若 `rubrics_b` 缺失/为空：透传 `rubrics_a`，并将 model 设为 `"passthrough"`

Step4 `step4_augment_rubrics.py`

- 输入：Step3 输出（需要 `merged_rubrics` + `response_a/response_b`）
- 输出新增：
  - `augmented_rubrics`: list[{title, description, weight}]（应只包含*新的*更严格条目）
  - `augmented_rubrics_model`
  - `augmented_rubrics_error`
  - 若回答缺失：输出 `[]`，并将 model 设为 `"skipped(no responses)"`（视为有效输出）

Step5 `step5_export_dataset.py`

- 输入：Step4 输出
- 输出：
  - `final.jsonl`: `{question, id, rubrics:[{criterion, points}]}`
  - `final.parquet`: 相同 schema，但 `rubrics` 是嵌套列表列（`list<struct<criterion: string, points: int32>>`）

导出规则（重要）：

- 拼接 `merged_rubrics + augmented_rubrics`
- 映射 `{description, weight}` → `{criterion, points}`
- 按归一化的 `criterion` 文本去重：保留 **points 更高** 的条目
- 可选：通过 `MAX_CRITERIA` / `--max-criteria` 限制最终标准数量

### 断点续跑（为什么可重复运行而不重复计费）

每个步骤遵循相同模式：

1) 若该步骤的输出 JSONL 已存在，则加载之
2) 构建 `__record_id__ → 行` 索引
3) 对每个输入行：
   - 若输出已有目标字段且无 `*_error`：复用
   - 否则：重新运行并将错误写回 `*_error`

`__record_id__` 生成（在 Step0 中）：

- 若提供了 `ID_COLUMN` 且字段非空：使用它（推荐；在打乱后保持稳定）
- 否则：使用输入行索引（`idx:<n>`）（改变输入顺序会破坏续跑对齐）

### LLM 调用：并发、重试以及为什么输出必须是 JSON

- 所有 LLM 调用使用 OpenAI Python SDK 异步客户端：`client.chat.completions.create(...)`
- `chat_completion_with_retry()` 用指数退避处理限流 / 超时 / 5xx
- `run_with_concurrency()` 使用异步工作队列，并周期性写入中间结果（`SAVE_EVERY`）

论文对齐说明：

- 论文附录 B / 表 5 指定了 RuRL rollout 采样超参数：**temperature=1.0** 和 **max response length=8192**。
  本仓库不包含 RL 训练，但 **Step1 回答采样** 通过 `run_data_synthesis.sh` 中的 `RESPONSE_TEMPERATURE` / `RESPONSE_MAX_TOKENS` 保留该设置。
- 论文未明确指定 rubric 合成步骤（reference/rubric/merge/augment）的 temperature / max_tokens；本仓库默认保持这些步骤为低温以提升稳定性。

Step2–Step4 要求输出是 **严格的 JSON 数组**：

- `extract_json_array()` 提取第一个有效的 `[...]` 并运行 `json.loads()`。
- 任何额外说明文本或非数组输出都会变成错误，并在续跑时被重试。

### Prompt 模板（优先在此处编辑策略）

占位符约定：

- `reference.txt`: `<<query>>`
- `rubric_generation.txt`: `<<query>>` + `<<reference>>`
- `rubric_aggregation.txt`: `{|prompt|}` + `{|rubrics1|}` + `{|rubrics2|}`
- `difficulty_evolution.txt`: `{|prompt|}` + `{|rubrics|}` + `{|response1|}` + `{|response2|}`

编辑技巧：

- 想要更"挑剔 / 有判别力"的标准：调整 `difficulty_evolution.txt`
- 想要更"切题 / 少漂移"：调整 `rubric_generation.txt`
- 想要更"保守的合并"：调整 `rubric_aggregation.txt`（仅在明显等价时才合并）

### Rubric 质量原则（论文附录 A / 表 4）

论文将"好的 rubric"的属性归纳为四个元维度：

- **一致性与对齐**：评分者应当一致；每条标准必须与查询相关
- **结构与范围**：覆盖显式/隐式要求；标准数量合理；标准应原子化且不重叠
- **清晰与质量**：避免含糊措辞；简洁清晰；语言与查询匹配
- **推理与可评估性**：标准应具有判别力且可验证/可检查；权重应合理

在本实现中，这些约束主要存在于 prompt 模板中：
`rubric_generation.txt` / `rubric_aggregation.txt` / `difficulty_evolution.txt`。

### 与"完整"论文系统的有意简化

- **权重范围**：论文允许更宽的范围（包括负惩罚权重）。为导出简化，本仓库裁剪到 `0–10`（见 `common.py`）。
- **回答对选择**：论文从更大候选池中选择；本仓库直接使用两个生成的回答。
- **无 grader / 无 RL 代码**：本仓库产出 `(q, R_q)` 数据，但未实现 grader 或 RuFT/RuRL 训练。
- **更少的模型**：论文常使用更多异构模型；本仓库为实用性默认 1–2 个 rubric 生成器 + 1 个合并模型。

### 此输出如何连接到 RuFT / RuRL（论文层面）

导出的 `final.parquet` 本质上就是论文的 **(q, R_q)**。
下游（不在此仓库中）通常形如：

- **RuFT（SFT 数据构建）**
  对每个 `(q, R_q)`，采样 `K` 个候选回答 `A={a_k}`，用 rubric 打分，按阈值 `τ` 筛选，
  选出高分 `a+`，构成 SFT 对 `(q, a+)`。

- **RuRL（rubric 作为奖励）**
  对每条标准 `c_i`，grader 输出二元 `b_i∈{0,1}`（规则或 LLM grader），
  奖励为归一化加权和：`r = (Σ w_i b_i) / (Σ w_i)`，然后运行 RL 优化策略。

---

## 使用自己的 LLM API 服务

本工具支持通过 OpenAI 兼容 API 调用任意 LLM 服务完成数据合成。

### 1. 修改脚本中的 API 配置

编辑 `run_data_synthesis.sh`，设置你的 API 地址和密钥：

```bash
# 参考模型
REFERENCE_BASE_URL="http://your-api/v1"
REFERENCE_API_KEY="sk-your-api-key"
REFERENCE_MODEL="your-model-name"

# 响应模型 A/B
RESPONSE_A_BASE_URL="http://your-api/v1"
RESPONSE_A_API_KEY="sk-your-api-key"
RESPONSE_A_MODEL="your-model-name"

# Rubric 生成模型 A/B
RUBRIC_A_BASE_URL="http://your-api/v1"
RUBRIC_A_API_KEY="sk-your-api-key"
RUBRIC_A_MODEL="your-model-name"

# 合并与增强模型
MERGE_BASE_URL="http://your-api/v1"
MERGE_API_KEY="sk-your-api-key"
MERGE_MODEL="your-model-name"

AUGMENT_BASE_URL="http://your-api/v1"
AUGMENT_API_KEY="sk-your-api-key"
AUGMENT_MODEL="your-model-name"
```

### 2. 命令行参数方式

```bash
python run_pipeline.py \
    --input raw.jsonl \
    --output output/ \
    --reference-base-url http://your-api/v1 \
    --reference-api-key sk-your-api-key \
    --reference-model your-model-name \
    --response-base-url-a http://your-api/v1 \
    --response-api-key-a sk-your-api-key \
    --response-model-a your-model-name
```

### 3. 本地 vLLM 服务

如果使用本地 vLLM 部署：

```bash
python run_pipeline.py \
    --input raw.jsonl \
    --output output/ \
    --reference-base-url http://127.0.0.1:8000/v1 \
    --reference-api-key EMPTY \
    --reference-model your-local-model-name \
    --response-base-url-a http://127.0.0.1:8000/v1 \
    --response-api-key-a EMPTY \
    --response-model-a your-local-model-name
```
