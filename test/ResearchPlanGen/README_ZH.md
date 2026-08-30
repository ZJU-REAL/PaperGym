# ResearchPlanGen

ResearchPlanGen 回答生成与 Rubric 评测工具。

## 文件说明

| 文件 | 说明 |
|------|------|
| `generate_responses.py` | 调用 LLM 生成研究计划回答 |
| `rubric_eval.py` | 根据 Rubric 对生成的计划进行评分（生成 + 评分） |
| `run_generate_responses_local.sh` | 本地 vLLM 批量生成回答 |
| `run_rubric_eval_local.sh` | 本地 vLLM 批量评测（生成 + 评分） |
| `run_rubric_eval_remote.sh` | 远程 API 批量评测 |
| `test_mock.py` | Mock 测试脚本，无需调用 LLM API |
| `data/` | Parquet 数据集目录（ResearchPlanGen_ml.parquet） |

## 运行方式

### 1. 本地生成回答

```bash
bash run_generate_responses_local.sh
```

### 2. 本地评测（生成 + 评分）

```bash
bash run_rubric_eval_local.sh
```

### 3. 远程 API 评测

```bash
bash run_rubric_eval_remote.sh
```

### 4. Mock 测试（无需 LLM API）

```bash
python test_mock.py
```

## 数据格式

输入数据为 Parquet 格式，包含以下字段：
- `Goal`: 研究场景描述
- `Rubric`: 评分标准列表
- `Reference solution`: 参考解决方案
- `article_id` / `q_id`: 样本标识

## 评测流程

1. **生成（Generate）**: 调用 LLM 根据 Goal 生成 Proposed Research Plan
2. **评分（Grade）**: 使用 GRADING_PROMPT 对 Proposed Plan 按 Rubric 逐项评分
3. **汇总（Summarize）**: 统计每个数据集和整体的通过率

## 依赖

- Python 3.8+
- pyarrow
- openai（可选，用于实际 LLM 调用）
- vllm（可选，用于本地模型部署）

## 使用自己的 LLM API 服务

本工具支持通过 OpenAI 兼容 API 调用任意 LLM 服务。配置方式如下：

### 1. 修改脚本中的 API 配置

编辑 `run_rubric_eval_remote.sh`，设置你的 API 地址和密钥：

```bash
# 通用配置
API_BASE="http://your-api-address/v1"      # 替换为你的 API 地址
API_KEY="sk-your-api-key"                  # 替换为你的 API 密钥

# 评分模型配置
JUDGE_API_BASE="http://your-judge-api/v1"  # 评分模型 API 地址
JUDGE_API_KEY="sk-your-judge-api-key"      # 评分模型 API 密钥
```

### 2. 命令行参数方式（推荐）

直接在命令行传入 API 参数：

```bash
# 远程 API 评测
API_BASE=http://your-api/v1 API_KEY=sk-your-key bash run_rubric_eval_remote.sh

# 分别指定生成和评分的 API
GEN_API_BASE=http://gen-api/v1 GEN_API_KEY=sk-gen-key \
JUDGE_API_BASE=http://judge-api/v1 JUDGE_API_KEY=sk-judge-key \
bash run_rubric_eval_remote.sh
```

### 3. Python 脚本直接调用

```bash
python rubric_eval.py \
    --base-url http://your-api/v1 \
    --api-key sk-your-key \
    --model your-model-name \
    --data-dir data/ \
    --output-dir results/
```

### 4. 本地 vLLM 部署

如果使用本地 vLLM 服务，参考 `run_rubric_eval_local.sh`：

```bash
bash run_rubric_eval_local.sh
```

本地服务默认使用 `http://127.0.0.1:8019/v1`，无需 API Key（设为 `EMPTY` 即可）。
