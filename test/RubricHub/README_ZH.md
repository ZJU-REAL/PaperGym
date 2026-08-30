# RubricHub

基于 Rubric 的 LLM 回答评分框架，支持生成回答、按 rubric 评分、结果可视化。

## 目录结构

```
RubricHub/
├── generate_responses.py              # 生成 LLM 回答
├── score_responses.py                 # 逐条 rubric 评分
├── score_responses_batch.py           # 批量 rubric 评分（推荐）
├── view_scores.py                     # 结果可视化
├── run_generate_responses.sh          # 本地 vLLM 生成脚本
├── run_generate_responses_remote.sh   # 远程 API 生成脚本
├── run_score_responses_batch.sh       # 批量评分脚本
├── dataset/
│   └── eval_set_Science_800.parquet   # 评测数据集
└── output/                            # 输出目录
```

## 快速开始

```bash
# 1. 安装依赖
pip install pandas numpy openai

# 2. 生成回答（本地 vLLM）
bash run_generate_responses.sh

# 3. 批量评分
bash run_score_responses_batch.sh

# 4. 查看结果
python view_scores.py --scores_file output/scores/result.json
```

## 核心概念

**Rubric**：每条评分标准包含 `id`、`description`、`weight`，LLM 按 rubric 逐条评判并给出 0/1 分。

## 主要脚本

| 脚本 | 功能 |
|------|------|
| `generate_responses.py` | 调用 LLM 生成回答 |
| `score_responses_batch.py` | 批量 rubric 评分（一次 LLM 调用评判所有 rubric） |
| `view_scores.py` | 表格形式展示评分结果 |

## 典型流程

```bash
# 生成回答 → 评分 → 查看结果
bash run_generate_responses.sh
bash run_score_responses_batch.sh
python view_scores.py --scores_file output/scores/result.json
```

## 使用自己的 LLM API 服务

本工具支持通过 OpenAI 兼容 API 调用任意 LLM 服务完成回答生成和评分。

### 1. 生成回答

#### 修改脚本配置

编辑 `run_generate_responses_remote.sh`，设置你的 API 地址和密钥：

```bash
API_BASE="http://your-api/v1"
API_KEY="sk-your-api-key"
DEFAULT_MODEL="your-model-name"
```

#### 命令行参数方式

```bash
python generate_responses.py \
    --data dataset/eval_set_Science_800.parquet \
    --output output/answers/responses.json \
    --api_base http://your-api/v1 \
    --api_key sk-your-api-key \
    --model your-model-name
```

### 2. 评分

#### 修改脚本配置

编辑 `run_score_responses_batch.sh`，设置评分模型的 API 地址和密钥：

```bash
API_BASE="http://your-api/v1"
API_KEY="sk-your-api-key"
MODEL="your-model-name"
```

#### 命令行参数方式

```bash
python score_responses_batch.py \
    --data dataset/eval_set_Science_800.parquet \
    --response output/answers/responses.json \
    --output output/scores/result.json \
    --api_base http://your-api/v1 \
    --api_key sk-your-api-key \
    --model your-model-name
```

### 3. 本地 vLLM 服务

如果使用本地 vLLM 部署：

```bash
# 生成回答
python generate_responses.py \
    --data dataset/eval_set_Science_800.parquet \
    --output output/answers/responses.json \
    --api_base http://127.0.0.1:8000/v1 \
    --api_key EMPTY \
    --model your-local-model-name

# 评分
python score_responses_batch.py \
    --data dataset/eval_set_Science_800.parquet \
    --response output/answers/responses.json \
    --output output/scores/result.json \
    --api_base http://127.0.0.1:8000/v1 \
    --api_key EMPTY \
    --model your-local-model-name
```
