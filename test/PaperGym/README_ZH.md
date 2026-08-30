# PaperGym

基于领域分类的模型响应生成与评分工具。

## 文件说明

| 文件 | 说明 |
|------|------|
| `generate_responses.py` | 调用 LLM 生成响应，支持多种模型和 API |
| `score_responses.py` | 根据 rubric 对生成的响应进行评分 |
| `show_scores.py` | 汇总并展示评分结果 |
| `show_scores_by_category.py` | 按领域类别展示评分结果 |
| `run_generate_design.sh` | 生成 design 领域响应 |
| `run_generate_innov.sh` | 生成 innov 领域响应 |
| `run_score.sh` | 运行评分流程 |
| `run_generate_all_remote.sh` | 批量生成所有领域响应 |
| `test_mock.py` | Mock 测试脚本，无需调用 LLM API |
| `data/` | Parquet 数据集目录 |

## 运行方式

### 1. 生成响应

```bash
# 生成 design 领域响应
bash run_generate_design.sh

# 生成 innov 领域响应
bash run_generate_innov.sh

# 批量生成所有领域响应
bash run_generate_all_remote.sh
```

### 2. 评分

```bash
bash run_score.sh
```

### 3. 查看结果

```bash
python show_scores.py
python show_scores_by_category.py
```

### 4. Mock 测试（无需 LLM API）

```bash
python test_mock.py
```

## 数据格式

输入数据为 Parquet 格式，包含以下字段：
- `id`: 问题唯一标识
- `problem`: 问题文本
- `prompt`: 对话格式的 prompt
- `data_source`: 领域来源（如 cs、econ 等）
- `rubric`: 评分标准列表

## 依赖

- Python 3.8+
- pandas
- numpy
- openai / anthropic（可选，用于实际 LLM 调用）

## 使用自己的 LLM API 服务

本工具支持通过 OpenAI 兼容 API 调用任意 LLM 服务完成响应生成和评分。

### 1. 生成响应

#### 修改脚本配置

编辑 `run_generate_all_remote.sh`，设置你的 API 地址和密钥：

```bash
# 在 MODELS 数组中添加你的模型
MODELS=(
    "http://your-api/v1|sk-your-key|your-model-name|output_name|Display Name"
)
```

#### 命令行参数方式

```bash
python generate_responses.py \
    --parquet_path data/design.parquet \
    --output_path answer/design/output.json \
    --api_key sk-your-api-key \
    --base_url http://your-api/v1 \
    --model your-model-name
```

### 2. 评分

#### 修改脚本配置

编辑 `run_score.sh`，设置评分模型的 API 地址和密钥：

```bash
SCORE_BASE_URL="http://your-api/v1"
SCORE_API_KEY="sk-your-api-key"
SCORE_MODEL="your-model-name"
```

#### 命令行参数方式

```bash
python score_responses.py \
    --parquet_path data/design.parquet \
    --answer_path answer/design/output.json \
    --output_path score/design/output_scores.json \
    --api_key sk-your-api-key \
    --base_url http://your-api/v1 \
    --model your-model-name
```

### 3. 本地 vLLM 服务

如果使用本地 vLLM 部署：

```bash
# 生成响应
python generate_responses.py \
    --parquet_path data/design.parquet \
    --output_path answer/design/output.json \
    --api_key EMPTY \
    --base_url http://127.0.0.1:8000/v1 \
    --model your-local-model-name

# 评分
python score_responses.py \
    --parquet_path data/design.parquet \
    --answer_path answer/design/output.json \
    --output_path score/design/output_scores.json \
    --api_key EMPTY \
    --base_url http://127.0.0.1:8000/v1 \
    --model your-local-model-name
```
