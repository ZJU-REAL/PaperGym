# compare

模型回答比较工具，支持两模型和三模型的盲评比较。

## 文件说明

| 文件 | 说明 |
|------|------|
| `compare_responses.py` | 两模型回答比较（调用 LLM API 评分） |
| `compare_three.py` | 三模型回答比较（每维度仅选出一个最佳） |
| `aggregate_results.py` | 聚合两模型比较结果 |
| `aggregate_three_results.py` | 聚合三模型比较结果 |
| `parse_results.py` | 解析比较结果 |
| `run_compare.sh` | 运行两模型比较 |
| `run_compare_three.sh` | 运行三模型比较 |
| `test_mock.py` | Mock 测试脚本，无需调用 LLM API |
| `data/` | 数据集目录 |
| `answer/` | 模型回答文件目录 |

## 运行方式

### 1. 两模型比较

```bash
bash run_compare.sh [数据集]
```

### 2. 三模型比较

```bash
bash run_compare_three.sh
```

### 3. Mock 测试（无需 LLM API）

```bash
python test_mock.py
```

## 比较流程

1. **加载回答**: 从 `answer/<数据集>/` 目录加载两个/三个模型的回答 JSON 文件
2. **盲评**: 随机分配 Plan A/B/C，调用 LLM 进行评分
3. **解析结果**: 从 LLM 输出中解析 XML 格式的评分结果
4. **汇总统计**: 统计每个模型在每个维度上的胜率

## 数据格式

回答文件为 JSON 格式，key 为 `idx_X`，value 包含：
- `question`: 问题文本
- `answer`: 模型回答

## 依赖

- Python 3.8+
- openai
- retry

## 使用自己的 LLM API 服务

本工具支持通过 OpenAI 兼容 API 调用任意 LLM 服务完成模型回答比较。

### 1. 修改脚本中的 API 配置

编辑 `run_compare.sh` 或 `run_compare_three.sh`，设置你的 API 地址和密钥：

```bash
BASE_URL="http://your-api/v1"
API_KEY="sk-your-api-key"
MODEL="your-model-name"
```

### 2. 命令行参数方式

#### 两模型比较

```bash
python compare_responses.py \
    --data data/your_data.parquet \
    --answer_a answer/model_a.json \
    --answer_b answer/model_b.json \
    --output output/compare_result.json \
    --api_key sk-your-api-key \
    --base_url http://your-api/v1 \
    --model your-model-name
```

#### 三模型比较

```bash
python compare_three.py \
    --data data/your_data.parquet \
    --answer_a answer/model_a.json \
    --answer_b answer/model_b.json \
    --answer_c answer/model_c.json \
    --output output/compare_three_result.json \
    --api_key sk-your-api-key \
    --base_url http://your-api/v1 \
    --model your-model-name
```

### 3. 本地 vLLM 服务

```bash
# 两模型比较
python compare_responses.py \
    --data data/your_data.parquet \
    --answer_a answer/model_a.json \
    --answer_b answer/model_b.json \
    --output output/compare_result.json \
    --api_key EMPTY \
    --base_url http://127.0.0.1:8000/v1 \
    --model your-local-model-name

# 三模型比较
python compare_three.py \
    --data data/your_data.parquet \
    --answer_a answer/model_a.json \
    --answer_b answer/model_b.json \
    --answer_c answer/model_c.json \
    --output output/compare_three_result.json \
    --api_key EMPTY \
    --base_url http://127.0.0.1:8000/v1 \
    --model your-local-model-name
```
