# data_leakage

数据集数据泄露检测工具。

## 文件说明

| 文件 | 说明 |
|------|------|
| `data_leakage_detector.py` | 数据泄露检测主脚本 |
| `test_mock.py` | Mock 测试脚本，无需调用 LLM API |
| `dataset/` | 数据集目录 |

## 运行方式

### 1. 运行数据泄露检测

```bash
python data_leakage_detector.py --datasets dataset/<数据集文件>
```

### 2. Mock 测试（无需 LLM API）

```bash
python test_mock.py
```

## 检测流程

1. **数据加载阶段（Phase 1）**: 读取不同格式的数据集，统一为 (question, rubrics) 格式
2. **数据泄露分析阶段（Phase 2）**: 对每个问题的所有 rubric 批量输入 LLM，一次性判断每条 rubric 内容是否可以从问题中直接推断
3. **统计并输出泄露比例**

## 支持的数据集格式

- **Parquet 格式**:
  - `RubricHub_Science_800.parquet` (prompt + Rubrics/reward_model 格式)
  - `ResearchPlanGen_ml.parquet` / `ResearchPlanGen_arxiv.parquet` (Goal + Rubric 格式)
  - `PaperGym_*.parquet` (prompt + extra_info.merged_rubrics 格式)
  - `ai-co-scientist.parquet` (Goal + Rubric 格式)
- **JSON 格式**:
  - `research_qa_test.json` (ResearchQA/data/)
- **JSONL 格式**:
  - `HealthBench.jsonl` (prompt + rubrics 格式)

## 数据加载器

| 加载器 | 说明 |
|--------|------|
| `CSNewParquetLoader` | 加载 cs_new.parquet 格式 |
| `ResearchQATestMiniLoader` | 加载 research_qa_test_mini.json 格式 |
| `HealthBenchJsonlLoader` | 加载 HealthBench.jsonl 格式 |
| `AICoScientistParquetLoader` | 加载 ai-co-scientist.parquet 格式 |
| `PaperGymParquetLoader` | 加载 PaperGym_*.parquet 格式 |
| `EvalSetScienceParquetLoader` | 加载 eval_set_Science_800.parquet 格式 |

## 依赖

- Python 3.8+
- pandas
- numpy
- pyarrow
- openai（可选，用于实际 LLM 调用）
- tqdm
- retry

## 使用自己的 LLM API 服务

本工具支持通过 OpenAI 兼容 API 调用任意 LLM 服务进行数据泄露检测。

### 1. 修改脚本中的默认 API 配置

编辑 `data_leakage_detector.py`，设置你的 API 地址和密钥：

```python
# 默认 API 配置（文件顶部）
DEFAULT_API_KEY = "sk-your-api-key"           # 替换为你的 API 密钥
DEFAULT_BASE_URL = "http://your-api/v1"       # 替换为你的 API 地址
DEFAULT_MODEL = "your-model-name"             # 替换为你的模型名称
```

### 2. 命令行参数方式

```bash
python data_leakage_detector.py \
    --datasets dataset/your_data.parquet \
    --api-key sk-your-api-key \
    --base-url http://your-api/v1 \
    --model your-model-name
```

### 3. 环境变量方式

```bash
export OPENAI_API_KEY="sk-your-api-key"
export OPENAI_BASE_URL="http://your-api/v1"
python data_leakage_detector.py --datasets dataset/your_data.parquet
```

### 4. 本地 vLLM 服务

如果使用本地 vLLM 部署的模型：

```bash
python data_leakage_detector.py \
    --datasets dataset/your_data.parquet \
    --api-key EMPTY \
    --base-url http://127.0.0.1:8000/v1 \
    --model your-local-model-name
```
