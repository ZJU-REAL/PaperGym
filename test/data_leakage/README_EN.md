# data_leakage

Dataset data leakage detection tool.

## File Description

| File | Description |
|------|------|
| `data_leakage_detector.py` | Main data leakage detection script |
| `test_mock.py` | Mock test script, no LLM API required |
| `dataset/` | Dataset directory |

## Usage

### 1. Run Data Leakage Detection

```bash
python data_leakage_detector.py --datasets dataset/<dataset_file>
```

### 2. Mock Testing (No LLM API)

```bash
python test_mock.py
```

## Detection Workflow

1. **Data Loading Phase (Phase 1)**: Read datasets in various formats and unify them into the (question, rubrics) format
2. **Data Leakage Analysis Phase (Phase 2)**: Batch-feed all rubrics of each question to the LLM to judge, in a single call, whether each rubric's content can be directly inferred from the question
3. **Compute and output leakage ratios**

## Supported Dataset Formats

- **Parquet format**:
  - `RubricHub_Science_800.parquet` (prompt + Rubrics/reward_model format)
  - `ResearchPlanGen_ml.parquet` / `ResearchPlanGen_arxiv.parquet` (Goal + Rubric format)
  - `PaperGym_*.parquet` (prompt + extra_info.merged_rubrics format)
  - `ai-co-scientist.parquet` (Goal + Rubric format)
- **JSON format**:
  - `research_qa_test.json` (ResearchQA/data/)
- **JSONL format**:
  - `HealthBench.jsonl` (prompt + rubrics format)

## Data Loaders

| Loader | Description |
|--------|------|
| `CSNewParquetLoader` | Loads cs_new.parquet format |
| `ResearchQATestMiniLoader` | Loads research_qa_test_mini.json format |
| `HealthBenchJsonlLoader` | Loads HealthBench.jsonl format |
| `AICoScientistParquetLoader` | Loads ai-co-scientist.parquet format |
| `PaperGymParquetLoader` | Loads PaperGym_*.parquet format |
| `EvalSetScienceParquetLoader` | Loads eval_set_Science_800.parquet format |

## Dependencies

- Python 3.8+
- pandas
- numpy
- pyarrow
- openai (optional, for actual LLM calls)
- tqdm
- retry

## Using Your Own LLM API Service

This tool supports calling any LLM service via an OpenAI-compatible API for data leakage detection.

### 1. Modify Default API Configuration in Scripts

Edit `data_leakage_detector.py` to set your API address and key:

```python
# Default API configuration (top of file)
DEFAULT_API_KEY = "sk-your-api-key"           # Replace with your API key
DEFAULT_BASE_URL = "http://your-api/v1"       # Replace with your API address
DEFAULT_MODEL = "your-model-name"             # Replace with your model name
```

### 2. Command-Line Arguments

```bash
python data_leakage_detector.py \
    --datasets dataset/your_data.parquet \
    --api-key sk-your-api-key \
    --base-url http://your-api/v1 \
    --model your-model-name
```

### 3. Environment Variables

```bash
export OPENAI_API_KEY="sk-your-api-key"
export OPENAI_BASE_URL="http://your-api/v1"
python data_leakage_detector.py --datasets dataset/your_data.parquet
```

### 4. Local vLLM Service

If using a locally deployed vLLM model:

```bash
python data_leakage_detector.py \
    --datasets dataset/your_data.parquet \
    --api-key EMPTY \
    --base-url http://127.0.0.1:8000/v1 \
    --model your-local-model-name
```
