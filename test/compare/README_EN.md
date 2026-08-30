# compare

Model answer comparison tool supporting blind comparison between two and three models.

## File Description

| File | Description |
|------|------|
| `compare_responses.py` | Two-model answer comparison (LLM API scoring) |
| `compare_three.py` | Three-model answer comparison (selects only one best per dimension) |
| `aggregate_results.py` | Aggregates two-model comparison results |
| `aggregate_three_results.py` | Aggregates three-model comparison results |
| `parse_results.py` | Parses comparison results |
| `run_compare.sh` | Runs two-model comparison |
| `run_compare_three.sh` | Runs three-model comparison |
| `test_mock.py` | Mock test script, no LLM API required |
| `data/` | Dataset directory |
| `answer/` | Model answer files directory |

## Usage

### 1. Two-Model Comparison

```bash
bash run_compare.sh [dataset]
```

### 2. Three-Model Comparison

```bash
bash run_compare_three.sh
```

### 3. Mock Testing (No LLM API)

```bash
python test_mock.py
```

## Comparison Workflow

1. **Load Answers**: Load answer JSON files of two/three models from `answer/<dataset>/` directory
2. **Blind Evaluation**: Randomly assign Plan A/B/C, call LLM to score
3. **Parse Results**: Parse XML-formatted scoring results from LLM output
4. **Aggregate Statistics**: Compute the win rate of each model across each dimension

## Data Format

Answer files are in JSON format, keyed by `idx_X`, where each value contains:
- `question`: Question text
- `answer`: Model answer

## Dependencies

- Python 3.8+
- openai
- retry

## Using Your Own LLM API Service

This tool supports calling any LLM service via an OpenAI-compatible API for model answer comparison.

### 1. Modify API Configuration in Scripts

Edit `run_compare.sh` or `run_compare_three.sh` to set your API address and key:

```bash
BASE_URL="http://your-api/v1"
API_KEY="sk-your-api-key"
MODEL="your-model-name"
```

### 2. Command-Line Arguments

#### Two-Model Comparison

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

#### Three-Model Comparison

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

### 3. Local vLLM Service

```bash
# Two-model comparison
python compare_responses.py \
    --data data/your_data.parquet \
    --answer_a answer/model_a.json \
    --answer_b answer/model_b.json \
    --output output/compare_result.json \
    --api_key EMPTY \
    --base_url http://127.0.0.1:8000/v1 \
    --model your-local-model-name

# Three-model comparison
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
