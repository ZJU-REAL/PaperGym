# ResearchPlanGen

ResearchPlanGen answer generation and Rubric evaluation tool.

## File Description

| File | Description |
|------|------|
| `generate_responses.py` | Calls LLM to generate research plan answers |
| `rubric_eval.py` | Scores generated plans against Rubrics (generation + scoring) |
| `run_generate_responses_local.sh` | Batch generation via local vLLM |
| `run_rubric_eval_local.sh` | Batch evaluation via local vLLM (generation + scoring) |
| `run_rubric_eval_remote.sh` | Batch evaluation via remote API |
| `test_mock.py` | Mock test script, no LLM API required |
| `data/` | Parquet dataset directory (ResearchPlanGen_ml.parquet) |

## Usage

### 1. Generate Answers Locally

```bash
bash run_generate_responses_local.sh
```

### 2. Local Evaluation (Generate + Score)

```bash
bash run_rubric_eval_local.sh
```

### 3. Remote API Evaluation

```bash
bash run_rubric_eval_remote.sh
```

### 4. Mock Testing (No LLM API)

```bash
python test_mock.py
```

## Data Format

Input data is in Parquet format with the following fields:
- `Goal`: Research scenario description
- `Rubric`: List of scoring criteria
- `Reference solution`: Reference solution
- `article_id` / `q_id`: Sample identifiers

## Evaluation Workflow

1. **Generate**: Call LLM to generate a Proposed Research Plan based on the Goal
2. **Grade**: Use GRADING_PROMPT to score the Proposed Plan against each Rubric item
3. **Summarize**: Compute pass rates for each dataset and overall

## Dependencies

- Python 3.8+
- pyarrow
- openai (optional, for actual LLM calls)
- vllm (optional, for local model deployment)

## Using Your Own LLM API Service

This tool supports calling any LLM service via an OpenAI-compatible API. Configuration methods are as follows:

### 1. Modify API Configuration in Scripts

Edit `run_rubric_eval_remote.sh` to set your API address and key:

```bash
# General configuration
API_BASE="http://your-api-address/v1"      # Replace with your API address
API_KEY="sk-your-api-key"                  # Replace with your API key

# Scoring model configuration
JUDGE_API_BASE="http://your-judge-api/v1"  # Scoring model API address
JUDGE_API_KEY="sk-your-judge-api-key"      # Scoring model API key
```

### 2. Command-Line Environment Variables (Recommended)

Pass API parameters directly on the command line:

```bash
# Remote API evaluation
API_BASE=http://your-api/v1 API_KEY=sk-your-key bash run_rubric_eval_remote.sh

# Specify separate APIs for generation and scoring
GEN_API_BASE=http://gen-api/v1 GEN_API_KEY=sk-gen-key \
JUDGE_API_BASE=http://judge-api/v1 JUDGE_API_KEY=sk-judge-key \
bash run_rubric_eval_remote.sh
```

### 3. Direct Python Script Invocation

```bash
python rubric_eval.py \
    --base-url http://your-api/v1 \
    --api-key sk-your-key \
    --model your-model-name \
    --data-dir data/ \
    --output-dir results/
```

### 4. Local vLLM Deployment

If using a local vLLM service, refer to `run_rubric_eval_local.sh`:

```bash
bash run_rubric_eval_local.sh
```

The local service defaults to `http://127.0.0.1:8019/v1` and does not require an API key (set to `EMPTY`).
