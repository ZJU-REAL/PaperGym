# RubricHub

A Rubric-based LLM answer scoring framework that supports answer generation, rubric-based scoring, and result visualization.

## Directory Structure

```
RubricHub/
├── generate_responses.py              # Generate LLM answers
├── score_responses.py                 # Per-rubric scoring
├── score_responses_batch.py           # Batch rubric scoring (recommended)
├── view_scores.py                     # Result visualization
├── run_generate_responses.sh          # Local vLLM generation script
├── run_generate_responses_remote.sh   # Remote API generation script
├── run_score_responses_batch.sh       # Batch scoring script
├── dataset/
│   └── eval_set_Science_800.parquet   # Evaluation dataset
└── output/                            # Output directory
```

## Quick Start

```bash
# 1. Install dependencies
pip install pandas numpy openai

# 2. Generate answers (local vLLM)
bash run_generate_responses.sh

# 3. Batch scoring
bash run_score_responses_batch.sh

# 4. View results
python view_scores.py --scores_file output/scores/result.json
```

## Core Concept

**Rubric**: Each scoring criterion contains `id`, `description`, and `weight`. The LLM judges each rubric item and assigns a 0/1 score.

## Main Scripts

| Script | Description |
|------|------|
| `generate_responses.py` | Calls LLM to generate answers |
| `score_responses_batch.py` | Batch rubric scoring (judges all rubrics in one LLM call) |
| `view_scores.py` | Displays scoring results in table form |

## Typical Workflow

```bash
# Generate answers → Score → View results
bash run_generate_responses.sh
bash run_score_responses_batch.sh
python view_scores.py --scores_file output/scores/result.json
```

## Using Your Own LLM API Service

This tool supports calling any LLM service via an OpenAI-compatible API for answer generation and scoring.

### 1. Generate Answers

#### Modify Script Configuration

Edit `run_generate_responses_remote.sh` to set your API address and key:

```bash
API_BASE="http://your-api/v1"
API_KEY="sk-your-api-key"
DEFAULT_MODEL="your-model-name"
```

#### Command-Line Arguments

```bash
python generate_responses.py \
    --data dataset/eval_set_Science_800.parquet \
    --output output/answers/responses.json \
    --api_base http://your-api/v1 \
    --api_key sk-your-api-key \
    --model your-model-name
```

### 2. Score

#### Modify Script Configuration

Edit `run_score_responses_batch.sh` to set the scoring model's API address and key:

```bash
API_BASE="http://your-api/v1"
API_KEY="sk-your-api-key"
MODEL="your-model-name"
```

#### Command-Line Arguments

```bash
python score_responses_batch.py \
    --data dataset/eval_set_Science_800.parquet \
    --response output/answers/responses.json \
    --output output/scores/result.json \
    --api_base http://your-api/v1 \
    --api_key sk-your-api-key \
    --model your-model-name
```

### 3. Local vLLM Service

If using a local vLLM deployment:

```bash
# Generate answers
python generate_responses.py \
    --data dataset/eval_set_Science_800.parquet \
    --output output/answers/responses.json \
    --api_base http://127.0.0.1:8000/v1 \
    --api_key EMPTY \
    --model your-local-model-name

# Score
python score_responses_batch.py \
    --data dataset/eval_set_Science_800.parquet \
    --response output/answers/responses.json \
    --output output/scores/result.json \
    --api_base http://127.0.0.1:8000/v1 \
    --api_key EMPTY \
    --model your-local-model-name
```
