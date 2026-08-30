# Stability

A toolkit for evaluating the stability of LLM scoring, analyzing scoring consistency through multiple rounds of independent evaluation.

## Directory Structure

```
stability/
├── stability_eval.py              # Main stability evaluation script
├── analyze_consistency.py         # Consistency analysis
├── analyze_rubric_agreement.py    # Rubric consistency analysis
├── compare_scoring_models.py    # Scoring model comparison
├── run_generate_stability.sh      # Convenience run script
├── data/
│   └── stability.parquet          # Input dataset
└── output/                        # Output directory (auto-created)
```

## Quick Start

```bash
# 1. Install dependencies
pip install pandas numpy openai

# 2. Edit run_generate_stability.sh to configure parameters

# 3. Run the stability evaluation
bash run_generate_stability.sh
```

## Core Features

Analyzes, through multiple rounds of independent evaluation (e.g., 10 rounds):
- **Scoring consistency**: Score fluctuation of the same answer across evaluation rounds
- **Rubric consistency**: Stability of scoring across different rubrics
- **Model comparison**: Consistency differences across different scoring models

## Main Scripts

| Script | Description |
|------|------|
| `stability_eval.py` | Main evaluation script, runs multiple rounds of independent scoring |
| `analyze_consistency.py` | Analyzes scoring consistency (mean, variance, standard deviation) |
| `analyze_rubric_agreement.py` | Analyzes the agreement of each rubric's scoring |
| `compare_scoring_models.py` | Compares the stability of different scoring models |

## Typical Workflow

```bash
# Run the stability evaluation
bash run_generate_stability.sh

# Analyze results
python analyze_consistency.py --input output/stability_results.json
python analyze_rubric_agreement.py --input output/stability_results.json
```

## Using Your Own LLM API Service

This tool supports calling any LLM service via an OpenAI-compatible API for stability evaluation.

### 1. Modify API Configuration in Scripts

Edit `run_generate_stability.sh` to set your API address and key:

```bash
# LLM model parameters
MODEL="your-model-name"
API_KEY="sk-your-api-key"
BASE_URL="http://your-api/v1"
```

### 2. Command-Line Arguments

```bash
python stability_eval.py \
    --input data/stability.parquet \
    --output output/stability_results.json \
    --api_key sk-your-api-key \
    --base_url http://your-api/v1 \
    --model your-model-name \
    --rounds 10
```

### 3. Local vLLM Service

```bash
python stability_eval.py \
    --input data/stability.parquet \
    --output output/stability_results.json \
    --api_key EMPTY \
    --base_url http://127.0.0.1:8000/v1 \
    --model your-local-model-name \
    --rounds 10
```
