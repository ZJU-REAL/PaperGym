# Rubric Rate

Analyzes the type composition ratio of **Rubrics** in academic paper datasets (Method vs Experiment).

## Directory Structure

```
rubric_rate/
├── analyze_rubric_types.py      # Core analysis script
├── run_analyze_rubric_types.sh  # Convenience run script
├── test_mock.py                 # Mock test script
├── data/
│   └── rubric_rate.parquet      # Input dataset
└── output/                      # Output directory (auto-created)
```

## Quick Start

```bash
# 1. Install dependencies
pip install pandas numpy openai retry

# 2. Edit run_analyze_rubric_types.sh to configure API parameters

# 3. Run the analysis
bash run_analyze_rubric_types.sh
```

## Core Features

Uses the LLM to automatically classify each rubric as **Method** or **Experiment**, outputting:
- Overall Method/Experiment ratio
- Ratio statistics within each subject category

## Main Parameters

```bash
python analyze_rubric_types.py \
    --parquet_path data/rubric_rate.parquet \  # Input data
    --output_dir output \                        # Output directory
    --rubric_key rubrics \                       # Rubric collection key
    --categories cs,econ,physics \                # Specify subjects (optional)
    --num_workers 10                             # Number of parallel threads
```

## Output Files

- `rubric_classification_detail.json` — Per-rubric classification results
- `rubric_classification_stats.json` — Statistical summary

## Mock Testing

```bash
python test_mock.py
```

Verifies core logic and path configuration without calling any LLM API.

## Using Your Own LLM API Service

This tool supports calling any LLM service via an OpenAI-compatible API for rubric type analysis.

### 1. Modify API Configuration in Scripts

Edit `run_analyze_rubric_types.sh` to set your API address and key:

```bash
# LLM model parameters
MODEL="your-model-name"
API_KEY="sk-your-api-key"
BASE_URL="http://your-api/v1"
```

### 2. Command-Line Arguments

```bash
python analyze_rubric_types.py \
    --parquet_path data/rubric_rate.parquet \
    --api_key sk-your-api-key \
    --base_url http://your-api/v1 \
    --model your-model-name
```

### 3. Local vLLM Service

```bash
python analyze_rubric_types.py \
    --parquet_path data/rubric_rate.parquet \
    --api_key EMPTY \
    --base_url http://127.0.0.1:8000/v1 \
    --model your-local-model-name
```
