# ResearchQA

A framework for LLM answer generation and scoring based on the ResearchQA dataset.

## Directory Structure

```
ResearchQA/
├── generate_responses.py              # Multi-threaded LLM answer generation
├── compute_coverage_ds.py             # Scoring script (parallel support)
├── show_results.py                    # Result display
├── researchqa_loader.py               # Data loading utility
├── run_generate_responses.sh          # Local vLLM generation script
├── run_generate_responses_remote.sh   # Remote API generation script
├── run_compute_coverage.sh            # Scoring script
├── data/
│   └── research_qa.json               # Dataset
├── answer/                            # Generated answers (auto-created)
└── output/                            # Scoring results (auto-created)
```

## Quick Start

```bash
# 1. Install dependencies
pip install openai tqdm retry

# 2. Generate answers (local vLLM)
bash run_generate_responses.sh

# 3. Score
bash run_compute_coverage.sh

# 4. View results
python show_results.py --dir output/
```

## Core Features

**Answer Generation**: Multi-threaded LLM API calls to generate answers for each question in the ResearchQA dataset.

**Scoring**: Uses LLM to judge answer quality against each rubric item, producing a 5-level score:
- Not at all → Barely → Moderately → Mostly → Completely

**Result Display**: Summarizes the average coverage of each model and outputs it in table form.

## Main Scripts

| Script | Description |
|------|------|
| `generate_responses.py` | Multi-threaded answer generation |
| `compute_coverage_ds.py` | Parallel rubric scoring |
| `show_results.py` | Result summary display |
| `researchqa_loader.py` | Loads ResearchQA JSON data |

## Typical Workflow

```bash
# Generate answers → Score → View results
bash run_generate_responses.sh
bash run_compute_coverage.sh
python show_results.py --dir output/
```

## Mock Testing

```bash
python test_mock.py
```

Verifies core logic and path configuration without calling any LLM API.

## Using Your Own LLM API Service

This tool supports calling any LLM service via an OpenAI-compatible API for answer generation and scoring.

### 1. Generate Answers

#### Modify Script Configuration

Edit `run_generate_responses_remote.sh` to set your API address and key:

```bash
API_BASE="http://your-api/v1"
API_KEY="sk-your-api-key"
MODEL="your-model-name"
```

#### Command-Line Arguments

```bash
python generate_responses.py \
    --data data/research_qa.json \
    --output answer/responses.json \
    --api_base http://your-api/v1 \
    --api_key sk-your-api-key \
    --model your-model-name
```

### 2. Score

#### Modify Script Configuration

Edit `run_compute_coverage.sh` to set the scoring model's API address and key:

```bash
API_BASE="http://your-api/v1"
API_KEY="sk-your-api-key"
MODEL="your-model-name"
```

#### Command-Line Arguments

```bash
python compute_coverage_ds.py \
    --input answer/responses.json \
    --output output/scores.json \
    --api_base http://your-api/v1 \
    --api_key sk-your-api-key \
    --model your-model-name
```

### 3. Local vLLM Service

If using a local vLLM deployment:

```bash
# Generate answers
python generate_responses.py \
    --data data/research_qa.json \
    --output answer/responses.json \
    --api_base http://127.0.0.1:8000/v1 \
    --api_key EMPTY \
    --model your-local-model-name

# Score
python compute_coverage_ds.py \
    --input answer/responses.json \
    --output output/scores.json \
    --api_base http://127.0.0.1:8000/v1 \
    --api_key EMPTY \
    --model your-local-model-name
```
