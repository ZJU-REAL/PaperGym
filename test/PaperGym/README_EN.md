# PaperGym

Category-based model response generation and scoring tool.

## File Description

| File | Description |
|------|------|
| `generate_responses.py` | Calls LLM to generate responses, supports multiple models and APIs |
| `score_responses.py` | Scores generated responses against rubrics |
| `show_scores.py` | Summarizes and displays scoring results |
| `show_scores_by_category.py` | Displays scoring results by category |
| `run_generate_design.sh` | Generates design-category responses |
| `run_generate_innov.sh` | Generates innov-category responses |
| `run_score.sh` | Runs the scoring workflow |
| `run_generate_all_remote.sh` | Batch-generates responses for all categories |
| `test_mock.py` | Mock test script, no LLM API required |
| `data/` | Parquet dataset directory |

## Usage

### 1. Generate Responses

```bash
# Generate design-category responses
bash run_generate_design.sh

# Generate innov-category responses
bash run_generate_innov.sh

# Batch-generate responses for all categories
bash run_generate_all_remote.sh
```

### 2. Score

```bash
bash run_score.sh
```

### 3. View Results

```bash
python show_scores.py
python show_scores_by_category.py
```

### 4. Mock Testing (No LLM API)

```bash
python test_mock.py
```

## Data Format

Input data is in Parquet format with the following fields:
- `id`: Unique question identifier
- `problem`: Question text
- `prompt`: Conversation-format prompt
- `data_source`: Category source (e.g., cs, econ, etc.)
- `rubric`: List of scoring criteria

## Dependencies

- Python 3.8+
- pandas
- numpy
- openai / anthropic (optional, for actual LLM calls)

## Using Your Own LLM API Service

This tool supports calling any LLM service via an OpenAI-compatible API for response generation and scoring.

### 1. Generate Responses

#### Modify Script Configuration

Edit `run_generate_all_remote.sh` to set your API address and key:

```bash
# Add your model to the MODELS array
MODELS=(
    "http://your-api/v1|sk-your-key|your-model-name|output_name|Display Name"
)
```

#### Command-Line Arguments

```bash
python generate_responses.py \
    --parquet_path data/design.parquet \
    --output_path answer/design/output.json \
    --api_key sk-your-api-key \
    --base_url http://your-api/v1 \
    --model your-model-name
```

### 2. Score

#### Modify Script Configuration

Edit `run_score.sh` to set the scoring model's API address and key:

```bash
SCORE_BASE_URL="http://your-api/v1"
SCORE_API_KEY="sk-your-api-key"
SCORE_MODEL="your-model-name"
```

#### Command-Line Arguments

```bash
python score_responses.py \
    --parquet_path data/design.parquet \
    --answer_path answer/design/output.json \
    --output_path score/design/output_scores.json \
    --api_key sk-your-api-key \
    --base_url http://your-api/v1 \
    --model your-model-name
```

### 3. Local vLLM Service

If using a local vLLM deployment:

```bash
# Generate responses
python generate_responses.py \
    --parquet_path data/design.parquet \
    --output_path answer/design/output.json \
    --api_key EMPTY \
    --base_url http://127.0.0.1:8000/v1 \
    --model your-local-model-name

# Score
python score_responses.py \
    --parquet_path data/design.parquet \
    --answer_path answer/design/output.json \
    --output_path score/design/output_scores.json \
    --api_key EMPTY \
    --base_url http://127.0.0.1:8000/v1 \
    --model your-local-model-name
```
