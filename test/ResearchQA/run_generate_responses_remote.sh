#!/bin/bash
#
# Script for running generate_responses_no_think.py
# Does not start a local vLLM; calls the remote LLM API service directly to generate answers
# Supports processing multiple remote models at once, running in a loop
#

# ========== Dataset path ==========
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_PATH="$SCRIPT_DIR/data/research_qa.json"

# ========== General parameters ==========
# Number of concurrent threads
NUM_WORKERS=10

# Maximum number of tokens to generate per answer
MAX_TOKENS=10240

# ========== Model configuration list ==========
# The four arrays correspond index-by-index; each index represents one remote model.
# To add a new model, append the corresponding entry at the end of each of the four arrays below.
#
# Note: if an API_KEY is not needed for a row (e.g. a custom vLLM that does not verify keys), fill in "EMPTY".
#
# Example (commented out):
#   "https://api.siliconflow.cn/v1"
#   "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
#   "Qwen/Qwen3-32B"
#   "${SCRIPT_DIR}/answer_mini/qwen3_silicon_responses.json"
# =================================

# API base URL list
API_BASES=(
    "http://xxx/"
)

# API key list (must correspond one-to-one with API_BASES order)
API_KEYS=(
    "sk-xxx"
)

# Model name list (must correspond one-to-one with API_BASES order)
MODELS=(
    "glm-5.1"
)

# Output file path list (must correspond one-to-one with API_BASES order)
OUTPUT_PATHS=(
    "$SCRIPT_DIR/answer/glm_5.1_responses.json"
)

# ========== No need to modify below ==========
NUM_MODELS=${#API_BASES[@]}

echo ""
echo "=========================================="
echo "  Remote model batch answer generation"
echo "  Dataset:    $DATA_PATH"
echo "  Models:     $NUM_MODELS"
echo "=========================================="

for (( i=0; i<NUM_MODELS; i++ )); do
    API_BASE="${API_BASES[$i]}"
    API_KEY="${API_KEYS[$i]}"
    MODEL="${MODELS[$i]}"
    OUTPUT_PATH="${OUTPUT_PATHS[$i]}"

    # Ensure output directory exists
    mkdir -p "$(dirname "$OUTPUT_PATH")"

    # Skip if the output file already exists
    if [ -f "$OUTPUT_PATH" ]; then
        echo ""
        echo "=================================="
        echo "  Skipping: output file already exists"
        echo "  Model:      $MODEL"
        echo "  Output:     $OUTPUT_PATH"
        echo "=================================="
        continue
    fi

    echo ""
    echo "=========================================="
    echo "  Generating answers [$((i+1))/$NUM_MODELS]"
    echo "  Data: $DATA_PATH"
    echo "  API:  $API_BASE"
    echo "  Model: $MODEL"
    echo "  Output:     $OUTPUT_PATH"
    echo "=========================================="

    python "$SCRIPT_DIR/generate_responses.py" \
        --data "$DATA_PATH" \
        --output "$OUTPUT_PATH" \
        --api_base "$API_BASE" \
        --api_key "$API_KEY" \
        --model "$MODEL" \
        --num_workers "$NUM_WORKERS" \
        --max_tokens "$MAX_TOKENS"

    echo ""
    echo "=================================="
    echo "  Finished [$((i+1))/$NUM_MODELS]"
    echo "  Output:     $OUTPUT_PATH"
    echo "=================================="
done

echo ""
echo "=========================================="
echo "  All models processed"
echo "=========================================="
