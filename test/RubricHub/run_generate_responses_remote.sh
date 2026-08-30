#!/bin/bash
#
# Remote LLM API answer generation script.
# Only sends requests to a remote OpenAI-compatible API to generate answers; does not start a local vLLM service.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ========== Remote API configuration (modify according to actual conditions) ==========

# Remote LLM API address (OpenAI-compatible format)
# API_BASE="http://xxx/"
# API_KEY="sk-xxx"
API_BASE="http://xxx/"
API_KEY="sk-xxx"
# Default model name (not used when overridden by the first field of MODELS)
DEFAULT_MODEL="Qwen/Qwen3-32B"

# ========== Data and output configuration ==========

DATA_PATH="$SCRIPT_DIR/dataset/eval_set_Science_800.parquet"
OUTPUT_DIR="$SCRIPT_DIR/output/answers"
# Note: this script uses SCRIPT_DIR as the base path, ensuring it runs correctly from any directory

# Multithreaded client parameters
NUM_WORKERS=10
MAX_TOKENS=10240

# Model list: model name|output file|display name
# Note: the model name is determined by the remote API (e.g. `deepseek-chat`, gpt-4o, etc.), not a local path
MODELS=(
    # "DeepSeek-V4-Flash|$OUTPUT_DIR/deeepseek_v4_flash_responses.json|DeepSeek-V4-Flash"
)

# ========== Signal handling ==========

PYTHON_PID=""

cleanup() {
    echo "Signal received, terminating Python process..."
    if [[ -n $PYTHON_PID ]] && kill -0 "$PYTHON_PID" 2>/dev/null; then
        kill -TERM "$PYTHON_PID" 2>/dev/null
        sleep 2
        if kill -0 "$PYTHON_PID" 2>/dev/null; then
            echo "Force-killing Python process..."
            kill -KILL "$PYTHON_PID" 2>/dev/null
        fi
    fi
    exit 1
}

trap cleanup SIGTERM SIGINT SIGQUIT SIGHUP

# ========== Run ==========

mkdir -p "$OUTPUT_DIR"

for model_entry in "${MODELS[@]}"; do
    IFS='|' read -r model_name output_path display_name <<< "$model_entry"

    if [[ -f "$output_path" ]]; then
        echo ""
        echo "========================================"
        echo "  Model: $display_name"
        echo "  Output file already exists, skipping: $output_path"
        echo "========================================"
        continue
    fi

    echo ""
    echo "========================================"
    echo "  Model: $display_name"
    echo "  Remote model name: $model_name"
    echo "  API: $API_BASE"
    echo "========================================"
    echo ""

    python -u "$SCRIPT_DIR/generate_responses.py" \
        --data "$DATA_PATH" \
        --output "$output_path" \
        --api_base "$API_BASE" \
        --api_key "$API_KEY" \
        --model "$model_name" \
        --num_workers "$NUM_WORKERS" \
        --max_tokens "$MAX_TOKENS" &
    PYTHON_PID=$!
    wait "$PYTHON_PID"
    PYTHON_PID=""

    echo ""
    echo "  $display_name generation complete"
done

echo ""
echo "========================================"
echo "  All generation complete"
echo "  Output files:"
for model_entry in "${MODELS[@]}"; do
    IFS='|' read -r _ output_path display_name <<< "$model_entry"
    echo "    - $output_path ($display_name)"
done
echo "========================================"
