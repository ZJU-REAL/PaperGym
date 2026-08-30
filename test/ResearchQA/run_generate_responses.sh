#!/bin/bash
#
# Start the vLLM service in sequence and call the multithreaded client to generate answers
#

set -e

# Specify GPU to use
export CUDA_VISIBLE_DEVICES=2

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ========== Configuration parameters ==========

# DATA_PATH="$SCRIPT_DIR/data/research_qa_test_micro.json"
DATA_PATH="$SCRIPT_DIR/data/research_qa.json"
OUTPUT_DIR="$SCRIPT_DIR/answer"
# DATA_PATH="$SCRIPT_DIR/data/research_qa_validate.json"
# OUTPUT_DIR="$SCRIPT_DIR/answer_validate"

# vLLM service port
API_PORT=8020
API_BASE="http://127.0.0.1:${API_PORT}/v1"
API_KEY="EMPTY"

# Multithreaded client parameters
NUM_WORKERS=20
MAX_TOKENS=10240

# vLLM service parameters
TENSOR_PARALLEL_SIZE=1
MAX_MODEL_LEN=20480
GPU_MEMORY_UTILIZATION=0.9

# Model list: path output file name port
# Model root (override via the MODEL_ROOT env var)
MODEL_ROOT="${MODEL_ROOT:-$HOME/.cache/modelscope/hub/models}"
MODELS=(
#   "${MODEL_ROOT}/Qwen/Qwen3-1.7B|$OUTPUT_DIR/qwen3_1b_responses.json|Qwen3-1B"
)

# ========== Signal handling ==========

VLLM_PID=""
PYTHON_PID=""

cleanup() {
    echo "Signal received, cleaning up..."
    if [[ -n $VLLM_PID ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Stopping the vLLM service (PID=$VLLM_PID)..."
        kill -TERM "$VLLM_PID" 2>/dev/null
        sleep 2
        kill -KILL "$VLLM_PID" 2>/dev/null
    fi
    if [[ -n $PYTHON_PID ]] && kill -0 "$PYTHON_PID" 2>/dev/null; then
        kill -TERM "$PYTHON_PID" 2>/dev/null
        sleep 2
        kill -KILL "$PYTHON_PID" 2>/dev/null
    fi
    exit 1
}

trap cleanup SIGTERM SIGINT SIGQUIT SIGHUP

# ========== Helper functions ==========

wait_for_server() {
    local port="$1"
    local model_name="$2"
    local max_retries=900
    local retry=0

    echo "Waiting for the $model_name service to become ready (port $port)..."
    while ! curl -s "http://127.0.0.1:${port}/v1/models" > /dev/null 2>&1; do
        retry=$((retry + 1))
        if [[ $retry -ge $max_retries ]]; then
            echo "Error: timed out waiting for the $model_name service to start"
            cleanup
            exit 1
        fi
        sleep 1
    done
    echo "$model_name service is ready"
}

run_model() {
    local model_path="$1"
    local output_path="$2"
    local model_name="$3"
    local port="$4"
    local api_base="http://127.0.0.1:${port}/v1"

    if [[ -f "$output_path" ]]; then
        echo ""
        echo "========================================"
        echo "  Model:  $model_name"
        echo "  Output file already exists, skipping: $output_path"
        echo "========================================"
        return 0
    fi

    echo ""
    echo "========================================"
    echo "  Model:      $model_name"
    echo "  Path:   $model_path"
    echo "  Port:   $port (reusing an already deployed vLLM service)"
    echo "========================================"
    echo ""

    # Reuse the already-deployed vLLM service (no longer auto-start/auto-stop)
    wait_for_server "$port" "$model_name"

    # Run the multithreaded client
    echo "Starting the multi-threaded client ($NUM_WORKERS threads)..."
    python -u "$SCRIPT_DIR/generate_responses.py" \
        --data "$DATA_PATH" \
        --output "$output_path" \
        --api_base "$api_base" \
        --api_key "$API_KEY" \
        --model "$model_path" \
        --num_workers "$NUM_WORKERS" \
        --max_tokens "$MAX_TOKENS" &
    PYTHON_PID=$!
    wait "$PYTHON_PID"
    PYTHON_PID=""

    echo ""
    echo "  Generation finished for $model_name"
}

# ========== Run ==========

# mkdir -p "$OUTPUT_DIR"

# Parse the model list with the | separator and run them one by one
for model_entry in "${MODELS[@]}"; do
    IFS='|' read -r model_path output_path model_name port <<< "$model_entry"
    run_model "$model_path" "$output_path" "$model_name" "$port"
done

echo ""
echo "========================================"
echo "  All done"
echo "  Output files:"
for model_entry in "${MODELS[@]}"; do
    IFS='|' read -r _ output_path model_name <<< "$model_entry"
    echo "    - $output_path ($model_name)"
done
echo "========================================"
