#!/bin/bash
#
# Sequentially start a vLLM server and call generate_responses.py to generate answers for the RubricHub evaluation set.
# Reference: ../ResearchQA/run_generate_responses_mini.sh
#

set -e

# Specify the GPU to use
export CUDA_VISIBLE_DEVICES=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ========== Configuration parameters ==========

DATA_PATH="$SCRIPT_DIR/dataset/eval_set_Science_800.parquet"
OUTPUT_DIR="$SCRIPT_DIR/output/answers"

# vLLM server port
API_PORT=8030
API_BASE="http://127.0.0.1:${API_PORT}/v1"
API_KEY="EMPTY"

# Multithreaded client parameters
NUM_WORKERS=10
MAX_TOKENS=10240

# vLLM server parameters
TENSOR_PARALLEL_SIZE=1
MAX_MODEL_LEN=20480
GPU_MEMORY_UTILIZATION=0.9

# Model list: path|output_file|name
# Model root (override via the MODEL_ROOT env var)
MODEL_ROOT="${MODEL_ROOT:-$HOME/.cache/modelscope/hub/models}"
MODELS=(
    # "${MODEL_ROOT}/Qwen/Qwen3-1.7B|$OUTPUT_DIR/qwen3_1b_responses.json|Qwen3-1B"
)

# ========== Signal handling ==========

VLLM_PID=""
PYTHON_PID=""

cleanup() {
    echo "Signal received, cleaning up..."
    if [[ -n $VLLM_PID ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Stopping vLLM server (PID=$VLLM_PID)..."
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
    local max_retries=2400
    local retry=0

    echo "Waiting for $model_name server to be ready (port $port)..."
    while ! curl -s "http://127.0.0.1:${port}/v1/models" > /dev/null 2>&1; do
        retry=$((retry + 1))
        if [[ $retry -ge $max_retries ]]; then
            echo "ERROR: $model_name server startup timed out"
            cleanup
            exit 1
        fi
        sleep 1
    done
    echo "$model_name server is ready"
}

run_model() {
    local model_path="$1"
    local output_path="$2"
    local model_name="$3"

    if [[ -f "$output_path" ]]; then
        echo ""
        echo "========================================"
        echo "  Model: $model_name"
        echo "  Output file already exists, skipping: $output_path"
        echo "========================================"
        return 0
    fi

    echo ""
    echo "========================================"
    echo "  Model: $model_name"
    echo "  Path:  $model_path"
    echo "========================================"
    echo ""

    # 1. Start the vLLM server
    echo "Starting vLLM server..."
    python -m vllm.entrypoints.openai.api_server \
        --model "$model_path" \
        --port "$API_PORT" \
        --host 0.0.0.0 \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" &
    VLLM_PID=$!

    wait_for_server "$API_PORT" "$model_name"

    # 2. Run the multithreaded client (calls RubricHub's generate_responses.py)
    echo "Starting multithreaded client ($NUM_WORKERS threads)..."
    python -u "$SCRIPT_DIR/generate_responses.py" \
        --data "$DATA_PATH" \
        --output "$output_path" \
        --api_base "$API_BASE" \
        --api_key "$API_KEY" \
        --model "$model_path" \
        --num_workers "$NUM_WORKERS" \
        --max_tokens "$MAX_TOKENS" &
    PYTHON_PID=$!
    wait "$PYTHON_PID"
    PYTHON_PID=""

    echo ""
    echo "  $model_name generation complete"

    # 3. Stop the vLLM server
    echo "Stopping vLLM server..."
    kill -TERM "$VLLM_PID" 2>/dev/null
    wait "$VLLM_PID" 2>/dev/null || true
    VLLM_PID=""
    sleep 2
}

# ========== Run ==========

mkdir -p "$OUTPUT_DIR"

# Parse the model list using the | delimiter and run each one
for model_entry in "${MODELS[@]}"; do
    IFS='|' read -r model_path output_path model_name <<< "$model_entry"
    run_model "$model_path" "$output_path" "$model_name"
done

echo ""
echo "========================================"
echo "  All generation complete"
echo "  Output files:"
for model_entry in "${MODELS[@]}"; do
    IFS='|' read -r _ output_path model_name <<< "$model_entry"
    echo "    - $output_path ($model_name)"
done
echo "========================================"
