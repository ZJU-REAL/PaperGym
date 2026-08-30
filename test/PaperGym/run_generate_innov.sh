#!/bin/bash
#
# Batch answer generation: for each model in turn, start the vLLM service -> generate answers -> stop vLLM
#
# Usage:
#   1. Add the models to process in the MODELS array
#   2. bash run_generate.sh
#

set -e

# Specify GPU to use
export CUDA_VISIBLE_DEVICES=0

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ========== Configuration parameters ==========

# ---- Dataset ----
PARQUET_PATH="$SCRIPT_DIR/data/innov.parquet"
MAX_ITEMS=1200

# ---- vLLM service parameters ----
GEN_PORT=8020
TENSOR_PARALLEL_SIZE=1
MAX_MODEL_LEN=20480
GPU_MEMORY_UTILIZATION=0.9

# ---- Concurrency (number of threads calling generate_responses.py) ----
NUM_WORKERS=15

# ---- Model list: model path|output file suffix|model display name ----
CACHE_BASE_DIR="${SCRIPT_DIR}/answer/innov"
# Model root (override via the MODEL_ROOT env var)
MODEL_ROOT="${MODEL_ROOT:-$HOME/.cache/modelscope/hub/models}"
MODELS=(
  "${MODEL_ROOT}/Qwen/Qwen3-1.7B|qwen3_1b_responses|Qwen3-1B"
)

# ========== Signal handling ==========

VLLM_PID=""
PYTHON_PID=""

cleanup() {
    echo ""
    echo "Signal received, cleaning up..."

    if [[ -n $PYTHON_PID ]] && kill -0 "$PYTHON_PID" 2>/dev/null; then
        echo "Stopping the Python process (PID=$PYTHON_PID)..."
        kill -TERM "$PYTHON_PID" 2>/dev/null
        sleep 1
    fi

    if [[ -n $VLLM_PID ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Stopping the vLLM service (PID=$VLLM_PID)..."
        kill -TERM "$VLLM_PID" 2>/dev/null
        sleep 2
        kill -KILL "$VLLM_PID" 2>/dev/null
    fi

    exit 1
}

trap cleanup SIGTERM SIGINT SIGQUIT SIGHUP

# ========== Helper functions ==========

wait_for_server() {
    local port="$1"
    local name="$2"
    local max_retries=3600
    local retry=0

    echo "Waiting for the ${name} vLLM service to become ready (port $port)..."
    while ! curl -s "http://127.0.0.1:${port}/v1/models" > /dev/null 2>&1; do
        retry=$((retry + 1))
        if [[ $retry -ge $max_retries ]]; then
            echo "Error: timed out waiting for the ${name} vLLM service to start"
            cleanup
        fi
        sleep 1
    done
    echo "${name} vLLM service is ready"
}

run_model() {
    local model_path="$1"
    local output_suffix="$2"
    local model_name="$3"
    local output_file="${CACHE_BASE_DIR}/${output_suffix}.json"
    local api_base="http://127.0.0.1:${GEN_PORT}/v1"
    local api_key="EMPTY"

    # Skip models that have already been generated
    if [[ -f "$output_file" ]]; then
        echo ""
        echo "==========================================="
        echo "  Skipping: ${model_name}"
        echo "  Output file already exists: ${output_file}"
        echo "==========================================="
        return 0
    fi

    echo ""
    echo "==========================================="
    echo "  Model:      ${model_name}"
    echo "  Path:       ${model_path}"
    echo "  Output:     ${output_file}"
    echo "==========================================="
    echo ""

    # Ensure the cache directory exists
    mkdir -p "$CACHE_BASE_DIR"

    # 1. Start the vLLM service
    echo ">>> Starting the vLLM service..."
    python -m vllm.entrypoints.openai.api_server \
        --model "$model_path" \
        --trust-remote-code \
        --port "$GEN_PORT" \
        --host 0.0.0.0 \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" &
    VLLM_PID=$!

    wait_for_server "$GEN_PORT" "$model_name"

    # 2. Generate answers
    echo ""
    echo ">>> Generating answers (${NUM_WORKERS} threads)..."
    echo ""

    python -u "$SCRIPT_DIR/generate_responses.py" \
        --parquet_path "$PARQUET_PATH" \
        --output_path "$output_file" \
        --num_workers "$NUM_WORKERS" \
        --max_items "$MAX_ITEMS" \
        --api_key "$api_key" \
        --base_url "$api_base" \
        --model "$model_path"

    echo ""
    echo "  [Done] Answer generation finished for ${model_name}"

    # 3. Stop the vLLM service
    echo ""
    echo ">>> Stopping the vLLM service..."
    kill -TERM "$VLLM_PID" 2>/dev/null
    wait "$VLLM_PID" 2>/dev/null || true
    VLLM_PID=""
    sleep 2
}

# ========== Run model by model ==========

echo ""
echo "==========================================="
echo "  Batch answer generation"
echo "  Dataset:      ${PARQUET_PATH}"
echo "  Items:        ${MAX_ITEMS}"
echo "  Models:       ${#MODELS[@]}"
echo "==========================================="

for model_entry in "${MODELS[@]}"; do
    IFS='|' read -r model_path output_suffix model_name <<< "$model_entry"
    run_model "$model_path" "$output_suffix" "$model_name"
done

echo ""
echo "==========================================="
echo "  Generation completed for all models"
echo "  Output dir:   ${CACHE_BASE_DIR}"
echo "==========================================="
echo ""
echo "Generated files:"
for model_entry in "${MODELS[@]}"; do
    IFS='|' read -r _ output_suffix model_name <<< "$model_entry"
    echo "  - ${CACHE_BASE_DIR}/${output_suffix}.json  (${model_name})"
done
echo ""
