#!/bin/bash
#
# Generate answers for the stability evaluation dataset using a local vLLM (Qwen3-4B)
# based on stability.parquet.
#
# Configuration:
#   - GPU:           3 (currently idle)
#   - Model:         $HOME/.cache/modelscope/hub/models/Qwen/Qwen3-4B
#   - Dataset:       data/stability.parquet (600 items, 200 each for cs/econ/physics)
#   - Output:        answer/stability/qwen3_4b_responses.json
#   - Temperature:   1.0 (hardcoded in generate_responses.py)
#   - MaxTokens:     no limit set (generate_responses.py does not pass max_tokens to the API)
#
# Usage:
#   bash run_generate_stability.sh
#

set -e

export CUDA_VISIBLE_DEVICES=3

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ========== Configuration parameters ==========

# ---- Python environment (vLLM installed in the opsd conda env) ----
# Override with: PYTHON=/path/to/conda/envs/opsd/bin/python bash run_generate_stability.sh
PYTHON="${PYTHON:-python}"

# ---- Dataset ----
PARQUET_PATH="${SCRIPT_DIR}/data/stability.parquet"
# Note: this script uses relative paths, assuming it runs inside the stability directory
MAX_ITEMS=600

# ---- vLLM service parameters ----
# Note: the model source path is on NFS, and vLLM loads it very slowly from NFS mmap (~850KB/s).
# Therefore, first copy it to local /tmp then load (NFS sequential read reaches ~39MB/s, local load takes ~1 min).
# Original path: $HOME/.cache/modelscope/hub/models/Qwen/Qwen3-4B
# After generation, please delete /tmp/qwen3-4b to free up space.
MODEL_PATH="/tmp/qwen3-4b"
GEN_PORT=8031
MAX_MODEL_LEN=20480
GPU_MEMORY_UTILIZATION=0.6

# ---- Concurrency ----
NUM_WORKERS=12

# ---- Output ----
OUTPUT_FILE="${SCRIPT_DIR}/answer/stability/qwen3_4b_responses.json"

# ========== Signal handling ==========

VLLM_PID=""

cleanup() {
    echo ""
    echo "Signal received, cleaning up..."
    if [[ -n $VLLM_PID ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Stopping vLLM server (PID=$VLLM_PID)..."
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
    local max_retries=3600
    local retry=0
    echo "Waiting for vLLM server to be ready (port $port)..."
    while ! curl -s "http://127.0.0.1:${port}/v1/models" > /dev/null 2>&1; do
        retry=$((retry + 1))
        if [[ $retry -ge $max_retries ]]; then
            echo "ERROR: vLLM server startup timed out"
            cleanup
        fi
        sleep 1
    done
    echo "vLLM server is ready"
}

# ========== Main ==========

mkdir -p "$(dirname "$OUTPUT_FILE")"

echo ""
echo "==========================================="
echo "  Generating stability dataset answers"
echo "  Model:       ${MODEL_PATH}"
echo "  Dataset:     ${PARQUET_PATH}"
echo "  Output:      ${OUTPUT_FILE}"
echo "  GPU:         ${CUDA_VISIBLE_DEVICES}"
echo "  Port:        ${GEN_PORT}"
echo "  MaxItems:    ${MAX_ITEMS}"
echo "  Temperature: 1.0 (hardcoded in generate_responses.py)"
echo "  MaxTokens:   unlimited"
echo "==========================================="

# Skip if output already exists
if [[ -f "$OUTPUT_FILE" ]]; then
    echo ""
    echo "Output file already exists, skipping generation: ${OUTPUT_FILE}"
    echo "To regenerate, please delete this file first."
    exit 0
fi

# 1. Start vLLM
echo ""
echo ">>> Starting vLLM server..."
"$PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --trust-remote-code \
    --port "$GEN_PORT" \
    --host 0.0.0.0 \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" &
VLLM_PID=$!

wait_for_server "$GEN_PORT"

# 2. Generate answers
echo ""
echo ">>> Generating answers (${NUM_WORKERS} threads, temperature=1.0, no max_tokens limit)..."
"$PYTHON" -u "$SCRIPT_DIR/generate_responses.py" \
    --parquet_path "$PARQUET_PATH" \
    --output_path "$OUTPUT_FILE" \
    --num_workers "$NUM_WORKERS" \
    --max_items "$MAX_ITEMS" \
    --api_key "EMPTY" \
    --base_url "http://127.0.0.1:${GEN_PORT}/v1" \
    --model "$MODEL_PATH"

echo ""
echo "  [Done] answer generation finished: ${OUTPUT_FILE}"

# 3. Stop vLLM
echo ""
echo ">>> Stopping vLLM server..."
kill -TERM "$VLLM_PID" 2>/dev/null
wait "$VLLM_PID" 2>/dev/null || true
VLLM_PID=""
sleep 2

echo ""
echo "==========================================="
echo "  Generation complete"
echo "  Output file: ${OUTPUT_FILE}"
echo "==========================================="
