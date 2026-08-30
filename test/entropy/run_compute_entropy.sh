#!/bin/bash
#
# Batch entropy measurement: for each model in turn, start the vLLM service -> compute entropy -> stop vLLM
#
# Usage:
#   1. Add the models to process in the MODELS array
#   2. Adjust GPU, port, and other configurations
#   3. bash run_compute_entropy.sh
#
# Note:
#   - Each model's result is saved to results/entropy_<suffix>.json
#   - To aggregate results, run run_summarize_entropy.sh separately
#

set -e

# Specify GPU to use
export CUDA_VISIBLE_DEVICES=3

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ========== Configuration parameters ==========

# ---- Dataset ----
PARQUET_PATH="${SCRIPT_DIR}/data/cs_new.parquet"
MAX_ITEMS=100          # 0 means process all

# ---- vLLM service parameters ----
GEN_PORT=8022
TENSOR_PARALLEL_SIZE=1
MAX_MODEL_LEN=20480
GPU_MEMORY_UTILIZATION=0.9

# ---- Entropy computation parameters ----
TOP_LOGPROBS=20
TEMPERATURE=1.0
NUM_WORKERS=10

# ---- Output directory ----
OUTPUT_DIR="${SCRIPT_DIR}/results_4b"

# ---- Model roots (override via environment variables) ----
MODEL_ROOT="${MODEL_ROOT:-$HOME/.cache/modelscope/hub/models}"
PAPER_RL_ROOT="${PAPER_RL_ROOT:-$HOME/paper-rl}"
RUBRICSD_ROOT="${RUBRICSD_ROOT:-$HOME/self-distilation/RubricSD}"

# ---- Model list ----
# Auto-generate using a loop: Base + the four model types at step-20 ~ step-200 (step size 20)
MODELS=()

# # 1. Base model
# MODELS+=("${MODEL_ROOT}/Qwen/Qwen3-1.7B|qwen3_1b|Qwen3-1B")

# # 2. Step checkpoints of the four model types
# # for step in 20 40 60 80 100 120 140 160 180 200; do
# for step in 20 40 60 80 100 120 140 160 180 200; do
#     # GRPO offline
#     MODELS+=("${PAPER_RL_ROOT}/model/qwen3-1b-cs-offline/step-${step}|qwen3_1b_grpo_${step}|Qwen3-1B-GRPO-${step}")
#     # OPSD-GRPO
#     MODELS+=("${PAPER_RL_ROOT}/model/qwen3-1b-cs-opsd-offline-37/step-${step}|qwen3_1b_opsd_grpo_${step}|Qwen3-1B-OPSD-GRPO-${step}")
#     # OPSD
#     MODELS+=("${RUBRICSD_ROOT}/models/cs/qwen3_1b_opsd/method-model-${step}|qwen3_1b_opsd_${step}|Qwen3-1B-OPSD-${step}")
#     # GRPO-OPSD
#     MODELS+=("${RUBRICSD_ROOT}/models/cs/qwen3_1b_opsd_grpo/method-model-${step}|qwen3_1b_grpo_opsd_${step}|Qwen3-1B-GRPO-OPSD-${step}")
#     # SFT
#     MODELS+=("${RUBRICSD_ROOT}/models/cs/qwen3_1b_sft/method-model-${step}|qwen3_1b_sft_${step}|Qwen3-1B-SFT-${step}")
# done

# 1. Base model
MODELS+=("${MODEL_ROOT}/Qwen/Qwen3-4B|qwen3_4b|Qwen3-4B")

# 2. Step checkpoints of the four model types
# for step in 20 40 60 80 100 120 140 160 180 200; do
for step in 200; do
    # GRPO offline
    # MODELS+=("${PAPER_RL_ROOT}/model/qwen3-4b-cs-grpo-offline/step-${step}|qwen3_4b_grpo_${step}|Qwen3-4B-GRPO-${step}")
    # OPSD-GRPO
    MODELS+=("${PAPER_RL_ROOT}/model/qwen3-4b-cs-opsd-grpo-offline/step-${step}|qwen3_4b_opsd_grpo_${step}|Qwen3-4B-OPSD-GRPO-${step}")
    # OPSD
    MODELS+=("${RUBRICSD_ROOT}/models/cs/qwen3_4b_opsd/method-model-${step}|qwen3_4b_opsd_${step}|Qwen3-4B-OPSD-${step}")
    # GRPO-OPSD
    # MODELS+=("${RUBRICSD_ROOT}/models/cs/qwen3_4b_opsd_grpo/method-model-${step}|qwen3_4b_grpo_opsd_${step}|Qwen3-4B-GRPO-OPSD-${step}")
    # SFT
    # MODELS+=("${RUBRICSD_ROOT}/models/cs/qwen3_4b_sft/method-model-${step}|qwen3_4b_sft_${step}|Qwen3-4B-SFT-${step}")
done

# ========== Signal handling ==========

VLLM_PID=""
PYTHON_PID=""

cleanup() {
    local exit_code=${1:-1}
    echo ""
    echo "Signal received, cleaning up..."

    # Stop compute_entropy.py (the PID was tracked since it runs in the background)
    if [[ -n $PYTHON_PID ]] && kill -0 "$PYTHON_PID" 2>/dev/null; then
        echo "Stopping the Python process (PID=$PYTHON_PID)..."
        kill -TERM "$PYTHON_PID" 2>/dev/null
        sleep 1
    fi

    # Stop the vLLM service
    if [[ -n $VLLM_PID ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Stopping the vLLM service (PID=$VLLM_PID)..."
        kill -TERM "$VLLM_PID" 2>/dev/null
        sleep 2
        kill -KILL "$VLLM_PID" 2>/dev/null || true
    fi

    # Fallback: kill any leftover process on GEN_PORT
    fuser -k "${GEN_PORT}/tcp" 2>/dev/null || true

    exit "$exit_code"
}

trap 'cleanup 1' SIGTERM SIGINT SIGQUIT SIGHUP

# ========== Helper functions ==========

wait_for_server() {
    local port="$1"
    local name="$2"
    local max_retries=2400
    local retry=0

    echo "Waiting for the ${name} vLLM service to become ready (port $port)..."
    while ! curl -s "http://127.0.0.1:${port}/v1/models" > /dev/null 2>&1; do
        retry=$((retry + 1))
        if [[ $retry -ge $max_retries ]]; then
            echo "Error: timed out waiting for the ${name} vLLM service to start"
            cleanup 1
        fi
        sleep 1
    done
    echo "${name} vLLM service is ready"
}

run_entropy() {
    local model_path="$1"
    local output_suffix="$2"
    local model_name="$3"
    local output_file="${OUTPUT_DIR}/entropy_${output_suffix}.json"
    local api_base="http://127.0.0.1:${GEN_PORT}/v1"
    local api_key="EMPTY"

    # Skip models that have already been computed
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

    # Ensure output directory exists
    mkdir -p "$OUTPUT_DIR"

    # 1. Start the vLLM service (in the background)
    echo ">>> Starting the vLLM service..."
    python -m vllm.entrypoints.openai.api_server \
        --model "$model_path" \
        --port "$GEN_PORT" \
        --host 0.0.0.0 \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" &
    VLLM_PID=$!

    wait_for_server "$GEN_PORT" "$model_name"

    # 2. Compute entropy (run in the background so the PID can be tracked and terminated on interrupt)
    echo ""
    echo ">>> Computing entropy..."
    echo "    Parameters: top_logprobs=${TOP_LOGPROBS}, temperature=${TEMPERATURE}, max_items=${MAX_ITEMS}"
    echo ""

    python -u "$SCRIPT_DIR/compute_entropy.py" \
        --parquet_path "$PARQUET_PATH" \
        --output_path "$output_file" \
        --model "$model_path" \
        --base_url "$api_base" \
        --api_key "$api_key" \
        --top_logprobs "$TOP_LOGPROBS" \
        --temperature "$TEMPERATURE" \
        --max_items "$MAX_ITEMS" \
        --num_workers "$NUM_WORKERS" &
    PYTHON_PID=$!
    wait "$PYTHON_PID" 2>/dev/null || true
    PYTHON_PID=""

    echo ""
    echo "  [Done] Entropy computation finished for ${model_name}"

    # 3. Stop the vLLM service
    echo ""
    echo ">>> Stopping the vLLM service..."
    kill -TERM "$VLLM_PID" 2>/dev/null
    wait "$VLLM_PID" 2>/dev/null || true
    VLLM_PID=""
    # Fallback again
    fuser -k "${GEN_PORT}/tcp" 2>/dev/null || true
    sleep 2
}

# ========== Run model by model ==========

echo ""
echo "==========================================="
echo "  Batch entropy measurement"
echo "  Dataset:      ${PARQUET_PATH}"
echo "  Items:        ${MAX_ITEMS}"
echo "  Models:       ${#MODELS[@]}"
echo "  Output dir:   ${OUTPUT_DIR}"
echo "==========================================="
echo ""

for model_entry in "${MODELS[@]}"; do
    IFS='|' read -r model_path output_suffix model_name <<< "$model_entry"
    run_entropy "$model_path" "$output_suffix" "$model_name"
done

echo ""
echo "==========================================="
echo "  Entropy measurement completed for all models"
echo "  Output dir:   ${OUTPUT_DIR}"
echo "==========================================="
echo ""

# Hint to run the summary
echo "Tip: run bash run_summarize_entropy.sh to view the aggregated results"
echo ""
