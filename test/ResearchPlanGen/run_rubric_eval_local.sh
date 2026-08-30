#!/bin/bash
#
# AI Co-Scientist Rubric batch evaluation script - start a vLLM server -> generate plans -> score them
#
# Usage:
#   bash run_rubric_eval_local.sh
#
# Configuration:
#   1. Define the local model list in the MODELS array; each entry has the format:
#      "model_path|engine_name(used for naming output files)"
#   2. The remote judge model defaults to DeepSeek-V4-Flash (see the JUDGE_* config below),
#      and can be overridden via environment variables: JUDGE_BASE_URL / JUDGE_API_KEY / JUDGE_MODEL
#   3. To let a local model handle both generation and scoring, set USE_REMOTE_JUDGE to false
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ========== GPU settings ==========
export CUDA_VISIBLE_DEVICES=0

# ========== vLLM server params (local model = generating proposed plans) ==========
BASE_PORT=8019              # starting port
TENSOR_PARALLEL_SIZE=1      # tensor parallel size
MAX_MODEL_LEN=20480         # max input length (local model only generates; 20480 is enough)
GPU_MEMORY_UTILIZATION=0.9  # GPU memory utilization

# ========== Evaluation params ==========
DATA_DIR="$SCRIPT_DIR/data"
OUTPUT_DIR="$SCRIPT_DIR/eval_results"
TEMPERATURE=0.0              # generation temperature
GEN_MAX_TOKENS=10240         # max tokens for generating plans
JUDGE_MAX_TOKENS=20480       # max tokens for the judge (scoring)
WORKERS=20                   # number of parallel workers

# Optional: evaluate only the specified datasets (leave empty = all arxiv ml), e.g. DATASETS=(arxiv)
DATASETS=(ml)
# Optional: process at most the first N records per dataset (leave empty = all)
LIMIT="800"
# Force regeneration (ignore existing intermediate results)
FORCE=""

# ========== Remote judge model config ==========
# Uses DeepSeek-V4-Flash for scoring by default; can be overridden via environment variables.
# NOTE: JUDGE_BASE_URL must point to your own DeepSeek service endpoint (set it before running).
# For security, never hardcode credentials here - always provide them via environment variables.
USE_REMOTE_JUDGE=true
JUDGE_BASE_URL="${JUDGE_BASE_URL:-https://api.example.com/v1}"
JUDGE_API_KEY="${JUDGE_API_KEY:-${DEEPSEEK_API_KEY:-}}"
JUDGE_MODEL="${JUDGE_MODEL:-DeepSeek-V4-Flash}"

# ========== Local model list ==========
# Format: "model_path|engine_name"
# engine_name is used for the vLLM served-model-name and the output directory name
MODELS=(
    # "/path/to/your/model|my-model-name"
)

# ========== Signal handling ==========
VLLM_PID=""
EVAL_PID=""

cleanup() {
    echo "Signal received, cleaning up..."
    if [[ -n $VLLM_PID ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Stopping vLLM server (PID=$VLLM_PID)..."
        kill -TERM "$VLLM_PID" 2>/dev/null
        sleep 2
        kill -KILL "$VLLM_PID" 2>/dev/null || true
    fi
    if [[ -n $EVAL_PID ]] && kill -0 "$EVAL_PID" 2>/dev/null; then
        kill -TERM "$EVAL_PID" 2>/dev/null
        sleep 2
        kill -KILL "$EVAL_PID" 2>/dev/null || true
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

    echo "  Waiting for $model_name server to be ready (port $port)..."
    while ! curl -s "http://127.0.0.1:${port}/v1/models" > /dev/null 2>&1; do
        retry=$((retry + 1))
        if [[ $retry -ge $max_retries ]]; then
            echo "  ERROR: $model_name server startup timed out (${max_retries}s)"
            cleanup
            exit 1
        fi
        sleep 1
    done
    echo "  $model_name server is ready"
}

run_eval() {
    local model_path="$1"
    local engine_name="$2"
    local port="$3"

    local api_base="http://127.0.0.1:${port}/v1"
    local result_dir="$OUTPUT_DIR/$engine_name"
    mkdir -p "$result_dir"

    echo ""
    echo "============================================"
    echo "  Local model (generation): $engine_name"
    echo "  Path:                     $model_path"
    echo "  Port:                     $port"
    echo "  API:                      $api_base"
    echo "  Judge:                    $JUDGE_MODEL"
    echo "  Output:                   $result_dir"
    echo "============================================"
    echo ""

    # ---- 1. Start the vLLM server (local model generates proposed plans) ----
    echo "[step 1/4] Starting vLLM server..."
    python -m vllm.entrypoints.openai.api_server \
        --model "$model_path" \
        --served-model-name "$engine_name" \
        --port "$port" \
        --host 0.0.0.0 \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" &
    VLLM_PID=$!

    wait_for_server "$port" "$engine_name"

    # ---- 2. Assemble rubric_eval.py arguments ----
    echo "[step 2/4] Running Rubric evaluation (generation + scoring)..."

    local gen_args=(
        --gen-base-url "$api_base"
        --gen-api-key "EMPTY"
        --gen-model "$engine_name"
        --gen-max-tokens "$GEN_MAX_TOKENS"
    )

    local judge_args=()
    if [[ "$USE_REMOTE_JUDGE" == true ]]; then
        judge_args=(
            --base-url "$JUDGE_BASE_URL"
            --api-key "$JUDGE_API_KEY"
            --model "$JUDGE_MODEL"
        )
        echo "  Judge uses remote model: $JUDGE_MODEL @ $JUDGE_BASE_URL"
    else
        # Use the local model for both generation and scoring
        judge_args=(
            --base-url "$api_base"
            --api-key "EMPTY"
            --model "$engine_name"
        )
        echo "  Judge uses local model: $engine_name"
    fi

    local extra_args=()
    if [[ ${#DATASETS[@]} -gt 0 ]]; then
        extra_args+=(--datasets "${DATASETS[@]}")
    fi
    if [[ -n "$LIMIT" ]]; then
        extra_args+=(--limit "$LIMIT")
    fi
    if [[ "$FORCE" == true ]]; then
        extra_args+=(--force)
    fi

    local log_file="$result_dir/eval.log"
    echo "  Log: $log_file"
    echo ""

    python -u "$SCRIPT_DIR/rubric_eval.py" \
        --data-dir "$DATA_DIR" \
        --output-dir "$result_dir" \
        --temperature "$TEMPERATURE" \
        --judge-max-tokens "$JUDGE_MAX_TOKENS" \
        --workers "$WORKERS" \
        "${gen_args[@]}" \
        "${judge_args[@]}" \
        "${extra_args[@]}" \
        2>&1 | tee "$log_file"

    EVAL_PID=""

    # ---- 3. Summarize the results for this model ----
    echo ""
    echo "[step 3/4] Result summary ($engine_name):"
    if [[ -f "$result_dir/summary.json" ]]; then
        cat "$result_dir/summary.json"
    else
        echo "  (summary.json not found)"
    fi
    echo ""

    # ---- 4. Stop the vLLM server ----
    echo "[step 4/4] Stopping vLLM server..."
    kill -TERM "$VLLM_PID" 2>/dev/null
    wait "$VLLM_PID" 2>/dev/null || true
    VLLM_PID=""
    sleep 2
}

# ========== Summary output file ==========
SUMMARY_FILE="$OUTPUT_DIR/summary.txt"

write_summary_header() {
    mkdir -p "$OUTPUT_DIR"
    cat > "$SUMMARY_FILE" <<'EOF'
============================================
  AI Co-Scientist Rubric batch evaluation summary
============================================

EOF
}

append_summary() {
    local engine_name="$1"
    local sum_file="$OUTPUT_DIR/$engine_name/summary.json"

    {
        echo "--------------------------------------------"
        echo "  Local model (generation): $engine_name"
        echo "  Judge:                    $JUDGE_MODEL"
        echo "--------------------------------------------"
        if [[ -f "$sum_file" ]]; then
            cat "$sum_file"
        else
            echo "  (no evaluation results)"
        fi
        echo ""
    } >> "$SUMMARY_FILE"
}

write_summary_tail() {
    {
        echo "============================================"
        echo "  Evaluation results saved to: $OUTPUT_DIR"
        echo "============================================"
    } >> "$SUMMARY_FILE"
}

# ========== Main flow ==========

write_summary_header

current_port=$BASE_PORT
for model_entry in "${MODELS[@]}"; do
    IFS='|' read -r model_path engine_name <<< "$model_entry"

    # Check whether the model path exists
    if [[ ! -d "$model_path" ]]; then
        echo "WARNING: model path does not exist, skipping: $model_path"
        continue
    fi

    run_eval "$model_path" "$engine_name" "$current_port"
    append_summary "$engine_name"
done

write_summary_tail

echo ""
echo "============================================"
echo "  All evaluations complete"
echo "  Detailed results: $OUTPUT_DIR/<engine_name>/"
echo "  Summary:          $SUMMARY_FILE"
echo "============================================"
