#!/bin/bash
#
# ResearchPlanGen answer generation — batch start local vLLM services → generate answers per model → collect results
#
# Used with generate_responses.py: this script only handles "start service + generate answers + collect",
# No scoring is performed. Each local model corresponds to an output file, with the same format as
# compare/answer/qwen3_1b_opsd_grpo_responses.json.
#
# Usage:
#   bash run_generate_responses_local.sh
#
# Configuration:
#   1. The MODELS array defines the local model list, each entry formatted as:
#        "model_path|engine_name|output_filename"
#      The output filename can be an absolute path; if relative, it is saved under OUTPUT_DIR.
#      If the third item is omitted, <engine_name>_responses.json is used by default.
#   2. Requires conda environment with vllm (default: opsd), script will auto-activate.
#   3. Datasets controlled by DATASETS / LIMIT (empty = all).
#
# Prerequisites:
#   - vllm installed and available on this machine (see ENV_NAME / CONDA_SH config).
#   - GPU memory sufficient for the heaviest model in MODELS (see GPU_MEMORY_UTILIZATION).
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ========== Runtime environment (conda with vllm) ==========
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-opsd}"
PYTHON="${PYTHON:-}"   # If empty, use python from activated environment

if [[ -f "$CONDA_SH" ]]; then
    # shellcheck disable=SC1090
    source "$CONDA_SH"
    conda activate "$ENV_NAME"
    echo "[env] Activated conda environment: $ENV_NAME"
else
    echo "[warn] Not found $CONDA_SH, using python from current PATH"
fi
if [[ -z "$PYTHON" ]]; then
    PYTHON="$(command -v python)"
fi
echo "[env] python: $PYTHON ($($PYTHON -c 'import vllm; print("vllm", vllm.__version__)' 2>/dev/null || echo 'no vllm'))"

# ========== GPU settings (services start sequentially, only one model occupies GPU at a time) ==========
export CUDA_VISIBLE_DEVICES=7
# ========== vLLM service parameters ==========
BASE_PORT=8305              # starting port
TENSOR_PARALLEL_SIZE=1      # tensor parallel size
MAX_MODEL_LEN=20480         # maximum input length
GPU_MEMORY_UTILIZATION=0.6  # GPU memory utilization (lower it if OOM, e.g. 0.7)

# ========== Answer generation parameters ==========
DATA_DIR="$SCRIPT_DIR/data"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/answers}"   # Output directory (relative to where output files are saved)
TEMPERATURE=0.0             # Generation temperature
GEN_MAX_TOKENS=10240        # Max tokens per answer
WORKERS=20                  # Number of parallel threads

# Optional: only generate the specified datasets (empty = all), e.g.: DATASETS=(ResearchPlanGen_ml)
DATASETS=(ResearchPlanGen_ml)
# Optional: process the first N items per dataset (empty = all)
LIMIT=""
# Force regeneration (ignore existing output, overwrite)
FORCE=""

# ========== Local model list ==========
# Format: "model_path|engine_name|output_filename"
# If output filename omitted, defaults to <engine_name>_responses.json (saved in $OUTPUT_DIR)
MODELS=(
    # "/path/to/your/model|my-model-name"
)

# ========== Signal handling ==========
VLLM_PID=""
GEN_PID=""

cleanup() {
    echo "Signal received, cleaning up..."
    if [[ -n $VLLM_PID ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Stopping vLLM service (PID=$VLLM_PID)..."
        kill -TERM "$VLLM_PID" 2>/dev/null
        sleep 2
        kill -KILL "$VLLM_PID" 2>/dev/null || true
    fi
    if [[ -n $GEN_PID ]] && kill -0 "$GEN_PID" 2>/dev/null; then
        kill -TERM "$GEN_PID" 2>/dev/null
        sleep 2
        kill -KILL "$GEN_PID" 2>/dev/null || true
    fi
    exit 1
}

trap cleanup SIGTERM SIGINT SIGQUIT SIGHUP

# ========== Helper functions ==========

wait_for_server() {
    local port="$1"
    local model_name="$2"
    local max_retries=3600
    local retry=0

    echo "  Waiting for $model_name service ready (port $port)..."
    while ! curl -s "http://127.0.0.1:${port}/v1/models" > /dev/null 2>&1; do
        retry=$((retry + 1))
        if [[ $retry -ge $max_retries ]]; then
            echo "  Error: $model_name service startup timeout (${max_retries}s)"
            cleanup
            exit 1
        fi
        sleep 1
    done
    echo "  $model_name service is ready"
}

resolve_output_path() {
    local engine_name="$1"
    local out_arg="$2"   # may be empty
    if [[ -z "$out_arg" ]]; then
        out_arg="${engine_name}_responses.json"
    fi
    if [[ "$out_arg" == /* ]]; then
        echo "$out_arg"
    else
        echo "$OUTPUT_DIR/$out_arg"
    fi
}

run_generate() {
    local model_path="$1"
    local engine_name="$2"
    local output_path="$3"
    local port="$4"

    local api_base="http://127.0.0.1:${port}/v1"

    echo ""
    echo "============================================"
    echo "  Local model: $engine_name"
    echo "  Path:       $model_path"
    echo "  Port:       $port"
    echo "  API:      $api_base"
    echo "  Output:     $output_path"
    echo "============================================"
    echo ""

    # ---- 1. Start the vLLM service ----
    echo "[step 1/3] Starting the vLLM service..."
    $PYTHON -m vllm.entrypoints.openai.api_server \
        --model "$model_path" \
        --served-model-name "$engine_name" \
        --port "$port" \
        --host 0.0.0.0 \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" &
    VLLM_PID=$!

    wait_for_server "$port" "$engine_name"

    # ---- 2. Assemble generate_responses.py arguments (generate answers only, no scoring) ----
    echo "[step 2/3] Generating answers..."

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

    local log_file="${output_path%.json}.gen.log"
    echo "  Log: $log_file"
    echo ""

    set +e
    $PYTHON -u "$SCRIPT_DIR/generate_responses.py" \
        --data-dir "$DATA_DIR" \
        --base-url "$api_base" \
        --api-key "EMPTY" \
        --model "$engine_name" \
        --output "$output_path" \
        --temperature "$TEMPERATURE" \
        --max-tokens "$GEN_MAX_TOKENS" \
        --workers "$WORKERS" \
        "${extra_args[@]}" \
        2>&1 | tee "$log_file"
    local gen_status=${PIPESTATUS[0]}
    set -e

    GEN_PID=""

    # ---- 3. Stopping vLLM service ----
    echo "[step 3/3] Stopping vLLM service..."
    kill -TERM "$VLLM_PID" 2>/dev/null
    wait "$VLLM_PID" 2>/dev/null || true
    VLLM_PID=""
    sleep 2

    # Summarize model results
    echo ""
    echo "  [summary] $engine_name"
    if [[ -f "$output_path" ]]; then
        local n_entries n_empty
        n_entries=$($PYTHON -c "import json;print(len(json.load(open('$output_path'))))" 2>/dev/null || echo "?")
        n_empty=$($PYTHON -c "
import json
d=json.load(open('$output_path'))
print(sum(1 for v in d.values() if not (v.get('answer') or '').strip()))" 2>/dev/null || echo "?")
        echo "    Answer entries: $n_entries, empty answers: $n_empty"
    fi
    if [[ $gen_status -ne 0 ]]; then
        echo "  [warn] $engine_name generation returned non-zero status ($gen_status), please check $log_file"
    fi
    echo ""
    return 0
}

# ========== Main workflow ==========

mkdir -p "$OUTPUT_DIR"

current_port=$BASE_PORT
for model_entry in "${MODELS[@]}"; do
    IFS='|' read -r model_path engine_name out_arg <<< "$model_entry"

    if [[ ! -d "$model_path" ]]; then
        echo "Warning: model path does not exist, skipping: $model_path"
        continue
    fi

    output_path="$(resolve_output_path "$engine_name" "${out_arg:-}")"

    # Skip if complete output exists and FORCE is not set (delete output or set FORCE=true to regenerate)
    if [[ "$FORCE" != true && -f "$output_path" ]]; then
        echo ""
        echo "  $engine_name already has an output file, skipping: $output_path"
        current_port=$((current_port))
        continue
    fi

    run_generate "$model_path" "$engine_name" "$output_path" "$current_port"
done

echo ""
echo "============================================"
echo "  All answer generation completed"
echo "  Output directory: $OUTPUT_DIR"
echo "============================================"
