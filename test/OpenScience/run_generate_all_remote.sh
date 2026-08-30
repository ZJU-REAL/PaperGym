#!/bin/bash
#
# Unified remote batch generation script
#
# Generates answers for the solution / experiment datasets at once
# through the remote LLM API, requiring no local vLLM service and no GPU resources.
#
# Usage:
#   1. Add the remote models to call in the MODELS array
#   2. Optionally modify the per-dataset configuration in DATASETS
#   3. bash run_generate_all_remote.sh
#

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ========== Remote API model list ==========
# Each line represents a remote model, with 5 fields separated by |, in order:
#
#   1. API base URL    — the full API base address of the LLM service, must end with /
#                        e.g. "http://<your-host>:<port>/"
#                             "https://api.siliconflow.cn/v1/"
#
#   2. API key         — the API key required to call the LLM service
#                        If the service does not verify keys, fill in "EMPTY"
#
#   3. Model name      — the model name registered in the LLM service
#                        e.g. "qwen3_32b"  "Qwen/Qwen3-32B"
#
#   4. Output file suffix — the name of the output JSON file (without the .json suffix)
#                        It is saved as: answer/<dataset>/<suffix>.json
#                        e.g. "qwen3_32b_responses"
#
#   5. Model display name — a friendly name used only for logging
#                        e.g. "Qwen3-32B"  "Qwen3-1B-GRPO-Offline-200"
#
# Example:
#   "https://api.siliconflow.cn/v1/|sk-xxx|Qwen/Qwen3-32B|qwen3_silicon_responses|Qwen3-32B-Silicon"
#
# This list is applied to all four datasets
# ========================================

MODELS=(
    # "https://api.siliconflow.cn/v1/|sk-xxx|Qwen/Qwen3-32B|qwen3_silicon_responses|Qwen3-32B-Silicon"
)

# ========== Dataset configuration list ==========
# Each line represents a dataset, with 4 fields separated by |, in order:
#
#   1. Dataset name      — the output subdirectory name, saved under answer/<name>/
#   2. parquet file path — the path to the evaluation dataset file
#   3. Number of threads — the number of threads for parallel requests
#   4. Max items         — the maximum number of items to process per dataset
# =====================================

DATASETS=(
  "innov|$SCRIPT_DIR/data/innov.parquet|8|1200"
  "design|$SCRIPT_DIR/data/design.parquet|8|1200"
)

# ========== Generate a single (dataset, model) combination ==========

run_model_for_dataset() {
    local api_base="$1"
    local api_key="$2"
    local model="$3"
    local output_suffix="$4"
    local model_name="$5"
    local dataset_name="$6"
    local parquet_path="$7"
    local num_workers="$8"
    local max_items="$9"

    local cache_base_dir="${SCRIPT_DIR}/answer/${dataset_name}"
    local output_file="${cache_base_dir}/${output_suffix}.json"

    # Skip models that have already been generated
    if [[ -f "$output_file" ]]; then
        echo "      [Skip] ${model_name} -> output file already exists"
        return 0
    fi

    mkdir -p "$cache_base_dir"

    echo "    >>> [${model_name}] Remote API: ${api_base}, model: ${model}"

    python -u "$SCRIPT_DIR/generate_responses.py" \
        --parquet_path "$parquet_path" \
        --output_path "$output_file" \
        --num_workers "$num_workers" \
        --max_items "$max_items" \
        --api_key "$api_key" \
        --base_url "${api_base}v1" \
        --model "$model"

    echo "    [Done] ${model_name} -> ${output_file}"
}

# ========== Main flow ==========

NUM_DATASETS=${#DATASETS[@]}
NUM_MODELS=${#MODELS[@]}

START_TIME=$(date +%s)

echo ""
echo "=============================================="
echo "  Remote API batch answer generation"
echo "  Datasets:            ${NUM_DATASETS}"
echo "  Models per dataset:  ${NUM_MODELS}"
echo "  Total tasks:         $((NUM_DATASETS * NUM_MODELS))"
echo "  Start time:          $(date)"
echo "=============================================="

DATASET_IDX=0
for dataset_entry in "${DATASETS[@]}"; do
    IFS='|' read -r dataset_name parquet_path num_workers max_items <<< "$dataset_entry"

    echo ""
    echo "=========================================="
    echo "  Dataset ($((++DATASET_IDX))/${NUM_DATASETS}): ${dataset_name}"
    echo "  Parquet: ${parquet_path}"
    echo "  Threads: ${num_workers}, items: ${max_items}"
    echo "=========================================="

    MODEL_IDX=0
    for model_entry in "${MODELS[@]}"; do
        IFS='|' read -r api_base api_key model output_suffix model_name <<< "$model_entry"

        echo ""
        echo "  --- [${dataset_name}] Model [$((++MODEL_IDX))/${NUM_MODELS}]: ${model_name} ---"

        run_model_for_dataset \
            "$api_base" "$api_key" "$model" "$output_suffix" "$model_name" \
            "$dataset_name" "$parquet_path" "$num_workers" "$max_items"
    done

    echo ""
    echo "  [Done] All models processed for dataset ${dataset_name}"
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

echo ""
echo "=============================================="
echo "  All done"
echo "  End time:            $(date)"
echo "  Elapsed:             ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo "=============================================="
