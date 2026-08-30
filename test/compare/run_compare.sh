#!/usr/bin/env bash
# run_compare.sh — Convenient runner for compare_responses.py
set -euo pipefail

# ===================== Adjustable parameters =====================
API_KEY="${PAPERGYM_API_KEY:-}"
BASE_URL="${PAPERGYM_BASE_URL:-http://your-api/v1}"
MODEL="glm-5.1"
# MODEL="dashscope/qwen3.5-plus"
# MODEL="kimi-k2.6"
# MODEL="DeepSeek-V4-Flash"
WORKERS=10
TEMPERATURE=0
# DATASET: select the dataset to compare (ml / cs_new / arxiv / all).
#   ml     -> answer/ml/     question source data/ml_data.parquet (685 questions)
#   cs_new -> answer/cs_new/ question source data/cs_new_solution.parquet (400 questions)
#   arxiv  -> answer/arxiv/  question source ai-co-scientist/data/arxiv_data.parquet (answers not yet generated)
#   all    -> all datasets
# Can also be overridden by a command-line argument: bash run_compare.sh cs_new
DATASET="${DATASET:-ml}"
# OUTPUT_DIR="results_kimi"
# By default, results are separated per dataset to avoid overwriting across datasets; can be overridden by env var
OUTPUT_DIR="${OUTPUT_DIR:-}"
# FORCE: whether to force re-comparison when the output file already exists (default false = skip if exists)
FORCE="${FORCE:-false}"
# ===================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/compare_responses.py"

# The first command-line argument can override the dataset: bash run_compare.sh arxiv
if [[ $# -ge 1 ]]; then
    DATASET="$1"
fi
# The output directory defaults to including the dataset name (ml / arxiv / all) to avoid overwriting across datasets
if [[ -z "$OUTPUT_DIR" ]]; then
    # OUTPUT_DIR="results_ds_${DATASET}"
    # OUTPUT_DIR="results_kimi_${DATASET}"
    # OUTPUT_DIR="results_qwen_${DATASET}"
    OUTPUT_DIR="results_glm_${DATASET}"
fi

ARGS=(
    --api-key "$API_KEY"
    --base-url "$BASE_URL"
    --model "$MODEL"
    --output-dir "$OUTPUT_DIR"
    --workers "$WORKERS"
    --temperature "$TEMPERATURE"
    --dataset "$DATASET"
)
if [[ "$FORCE" == "true" ]]; then
    ARGS+=(--force)
fi

echo "Running: $PY_SCRIPT"
echo "  model=$MODEL dataset=$DATASET workers=$WORKERS temperature=$TEMPERATURE output-dir=$OUTPUT_DIR force=$FORCE"

python3 "$PY_SCRIPT" "${ARGS[@]}"
