#!/usr/bin/env bash
#
# Convenient script for running analyze_rubric_types.py.
# Modify the parameters in the "configuration area" below, then run: bash run_analyze_rubric_types.sh
#
set -euo pipefail

# ===================== Configuration area (modify as needed) =====================
# Dataset file (merged format, containing subjects such as cs/econ/physics, schema includes solution.<rubric_key>)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARQUET_PATH="$SCRIPT_DIR/data/rubric_rate.parquet"

# Output directory
OUTPUT_DIR="$SCRIPT_DIR/output"

# LLM model parameters
MODEL="DeepSeek-V4-Flash"
API_KEY="sk-xxx"
BASE_URL="http://xxx/"
TEMPERATURE=0

# The rubric set to analyze (a key within the solution dict of the merged format):
#   merged_rubrics / problem_rubrics / solution_rubrics / rubrics
RUBRIC_KEY="rubrics"

# Only analyze the specified subjects (comma-separated; empty means all subjects)
CATEGORIES="cs,econ,physics"

# Maximum number of questions to process (empty means all)
MAX_ITEMS=""

# Number of parallel threads (LLM calls are I/O-bound, recommended 4~16)
NUM_WORKERS=10
# =============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/analyze_rubric_types.py"

ARGS=(--parquet_path "$PARQUET_PATH" --output_dir "$OUTPUT_DIR")
ARGS+=(--model "$MODEL" --api_key "$API_KEY" --base_url "$BASE_URL" --temperature "$TEMPERATURE")
ARGS+=(--rubric_key "$RUBRIC_KEY")
if [ -n "$CATEGORIES" ]; then
    ARGS+=(--categories "$CATEGORIES")
fi
if [ -n "$MAX_ITEMS" ]; then
    ARGS+=(--max_items "$MAX_ITEMS")
fi
ARGS+=(--num_workers "$NUM_WORKERS")

echo ">>> Running: python $PYTHON_SCRIPT ${ARGS[*]}"
python "$PYTHON_SCRIPT" "${ARGS[@]}"
