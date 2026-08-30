#!/bin/bash
# run_compare_three.sh — Run the three-model comparison (ml_1b vs ml_1b_rubrichub vs ml_1b_cs)

set -euo pipefail

cd "$(dirname "$0")"

# Configuration (kept consistent with run_compare.sh)
API_KEY="${OPENSCIENCE_API_KEY:-}"
BASE_URL="${OPENSCIENCE_BASE_URL:-http://your-api/v1}"
# API_KEY="sk-xxx"
# BASE_URL="http://xxx/"
MODEL="DeepSeek-V4-Flash"
# MODEL="glm-5.1"
# MODEL="kimi-k2.6"
# MODEL="gemini-3.7-flash"
WORKERS=10
TEMPERATURE=0

# Output directory
# OUTPUT_DIR="results_three/ds"
# OUTPUT_DIR="results_three/glm"
# OUTPUT_DIR="results_three/kimi"
# OUTPUT_DIR="results_three/gemini"
OUTPUT_DIR="results_three/test"

# Build arguments
ARGS=(
    --output-dir "$OUTPUT_DIR"
    --temperature "$TEMPERATURE"
    --workers "$WORKERS"
    --api-key "$API_KEY"
    --base-url "$BASE_URL"
    --model "$MODEL"
)

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)
            ARGS+=(--force)
            shift
            ;;
        --max-entries)
            if [[ -n "${2:-}" ]]; then
                ARGS+=(--max-entries "$2")
                shift 2
            else
                echo "Error: --max-entries requires a numeric value" >&2
                exit 1
            fi
            ;;
        *)
            echo "Warning: unknown argument $1" >&2
            shift
            ;;
    esac
done

echo "========================================"
echo "Three-model comparison"
echo "  model=$MODEL"
echo "  workers=$WORKERS"
echo "  temperature=$TEMPERATURE"
echo "  output-dir=$OUTPUT_DIR"
echo "========================================"

python3 compare_three.py "${ARGS[@]}"
