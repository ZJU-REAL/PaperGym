#!/bin/bash
#
# Script for running compute_coverage_ds.py
# Uses an OpenAI-compatible API for scoring

# ========== Configuration parameters (modify according to actual paths) ==========

# Dataset JSON file (local ResearchQA format)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_PATH="$SCRIPT_DIR/data/research_qa.json"

# LLM configuration
# API_BASE="http://xxx"
# API_KEY="sk-xxx"
# MODEL="deepseek-v4-flash"
API_BASE="${PAPERGYM_BASE_URL:-http://your-api/v1}"
API_KEY="${PAPERGYM_API_KEY:-}"
MODEL="DeepSeek-V4-Flash"

# Rubric batch size
BATCH_SIZE=8

# Number of concurrent threads (scoring stage)
NUM_WORKERS=10

# Output directory
INPUT_DIR="$SCRIPT_DIR/answer_ablation"
OUTPUT_DIR="$SCRIPT_DIR/output_ablation"

# Answer file list: answer-map JSON|output scoring JSON|name
ANSWER_FILES=(
  "$INPUT_DIR/qwen3_1b_responses.json|$OUTPUT_DIR/qwen3_1b_coverage_results.json|Qwen3-1B"
)

# ========== No need to modify below ==========

# The script directory (already defined)

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Function: signal handling
cleanup() {
    echo "Signal received, terminating the Python process..."
    if [[ -n $PYTHON_PID ]]; then
        kill -TERM "$PYTHON_PID" 2>/dev/null
        sleep 2
        if kill -0 "$PYTHON_PID" 2>/dev/null; then
            echo "Force-terminating the Python process..."
            kill -KILL "$PYTHON_PID" 2>/dev/null
        fi
    fi
    exit 0
}

trap cleanup SIGTERM SIGINT SIGQUIT SIGHUP

# Score one by one
for entry in "${ANSWER_FILES[@]}"; do
    IFS='|' read -r responses_path output_path name <<< "$entry"

    # Skip scoring if the output file already exists
    if [ -f "$output_path" ]; then
        echo ""
        echo "=================================="
        echo "  Skipping: $name (output file already exists)"
        echo "  Output:      $output_path"
        echo "=================================="
        continue
    fi

    echo ""
    echo "=================================="
    echo "  Scoring:     $name"
    echo "  Answer map:  $responses_path"
    echo "  Output:     $output_path"
    echo "=================================="

    python "$SCRIPT_DIR/compute_coverage_ds.py" \
        --data "$DATA_PATH" \
        --responses "$responses_path" \
        --output "$output_path" \
        --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" \
        --model "$MODEL" \
        --api_base "$API_BASE" \
        --api_key "$API_KEY" &

    PYTHON_PID=$!
    wait "$PYTHON_PID"

    echo "  Scoring finished for $name"
done

echo ""
echo "=================================="
echo "  All scoring completed"
echo "  Output files:"
for entry in "${ANSWER_FILES[@]}"; do
    IFS='|' read -r _ output_path name <<< "$entry"
    echo "    - $output_path ($name)"
done
echo "=================================="
