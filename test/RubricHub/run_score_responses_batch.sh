#!/bin/bash
#
# Run score_responses.py to automatically score the RubricHub evaluation set.
# Reference: ../ResearchQA/run_compute_coverage_mini.sh
#

# ========== Configuration parameters (modify according to actual paths) ==========

# The script directory (uses relative paths to ensure portability)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Evaluation dataset parquet file
DATA_PATH="$SCRIPT_DIR/dataset/eval_set_Science_800.parquet"

# LLM judge configuration (uses an OpenAI-compatible API such as DeepSeek)
# API_BASE="http://xxx/v1"
# API_KEY="sk-xxx"
# MODEL="deepseek-v4-flash"
API_BASE="http://xxx/"
API_KEY="sk-xxx"
MODEL="DeepSeek-V4-Flash"

# Number of concurrent threads (scoring stage)
NUM_WORKERS=10

# Output directory
OUTPUT_DIR="$SCRIPT_DIR/output/scores_ds"

# Answer file list: answer-response JSON|output scoring JSON|name
# Uses SCRIPT_DIR as the base path to ensure portability
ANSWER_FILES=(
    # "${SCRIPT_DIR}/output/answers/qwen3_1b_responses.json|$OUTPUT_DIR/qwen3_1b_scores.json|Qwen3-1B"
)

# ========== No need to modify below ==========

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Signal handling
PYTHON_PID=""

cleanup() {
    echo "Signal received, terminating Python process..."
    if [[ -n $PYTHON_PID ]]; then
        kill -TERM "$PYTHON_PID" 2>/dev/null
        sleep 2
        if kill -0 "$PYTHON_PID" 2>/dev/null; then
            echo "Force-killing Python process..."
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
        echo "  Output: $output_path"
        echo "=================================="
        continue
    fi

    echo ""
    echo "=================================="
    echo "  Scoring: $name"
    echo "  Response file: $responses_path"
    echo "  Output: $output_path"
    echo "=================================="

    python "$SCRIPT_DIR/score_responses_batch.py" \
        --data "$DATA_PATH" \
        --response "$responses_path" \
        --output "$output_path" \
        --num_workers "$NUM_WORKERS" \
        --model "$MODEL" \
        --api_base "$API_BASE" \
        --api_key "$API_KEY" &

    PYTHON_PID=$!
    wait "$PYTHON_PID"

    echo "  $name scoring complete"
done

echo ""
echo "=================================="
echo "  All scoring complete"
echo "  Output files:"
for entry in "${ANSWER_FILES[@]}"; do
    IFS='|' read -r _ output_path name <<< "$entry"
    echo "    - $output_path ($name)"
done
echo "=================================="
