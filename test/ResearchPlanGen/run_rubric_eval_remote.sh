#!/bin/bash
#
# ResearchPlanGen Rubric batch evaluation script — remote API version
# Does not start a local vLLM service; directly calls a remote OpenAI-compatible API
# to perform both "answer generation + scoring".
#
# Usage:
#   bash run_rubric_eval_remote.sh
#
# Configuration:
#   1. Define the remote models to evaluate in the MODELS array, each entry formatted as:
#      "remote_model_name|output_file|display_name"
#      Removing the .json suffix from the output file path gives that model's result directory,
#      which contains plans/, grades/, summary.json, and all_results.json.
#   2. The judge defaults to the same model being evaluated (the remote service handles generation + scoring);
#      it can be overridden to a fixed scoring model via the JUDGE_MODEL environment variable, e.g.:
#        JUDGE_MODEL=DeepSeek-V4-Flash bash run_rubric_eval_remote.sh
#   3. The remote API base / key can be overridden via the API_BASE / API_KEY environment variables.
#   4. Generation and scoring can specify API base and key separately:
#        GEN_API_BASE=<url> GEN_API_KEY=<key> JUDGE_API_BASE=<url> JUDGE_API_KEY=<key> bash run_rubric_eval_remote.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ========== Remote API configuration ==========
# Generation and scoring can be specified separately; if not specified, they fall back to the general config
API_BASE="${API_BASE:-http://xxx/}"
API_KEY="${API_KEY:-sk-xxx}"

# Generation model config (if empty, falls back to API_BASE/API_KEY/model_name)
GEN_API_BASE="${GEN_API_BASE:-}"
GEN_API_KEY="${GEN_API_KEY:-}"
GEN_MODEL="${GEN_MODEL:-}"          # empty = same as the model being evaluated

# Scoring model config (if empty, falls back to API_BASE/API_KEY/model_name)
JUDGE_API_BASE="${JUDGE_API_BASE:-http://xxx/}"
JUDGE_API_KEY="${JUDGE_API_KEY:-sk-xxx}"
JUDGE_MODEL="${JUDGE_MODEL:-DeepSeek-V4-Flash}"

# ========== Evaluation parameters ==========
DATA_DIR="$SCRIPT_DIR/data"
OUTPUT_DIR="$SCRIPT_DIR/eval_results_remote"
TEMPERATURE=0.0               # generation temperature
GEN_MAX_TOKENS=10240          # max tokens for generating the plan
JUDGE_MAX_TOKENS=20480        # max tokens for the judge (scoring) stage
WORKERS=10                    # number of parallel threads

# Optional: only evaluate the specified datasets (empty = all), e.g.: DATASETS=(ResearchPlanGen_ml)
DATASETS=(ResearchPlanGen_ml)
# Optional: process at most the first N items per dataset (empty = all)
LIMIT="200"
# Force regeneration (ignore existing intermediate results)
FORCE=""

# ========== Remote model list ==========
# Format: "model_name|output_file|display_name"
# Note: the model name is determined by the remote API (e.g. `kimi-k2.6`, `Qwen3-235B-A22B`, etc.), not a local path.
MODELS=(
    # "/path/to/your/model|my-model-name"
)

# ========== Signal handling ==========
EVAL_PID=""

cleanup() {
    echo "Signal received, cleaning up..."
    if [[ -n $EVAL_PID ]] && kill -0 "$EVAL_PID" 2>/dev/null; then
        echo "Terminating the scoring Python process (PID=$EVAL_PID)..."
        kill -TERM "$EVAL_PID" 2>/dev/null
        sleep 2
        kill -KILL "$EVAL_PID" 2>/dev/null || true
    fi
    exit 1
}

trap cleanup SIGTERM SIGINT SIGQUIT SIGHUP

# ========== Helper functions ==========

# Normalize the API base: ensure it ends with /v1 (OpenAI-compatible client convention)
normalize_api_base() {
    if [[ "$API_BASE" != */v1 ]]; then
        API_BASE="${API_BASE%/}/v1"
    fi
}

check_api() {
    echo "Checking remote API connectivity: $API_BASE/models"
    if ! curl -s --max-time 15 -H "Authorization: Bearer $API_KEY" \
        "$API_BASE/models" > /dev/null 2>&1; then
        echo "Error: unable to reach the remote API ($API_BASE). Check the network / API_KEY."
        exit 1
    fi
    echo "  API connectivity OK"
}

run_eval() {
    local model_name="$1"
    local output_path="$2"
    local display_name="$3"

    # The result directory = the output file path with the .json suffix removed
    local result_dir="${output_path%.json}"

    mkdir -p "$result_dir"

    # Scoring model: if JUDGE_MODEL is empty, use the same model being evaluated
    local judge_model="${JUDGE_MODEL:-$model_name}"

    echo ""
    echo "============================================"
    echo "  Remote model (gen):   $display_name"
    echo "  Remote model name:    $model_name"
    echo "  Generation API:       $gen_base"
    echo "  Generation model:     $gen_model"
    echo "  Judge model:          $judge_model"
    echo "  Judge API:            $judge_base"
    echo "  Output:               $result_dir"
    echo "============================================"
    echo ""

    # ---- 1. Assemble the rubric_eval.py arguments (both generation and scoring use the remote API) ----
    echo "[step 1/2] Running rubric evaluation (remote generation + remote scoring)..."

    # Generation model params (prefer GEN_API_BASE/GEN_API_KEY/GEN_MODEL; otherwise fall back to the general config)
    local gen_base="${GEN_API_BASE:-$API_BASE}"
    local gen_key="${GEN_API_KEY:-$API_KEY}"
    local gen_model="${GEN_MODEL:-$model_name}"

    # Scoring model params (prefer JUDGE_API_BASE/JUDGE_API_KEY; otherwise fall back to the general config)
    local judge_base="${JUDGE_API_BASE:-$API_BASE}"
    local judge_key="${JUDGE_API_KEY:-$API_KEY}"

    local gen_args=(
        --gen-base-url "$gen_base"
        --gen-api-key "$gen_key"
        --gen-model "$gen_model"
        --gen-max-tokens "$GEN_MAX_TOKENS"
    )

    local judge_args=(
        --base-url "$judge_base"
        --api-key "$judge_key"
        --model "$judge_model"
    )

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

    # ---- 2. Summarize the results of this model ----
    echo ""
    echo "[step 2/2] Result summary ($display_name):"
    if [[ -f "$result_dir/summary.json" ]]; then
        cat "$result_dir/summary.json"
    else
        echo "  (summary.json not found)"
    fi
    echo ""
}

# ========== Summary result file ==========
SUMMARY_FILE="$OUTPUT_DIR/summary.txt"

write_summary_header() {
    mkdir -p "$OUTPUT_DIR"
    cat > "$SUMMARY_FILE" <<'EOF'
============================================
  ResearchPlanGen Rubric Batch Evaluation Summary (Remote API)
============================================

EOF
}

append_summary() {
    local display_name="$1"
    local output_path="$2"
    local result_dir="${output_path%.json}"
    local sum_file="$result_dir/summary.json"
    local judge_model="${JUDGE_MODEL:-$display_name}"
    local gen_base_disp="${GEN_API_BASE:-$API_BASE}"
    local judge_base_disp="${JUDGE_API_BASE:-$API_BASE}"

    {
        echo "--------------------------------------------"
        echo "  Remote model (gen):   $display_name"
        echo "  Generation API:       $gen_base_disp"
        echo "  Judge model:          $judge_model"
        echo "  Judge API:            $judge_base_disp"
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

normalize_api_base
check_api
write_summary_header

for model_entry in "${MODELS[@]}"; do
    IFS='|' read -r model_name output_path display_name <<< "$model_entry"

    # Skip complete results that already exist (optional: delete the result directory to re-evaluate)
    if [[ -f "${output_path%.json}/summary.json" ]] && [[ "$FORCE" != true ]]; then
        echo ""
        echo "  $display_name already has results, skipping: ${output_path%.json}/summary.json"
        continue
    fi

    run_eval "$model_name" "$output_path" "$display_name"
    append_summary "$display_name" "$output_path"
done

write_summary_tail

echo ""
echo "============================================"
echo "  All evaluations completed"
echo "  Detailed results: $OUTPUT_DIR/<model>_responses/"
echo "  Summary:          $SUMMARY_FILE"
echo "============================================"
