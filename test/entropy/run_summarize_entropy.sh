#!/bin/bash
#
# Summarize the entropy measurement results: scans all entropy_*.json files under the results/ directory,
# extracts the statistics, and generates a summary table and JSON file.
#
# Usage:
#   bash run_summarize_entropy.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/results_4b"
SUMMARY_OUTPUT="${OUTPUT_DIR}/entropy_summary.json"

echo ""
echo "==========================================="
echo "  Entropy result summary"
echo "  Scan dir:      ${OUTPUT_DIR}"
echo "  Summary output: ${SUMMARY_OUTPUT}"
echo "==========================================="
echo ""

if [[ ! -d "$OUTPUT_DIR" ]]; then
    echo "Error: results directory does not exist: ${OUTPUT_DIR}"
    echo "Please run run_compute_entropy.sh first"
    exit 1
fi

# List the result file of each model
echo "Per-model results:"
for result_file in "${OUTPUT_DIR}"/entropy_*.json; do
    if [[ ! -f "$result_file" ]]; then
        continue
    fi
    basename=$(basename "$result_file")
    model_name="${basename#entropy_}"
    model_name="${model_name%.json}"

    # Extract the statistics
    mean_val=$(python3 -c "
import json
with open('${result_file}') as f:
    data = json.load(f)
per_item_means = [r.get('entropy_stats', {}).get('mean_entropy', float('nan')) for r in data if 'entropy_stats' in r and r.get('entropy_stats')]
if per_item_means:
    import numpy as np
    vals = np.array(per_item_means)
    print(f'overall_mean={np.mean(vals):.4f}, overall_std={np.std(vals):.4f}, overall_min={np.min(vals):.4f}, overall_max={np.max(vals):.4f}, n={len(vals)}')
else:
    print('(no data)')
" 2>/dev/null || echo "(extract failed)")

    echo "  - ${model_name}: ${mean_val}"
done

echo ""

# Generate the summary JSON
echo ">>> Generating the summary file: ${SUMMARY_OUTPUT}"
python3 -c "
import json, os, glob
import numpy as np

results_dir = '${OUTPUT_DIR}'
summary = {'models': []}

files = sorted(glob.glob(os.path.join(results_dir, 'entropy_*.json')))
if not files:
    print('  No entropy_*.json files found')
    exit(0)

for fpath in files:
    try:
        with open(fpath) as f:
            data = json.load(f)
        per_item_means = [
            r.get('entropy_stats', {}).get('mean_entropy', None)
            for r in data if r.get('entropy_stats')
        ]
        per_item_means = [v for v in per_item_means if v is not None]
        if per_item_means:
            arr = np.array(per_item_means)
            basename = os.path.basename(fpath).replace('entropy_', '').replace('.json', '')
            summary['models'].append({
                'model': basename,
                'file': fpath,
                'num_responses': len(per_item_means),
                'overall_mean_entropy': round(float(np.mean(arr)), 4),
                'overall_std_entropy': round(float(np.std(arr)), 4),
                'overall_min_entropy': round(float(np.min(arr)), 4),
                'overall_max_entropy': round(float(np.max(arr)), 4),
                'overall_median_entropy': round(float(np.median(arr)), 4),
            })
    except Exception as e:
        print(f'  [Warning] Failed to read {fpath}: {e}')

os.makedirs(results_dir, exist_ok=True)
with open('${SUMMARY_OUTPUT}', 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f'  Summarized {len(summary[\"models\"])} models in total')
print()
for m in summary['models']:
    print(f'    {m[\"model\"]}:')
    print(f'      mean={m[\"overall_mean_entropy\"]:.4f}, std={m[\"overall_std_entropy\"]:.4f}')
    print(f'      min={m[\"overall_min_entropy\"]:.4f}, max={m[\"overall_max_entropy\"]:.4f}')
    print(f'      n={m[\"num_responses\"]}')
" || echo "  [Warning] Failed to generate the summary"

echo ""
echo "==========================================="
echo "  Summary completed"
echo "  Summary file: ${SUMMARY_OUTPUT}"
echo "==========================================="
echo ""
