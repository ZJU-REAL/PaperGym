"""
Analyze the rubric composition of a merged dataset that spans multiple
disciplines (cs/econ/physics, etc.).
Each rubric is labeled as either "Method" or "Experiment".
Output: (1) overall Method/Experiment ratios and their quotient;
        (2) Method/Experiment ratios within each discipline category.

Supports the merged format (e.g. merged_dataset_1k.parquet), whose schema is:
    text / problem / solution / primary_category / title / arxiv_id
Here solution is a dict containing merged_rubrics / problem_rubrics /
solution_rubrics / rubrics.

Usage:
    python analyze_rubric_types.py [--parquet_path PARQUET_PATH] [--output_dir OUTPUT_DIR]
        [--model MODEL] [--api_key API_KEY] [--base_url BASE_URL] [--temperature TEMPERATURE]
        [--rubric_key RUBRIC_KEY] [--categories cs,econ,physics] [--max_items N] [--num_workers N]
"""

import os
import json
import re
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from openai import OpenAI
from retry import retry
import pandas as pd

# =============================================================================
# Configuration
# =============================================================================
API_KEY = "sk-xxx"
BASE_URL = "http://xxx/"
MODEL = "DeepSeek-V4-Flash"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PARQUET = os.path.join(SCRIPT_DIR, "data", "rubric_rate.parquet")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")


# =============================================================================
# Utility functions
# =============================================================================
def clean_llm_output(text: str) -> str:
    """Clean the LLM output, removing the <think> tag and its content."""
    if not isinstance(text, str):
        return ""
    think_end_tag = "</think>"
    tag_index = text.find(think_end_tag)
    if tag_index != -1:
        return text[tag_index + len(think_end_tag):].strip()
    return text.strip()


def extract_problem_text(item: dict) -> str:
    """Extract the question text: prefer the problem field of the merged format, and fall back to the old prompt field."""
    problem = item.get("problem", "")
    if isinstance(problem, str) and problem.strip():
        return problem

    prompt_data = item.get("prompt", "")
    if isinstance(prompt_data, np.ndarray):
        if len(prompt_data) > 0:
            first = prompt_data.tolist()[0]
            if isinstance(first, dict):
                return first.get("content", str(first))
            return str(first)
        return ""
    elif isinstance(prompt_data, list) and len(prompt_data) > 0:
        if isinstance(prompt_data[0], dict):
            return prompt_data[0].get("content", str(prompt_data[0]))
        return str(prompt_data[0])
    else:
        return str(prompt_data)[:500]


def extract_rubrics(item: dict, rubric_key: str) -> list:
    """Extract rubrics: prefer solution.<rubric_key> of the merged format, and fall back to the old extra_info.rubrics.

    Returns [{"id": ..., "description": ..., "weight": ...}]; returns [] when there are no rubrics.
    """
    solution = item.get("solution", {})
    if isinstance(solution, dict):
        raw = solution.get(rubric_key, [])
        if isinstance(raw, np.ndarray):
            raw = raw.tolist()
        elif not isinstance(raw, list):
            raw = list(raw) if raw else []
        rubrics = [
            {"id": r.get("id"), "description": r.get("description"), "weight": r.get("weight", 1)}
            for r in raw
            if isinstance(r, dict) and "id" in r and r.get("description")
        ]
        if rubrics:
            return rubrics

    # Old-format fallback: extra_info.rubrics
    extra_info = item.get("extra_info", {})
    raw_rubrics = extra_info.get("rubrics", []) if isinstance(extra_info, dict) else []
    if isinstance(raw_rubrics, np.ndarray):
        raw_rubrics = raw_rubrics.tolist()
    elif not isinstance(raw_rubrics, list):
        raw_rubrics = list(raw_rubrics) if raw_rubrics else []
    return [
        {"id": r.get("id"), "description": r.get("description"), "weight": r.get("weight", 1)}
        for r in raw_rubrics
        if isinstance(r, dict) and "id" in r and r.get("description")
    ]


def extract_category(item: dict) -> str:
    """Extract the subject category: take the dotted prefix of primary_category (e.g. 'cs.LG' -> 'cs'),
    and fall back to the old data_source / ability fields."""
    pc = item.get("primary_category", "")
    if isinstance(pc, str) and pc.strip():
        return pc.split(".")[0].strip() or "unknown"
    return str(item.get("data_source") or item.get("ability") or "unknown")


# =============================================================================
# LLM Client
# =============================================================================
class LLMClient:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None,
                 temperature: float = 0.1):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature

    @retry(tries=5, delay=1, backoff=2)
    def generate_response(self, prompt: str):
        chat_completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return clean_llm_output(chat_completion.choices[0].message.content)


# =============================================================================
# Prompt construction
# =============================================================================
CLASSIFICATION_PROMPT_TEMPLATE = """You are an expert reviewer of academic papers. For each rubric item below, classify it as either "Method" or "Experiment" based on these definitions:

1. **Method**: Involves proposing a new method, algorithm, model, framework, technique, theoretical contribution, or novel aspects of the solution design itself.
2. **Experiment**: Involves experimental setup, evaluation metrics, baseline comparisons, ablation studies, dataset selection, human-machine comparison experiments, cross-institution/cross-user comparisons, robustness verification, and other experiment-related designs.

Output only one label per line, one for each rubric, in the same order as listed below, and nothing else.

Problem context:
{problem_context}

Rubrics:
{rubrics_text}

Output:"""


def build_classification_prompt(problem_text: str, rubrics: list) -> str:
    """Build the classification prompt, one rubric per line."""
    rubrics_text = "\n".join(
        f"{r['id']}. {r['description']}"
        for r in rubrics
    )
    return CLASSIFICATION_PROMPT_TEMPLATE.format(
        problem_context=problem_text,
        rubrics_text=rubrics_text
    )


def parse_classification_response(response_text: str, rubrics: list):
    """
    Parse the classification result returned by the LLM line by line.
    Each line should be "Method" or "Experiment", in the same order as the rubrics.
    Returns a list or None (when parsing fails).
    Refer to the parse_batch_verdicts pattern in data_leakage_detector.py.
    """
    cleaned = response_text.strip()
    lines = cleaned.split('\n')
    lines = [line.strip() for line in lines if line.strip()]

    if len(lines) != len(rubrics):
        print(f"  [WARNING] Row count mismatch: expected {len(rubrics)}, got {len(lines)}")
        return None

    result = []
    for i, line in enumerate(lines):
        # Strip any possible numeric prefix (e.g. "1. Method" -> "Method")
        line = re.sub(r'^\d+[\.:\)]\s*', '', line).strip()
        if line == "Method":
            result.append({"id": rubrics[i]["id"], "type": "Method"})
        elif line == "Experiment":
            result.append({"id": rubrics[i]["id"], "type": "Experiment"})
        else:
            print(f"  [WARNING] Could not parse line {i+1}: '{line}'")
            return None

    return result


def classify_with_retry(llm_client: LLMClient, prompt: str, rubrics: list, max_retries: int = 3):
    """Call the LLM and parse the result; retry on parse failure, up to max_retries times. Returns a list or None (when all retries fail)."""
    for attempt in range(1, max_retries + 1):
        response = llm_client.generate_response(prompt)
        classification = parse_classification_response(response, rubrics)
        if classification is not None:
            return classification
        if attempt < max_retries:
            print(f"  [RETRY] Parsing failed on attempt {attempt}, retrying ({attempt + 1}/{max_retries})...", flush=True)
    print(f"  [ERROR] Parsing still failed after {max_retries} attempts, skipping this item", flush=True)
    return None


def process_item(idx: int, item: dict, llm_client: LLMClient, rubric_key: str):
    """Process a single item in the thread pool.

    Extracts the question text / subject category / rubrics, calls the LLM to classify, and tallies.
    Returns a dict (containing index, category, Method/Experiment counts, and per-rubric results),
    or None when skipped (no rubrics or classification failure).
    """
    problem_text = extract_problem_text(item)
    category = extract_category(item)
    rubrics = extract_rubrics(item, rubric_key)

    if len(rubrics) == 0:
        print(f"  [{idx + 1}] Skipped: no rubrics", flush=True)
        return None

    prompt = build_classification_prompt(problem_text, rubrics)
    classification = classify_with_retry(llm_client, prompt, rubrics, max_retries=3)

    if classification is None:
        return None

    method_count = sum(1 for c in classification if c.get("type") == "Method")
    experiment_count = sum(1 for c in classification if c.get("type") == "Experiment")
    unknown_count = sum(1 for c in classification if c.get("type") not in ("Method", "Experiment"))

    detail_rubrics = []
    for rubric in rubrics:
        matched = [c for c in classification if c.get("id") == rubric["id"]]
        rubric_type = matched[0]["type"] if matched else "unknown"
        detail_rubrics.append({
            "id": rubric["id"],
            "description": rubric["description"],
            "type": rubric_type,
        })

    return {
        "index": idx,
        "category": category,
        "total": len(rubrics),
        "Method": method_count,
        "Experiment": experiment_count,
        "unknown": unknown_count,
        "rubrics": detail_rubrics,
    }


# =============================================================================
# Statistics
# =============================================================================
def build_stats(total_questions: int, overall: dict, per_category: dict) -> dict:
    """Build the overall + per-category statistics results."""
    stats = {
        "total_questions": total_questions,
        "total_rubrics": overall["total"],
        "Method_count": overall["Method"],
        "Experiment_count": overall["Experiment"],
        "Method_ratio": round(overall["Method"] / overall["total"] * 100, 2) if overall["total"] else 0,
        "Experiment_ratio": round(overall["Experiment"] / overall["total"] * 100, 2) if overall["total"] else 0,
        "experiment_to_method_ratio": round(overall["Experiment"] / overall["Method"], 4) if overall["Method"] else None,
        "method_to_experiment_ratio": round(overall["Method"] / overall["Experiment"], 4) if overall["Experiment"] else None,
        "per_category": {},
    }

    for cat, c in sorted(per_category.items()):
        cat_total = c["total"]
        stats["per_category"][cat] = {
            "questions": c["questions"],
            "total_rubrics": cat_total,
            "rubric_share": round(cat_total / overall["total"] * 100, 2) if overall["total"] else 0,
            "Method_count": c["Method"],
            "Experiment_count": c["Experiment"],
            "Method_ratio": round(c["Method"] / cat_total * 100, 2) if cat_total else 0,
            "Experiment_ratio": round(c["Experiment"] / cat_total * 100, 2) if cat_total else 0,
            "experiment_to_method_ratio": round(c["Experiment"] / c["Method"], 4) if c["Method"] else None,
        }
    return stats


def print_stats(stats: dict) -> None:
    """Print the statistics overview."""
    print("\n" + "=" * 60)
    print("Overall statistics (all disciplines)")
    print("=" * 60)
    print(f"Total questions: {stats['total_questions']}")
    print(f"Total rubrics: {stats['total_rubrics']}")
    print(f"Method: {stats['Method_count']} ({stats['Method_ratio']}%)")
    print(f"Experiment: {stats['Experiment_count']} ({stats['Experiment_ratio']}%)")
    if stats["experiment_to_method_ratio"] is not None:
        print(f"Experiment : Method = {stats['experiment_to_method_ratio']} : 1")

    print("\nPer-discipline statistics:")
    for cat, c in stats["per_category"].items():
        print(f"  [{cat}] questions={c['questions']}, rubrics={c['total_rubrics']} "
              f"(share {c['rubric_share']}%), "
              f"Method={c['Method_count']} ({c['Method_ratio']}%), "
              f"Experiment={c['Experiment_count']} ({c['Experiment_ratio']}%)")


# =============================================================================
# Main logic
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Analyze rubric type ratios in a merged dataset (overall + per discipline)")
    parser.add_argument("--parquet_path", type=str, default=DEFAULT_PARQUET,
                        help="Path to the input parquet file")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Path to the output directory")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="LLM model name")
    parser.add_argument("--api_key", type=str, default=API_KEY,
                        help="LLM API Key")
    parser.add_argument("--base_url", type=str, default=BASE_URL,
                        help="LLM Base URL")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="LLM sampling temperature")
    parser.add_argument("--rubric_key", type=str, default="merged_rubrics",
                        choices=["merged_rubrics", "problem_rubrics", "solution_rubrics", "rubrics"],
                        help="Rubric set to analyze (inside the solution dict of the merged format)")
    parser.add_argument("--categories", type=str, default=None,
                        help="Only analyze the given disciplines, comma-separated (e.g. 'cs,econ,physics'; empty means all)")
    parser.add_argument("--max_items", type=int, default=None,
                        help="Maximum number of questions to process (for testing)")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of parallel threads (LLM calls are I/O bound, default 8)")
    args = parser.parse_args()

    categories = [c.strip() for c in args.categories.split(",") if c.strip()] if args.categories else None

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize the LLM client
    llm_client = LLMClient(api_key=args.api_key, base_url=args.base_url,
                           model=args.model, temperature=args.temperature)

    # Read the data
    print(f"Reading data: {args.parquet_path}")
    df = pd.read_parquet(args.parquet_path)
    items = df.to_dict("records")
    if args.max_items is not None:
        items = items[:args.max_items]
    if categories:
        items = [it for it in items if extract_category(it) in categories]
    total = len(items)
    print(f"{total} question records in total" +
          (f" (limited to {args.max_items})" if args.max_items else "") +
          (f", disciplines={categories}" if categories else "") +
          f", threads={args.num_workers}")

    # =====================================================================
    # Multithreaded parallel processing (LLM calls are I/O-bound, use a thread pool)
    # =====================================================================
    all_results = []
    overall = {"Method": 0, "Experiment": 0, "total": 0}
    per_category = {}

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        # Submit all tasks; the completion order of futures is arbitrary, so use the index to map back to the original data
        future_to_idx = {
            executor.submit(process_item, idx, item, llm_client, args.rubric_key): idx
            for idx, item in enumerate(items)
        }

        completed = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            completed += 1
            try:
                result = future.result()
            except Exception as e:
                print(f"  [{completed}/{total}] Error processing item {idx + 1}: {e}", flush=True)
                continue

            if result is None:
                continue

            overall["Method"] += result["Method"]
            overall["Experiment"] += result["Experiment"]
            overall["total"] += result["total"]

            cat_stats = per_category.setdefault(
                result["category"], {"Method": 0, "Experiment": 0, "total": 0, "questions": 0})
            cat_stats["Method"] += result["Method"]
            cat_stats["Experiment"] += result["Experiment"]
            cat_stats["total"] += result["total"]
            cat_stats["questions"] += 1

            all_results.append({
                "index": result["index"],
                "category": result["category"],
                "statistics": {
                    "total": result["total"],
                    "Method": result["Method"],
                    "Experiment": result["Experiment"],
                    "unknown": result["unknown"],
                },
                "rubrics": result["rubrics"],
            })

            print(f"  [{completed}/{total}] [{result['category']}] "
                  f"Method={result['Method']}, Experiment={result['Experiment']}", flush=True)

    # Sort by the original index to ensure a deterministic output order
    all_results.sort(key=lambda r: r["index"])

    # =====================================================================
    # Output the detailed results JSON
    # =====================================================================
    detail_path = os.path.join(args.output_dir, "rubric_classification_detail.json")
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to: {detail_path}")

    # =====================================================================
    # Statistics overview
    # =====================================================================
    stats = build_stats(len(all_results), overall, per_category)

    stats_path = os.path.join(args.output_dir, "rubric_classification_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Statistics saved to: {stats_path}")

    # Print the statistics
    print_stats(stats)


if __name__ == "__main__":
    main()
