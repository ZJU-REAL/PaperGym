"""
Compare two model responses (different answers to the same question), call LLM API for scoring.

Will compare two model pairs:
  - qwen3_1b vs qwen3_1b_opsd_grpo
  - qwen3_4b vs qwen3_4b_opsd_grpo

For each question, randomly assign Plan A and Plan B, call LLM for blind evaluation.
"""

import os
import re
import sys
import json
import copy
import random
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from retry import retry

# =============================================================================
# Configuration constants
# =============================================================================
ANSWER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "answer")

API_KEY = os.getenv("OPENSCIENCE_API_KEY", "")
BASE_URL = os.getenv("OPENSCIENCE_BASE_URL", "http://your-api/v1")
MODEL = "DeepSeek-V4-Flash"

MAX_RETRIES = 5
PROGRESS_INTERVAL = 100  # print progress every 100 processed items; no intermediate saving, the Output file is written only after everything completes

# Dataset config: short name -> {dir: subdirectory under answer root, pairs: file pairs to compare for this dataset}
# Each dataset corresponds to a question source parquet (used when generating answers), here only its answer subdirectory and file pairs are declared.
DATASET_CONFIG = {
    # Question source: data/ml_data.parquet (685 questions)
    "ml": {
        "dir": "ml",
        "pairs": [
            {
                "name": "ml_1b_vs_ml_1b_cs",
                "file_a": "ml_1b_responses.json",
                "file_b": "ml_1b_cs_responses.json",
            },
            {
                "name": "ml_1b_vs_ml_1b_cs_final",
                "file_a": "ml_1b_responses.json",
                "file_b": "ml_1b_cs_final_responses.json",
            },
            # {
            #     "name": "ml_1b_vs_ml_1b_rubrichub",
            #     "file_a": "ml_1b_responses.json",
            #     "file_b": "ml_1b_rubrichub_responses.json",
            # },
            # {
            #     "name": "ml_1b_vs_ml_1b_researchqa",
            #     "file_a": "ml_1b_responses.json",
            #     "file_b": "ml_1b_researchqa_responses.json",
            # },
            {
                "name": "ml_1b_rubrichub_vs_ml_1b_cs",
                "file_a": "ml_1b_rubrichub_responses.json",
                "file_b": "ml_1b_cs_responses.json",
            },
            # {
            #     "name": "ml_1b_researchqa_vs_ml_1b_cs",
            #     "file_a": "ml_1b_researchqa_responses.json",
            #     "file_b": "ml_1b_cs_responses.json",
            # },
            # {
            #     "name": "ml_4b_vs_ml_4b_cs",
            #     "file_a": "ml_4b_responses.json",
            #     "file_b": "ml_4b_cs_responses.json",
            # },
            # {
            #     "name": "ml_4b_vs_ml_4b_researchqa",
            #     "file_a": "ml_4b_responses.json",
            #     "file_b": "ml_4b_researchqa_responses.json",
            # },
            # {
            #     "name": "ml_4b_cs_vs_ml_4b_researchqa",
            #     "file_a": "ml_4b_cs_responses.json",
            #     "file_b": "ml_4b_researchqa_responses.json",
            # },
        ],
    },
    # Question source: data/cs_new_solution.parquet (400 questions, RL format)
    # "cs_new": {
    #     "dir": "cs_new",
    #     "pairs": [
    #         {
    #             "name": "1b_vs_opsd_grpo",
    #             "file_a": "qwen3_1b_responses.json",
    #             "file_b": "qwen3_1b_opsd_grpo_responses.json",
    #         },
    #         {
    #             "name": "4b_vs_opsd_grpo",
    #             "file_a": "qwen3_4b_responses.json",
    #             "file_b": "qwen3_4b_opsd_grpo_responses.json",
    #         },
    #     ],
    # },
    # Question source: ai-co-scientist/data/arxiv_data.parquet (1496 questions, answers not yet generated)
    # "arxiv": {
    #     "dir": "arxiv",
    #     "pairs": [
    #         {
    #             "name": "arxiv_4b_vs_arxiv_4b_cs",
    #             "file_a": "arxiv_4b_responses.json",
    #             "file_b": "arxiv_4b_cs_responses.json",
    #         },
    #         {
    #             "name": "arxiv_4b_vs_arxiv_4b_researchqa",
    #             "file_a": "arxiv_4b_responses.json",
    #             "file_b": "arxiv_4b_researchqa_responses.json",
    #         },
    #         {
    #             "name": "arxiv_4b_cs_vs_arxiv_4b_researchqa",
    #             "file_a": "arxiv_4b_cs_responses.json",
    #             "file_b": "arxiv_4b_researchqa_responses.json",
    #         },
    #     ],
    # },
}


def build_file_pairs(datasets, answer_root) -> list:
    """Return file pairs to compare based on list of dataset short names.

    Each dataset corresponds to a subdirectory under answer_root (answer_root/<dir>),
    Answer files are read from the corresponding subdirectory.
    """
    pairs = []
    for ds in datasets:
        cfg = DATASET_CONFIG.get(ds)
        if cfg is None:
            raise SystemExit(
                f"Unknown dataset: {ds}. Optional: {list(DATASET_CONFIG)} or all"
            )
        ds_dir = os.path.join(answer_root, cfg["dir"])
        for t in cfg["pairs"]:
            pairs.append(
                {
                    "name": t["name"],
                    "file_a": t["file_a"],
                    "file_b": t["file_b"],
                    "answer_dir": ds_dir,
                }
            )
    return pairs

# =============================================================================
# LLM output cleanup
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


# =============================================================================
# LLM Client
# =============================================================================
class LLMClient:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    @retry(tries=MAX_RETRIES, delay=2, backoff=2)
    def generate_response(self, prompt: str, temperature: float = 0.0):
        """Generate answer (with auto-retry, exponential backoff)"""
        chat_completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return clean_llm_output(chat_completion.choices[0].message.content)


# =============================================================================
# Data loading
# =============================================================================
def load_json(filepath: str) -> dict:
    """Load JSON file, return dict (key is idx_X)."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


# =============================================================================
# Build evaluation prompt
# =============================================================================
def build_evaluation_prompt(scenario: str, plan_a: str, plan_b: str) -> str:
    """Build prompt based on scoring template."""
    prompt = f"""You are tasked with comparing two research plans for the same research scenario. You need to evaluate them based on specific criteria and provide scores. The order they are provided here is randomized.

# Research Scenario
{scenario}

# Research Plan A
{plan_a}

# Research Plan B
{plan_b}

# Evaluation Criteria
Evaluate each research plan based on the following criteria:

1. Goal Alignment
Which plan better understands and addresses the core research goal? A strong plan should identify the key scientific challenges, formulate relevant research questions or hypotheses, and avoid unnecessary experiments or directions that do not contribute to the stated objective.

2. Novel Insight
Which plan demonstrates stronger originality and research insight? A strong plan should propose novel hypotheses, perspectives, methodologies, or experimental designs rather than simply applying existing approaches. Novelty should be meaningful and connected to the research goal.

3. Scientific Soundness
Which plan provides a more rigorous and convincing scientific investigation? A strong plan should have a coherent reasoning chain from hypothesis to methodology to evidence. It should consider appropriate baselines, controls, confounding factors, alternative explanations, and experimental validation strategies.

4. Execution Quality
Which plan is clearer, more feasible, and better structured for implementation? A strong plan should specify what needs to be done, how it will be done, and why each step is necessary. It should balance ambition and practicality, avoiding unnecessary complexity while retaining essential experiments.

5. Expected Research Impact
Which plan is more likely to generate meaningful research outcomes if successfully executed? A strong plan should lead to interpretable findings, advance understanding of the research problem, or provide valuable methodological contributions.

# Overall Score

Finally, provide an overall score (1-10) for each research plan.

The score reflects how valuable the plan would be as a research blueprint if assigned to an average graduate student in the relevant field. Consider both scientific quality and practical usefulness.

Score guidelines:

10: Exceptional research plan.
The plan is highly insightful, rigorous, and well-designed. If executed as described, it would likely produce a strong research contribution with minimal additional guidance.

9: Excellent research plan.
The plan has a strong research idea, solid methodology, and clear execution path. Only minor improvements are needed.

8: Strong research plan.
The plan is coherent and useful, with good scientific reasoning. It may contain some limitations in novelty, rigor, or details, but can likely lead to a successful study with moderate refinement.

7: Good research plan.
The plan addresses the research goal appropriately and contains valuable ideas, but has noticeable weaknesses in methodology, clarity, feasibility, or depth.

6: Moderately useful research plan.
The plan provides a reasonable direction but lacks important details, has methodological gaps, or requires substantial additional insight to become a strong study.

5: Mixed-quality research plan.
The plan contains some relevant ideas but also significant weaknesses. It may be partially useful as a reference, but requires major improvement before execution.

4: Weak research plan.
The plan is related to the research goal but has substantial problems in reasoning, methodology, feasibility, or research focus.

3: Poor research plan.
The plan contains major flaws and would likely mislead a researcher if followed without significant correction.

2: Very poor research plan.
The plan is largely ineffective, with serious conceptual or methodological issues.

1: Invalid research plan.
The plan is irrelevant to the research goal, fundamentally incorrect, or unusable.

# Output Format
For each criteria, choose:
• A if Plan A is better
• B if Plan B is better
• Tie if both plans are roughly equal on this aspect

Provide your analysis and judgment in the following XML format:
<evaluation>
    <reasoning>
    Think critically and skeptically about both plans. Consider their weaknesses and strengths. Compare them relative to each other on each criteria.
    </reasoning>
    <criteria_judgments>
        <goal_alignment>A, B, or Tie</goal_alignment>
        <novel_insight>A, B, or Tie</novel_insight>
        <scientific_soundness>A, B, or Tie</scientific_soundness>
        <execution_quality>A, B, or Tie</execution_quality>
        <expected_impact>A, B, or Tie</expected_impact>
    </criteria_judgments>
    <overall_score_a>Score from 1-10 for Plan A</overall_score_a>
    <overall_score_b>Score from 1-10 for Plan B</overall_score_b>
</evaluation>"""
    return prompt


# =============================================================================
# XML result parsing
# =============================================================================
def parse_xml_result(text: str) -> dict:
    """Parse XML-formatted scoring results from LLM output."""
    result = {}

    # Extract reasoning
    reasoning_match = re.search(
        r"<reasoning>(.*?)</reasoning>", text, re.DOTALL
    )
    if reasoning_match:
        result["reasoning"] = reasoning_match.group(1).strip()

    # Extract criteria judgments
    for tag in [
        "goal_alignment",
        "novel_insight",
        "scientific_soundness",
        "execution_quality",
        "expected_impact",
    ]:
        match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
        if match:
            result[tag] = match.group(1).strip()
        else:
            result[tag] = "ParseError"

    # Extract overall score
    score_a_match = re.search(
        r"<overall_score_a>(.*?)</overall_score_a>", text, re.DOTALL
    )
    if score_a_match:
        result["overall_score_a"] = score_a_match.group(1).strip()

    score_b_match = re.search(
        r"<overall_score_b>(.*?)</overall_score_b>", text, re.DOTALL
    )
    if score_b_match:
        result["overall_score_b"] = score_b_match.group(1).strip()

    return result


# =============================================================================
# Save intermediate results (resume support)
# =============================================================================
def save_results(results: list, output_path: str):
    """Save scoring results to JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[Save] Saved {len(results)} results to {output_path}", flush=True)


# =============================================================================
# Main comparison logic
# =============================================================================
def evaluate_single(
    llm_client: LLMClient,
    pair_config: dict,
    idx_key: str,
    entry_a: dict,
    entry_b: dict,
    temperature: float,
):
    """Call the LLM to score a single question, returning a result entry (thread-safe)."""
    question = entry_a.get("question", "")
    answer_a = entry_a.get("answer", "")
    answer_b = entry_b.get("answer", "")
    index_val = entry_a.get("index", idx_key)

    if not answer_a or not answer_b:
        return None

    # Randomly assign Plan A and Plan B (blind evaluation)
    swap = random.choice([True, False])
    if swap:
        plan_a_display = f"File_B ({pair_config['file_b']})"
        plan_b_display = f"File_A ({pair_config['file_a']})"
        actual_a_content = answer_b
        actual_b_content = answer_a
    else:
        plan_a_display = f"File_A ({pair_config['file_a']})"
        plan_b_display = f"File_B ({pair_config['file_b']})"
        actual_a_content = answer_a
        actual_b_content = answer_b

    prompt = build_evaluation_prompt(question, actual_a_content, actual_b_content)

    try:
        raw_response = llm_client.generate_response(prompt, temperature=temperature)
    except Exception as e:
        print(f"    [Error] LLM call failed for idx={idx_key}: {e}", flush=True)
        raw_response = ""

    parsed = parse_xml_result(raw_response)

    result_entry = {
        "index": index_val,
        "idx_key": idx_key,
        "question": question,
        "file_a": pair_config["file_a"],
        "file_b": pair_config["file_b"],
        "plan_a_source": plan_a_display,
        "plan_b_source": plan_b_display,
        "swapped": swap,
        "plan_a_actual_source": (
            pair_config["file_b"] if swap else pair_config["file_a"]
        ),
        "plan_b_actual_source": (
            pair_config["file_a"] if swap else pair_config["file_b"]
        ),
        "raw_response": raw_response,
        "parsed": parsed,
    }
    return result_entry


def compare_pair(
    llm_client: LLMClient,
    pair_config: dict,
    output_dir: str,
    answer_dir: str = ANSWER_DIR,
    force: bool = False,
    temperature: float = 0.0,
    max_workers: int = 5,
):
    """Compare all matching questions in a pair of files (concurrent LLM calls).

    When force=False, if output file for this pair exists, it is considered complete and skipped.
    """
    name = pair_config["name"]
    pair_answer_dir = pair_config.get("answer_dir", answer_dir)
    file_a = os.path.join(pair_answer_dir, pair_config["file_a"])
    file_b = os.path.join(pair_answer_dir, pair_config["file_b"])
    output_path = os.path.join(output_dir, f"{name}_comparison_results.json")

    print(f"\n{'='*60}", flush=True)
    print(f"Processing comparison pair: {name}", flush=True)
    print(f"  File A: {pair_config['file_a']}", flush=True)
    print(f"  File B: {pair_config['file_b']}", flush=True)
    print(f"  Output: {output_path}", flush=True)
    print(f"  Number of concurrent threads: {max_workers}", flush=True)

    # If the Output file already exists and not forcing a rerun: treat this comparison pair as complete and skip
    if not force and os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing_list = json.load(f)
            print(
                f"  [Skip] Output file already exists ({len(existing_list)} results), skipping comparison: "
                f"{output_path}",
                flush=True,
            )
            return existing_list
        except (json.JSONDecodeError, IOError) as e:
            print(
                f"  [Warning] Output file exists but cannot be parsed ({e}), re-running in full: {output_path}",
                flush=True,
            )

    # Loading data
    data_a = load_json(file_a)
    data_b = load_json(file_b)

    # Finding common indices
    common_indices = sorted(set(data_a.keys()) & set(data_b.keys()))
    print(f"  Common questions count: {len(common_indices)}", flush=True)

    # Runtime validation: questions under the same idx in both files must be identical.
    # If inconsistent, it means the two files were not generated from the same set of questions (mismatch),
    # comparison results are meaningless, skip this idx and warn.
    mismatch_keys = []
    for idx_key in common_indices:
        qa = (data_a[idx_key].get("question") or "").strip()
        qb = (data_b[idx_key].get("question") or "").strip()
        if qa != qb:
            mismatch_keys.append(idx_key)
    if mismatch_keys:
        shown = ", ".join(mismatch_keys[:10])
        more = " ..." if len(mismatch_keys) > 10 else ""
        print(
            f"  [Warning] {len(mismatch_keys)} idx have inconsistent questions between the two files, skipped: "
            f"{shown}{more}",
            flush=True,
        )
        common_indices = [k for k in common_indices if k not in set(mismatch_keys)]

    if not common_indices:
        print("  No comparable idx (file is empty or all questions mismatch), skip this pair", flush=True)
        return []

    # Collect tasks to process (full comparison from scratch, no resume support)
    tasks = []
    for idx_key in common_indices:
        if idx_key not in data_a or idx_key not in data_b:
            continue
        tasks.append((idx_key, data_a[idx_key], data_b[idx_key]))
    print(f"  Pending: {len(tasks)} items", flush=True)

    results = []

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                evaluate_single,
                llm_client,
                pair_config,
                idx_key,
                entry_a,
                entry_b,
                temperature,
            ): idx_key
            for idx_key, entry_a, entry_b in tasks
        }
        for future in as_completed(future_to_idx):
            idx_key = future_to_idx[future]
            try:
                entry = future.result()
            except Exception as e:
                print(f"    [Error] Exception on idx={idx_key}: {e}", flush=True)
                entry = None
            if entry is not None:
                results.append(entry)
            completed += 1
            if completed % PROGRESS_INTERVAL == 0:
                print(
                    f"  [Progress] {completed}/{len(tasks)} done",
                    flush=True,
                )

    # Final save
    save_results(results, output_path)
    print(f"Completed comparison pair: {name}, {len(results)} results processed", flush=True)

    return results


# =============================================================================
# Summary statistics
# =============================================================================
def summarize_results(all_results: dict):
    """Compute summary statistics for all comparison results.

    Since Plan A/B are randomly swapped during scoring, judgments are mapped back to real file sources
    (file_a / file_b), to get real comparison between models.
    """
    print(f"\n{'='*60}", flush=True)
    print("Summary statistics", flush=True)
    print(f"{'='*60}", flush=True)

    dimensions = [
        "goal_alignment",
        "novel_insight",
        "scientific_soundness",
        "execution_quality",
        "expected_impact",
    ]

    for pair_name, results in all_results.items():
        print(f"\n--- {pair_name} ---", flush=True)
        valid = [r for r in results if r.get("parsed", {})]

        if not valid:
            print("  No valid results", flush=True)
            continue

        file_a_name = valid[0]["file_a"]
        file_b_name = valid[0]["file_b"]

        # Count per-dimension wins by the real file source
        # src_counts[d] = {file_a_name: x, file_b_name: y, "Tie": z, "ParseError": w}
        src_counts = {
            d: {file_a_name: 0, file_b_name: 0, "Tie": 0, "ParseError": 0}
            for d in dimensions
        }

        total_score_a = 0  # total score of file_a
        total_score_b = 0  # total score of file_b
        score_count = 0

        a_wins = 0
        b_wins = 0
        ties = 0

        for r in valid:
            parsed = r["parsed"]
            swapped = r.get("swapped", False)

            file_a_src = r["plan_a_actual_source"] if not swapped else r["plan_b_actual_source"]
            file_b_src = r["plan_b_actual_source"] if not swapped else r["plan_a_actual_source"]

            for d in dimensions:
                judgment = parsed.get(d, "ParseError")
                if judgment == "A":
                    src_counts[d][file_a_src] += 1
                elif judgment == "B":
                    src_counts[d][file_b_src] += 1
                elif judgment == "Tie":
                    src_counts[d]["Tie"] += 1
                else:
                    src_counts[d]["ParseError"] += 1

            try:
                score_a = float(parsed.get("overall_score_a", 0))
                score_b = float(parsed.get("overall_score_b", 0))
                # Map scores back to the real file source
                if not swapped:
                    total_score_a += score_a
                    total_score_b += score_b
                else:
                    total_score_a += score_b
                    total_score_b += score_a
                score_count += 1

                # Wins/losses between the real file sources
                if not swapped:
                    sa, sb = score_a, score_b
                else:
                    sa, sb = score_b, score_a
                if sa > sb:
                    a_wins += 1
                elif sb > sa:
                    b_wins += 1
                else:
                    ties += 1
            except (ValueError, TypeError):
                pass

        # Print per-dimension statistics (by real file source)
        print(f"\n  Per-dimension verdict distribution ({len(valid)} valid items):", flush=True)
        print(f"    file_a = {file_a_name}")
        print(f"    file_b = {file_b_name}")
        for d in dimensions:
            counts = src_counts[d]
            print(
                f"    {d}: {file_a_name}={counts[file_a_name]}, "
                f"{file_b_name}={counts[file_b_name]}, "
                f"Tie={counts['Tie']}, ParseError={counts['ParseError']}",
                flush=True,
            )

        # Print score statistics (by real file source)
        if score_count > 0:
            avg_a = total_score_a / score_count
            avg_b = total_score_b / score_count
            print("\n  Average overall score:", flush=True)
            print(f"    {file_a_name}: {avg_a:.2f} ({score_count} items)", flush=True)
            print(f"    {file_b_name}: {avg_b:.2f} ({score_count} items)", flush=True)
            print(f"    Difference (file_a - file_b): {avg_a - avg_b:.2f}", flush=True)
            print(
                f"    {file_a_name} wins: {a_wins}, "
                f"{file_b_name} wins: {b_wins}, Ties: {ties}",
                flush=True,
            )
        else:
            print("  No valid scores", flush=True)


# =============================================================================
# Main entry
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Compare two model responses and call LLM API for scoring"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
        help="Output directory (default: results/)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-comparison even if the output file already exists (default: skip when it exists)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of threads for concurrent LLM API calls (default: 5)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=API_KEY,
        help="LLM API Key",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=BASE_URL,
        help="LLM API Base URL",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL,
        help="LLM model name",
    )
    parser.add_argument(
        "--answer-dir",
        type=str,
        default=ANSWER_DIR,
        help="Answer file root directory (default: <script_dir>/answer), actually reads files under <answer-dir>/<dataset>/ subdirectory",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ml",
        help="Dataset short name to compare: ml / cs_new / arxiv, or all for all (default: ml)",
    )
    args = parser.parse_args()

    # Create the Output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Expand file pairs based on --dataset (ml / cs_new / arxiv / all)
    ds_arg = args.dataset.strip().lower()
    if ds_arg == "all":
        selected_datasets = list(DATASET_CONFIG)
    else:
        selected_datasets = [ds_arg]
    file_pairs = build_file_pairs(selected_datasets, args.answer_dir)

    # Filter out comparison pairs with missing answer files (e.g. when arxiv answers are not yet generated), and give hints
    valid_pairs = []
    for pair_config in file_pairs:
        fa = os.path.join(pair_config["answer_dir"], pair_config["file_a"])
        fb = os.path.join(pair_config["answer_dir"], pair_config["file_b"])
        missing = [p for p in (fa, fb) if not os.path.exists(p)]
        if missing:
            print(f"[Skip] Comparison pair {pair_config['name']} is missing answer files, skipping:", flush=True)
            for p in missing:
                print(f"        {p}", flush=True)
            continue
        valid_pairs.append(pair_config)
    if not valid_pairs:
        print(
            f"Error: dataset '{args.dataset}' has no usable answer files under {args.answer_dir}.",
            file=sys.stderr,
        )
        print(
            "      Generate answers for the corresponding parquet first, then place the answer JSONs in",
            file=sys.stderr,
        )
        print(
            f"      the {args.answer_dir}/<dataset>/ subdirectory (file names must match",
            file=sys.stderr,
        )
        print(
            "      file_a / file_b of the pairs in DATASET_CONFIG).",
            file=sys.stderr,
        )
        sys.exit(1)

    # Initializing LLM Client
    llm_client = LLMClient(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
    )

    # Processing all comparison pairs
    all_results = {}
    for pair_config in valid_pairs:
        results = compare_pair(
            llm_client=llm_client,
            pair_config=pair_config,
            output_dir=output_dir,
            answer_dir=args.answer_dir,
            force=args.force,
            temperature=args.temperature,
            max_workers=args.workers,
        )
        all_results[pair_config["name"]] = results

    # Summary statistics
    summarize_results(all_results)

    print(f"\nAll done! Results saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
