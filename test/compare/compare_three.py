#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_three.py — Three-model response comparison (select only one best per dimension per round).

Unlike compare_responses.py (pairwise comparison), this script sends three responses simultaneously into a
prompt, letting the LLM select a unique Winner per dimension, and giving each of the three responses
an Overall Score (1-10). Finally, calculate each model's "accuracy" (i.e., the proportion judged as best
per dimension).

Currently only supports one comparison group:
  ml_1b_responses.json
  ml_1b_rubrichub_responses.json
  ml_1b_cs_responses.json
"""

import os
import re
import sys
import json
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
PROGRESS_INTERVAL = 20

# Three-model comparison config: name -> three response files (under answer/ml/)
TRIPLE_CONFIG = {
    "name": "ml_1b_vs_rubrichub_vs_cs",
    "dir": "ml",
    "files": [
        "ml_1b_responses.json",
        "ml_1b_rubrichub_responses.json",
        "ml_1b_cs_responses.json",
    ],
}

DIMENSIONS = [
    "goal_alignment",
    "novel_insight",
    "scientific_soundness",
    "execution_quality",
    "expected_impact",
]


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
# Build three-model scoring prompt
# =============================================================================
def build_evaluation_prompt(scenario: str, plan_a: str, plan_b: str, plan_c: str) -> str:
    """Build three-model comparison prompt based on scoring template."""
    prompt = f"""You are tasked with comparing three research plans for the same research scenario. You need to evaluate them based on specific criteria and provide scores. The order they are provided here is randomized.

# Research Scenario
{scenario}

# Research Plan A
{plan_a}

# Research Plan B
{plan_b}

# Research Plan C
{plan_c}

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

For each criteria, choose exactly ONE winner: A, B, or C. Do not choose Tie.

Provide your analysis and judgment in the following XML format:
<evaluation>
    <reasoning>
    Think critically and skeptically about all three plans. Consider their weaknesses and strengths. Compare them relative to each other on each criteria.
    </reasoning>
    <criteria_judgments>
        <goal_alignment>A, B, or C</goal_alignment>
        <novel_insight>A, B, or C</novel_insight>
        <scientific_soundness>A, B, or C</scientific_soundness>
        <execution_quality>A, B, or C</execution_quality>
        <expected_impact>A, B, or C</expected_impact>
    </criteria_judgments>
    <overall_score_a>Score from 1-10 for Plan A</overall_score_a>
    <overall_score_b>Score from 1-10 for Plan B</overall_score_b>
    <overall_score_c>Score from 1-10 for Plan C</overall_score_c>
</evaluation>"""
    return prompt


# =============================================================================
# XML result parsing
# =============================================================================
def parse_xml_result(text: str) -> dict:
    """Parse XML-formatted scoring results from LLM output."""
    result = {}

    # Extract reasoning
    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
    if reasoning_match:
        result["reasoning"] = reasoning_match.group(1).strip()

    # Extract criteria judgments
    for tag in DIMENSIONS:
        match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
        if match:
            val = match.group(1).strip()
            if val in ("A", "B", "C"):
                result[tag] = val
            else:
                result[tag] = "ParseError"
        else:
            result[tag] = "ParseError"

    # Extract overall scores
    for label in ("a", "b", "c"):
        match = re.search(f"<overall_score_{label}>(.*?)</overall_score_{label}>", text, re.DOTALL)
        if match:
            result[f"overall_score_{label}"] = match.group(1).strip()

    return result


# =============================================================================
# Save results
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
    config: dict,
    idx_key: str,
    entries: dict,
    temperature: float,
):
    """Call the LLM to score a single question, returning a result entry (thread-safe)."""
    question = entries["question"]
    answers = entries["answers"]
    index_val = entries.get("index", idx_key)

    if not all(answers):
        return None

    # Randomly shuffle the Plan A/B/C assignment (blind evaluation)
    #
    # Rationale:
    #   file_names = [F0, F1, F2]  — three real files, in fixed order
    #   answers    = [C0, C1, C2]  — the corresponding contents
    #
    # shuffle is a random permutation of [0,1,2], e.g. [2, 0, 1]
    #   plan_contents[0] = answers[2] = C2  →  Plan A shows F2's content in the prompt
    #   plan_contents[1] = answers[0] = C0  →  Plan B shows F0's content in the prompt
    #   plan_contents[2] = answers[1] = C1  →  Plan C shows F1's content in the prompt
    #
    # Therefore display_to_actual must satisfy:
    #   display_to_actual["A"] = file_names[shuffle[0]] = F2
    #   display_to_actual["B"] = file_names[shuffle[1]] = F0
    #   display_to_actual["C"] = file_names[shuffle[2]] = F1
    file_names = config["files"]
    shuffle = list(range(3))
    random.shuffle(shuffle)
    plan_contents = [answers[i] for i in shuffle]
    plan_labels = ["A", "B", "C"]
    display_to_actual = {plan_labels[i]: file_names[shuffle[i]] for i in range(3)}

    # Self-check: ensure plan_contents[i] indeed comes from the file corresponding to display_to_actual[plan_labels[i]]
    for i, label in enumerate(plan_labels):
        actual_file = display_to_actual[label]
        file_idx = file_names.index(actual_file)
        assert answers[file_idx] == plan_contents[i], (
            f"Shuffle mapping error: {label} maps to {actual_file} "
            f"but plan_contents[{i}] does not match"
        )

    prompt = build_evaluation_prompt(question, plan_contents[0], plan_contents[1], plan_contents[2])

    try:
        raw_response = llm_client.generate_response(prompt, temperature=temperature)
    except Exception as e:
        print(f"    [Error] LLM call failed for idx={idx_key}: {e}", flush=True)
        raw_response = ""

    parsed = parse_xml_result(raw_response)

    # Map the A/B/C winner back to the real file name
    dim_winners = {}
    for d in DIMENSIONS:
        winner = parsed.get(d, "ParseError")
        dim_winners[d] = display_to_actual.get(winner, "ParseError") if winner in ("A", "B", "C") else "ParseError"

    # Map overall_score_a/b/c back to the real file name
    score_map = {}
    for label in ("a", "b", "c"):
        actual_file = display_to_actual.get(label.upper())
        if actual_file and f"overall_score_{label}" in parsed:
            score_map[actual_file] = parsed[f"overall_score_{label}"]

    result_entry = {
        "index": index_val,
        "idx_key": idx_key,
        "question": question,
        "files": file_names,
        "raw_response": raw_response,
        "parsed": {
            "reasoning": parsed.get("reasoning", ""),
            **{d: dim_winners[d] for d in DIMENSIONS},
            "scores": score_map,
        },
        "display_to_actual": display_to_actual,
    }
    return result_entry


def compare_triple(
    llm_client: LLMClient,
    config: dict,
    output_dir: str,
    answer_dir: str = ANSWER_DIR,
    force: bool = False,
    temperature: float = 0.0,
    max_workers: int = 5,
    max_entries: int = None,
):
    """Compare all matching questions in three model responses (concurrent LLM calls)."""
    name = config["name"]
    pair_answer_dir = os.path.join(answer_dir, config["dir"])
    file_paths = [os.path.join(pair_answer_dir, f) for f in config["files"]]
    output_path = os.path.join(output_dir, f"{name}_comparison_results.json")

    print(f"\n{'='*60}", flush=True)
    print(f"Processing three-model comparison: {name}", flush=True)
    for i, fp in enumerate(config["files"]):
        print(f"  File {i+1}: {fp}", flush=True)
    print(f"  Output:   {output_path}", flush=True)
    print(f"  Threads:  {max_workers}", flush=True)

    # Output file exists and force is not set: skip
    if not force and os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing_list = json.load(f)
            print(f"  [Skip] Output file already exists ({len(existing_list)} results), skipping comparison: {output_path}", flush=True)
            return existing_list
        except (json.JSONDecodeError, IOError) as e:
            print(f"  [Warning] Output file exists but cannot be parsed ({e}), re-running in full: {output_path}", flush=True)

    # Loading three files
    data_list = [load_json(fp) for fp in file_paths]
    common_indices = sorted(set.intersection(*(set(d.keys()) for d in data_list)))
    print(f"  Common questions count: {len(common_indices)}", flush=True)

    # Runtime validation: questions under the same idx in all three files must be identical
    mismatch_keys = []
    for idx_key in common_indices:
        qs = [d[idx_key].get("question", "").strip() for d in data_list]
        if len(set(qs)) > 1:
            mismatch_keys.append(idx_key)
    if mismatch_keys:
        shown = ", ".join(mismatch_keys[:10])
        more = " ..." if len(mismatch_keys) > 10 else ""
        print(f"  [Warning] {len(mismatch_keys)} indices with inconsistent questions in three files, skipped: {shown}{more}", flush=True)
        common_indices = [k for k in common_indices if k not in set(mismatch_keys)]

    if not common_indices:
        print("  No comparable idx (file is empty or all questions mismatch), skip", flush=True)
        return []

    tasks = []
    for idx_key in common_indices:
        entries = {
            "question": data_list[0][idx_key].get("question", ""),
            "answers": [d[idx_key].get("answer", "") for d in data_list],
            "index": data_list[0][idx_key].get("index", idx_key),
        }
        tasks.append((idx_key, entries))

    # If --max-entries is specified, randomly sample to limit entries
    if max_entries is not None and max_entries > 0 and len(tasks) > max_entries:
        import random
        random.shuffle(tasks)
        tasks = tasks[:max_entries]
        print(f"  [Limit] Randomly sampled down to {max_entries} items", flush=True)

    print(f"  Pending: {len(tasks)} items", flush=True)

    results = []
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(evaluate_single, llm_client, config, idx_key, entries, temperature): idx_key
            for idx_key, entries in tasks
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
                print(f"  [Progress] {completed}/{len(tasks)} done", flush=True)

    save_results(results, output_path)
    print(f"Completed three-model comparison: {name}, {len(results)} results processed", flush=True)
    return results


# =============================================================================
# Summary statistics
# =============================================================================
def summarize_results(results: list, config: dict):
    """Summarize the three-model comparison results.

    Counts each model's proportion of being judged best per dimension (accuracy),
    as well as the average Overall Score and the overall-score win rate.
    """
    print(f"\n{'='*60}", flush=True)
    print("Three-model comparison summary statistics", flush=True)
    print(f"{'='*60}", flush=True)

    files = config["files"]
    n = len(results)
    if n == 0:
        print("  No valid results", flush=True)
        return

    # Dimension statistics: number of wins per model per dimension
    dim_wins = {d: {f: 0 for f in files} for d in DIMENSIONS}
    dim_parse_errors = {d: 0 for d in DIMENSIONS}

    # Overall Score statistics
    score_sums = {f: 0.0 for f in files}
    score_counts = {f: 0 for f in files}
    score_wins = {f: 0 for f in files}

    for r in results:
        parsed = r["parsed"]
        # Dimension winner
        for d in DIMENSIONS:
            winner = parsed.get(d, "ParseError")
            if winner in files:
                dim_wins[d][winner] += 1
            else:
                dim_parse_errors[d] += 1
        # Overall Score
        scores = parsed.get("scores", {})
        valid_scores = {}
        for f in files:
            if f in scores:
                try:
                    valid_scores[f] = float(scores[f])
                except (ValueError, TypeError):
                    pass
        if len(valid_scores) == len(files):
            for f, sc in valid_scores.items():
                score_sums[f] += sc
                score_counts[f] += 1
            best = max(valid_scores, key=valid_scores.get)
            score_wins[best] += 1

    # Print per-dimension accuracy
    print("\n  Best verdict per dimension (accuracy = times judged best / total items):")
    print(f"  {'Dimension':<20} {' '.join([f'{short_name(f):<18}' for f in files])}")
    print(f"  {'-' * (20 + 18 * len(files))}")
    for d in DIMENSIONS:
        row = f"  {d:<20}"
        for f in files:
            acc = dim_wins[d][f] / n * 100 if n > 0 else 0
            row += f" {acc:>5.1f}% ({dim_wins[d][f]:>3}/{n})"
        print(row, flush=True)
        if dim_parse_errors[d] > 0:
            print(f"    [ParseError: {dim_parse_errors[d]} items]", flush=True)

    # Print Overall Score statistics
    print("\n  Overall score statistics:")
    print(f"  {'Model':<30} {'Avg Score':>8} {'Win Rate':>10} {'Valid Items':>8}")
    print(f"  {'-' * 70}")
    for f in files:
        avg = score_sums[f] / score_counts[f] if score_counts[f] > 0 else 0
        win_rate = score_wins[f] / n * 100 if n > 0 else 0
        print(f"  {short_name(f):<30} {avg:>8.2f} {win_rate:>9.1f}% {score_counts[f]:>8}", flush=True)

    # Print each model's overall accuracy (all dimensions + total score)
    print("\n  Overall accuracy (mean of per-dimension best verdicts and total score win rate):")
    for f in files:
        total_wins = sum(dim_wins[d][f] for d in DIMENSIONS) + score_wins[f]
        total_possible = len(DIMENSIONS) * n + n
        overall_acc = total_wins / total_possible * 100 if total_possible > 0 else 0
        print(f"  {short_name(f):<30} {overall_acc:>6.2f}% ({total_wins}/{total_possible})", flush=True)


def short_name(name: str) -> str:
    return name.replace("_responses.json", "")


# =============================================================================
# Main entry
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Three-model response comparison (select only one best per dimension)")
    parser.add_argument("--output-dir", type=str, default="results_three",
                        help="Output directory (default: results_three/)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-comparison even if output file exists")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="LLM sampling temperature (default: 0.0)")
    parser.add_argument("--workers", type=int, default=5,
                        help="Number of threads for concurrent LLM API calls (default: 5)")
    parser.add_argument("--api-key", type=str, default=API_KEY, help="LLM API Key")
    parser.add_argument("--base-url", type=str, default=BASE_URL, help="LLM API Base URL")
    parser.add_argument("--model", type=str, default=MODEL, help="LLM model name")
    parser.add_argument("--answer-dir", type=str, default=ANSWER_DIR,
                        help="Answer file root directory (default: <script_dir>/answer)")
    parser.add_argument("--max-entries", type=int, default=None,
                        help="Limit number of entries to compare (default: unlimited)")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Verify answer files exist
    pair_answer_dir = os.path.join(args.answer_dir, TRIPLE_CONFIG["dir"])
    missing = []
    for f in TRIPLE_CONFIG["files"]:
        fp = os.path.join(pair_answer_dir, f)
        if not os.path.exists(fp):
            missing.append(fp)
    if missing:
        print("Error: missing answer files:", file=sys.stderr)
        for p in missing:
            print(f"  {p}", file=sys.stderr)
        sys.exit(1)

    # Initialize the LLM Client
    llm_client = LLMClient(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
    )

    # Run the comparison
    results = compare_triple(
        llm_client=llm_client,
        config=TRIPLE_CONFIG,
        output_dir=output_dir,
        answer_dir=args.answer_dir,
        force=args.force,
        temperature=args.temperature,
        max_workers=args.workers,
        max_entries=args.max_entries,
    )

    # Summary statistics
    summarize_results(results, TRIPLE_CONFIG)

    print(f"\nAll done! Results saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
