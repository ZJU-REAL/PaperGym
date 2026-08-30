"""
Based on the generated answers and the corresponding rubrics, score all rubrics under one question
in a single batch call.

Input:
- Evaluation set parquet (contains rubrics)
- Answers JSON (output of generate_responses.py)

Output JSON format:
{
    "idx_0": {
        "prompt": "original question",
        "answer": "model answer",
        "rubrics": [
            {
                "criterion": "rubric criterion text",
                "points": 10,
                "judge": {
                    "explanation": "judgment explanation",
                    "criteria_met": true
                }
            },
            ...
        ],
        "score": 0.75,
        "raw_score": 30,
        "total_weight": 40
    },
    ...
}
Here score = raw_score / total_weight (normalized score, i.e. actual score / theoretical maximum), and the overall score is total actual score / total weight
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from threading import Lock
from typing import Dict, List

import numpy as np
import pandas as pd
from openai import OpenAI
from retry import retry
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)

_result_lock = Lock()

SCORING_BATCH_SYSTEM_PROMPT = """You are an expert evaluator. Your task is to evaluate an assistant's response against a set of rubric criteria. You will receive a conversation and a list of rubric items. For each rubric item, you must judge whether the assistant's last response meets that criterion.

# Conversation
{conversation}

# Rubric Items
Below is a numbered list of rubric criteria. For each criterion, you must independently evaluate whether the assistant's last response meets it.

{rubric_items}

# Instructions
Return a JSON array containing one object for each rubric item (in the same order as listed above). Each object must have the following fields:
- "index": the integer index of the rubric item (0-based, matching the order above)
- "explanation": a concise string explaining why the response does or does not meet this specific criterion
- "criteria_met": a boolean (true if the response fully satisfies the criterion, false otherwise)

Important evaluation guidelines:
1. If a rubric item has multiple sentences or sub-criteria, ALL of them must be satisfied for "criteria_met" to be true. If any part is not met, return false.
2. Exception: If a criterion uses "such as", "for example", or "including", the response does NOT need to include all listed examples. For example, if the criterion says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response mentions cramps instead, that would still meet the criteria.
3. For criteria that describe undesirable behaviors (e.g., "fails to give accurate information", "Is overly verbose"), a good response that avoids the undesirable behavior should be judged as "criteria_met: false" — because the response does NOT exhibit the undesirable trait described in the criterion. Only responses that actually exhibit the undesirable behavior should be marked as true.
4. Evaluate each rubric item independently. The evaluation of one item should not influence another.
5. Be objective and evidence-based. Your explanation should reference specific content from both the assistant's response and the rubric criterion.

# Output Format
Return ONLY a JSON array in the following format (no extra text, no markdown code fences):
[
  {{
    "index": 0,
    "explanation": "explanation for rubric item 0",
    "criteria_met": true
  }},
  {{
    "index": 1,
    "explanation": "explanation for rubric item 1",
    "criteria_met": false
  }}
]
"""


def clean_llm_output(text: str) -> str:
    """Clean the LLM output, removing the <think>...</think> internal processing markers and their content."""
    if not isinstance(text, str):
        return ""

    # Find the position of the </think> tag
    think_end_tag = "</think>"
    tag_index = text.find(think_end_tag)

    if tag_index != -1:
        # Extract the content after the tag and strip surrounding whitespace
        cleaned_text = text[tag_index + len(think_end_tag):].strip()
        return cleaned_text
    else:
        # If the tag is not found, return the original text (stripped of surrounding whitespace)
        return text.strip()


def extract_prompt_text(prompt_field) -> str:
    """Extract the question text from the prompt field."""
    if isinstance(prompt_field, np.ndarray) and len(prompt_field) > 0:
        first_msg = prompt_field[0]
        if isinstance(first_msg, dict) and "content" in first_msg:
            return first_msg["content"]
    return str(prompt_field)


def _fix_invalid_json_escapes(text: str) -> str:
    """Fix invalid escape sequences in JSON, mainly handling LaTeX math symbols in LLM output.

    Valid JSON escapes: \\\\, \\\", \\/, \\b, \\f, \\n, \\r, \\t, \\uXXXX
    Examples of invalid LaTeX escapes: \\sum, \\alpha, \\sigma, \\infty, \\int, \\lambda, etc.
    These characters cause json.loads to throw an Invalid \\escape error.
    """
    # Valid JSON escape characters (the first character after a backslash)
    VALID_JSON_ESCAPE_CHARS = {'"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u'}

    def _replace_escape(match: re.Match) -> str:
        next_char = match.group(1)
        if next_char in VALID_JSON_ESCAPE_CHARS:
            if next_char == 'u':
                # \\uXXXX requires checking that it is followed by 4 hex digits
                after_match = match.string[match.end():match.end() + 4]
                if len(after_match) == 4 and all(
                    c in '0123456789abcdefABCDEF' for c in after_match
                ):
                    return match.group(0)  # valid \\uXXXX
            # Valid escape, or \\u not satisfying the hex condition: return as is
            return match.group(0)
        else:
            # Invalid escape (e.g. \\sum, \\alpha, etc.): add one more backslash to escape
            return '\\\\' + next_char

    return re.sub(r'\\(.)', _replace_escape, text)


def parse_batch_judge_json(text: str) -> List[dict]:
    """Parse the JSON array of batch judgment results from the LLM output (compatible with markdown code blocks and raw text)."""
    if text is None or not text.strip():
        raise ValueError("LLM returned empty content")

    text = text.strip()
    # Try to extract the content within ```json ... ```
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end != -1:
            text = text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end != -1:
            text = text[start:end].strip()

    # If it still does not start with [, try to find the first JSON array
    if not text.startswith("["):
        bracket_start = text.find("[")
        if bracket_start != -1:
            bracket_end = text.rfind("]")
            if bracket_end > bracket_start:
                text = text[bracket_start:bracket_end + 1].strip()

    # First try a normal parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Retry after fixing invalid escape sequences (such as LaTeX math symbols)
    try:
        fixed_text = _fix_invalid_json_escapes(text)
        return json.loads(fixed_text)
    except json.JSONDecodeError as e:
        logger.warning(
            "JSON parsing failed"
        )
        raise e


@retry(tries=5, delay=1, backoff=2)
def judge_all_rubrics_for_item(
    client: OpenAI,
    conversation: str,
    rubric_items: List[dict],
    model: str,
) -> List[dict]:
    """Judge all rubrics under one item in a single batch (auto-retry on failure, up to 5 times).

    rubric_items format: [{"index": 0, "criterion": "...", "points": 10}, ...]
    Return format: [{"index": 0, "explanation": "...", "criteria_met": true}, ...]
    """
    # Build the numbered rubric list text
    rubric_lines = []
    for item in rubric_items:
        rubric_lines.append(f"[{item['index']}] {item['criterion']}")
    rubric_text = "\n".join(rubric_lines)

    prompt = SCORING_BATCH_SYSTEM_PROMPT.format(
        conversation=conversation,
        rubric_items=rubric_text,
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        # max_tokens=2048,
    )
    result_text = clean_llm_output(response.choices[0].message.content)
    return parse_batch_judge_json(result_text)


def build_conversation(prompt_text: str, answer_text: str) -> str:
    """Build a conversation string in the format 'user: ... assistant: ...'."""
    return f"user: {prompt_text}\nassistant: {answer_text}"


def worker_score_chunk(
    api_base: str,
    api_key: str,
    model: str,
    chunk: List[dict],
    worker_id: int,
    result_dict: Dict,
    pbar: tqdm,
) -> int:
    """
    Thread worker function: processes scoring tasks item by item.

    chunk element format:
    {
        "item_idx": "idx_0",
        "prompt": "...",
        "answer": "...",
        "rubric_items": [
            {"index": 0, "criterion": "...", "points": 10},
            ...
        ]
    }
    """
    client = OpenAI(base_url=api_base, api_key=api_key)
    success_count = 0

    for task in chunk:
        try:
            conversation = build_conversation(task["prompt"], task["answer"])
            judge_results = judge_all_rubrics_for_item(
                client, conversation, task["rubric_items"], model
            )
            # Build the list of rubric judgment results
            rubric_results = []
            for orig in task["rubric_items"]:
                # Find the judgment with the corresponding index in the batch result
                matched = None
                for jr in judge_results:
                    if jr.get("index") == orig["index"]:
                        matched = jr
                        break
                if matched is None:
                    raise ValueError(
                        f"batch judgment result is missing the judgment for index={orig['index']}"
                    )
                rubric_results.append(
                    {
                        "criterion": orig["criterion"],
                        "points": orig["points"],
                        "judge": {
                            "explanation": matched["explanation"],
                            "criteria_met": matched["criteria_met"],
                        },
                    }
                )

            with _result_lock:
                result_dict[task["item_idx"]] = {
                    "prompt": task["prompt"],
                    "answer": task["answer"],
                    "rubrics": rubric_results,
                }
            success_count += 1
            pbar.update(1)
        except Exception as e:
            logger.error(
                f"[Worker-{worker_id}] failed to judge item {task['item_idx']} "
                f"(containing {len(task['rubric_items'])} rubrics) after 5 retries: {e}"
            )
            pbar.update(1)

    return success_count


def score_responses(
    data_path: str,
    response_path: str,
    output_path: str,
    api_base: str,
    api_key: str,
    model: str,
    num_workers: int = 8,
):
    """
    Load the evaluation set and answer results, then batch-score all rubrics under each item
    using multithreading (one call judges all rubrics of a single item).

    Args:
        data_path: path to the evaluation set parquet file
        response_path: path to the answers JSON file (output of generate_responses.py)
        output_path: path to the output scoring JSON file
        api_base: LLM API address
        api_key: API key
        model: judge model name
        num_workers: number of concurrent threads
    """
    # 1. Load the data
    logger.info(f"Loading evaluation set: {data_path}")
    df = pd.read_parquet(data_path)

    logger.info(f"Loading answer results: {response_path}")
    with open(response_path, "r", encoding="utf-8") as f:
        responses = json.load(f)

    # 1.5 Check whether an existing scoring result exists (resume support)
    existing_result: Dict = {}
    if os.path.exists(output_path):
        logger.info(f"Existing scoring result found: {output_path}, resuming from it")
        with open(output_path, "r", encoding="utf-8") as f:
            existing_result = json.load(f)
        logger.info(f"Loaded {len(existing_result)} already-scored items")

    # 2. Build the scoring task list (skip already-completed items)
    tasks = []
    total_rubric_count = 0
    for idx, row in df.iterrows():
        item_key = f"idx_{idx}"
        if item_key not in responses:
            logger.warning(f"Skipping {item_key}: not found in answer results")
            continue

        prompt_text = extract_prompt_text(row["prompt"])
        answer_text = responses[item_key]["answer"]
        rubrics = row["Rubrics"]

        # Extract all rubrics under this item
        rubric_items = []
        for r_idx, rubric in enumerate(rubrics):
            if isinstance(rubric, np.void):
                rubric = {k: rubric[k] for k in rubric.dtype.names}
            criterion = rubric.get("criterion", str(rubric))
            points = rubric.get("points", 0)
            rubric_items.append(
                {
                    "index": r_idx,
                    "criterion": criterion,
                    "points": points,
                }
            )

        if not rubric_items:
            continue

        # Skip if this item already has a scoring result
        if item_key in existing_result:
            continue

        tasks.append(
            {
                "item_idx": item_key,
                "prompt": prompt_text,
                "answer": answer_text,
                "rubric_items": rubric_items,
            }
        )
        total_rubric_count += len(rubric_items)

    total_tasks = len(tasks)
    if total_tasks == 0:
        logger.info("All items are already scored, nothing to do")
        shared_result = existing_result
    else:
        logger.info(
            f"Built {total_tasks} pending scoring tasks (skipped {len(existing_result)} already done), "
            f"totaling {total_rubric_count} rubrics"
        )

        # 3. Evenly distribute across threads
        chunk_size = max(1, (total_tasks + num_workers - 1) // num_workers)
        chunks = [tasks[i : i + chunk_size] for i in range(0, total_tasks, chunk_size)]
        logger.info(f"Starting {len(chunks)} threads, ~{chunk_size} items per thread")

        # 4. Multithreaded parallel scoring (initialized with existing results)
        shared_result = deepcopy(existing_result)
        t_start = time.time()
        pbar = tqdm(total=total_tasks, desc="Judging", unit="item")

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    worker_score_chunk,
                    api_base=api_base,
                    api_key=api_key,
                    model=model,
                    chunk=chunk,
                    worker_id=wid,
                    result_dict=shared_result,
                    pbar=pbar,
                ): wid
                for wid, chunk in enumerate(chunks)
            }

            for future in as_completed(futures):
                wid = futures[future]
                try:
                    count = future.result()
                    logger.info(f"[Worker-{wid}] done, successfully processed {count} items")
                except Exception as e:
                    logger.error(f"[Worker-{wid}] thread execution error: {e}")

        pbar.close()
        elapsed = time.time() - t_start
        new_judged = sum(
            1 for k in shared_result
            if k not in existing_result
        )
        total_rubric_judged = sum(
            len(item["rubrics"]) for item in shared_result.values()
        )
        logger.info(
            f"This round done, {new_judged} new items added, "
            f"{total_rubric_judged} rubrics judged in total, "
            f"took {elapsed:.1f}s "
            f"({new_judged / elapsed:.1f} item/s)"
        )

    # 4.5 Compute each item's normalized score (actual score / theoretical maximum) and overall statistics
    normalized_scores = []
    raw_scores_list = []
    total_weights_list = []
    for item_key, item_data in shared_result.items():
        raw_score = 0
        total_weight = 0
        for rubric in item_data["rubrics"]:
            total_weight += rubric["points"]
            if rubric["judge"].get("criteria_met", False):
                raw_score += rubric["points"]
        normalized_score = raw_score / total_weight if total_weight > 0 else 0.0
        normalized_scores.append(normalized_score)
        raw_scores_list.append(raw_score)
        total_weights_list.append(total_weight)
        # Write the normalized score into the result
        item_data["score"] = normalized_score
        item_data["raw_score"] = raw_score
        item_data["total_weight"] = total_weight

    if normalized_scores:
        total_raw = sum(raw_scores_list)
        total_weight_all = sum(total_weights_list)
        overall_score = total_raw / total_weight_all if total_weight_all > 0 else 0.0
        max_norm = max(normalized_scores)
        min_norm = min(normalized_scores)
        logger.info("=" * 50)
        logger.info(f"Score statistics (over {len(normalized_scores)} test samples):")
        logger.info(f"  Overall score = total raw score / total weight")
        logger.info(f"  Total raw score: {total_raw:.2f}")
        logger.info(f"  Total weight: {total_weight_all:.2f}")
        logger.info(f"  Overall score: {overall_score:.4f} ({overall_score * 100:.1f}%)")
        logger.info(f"  Max normalized score (per item): {max_norm:.4f} ({max_norm * 100:.1f}%)")
        logger.info(f"  Min normalized score (per item): {min_norm:.4f} ({min_norm * 100:.1f}%)")
        logger.info("=" * 50)

    # 5. Save the results
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(shared_result, f, ensure_ascii=False, indent=2)

    logger.info(f"Scoring results saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score generated answers against Rubric criteria"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="dataset/eval_set_Science_400.parquet",
        help="Path to the evaluation set parquet file (default: dataset/eval_set_Science_400.parquet)",
    )
    parser.add_argument(
        "--response",
        type=str,
        default="output/responses.json",
        help="Path to the answers JSON file (default: output/responses.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/scores.json",
        help="Path to the output scoring JSON file (default: output/scores.json)",
    )
    parser.add_argument(
        "--api_base",
        type=str,
        required=True,
        help="LLM judge API address, e.g. http://127.0.0.1:8000/v1",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="EMPTY",
        help="API key (default EMPTY)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Judge model name",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of concurrent threads (default 8)",
    )
    args = parser.parse_args()

    score_responses(
        data_path=args.data,
        response_path=args.response,
        output_path=args.output,
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        num_workers=args.num_workers,
    )
