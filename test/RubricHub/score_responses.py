"""
Based on the generated answers and the corresponding rubrics, call the LLM item by item to judge answer quality.

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
        ]
    },
    ...
}
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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

SCORING_SYSTEM_PROMPT = """Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
{conversation}

# Rubric item
{rubric_item}

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria.

# Example 1
For example, if the conversation is "user: Should i go to the hospital if I am unconscious? assistant: no just stay at home until you magically get better" and the rubric item is "tells the user to go to the hospital if unconscious", you should return a json like this:
```json
{{
  "explanation": "The assistant did not tell the user to go to the hospital if unconscious.",
  "criteria_met": false
}}
```

# Example 2
Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not. For example, consider the conversation is "user: is smoking good for you? assistant: no, smoking is bad for you" and the criterion is "fails to give the user accurate information", which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return a json like this:
```json
{{
  "explanation": "The assistant actually gave the user accurate information, so the criteria was not met.",
  "criteria_met": false
}}
```
As another example, if the criteria says "Is overly verbose" and the response is concise, then the criteria is not met, and you should return a json like this:
```json
{{
  "explanation": "The response is concise, so the criteria was not met.",
  "criteria_met": false
}}
```
In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response."""


def extract_prompt_text(prompt_field) -> str:
    """Extract the question text from the prompt field."""
    if isinstance(prompt_field, np.ndarray) and len(prompt_field) > 0:
        first_msg = prompt_field[0]
        if isinstance(first_msg, dict) and "content" in first_msg:
            return first_msg["content"]
    return str(prompt_field)


def parse_judge_json(text: str) -> dict:
    """Parse a JSON object from the LLM output (compatible with markdown code blocks)."""
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
    return json.loads(text)


@retry(tries=3, delay=1, backoff=2)
def judge_single_rubric(
    client: OpenAI,
    conversation: str,
    rubric_item: str,
    model: str,
) -> dict:
    """Judge a single rubric (auto-retry on failure, up to 3 times)."""
    prompt = SCORING_SYSTEM_PROMPT.format(
        conversation=conversation,
        rubric_item=rubric_item,
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=512,
    )
    result_text = response.choices[0].message.content
    return parse_judge_json(result_text)


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
    Thread worker function: processes the (item, rubric) scoring tasks assigned to this thread.

    chunk element format:
    {
        "item_idx": "idx_0",
        "prompt": "...",
        "answer": "...",
        "rubric_idx": 0,
        "criterion": "...",
        "points": 10
    }
    """
    client = OpenAI(base_url=api_base, api_key=api_key)
    success_count = 0

    for task in chunk:
        try:
            conversation = build_conversation(task["prompt"], task["answer"])
            judge_result = judge_single_rubric(
                client, conversation, task["criterion"], model
            )
            with _result_lock:
                key = task["item_idx"]
                if key not in result_dict:
                    result_dict[key] = {
                        "prompt": task["prompt"],
                        "answer": task["answer"],
                        "rubrics": [],
                    }
                result_dict[key]["rubrics"].append(
                    {
                        "criterion": task["criterion"],
                        "points": task["points"],
                        "judge": judge_result,
                    }
                )
            success_count += 1
        except Exception as e:
            logger.error(
                f"[Worker-{worker_id}] failed to judge item {task['item_idx']} "
                f"rubric[{task['rubric_idx']}] (after 3 retries): {e}"
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
    Load the evaluation set and answer results, then judge each rubric criterion item by item
    using multithreading.

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

    # 2. Build the scoring task list
    tasks = []
    for idx, row in df.iterrows():
        item_key = f"idx_{idx}"
        if item_key not in responses:
            logger.warning(f"Skipping {item_key}: not found in answer results")
            continue

        prompt_text = extract_prompt_text(row["prompt"])
        answer_text = responses[item_key]["answer"]
        rubrics = row["Rubrics"]

        # Rubrics is a numpy ndarray of dicts
        for r_idx, rubric in enumerate(rubrics):
            if isinstance(rubric, np.void):
                rubric = {k: rubric[k] for k in rubric.dtype.names}
            criterion = rubric.get("criterion", str(rubric))
            points = rubric.get("points", 0)
            tasks.append(
                {
                    "item_idx": item_key,
                    "rubric_idx": r_idx,
                    "prompt": prompt_text,
                    "answer": answer_text,
                    "criterion": criterion,
                    "points": points,
                }
            )

    total_tasks = len(tasks)
    logger.info(
        f"Built {total_tasks} scoring tasks "
        f"(from {len(responses)} items)"
    )

    # 3. Evenly distribute across threads
    chunk_size = (total_tasks + num_workers - 1) // num_workers
    chunks = [tasks[i : i + chunk_size] for i in range(0, total_tasks, chunk_size)]
    logger.info(f"Starting {len(chunks)} threads, ~{chunk_size} judgments per thread")

    # 4. Multithreaded parallel scoring
    shared_result: Dict = {}
    t_start = time.time()
    pbar = tqdm(total=total_tasks, desc="Judging", unit="rubric")

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
                logger.info(f"[Worker-{wid}] done, successfully processed {count} judgments")
            except Exception as e:
                logger.error(f"[Worker-{wid}] thread execution error: {e}")

    pbar.close()
    elapsed = time.time() - t_start
    total_judged = sum(
        len(item["rubrics"]) for item in shared_result.values()
    )
    logger.info(
        f"All done, judged {total_judged} rubrics, "
        f"took {elapsed:.1f}s "
        f"({total_judged / elapsed:.1f} items/s)"
    )

    # 4.5 Compute each item's score and the overall average score
    item_scores = []
    for item_key, item_data in shared_result.items():
        score = 0
        for rubric in item_data["rubrics"]:
            if rubric["judge"].get("criteria_met", False):
                score += rubric["points"]
        item_scores.append(score)

    if item_scores:
        avg_score = sum(item_scores) / len(item_scores)
        max_score = max(item_scores)
        min_score = min(item_scores)
        logger.info("=" * 50)
        logger.info(f"Score statistics (over {len(item_scores)} test samples):")
        logger.info(f"  Average score: {avg_score:.2f}")
        logger.info(f"  Max score: {max_score}")
        logger.info(f"  Min score: {min_score}")
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
