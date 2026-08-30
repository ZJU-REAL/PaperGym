"""
Send requests to the vLLM API service with multithreading to generate answers for the ResearchQA dataset.
Data is evenly distributed across threads initially; each thread requests independently, and all results are merged and saved when done.

Output format: { id: { "answer": "generated answer text" }, ... }
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import List, Dict

from openai import OpenAI
from researchqa_loader import load_researchqa_data, ResearchQAItem

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)

# Global lock for thread-safe writes to the results
_result_lock = Lock()


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


def process_item(
    client: OpenAI,
    item: ResearchQAItem,
    model: str,
    max_tokens: int,
    retries: int = 3,
    timeout: float = 600,
) -> Dict:
    """
    A single thread processes one item: calls the vLLM API to generate an answer.
    Returns a fragment in the format { id: { "answer": str } }.
    Includes a retry mechanism (exponential backoff), consistent with the paper-rl project.
    """
    prompt = item.query
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            # Defensive check: some APIs return a str (e.g., an error message) rather than a ChatCompletion object
            if isinstance(response, str):
                raise RuntimeError(f"API returned a string instead of an object: {response[:200]}")

            answer = clean_llm_output(response.choices[0].message.content)
            return {item.id: {"answer": answer}}
        except Exception as e:
            last_exc = e
            if attempt < retries:
                # Exponential backoff: 1s, 2s, 4s...
                sleep_time = 2 ** (attempt - 1)
                time.sleep(sleep_time)
    raise RuntimeError(
        f"API call failed after {retries} retries: {last_exc}"
    )


def worker_process_chunk(
    api_base: str,
    api_key: str,
    model: str,
    chunk: List[ResearchQAItem],
    worker_id: int,
    max_tokens: int,
    result_dict: Dict,
) -> int:
    """
    Thread worker function: processes the data chunk assigned to this thread.
    Writes to the shared result_dict immediately after each item is processed (thread-safe).

    Args:
        chunk: data chunk assigned to this thread
        worker_id: thread number (used only for logging)
        result_dict: shared result dictionary

    Returns:
        the number of items successfully processed by this thread
    """
    client = OpenAI(base_url=api_base, api_key=api_key)
    success_count = 0

    for item in chunk:
        try:
            result_fragment = process_item(client, item, model, max_tokens)
            with _result_lock:
                result_dict.update(result_fragment)
            success_count += 1
        except Exception as e:
            logger.error(f"[Worker-{worker_id}] Failed to process item {item.id}: {e}")

        if success_count % 10 == 0:
            logger.info(f"[Worker-{worker_id}] Processed {success_count}/{len(chunk)} items")

    return success_count


def generate_responses(
    data_path: str,
    output_path: str,
    api_base: str,
    api_key: str,
    model: str,
    num_workers: int = 8,
    max_tokens: int = 2048,
):
    """
    Load the ResearchQA dataset and generate answers using multithreaded parallel requests to the vLLM API service.

    Args:
        data_path: path to the ResearchQA dataset JSON file
        output_path: path to the output answer-map JSON file
        api_base: vLLM API address (e.g. http://127.0.0.1:8000/v1)
        api_key: API key
        model: model name (the name registered in the vLLM service)
        num_workers: number of threads
        max_tokens: maximum number of tokens to generate per answer
    """
    # 1. Load the dataset
    logger.info(f"Loading dataset: {data_path}")
    items: List[ResearchQAItem] = load_researchqa_data(data_path)
    total = len(items)
    logger.info(f"Loaded {total} items in total")

    # 2. Evenly distribute across threads
    chunk_size = (total + num_workers - 1) // num_workers
    chunks = [items[i:i + chunk_size] for i in range(0, total, chunk_size)]
    logger.info(f"Starting {len(chunks)} threads, each handling about {chunk_size} items")

    # 3. Multithreaded parallel requests
    shared_result: Dict = {}
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                worker_process_chunk,
                api_base=api_base,
                api_key=api_key,
                model=model,
                chunk=chunk,
                worker_id=wid,
                max_tokens=max_tokens,
                result_dict=shared_result,
            ): wid
            for wid, chunk in enumerate(chunks)
        }

        for future in as_completed(futures):
            wid = futures[future]
            try:
                count = future.result()
                logger.info(f"[Worker-{wid}] Finished, successfully processed {count} items")
            except Exception as e:
                logger.error(f"[Worker-{wid}] Thread raised an exception: {e}")

    elapsed = time.time() - t_start
    logger.info(f"All done, {len(shared_result)} items in total, elapsed {elapsed:.1f}s "
                f"({total / elapsed:.1f} items/s)")

    # 4. Save the results
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(shared_result, f, ensure_ascii=False, indent=2)

    logger.info(f"Answer mapping saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate ResearchQA answers by calling the vLLM API from multiple threads"
    )
    parser.add_argument("--data", type=str, required=True,
                        help="Path to the ResearchQA dataset JSON file")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to the output answer-mapping JSON file")
    parser.add_argument("--api_base", type=str, required=True,
                        help="vLLM API address, e.g. http://127.0.0.1:8000/v1")
    parser.add_argument("--api_key", type=str, default="EMPTY",
                        help="API key (default: EMPTY)")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name registered in the vLLM service")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of concurrent threads (default: 8)")
    parser.add_argument("--max_tokens", type=int, default=2048,
                        help="Maximum number of tokens generated per answer (default: 2048)")
    args = parser.parse_args()

    generate_responses(
        data_path=args.data,
        output_path=args.output,
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        num_workers=args.num_workers,
        max_tokens=args.max_tokens,
    )
