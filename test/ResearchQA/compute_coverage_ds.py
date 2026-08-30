import argparse
import json
from tqdm import tqdm
from researchqa_loader import load_researchqa_data
from typing import List
import openai
import logging
from retry import retry
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from openai import (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom exception for batch processing failures
class BatchProcessingError(Exception):
    pass

# Global OpenAI client (thread-safe)
_client = None
_client_lock = threading.Lock()

def _init_client(api_base: str, api_key: str):
    """Initialize the global OpenAI client with custom API base and key."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = openai.OpenAI(api_key=api_key, base_url=api_base)

def call_gpt(prompt: str, model: str) -> str:
    """Call LLM API using the OpenAI-compatible interface."""
    response = _client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        timeout=120
    )
    # print("Response is:", response, flush=True)
    return response.choices[0].message.content.strip()

def build_prompt(response: str, questions: List[str]) -> str:
    prompt = (
        "Please judge the following questions based on the response below.\n"
        "For each question, select one of the following ratings to indicate the extent to which the response addresses the question:\n"
        "Not at all, Barely, Moderately, Mostly, Completely\n\n"
        "Definitions:\n"
        "- Not at all: *totally uninferable*\n"
        "- Barely: *unmentioned but inferrable*\n"
        "- Moderately: *mentioned but misses important details*\n"
        "- Mostly: *mentioned but misses some details*\n"
        "- Completely: *mentioned with sufficient details*\n\n"
        "Only output one of the five phrases for each question, separated by newlines, and nothing else.\n\n"
        f"Response: {response}\n"
        f"Questions:\n" + '\n'.join(questions) + "\n\nOutput:"
    )
    return prompt

def convert_text_to_score(text):
    text_to_score = {
        'Not at all': 1,
        'Barely': 2,
        'Moderately': 3,
        'Mostly': 4,
        'Completely': 5
    }
    score = text_to_score[text]
    return score

def normalize_5scale(x):
    return (x - 1) / 4

def compute_coverage_score(rubric_judges):
    """
    rubric_judges: list of dicts, each with a 'score' key (text label)
    Returns: average normalized coverage score (float) in [0, 1]
    """
    numeric_scores = [convert_text_to_score(j['score']) for j in rubric_judges]
    normalized_scores = [normalize_5scale(x) for x in numeric_scores]
    return sum(normalized_scores) / len(normalized_scores)

@retry(tries=3, delay=1, backoff=2, exceptions=BatchProcessingError)
def process_batch_with_retry(prompt: str, batch: List[str], model: str, item_id: str) -> List[str]:
    """
    Process a batch of rubric items with retry logic.
    
    Args:
        prompt: The prompt to send to GPT
        batch: List of rubric items in the batch
        model: The model name for logging purposes
        item_id: The item ID for logging purposes
    
    Returns:
        List of scores for the batch
    """
    try:
        gpt_output = call_gpt(prompt, model=model)
    except (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError) as e:
        logger.warning(f"API call failed for item {item_id}: {type(e).__name__}: {e}. Retrying...")
        raise BatchProcessingError(f"API error: {type(e).__name__}: {e}")
    scores = gpt_output.split('\n')
    
    # Filter out empty lines
    scores = [score.strip() for score in scores if score.strip()]
    
    if len(scores) == len(batch):
        return scores
    else:
        logger.warning(f"Expected {len(batch)} results for item {item_id}, got {len(scores)}. Retrying...")
        logger.warning(f"Questions: {batch}")
        raise BatchProcessingError(f"Expected {len(batch)} results, got {len(scores)}")

def process_item(item, response_map, batch_size, model):
    """Process a single item's rubric coverage. Returns (item_id, rubric_judges) or None on failure."""
    item_id = item.id
    if item_id not in response_map:
        logger.warning(f"Skipping item {item_id} as it has no corresponding response")
        return None

    answer = response_map[item_id]['answer']
    rubric_items = [r.rubric_item for r in item.rubric]
    rubric_judges = []

    for i in range(0, len(rubric_items), batch_size):
        batch = rubric_items[i:i + batch_size]
        prompt = build_prompt(answer, batch)
        try:
            scores = process_batch_with_retry(prompt, batch, model, item_id)
            for rubric, score in zip(batch, scores):
                rubric_judges.append({
                    'rubric': rubric,
                    'score': score,
                })
        except Exception as e:
            logger.error(f"Failed to process batch for item {item_id} after all retries. Error: {e}")
            return None

    return item_id, rubric_judges


def compute_coverage(
    data_path: str,
    response_map_path: str,
    output_path: str,
    batch_size: int = 8,
    model: str = "gpt-4.1-mini",
    api_base: str = None,
    api_key: str = None,
    num_workers: int = 1,
):
    """Compute rubric coverage with parallel processing for items."""
    # Initialize the global OpenAI client once (thread-safe)
    _init_client(api_base, api_key)

    # Load local dataset
    items = load_researchqa_data(data_path)
    # Load responses
    with open(response_map_path, 'r', encoding='utf-8') as f:
        response_map = json.load(f)

    results = {}
    coverages = []
    results_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_item, item, response_map, batch_size, model): item
            for item in items
        }

        with tqdm(total=len(items), desc="Computing rubric coverage") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        item_id, rubric_judges = result
                        with results_lock:
                            results[item_id] = rubric_judges
                            item_coverage = compute_coverage_score(rubric_judges)
                            coverages.append(item_coverage)
                            logger.info(f"Coverage for {item_id}: {item_coverage:.3f}")
                except Exception as e:
                    item = futures[future]
                    logger.error(f"Error processing item {item.id}: {e}")
                finally:
                    pbar.update(1)

    # Save results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved rubric coverage results to {output_path}")

    if coverages:
        avg_coverage = sum(coverages) / len(coverages)
        logger.info(f"Average rubric coverage: {avg_coverage:.3f}")
    else:
        logger.info("No coverage scores computed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute rubric coverage using LLM API")
    parser.add_argument('--data', type=str, required=True,
                        help='Path to local ResearchQA JSON file')
    parser.add_argument('--responses', type=str, required=True,
                        help='Path to response map JSON file (id -> response dict)')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save rubric coverage results')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for rubric items')
    parser.add_argument('--model', type=str, default='gpt-4.1-mini',
                        help='LLM model name')
    parser.add_argument('--api_base', type=str, default=None,
                        help='Custom API base URL (e.g. http://xxx)')
    parser.add_argument('--api_key', type=str, default=None,
                        help='API key for the custom API endpoint')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of parallel worker threads')
    args = parser.parse_args()
    compute_coverage(
        args.data, args.responses, args.output,
        args.batch_size, args.model,
        args.api_base, args.api_key, args.num_workers
    ) 