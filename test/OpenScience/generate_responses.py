"""
Answer generation module: reads questions from the evaluation dataset, calls the LLM
(can be deployed locally via vLLM) to generate answers, and stores all results as a single
JSON file. Supports resume support and automatic retry.
"""

import os
import json
import copy
import threading
import concurrent.futures
import argparse
from openai import OpenAI
from retry import retry


# =============================================================================
# LLM output cleaning
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

    @retry(tries=3, delay=1, backoff=2)
    def generate_response(self, prompt: str, max_tokens: int = 10240,
                          temperature: float = 0.7, n: int = 1):
        """Generate an answer (with automatic retry, up to 3 times, exponential backoff)"""
        chat_completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            n=n
        )
        if n == 1:
            return clean_llm_output(chat_completion.choices[0].message.content)
        else:
            return [
                clean_llm_output(choice.message.content)
                for choice in chat_completion.choices
            ]


# =============================================================================
# Cache utility functions (single-file storage)
# =============================================================================
def load_existing_answers(output_path: str) -> dict:
    """Load an existing answer cache file; return an empty dict if it does not exist."""
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"  [Cache] Loaded existing results, {len(data)} items already processed", flush=True)
        return data
    return {}


def save_answers(output_path: str, results: dict) -> None:
    """Write all answers into a single JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_path = output_path + ".tmp"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, output_path)
    print(f"  [Save] Saved {len(results)} results to {output_path}", flush=True)


# =============================================================================
# Answer generation
# =============================================================================
def _extract_prompt_text(item: dict) -> str:
    """Extract the prompt text from a data item.

    Supports multiple data formats:
      - Field names: problem / question / prompt
      - Field types: str / list[dict] / numpy.ndarray[dict]
    """
    # Check the prompt field (may be a numpy array or list)
    prompt = item.get('prompt')
    if prompt is not None:
        # numpy.ndarray or list format: [{"role": "user", "content": "..."}]
        if hasattr(prompt, '__iter__') and not isinstance(prompt, (str, bytes)):
            prompt_list = prompt.tolist() if hasattr(prompt, 'tolist') else prompt
            if isinstance(prompt_list, list) and len(prompt_list) > 0:
                msg = prompt_list[0]
                if isinstance(msg, dict):
                    content = msg.get('content', '')
                    if content:
                        return content
        # It is directly a string
        if isinstance(prompt, str):
            return prompt

    # Fallback: check the problem / question fields
    question = item.get('problem', '') or item.get('question', '')
    if isinstance(question, str):
        return question
    if isinstance(question, list) and len(question) > 0:
        return question[0].get('content', '') if isinstance(question[0], dict) else str(question[0])
    return str(question) if question else ''


def _generate_single_item(client: LLMClient, item: dict, index: int,
                          shared_results: dict, lock: threading.Lock) -> None:
    """Generate an answer for a single item, writing to the shared dictionary thread-safely."""
    prompt_text = _extract_prompt_text(item)
    if not prompt_text:
        print(f"  [Warning] Item {index}: empty prompt, skipping", flush=True)
        return

    # Check whether it has already been processed by another thread
    item_key = f"idx_{index}"
    with lock:
        if item_key in shared_results:
            print(f"  [Cache HIT] Item {index}: already in results, skipping", flush=True)
            return

    print(f"  [Generate] Item {index}: generating answer...", flush=True)
    try:
        answer = client.generate_response(prompt_text, max_tokens=2048, temperature=1.0)
        if not answer:
            print(f"  [Warning] Item {index}: generated answer is empty", flush=True)
            return

        result = {'index': index, 'question': prompt_text, 'answer': answer}
        with lock:
            shared_results[item_key] = result
        print(f"  [Done] Item {index}: answer length={len(answer)}", flush=True)

    except Exception as e:
        print(f"  [Error] Item {index}: generation failed after retries: {str(e)}", flush=True)


# =============================================================================
# Main function
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Generate answers via LLM (single-file cache)")
    parser.add_argument("--parquet_path", type=str, required=True,
                        help="Path to the input parquet file")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to the output answers JSON file")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of parallel workers")
    parser.add_argument("--max_items", type=int, default=400,
                        help="Max items to process. 0 or negative for all.")

    parser.add_argument("--api_key", type=str, default="sk-xxx",
                        help="API Key for Generator LLM")
    parser.add_argument("--base_url", type=str, default="http://xxx/",
                        help="Base URL for Generator LLM")
    parser.add_argument("--model", type=str, default="qwen3_8b",
                        help="Model name for Generator LLM")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80, flush=True)
    print("Generate Responses (Single-File Cache)", flush=True)
    print("=" * 80, flush=True)
    print(f"Parquet:  {args.parquet_path}", flush=True)
    print(f"Output:   {args.output_path}", flush=True)
    print(f"Model:    {args.model} @ {args.base_url}", flush=True)
    print(f"Workers:  {args.num_workers}, Max items: {args.max_items}", flush=True)

    # Read the data
    print(f"\n[Loading] Reading data from {args.parquet_path}...", flush=True)
    try:
        import pandas as pd
        df = pd.read_parquet(args.parquet_path)
        print(f"[Loading] Loaded {len(df)} items", flush=True)
    except Exception as e:
        print(f"[Error] Failed to load data: {str(e)}", flush=True)
        return

    items = df.to_dict('records')
    if args.max_items and args.max_items > 0:
        items = items[:args.max_items]
    total_items = len(items)
    print(f"[Loading] Processing {total_items} items", flush=True)

    # Load the existing cache (resume support)
    existing = load_existing_answers(args.output_path)

    # Initialize the shared results dictionary (deep-copy existing results to avoid overwriting processed data)
    shared_results = copy.deepcopy(existing)
    lock = threading.Lock()

    # Extract the data that has not yet been processed
    pending_indices = [
        idx for idx in range(total_items)
        if f"idx_{idx}" not in shared_results
    ]
    print(f"[Cache] Existing: {len(existing)} items, still to process: {len(pending_indices)} items", flush=True)

    if not pending_indices:
        print("[Cache] All items already generated, nothing to do", flush=True)
        return

    # Initialize the LLM client
    client = LLMClient(api_key=args.api_key, base_url=args.base_url, model=args.model)

    # Only process unfinished items
    print(f"\n{'=' * 80}", flush=True)
    print(f"Generating answers (parallel, {args.num_workers} workers)", flush=True)
    print(f"Processing {len(pending_indices)} remaining items", flush=True)
    print(f"{'=' * 80}", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                _generate_single_item, client, items[idx], idx,
                shared_results, lock
            ): idx
            for idx in pending_indices
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                idx = futures[future]
                print(f"[Error] Item {idx} generation error: {str(e)}", flush=True)

    # Save the results
    save_answers(args.output_path, shared_results)

    print(f"\n{'=' * 80}", flush=True)
    print(f"Answer generation completed. Output: {args.output_path}", flush=True)
    print(f"Total: {total_items}, Generated/Skipped: {len(shared_results)}", flush=True)
    print(f"{'=' * 80}", flush=True)


if __name__ == "__main__":
    main()
