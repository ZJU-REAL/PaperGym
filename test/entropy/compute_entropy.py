"""
compute_entropy.py

Measure entropy during model response generation.

Get model-generated token-level top logprobs via OpenAI-compatible API (e.g. vLLM locally deployed API service)
Calculate entropy (uncertainty) at each token position, and aggregate output
statistics.

Usage example:
    # Local vLLM service
    python compute_entropy.py \\
        --model /path/to/model \\
        --base_url http://127.0.0.1:8019/v1 \\
        --parquet_path data/cs_new.parquet \\
        --top_logprobs 20 \\
        --max_items 10

    # Remote API
    python compute_entropy.py \\
        --model qwen3_8b \\
        --base_url http://xxx/ \\
        --api_key sk-xxx \\
        --parquet_path data/cs_new.parquet
"""

import os
import json
import argparse
import concurrent.futures
import numpy as np
from typing import Optional
from openai import OpenAI


# =============================================================================
# Entropy calculation
# =============================================================================

def softmax(logprobs_list: list) -> np.ndarray:
    """Normalize log probabilities to probability distribution via softmax."""
    logprobs = np.array(logprobs_list, dtype=np.float64)
    # Subtract the maximum value for numerical stability
    logprobs = logprobs - np.max(logprobs)
    probs = np.exp(logprobs)
    probs = probs / np.sum(probs)
    return probs


def compute_token_entropy(top_logprobs: dict) -> float:
    """Calculate entropy for a single token position.

    Use top-k logprobs to approximate the probability distribution at this position, then calculate entropy:
        H = -sum(p_i * log(p_i))

    Args:
        top_logprobs: dict {token_str: logprob}

    Returns:
        entropy (nats)
    """
    logprob_values = list(top_logprobs.values())
    if not logprob_values:
        return 0.0
    probs = softmax(logprob_values)
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    return float(entropy)


def compute_response_entropy(logprob_data: list) -> dict:
    """Calculate aggregated statistics for the entire response (do not save per-token data to save space)."""
    if not logprob_data:
        return {}

    token_entropies = []
    for info in logprob_data:
        top_logprobs = info.get('top_logprobs', {})
        if top_logprobs:
            entropy = compute_token_entropy(top_logprobs)
            token_entropies.append(entropy)

    if not token_entropies:
        return {}

    arr = np.array(token_entropies)

    return {
        'num_tokens': len(arr),
        'mean_entropy': float(np.mean(arr)),
        'std_entropy': float(np.std(arr)),
        'max_entropy': float(np.max(arr)),
        'min_entropy': float(np.min(arr)),
        'median_entropy': float(np.median(arr)),
    }


# =============================================================================
# LLM call (get logprobs)
# =============================================================================

def generate_with_logprobs(
    client: OpenAI,
    model: str,
    messages: list,
    max_tokens: int = 2048,
    temperature: float = 1.0,
    top_logprobs: int = 10,
) -> dict:
    """Call model to generate answer and get logprobs information."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        logprobs=True,
        top_logprobs=top_logprobs,
    )

    choice = response.choices[0]
    content = choice.message.content

    # Extract the logprob information
    logprob_data_choice = choice.logprobs
    logprob_list = []
    if logprob_data_choice and logprob_data_choice.content:
        for token_info in logprob_data_choice.content:
            top_k = {}
            if token_info.top_logprobs:
                for tl in token_info.top_logprobs:
                    top_k[tl.token] = tl.logprob
            logprob_list.append({
                'token': token_info.token,
                'logprob': token_info.logprob,
                'top_logprobs': top_k,
            })

    return {
        'content': content,
        'logprob_data': logprob_list,
    }


# =============================================================================
# Data loading and processing
# =============================================================================

def load_prompts(parquet_path: str, max_items: Optional[int] = None):
    """Load prompts from parquet file."""
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    items = df.to_dict('records')
    if max_items and max_items > 0:
        items = items[:max_items]
    return items


def extract_messages(item: dict) -> list:
    """Extract messages format from data item.

    Compatible with two formats:
        - prompt field is list[dict], e.g. [{"role": "user", "content": "..."}]
        - prompt field is str, wrap as user message
    """
    prompt = item.get('prompt')
    if prompt is not None:
        if isinstance(prompt, list):
            # Already in messages format, return directly
            return prompt
        if isinstance(prompt, str):
            # String format, wrap as a user message
            return [{"role": "user", "content": prompt}]
    # Fallback
    return [{"role": "user", "content": str(item)}]


# =============================================================================
# Main function
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute response entropy for a model given a parquet dataset'
    )
    parser.add_argument(
        '--parquet_path', type=str, required=True,
        help='Path to the input parquet file'
    )
    parser.add_argument(
        '--output_path', type=str, default=None,
        help='Path to save results JSON (optional)'
    )
    parser.add_argument(
        '--max_items', type=int, default=0,
        help='Max items to process (0 = all)'
    )
    parser.add_argument(
        '--top_logprobs', type=int, default=10,
        help='Number of top logprobs to request (default: 10)'
    )
    parser.add_argument(
        '--max_tokens', type=int, default=2048,
        help='Max tokens for generation'
    )
    parser.add_argument(
        '--temperature', type=float, default=1.0,
        help='Sampling temperature'
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='Number of parallel workers (default: 4)'
    )

    # Model configuration
    parser.add_argument(
        '--api_key', type=str, default='EMPTY',
        help='API Key for the LLM (default: EMPTY for local vLLM)'
    )
    parser.add_argument(
        '--base_url', type=str, required=True,
        help='Base URL for the LLM API (e.g., http://127.0.0.1:8019/v1)'
    )
    parser.add_argument(
        '--model', type=str, required=True,
        help='Model name or path'
    )
    return parser.parse_args()


def _process_one(args, client, idx, item):
    """Process single data item (for parallel calls)."""
    messages = extract_messages(item)
    try:
        result = generate_with_logprobs(
            client, args.model, messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_logprobs=args.top_logprobs,
        )
        content = result['content']
        logprob_data = result['logprob_data']
        entropy_stats = compute_response_entropy(logprob_data)

        num_tokens = len(logprob_data)
        mean_ent = entropy_stats.get('mean_entropy', 0.0) if entropy_stats else 0.0

        return {
            'index': idx,
            'ok': True,
            'num_tokens': num_tokens,
            'mean_entropy': mean_ent,
            'content': content,
            'entropy_stats': entropy_stats,
        }
    except Exception as e:
        return {'index': idx, 'ok': False, 'error': str(e)}


def main():
    args = parse_args()

    # 1. Load the data
    items = load_prompts(args.parquet_path, args.max_items)
    total = len(items)
    print(f'Loaded {total} prompts, model: {args.model}, workers: {args.num_workers}')
    print()

    # 2. Initialize the client
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    try:
        client.chat.completions.create(
            model=args.model, messages=[{"role": "user", "content": "ping"}],
            max_tokens=1, logprobs=True, top_logprobs=1
        )
    except Exception as e:
        print(f'Connection failed: {e}')
        return
    print()

    # 3. Parallel processing
    all_results = [None] * total
    token_counts = []
    entropies = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(_process_one, args, client, idx, items[idx]): idx
            for idx in range(total)
        }
        for f in concurrent.futures.as_completed(futures):
            idx = futures[f]
            res = f.result()
            all_results[idx] = res
            if res['ok']:
                token_counts.append(res['num_tokens'])
                entropies.append(res['mean_entropy'])
                print(f'  [{res["index"]+1}/{total}] len={res["num_tokens"]} tokens, entropy={res["mean_entropy"]:.4f}')
            else:
                print(f'  [{idx+1}/{total}] Error: {res["error"]}')

    # 4. Summary
    print()
    print('=' * 50)
    success = sum(1 for r in all_results if r['ok'])
    print(f'  Total: {total}, succeeded: {success}')
    if entropies:
        print(f'  Average token count: {np.mean(token_counts):.1f}')
        print(f'  Average entropy:  {np.mean(entropies):.4f} nats')
    print('=' * 50)

    # Save the results
    if args.output_path:
        out_list = []
        for r in all_results:
            if r['ok']:
                out_list.append({
                    'index': r['index'],
                    'response_content': r['content'],
                    'response_length_tokens': r['num_tokens'],
                    'entropy_stats': r['entropy_stats'],
                })
            else:
                out_list.append({'index': r['index'], 'error': r['error']})
        os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(out_list, f, ensure_ascii=False, indent=2)
        print(f'Results saved to: {args.output_path}')


if __name__ == '__main__':
    main()
