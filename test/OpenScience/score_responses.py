"""
Scoring module: loads cached LLM answers (single JSON file), scores them by rubric
using the judge model, and stores all scoring results as a single JSON file.
Supports resume support and automatic retry.
"""

import os
import re
import json
import copy
import threading
import concurrent.futures
import argparse
import numpy as np
from openai import OpenAI
from retry import retry


# =============================================================================
# Configuration constants
# =============================================================================
MAX_SCORE_RETRIES = 3  # maximum number of retries when a single score fails


# =============================================================================
# JSON preprocessing and parsing
# =============================================================================
def preprocess_json_string(json_str: str) -> str:
    """Preprocess the JSON string, fixing possible escape issues."""
    json_str = json_str.replace("'", '"')
    try:
        json.loads(json_str)
        return json_str
    except json.JSONDecodeError:
        pass

    repaired = ""
    i = 0
    while i < len(json_str):
        char = json_str[i]
        if char == '\\':
            if i + 1 < len(json_str):
                next_char = json_str[i + 1]
                if next_char in ['"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u']:
                    repaired += char + next_char
                    i += 2
                else:
                    repaired += next_char
                    i += 2
            else:
                break
        else:
            repaired += char
            i += 1
    return repaired


def _find_outermost_json_array(text: str) -> str:
    """Find the first complete outermost JSON array in the text by bracket counting."""
    start = text.find('[')
    if start == -1:
        return ""
    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == '\\':
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return ""


def _try_parse_json_array(json_str: str) -> list:
    """Try to parse the string as a JSON array; on failure, fix it and try again."""
    try:
        data = json.loads(json_str)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Try to fix common JSON issues
    fixed = preprocess_json_string(json_str)
    try:
        data = json.loads(fixed)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Try to remove trailing commas
    fixed = re.sub(r',\s*([}\]])', r'\1', json_str)
    try:
        data = json.loads(fixed)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    return []


def _parse_llm_response(response: str) -> list:
    """Parse the JSON response from the LLM in layers, precisely extracting the JSON array using bracket counting."""
    if not response:
        return []

    processed_response = response.strip()

    # Strategy 1: try to parse the entire response directly as a JSON array (fastest path)
    result = _try_parse_json_array(processed_response)
    if result:
        return result

    # Strategy 2: extract from a ```json ... ``` block
    for pattern in [r'```json\s*(.*?)\s*```', r'```\s*(.*?)\s*```']:
        match = re.search(pattern, processed_response, re.DOTALL)
        if match:
            block_content = match.group(1).strip()
            json_str = _find_outermost_json_array(block_content)
            if json_str:
                result = _try_parse_json_array(json_str)
                if result:
                    return result

    # Strategy 3: extract the outermost JSON array from the entire response
    json_str = _find_outermost_json_array(processed_response)
    if json_str:
        result = _try_parse_json_array(json_str)
        if result:
            return result

    # All strategies have failed
    print(f"[WARNING] Failed to parse valid JSON array from response", flush=True)
    print(f"  Response preview (first 500 chars): {processed_response[:500]}", flush=True)
    return []


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
# LLM Client (for scoring)
# =============================================================================
class LLMClient:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    @retry(tries=5, delay=1, backoff=2)
    def generate_response(self, prompt: str, max_tokens: int = 10240,
                          temperature: float = 0, n: int = 1):
        """Generate an answer (with automatic retry, up to 5 times, exponential backoff)"""
        chat_completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            # max_tokens=max_tokens,
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
# Scoring functions
# =============================================================================
def _score_with_rubrics(llm_client: LLMClient, question: str,
                         answer: str, rubrics: list, temperature: float = 0):
    """Generic scoring function that scores an answer according to the given rubrics."""
    # print(f"[Debug] Answer is: {answer[:20]}", flush=True)

    rubrics_text = "\n".join([
        f"{i+1}. {rubric.get('description', rubric) if isinstance(rubric, dict) else rubric}"
        for i, rubric in enumerate(rubrics)
    ])
    num_rubrics = len(rubrics)

    prompt = f"""You are an exceptionally rigorous and professional academic peer reviewer and domain expert. Your task is to conduct an objective, impartial, and high-standard evaluation of complex scientific research tasks, which may include methodology design, experimental setup, or data analysis pipelines.

### Evaluation Task
You must evaluate the provided answer against the given scoring rubrics. For each rubric, you must assign a strict binary score (0 or 1) indicating whether the answer completely satisfies the criterion.

### Input Data
<Question>
{question}
</Question>

<Answer>
{answer}
</Answer>

<Rubrics>
{rubrics_text}
</Rubrics>

### Strict Scoring Guidelines (MUST Follow):
1. **Score 1 (Completely Satisfies)**: The answer fully, explicitly, and comprehensively meets all requirements of the specific rubric.
2. **Score 0 (Does Not Satisfy)**: Assign 0 if ANY of the following apply:
   - The answer only partially meets the criterion or exhibits obvious methodological, experimental, or analytical flaws.
   - The content is vague, generic, or lacks specific technical details.
   - Important elements required by the rubric are omitted.
   - The answer contains factual errors, logical fallacies, or scientific inaccuracies.
   - The response is too brief to demonstrate a complete research methodology.
3. **ZERO Leniency**: Do NOT award a 1 simply because the answer is "on the right track" or "mostly correct". Strict precision is required for a score of 1.

### Output Requirements:
- You MUST output ONLY a valid JSON array. Do not output any thinking process or conversational filler.
- **CRITICAL**: You must provide the `reason` FIRST, and then the `score`. 
- **IMPORTANT**: The number of output items in the JSON array MUST match the number of input rubrics EXACTLY. You have {num_rubrics} rubrics, so you must output exactly {num_rubrics} items.

### Output Format:
Please wrap your output in ```json ``` blocks exactly like this:
```json
[
    {{
        "rubric_id": 1,
        "reason": "Detailed explanation of why the answer fails to meet the standard, pointing out specific flaws or omissions compared to the reference answer.",
        "score": 0
    }},
    {{
        "rubric_id": 2,
        "reason": "Detailed explanation of exactly how the answer completely meets all requirements of this standard, comparing favorably with the reference answer.",
        "score": 1
    }},
    {{
        "rubric_id": 3,
        "reason": "...",
        "score": ...
    }}
]
```"""
    try:
        response = llm_client.generate_response(prompt, temperature=temperature)
        response = clean_llm_output(response)
        individual_scores = _parse_llm_response(response)
    except Exception as e:
        print(f"[Error] Error during model inference after retries: {str(e)}", flush=True)
        return None, None

    try:
        if len(individual_scores) == 0:
            print(f"[Error] No individual scores found in the response", flush=True)
            return None, None

        required_fields = {'rubric_id', 'score', 'reason'}
        for idx, score_item in enumerate(individual_scores):
            if not isinstance(score_item, dict):
                print(f"[Error] Item at index {idx} is not a dictionary: {score_item}")
                return None, None
            missing_fields = required_fields - set(score_item.keys())
            if missing_fields:
                print(f"[Error] Missing fields {missing_fields} in item at index {idx}: {score_item}")
                return None, None

        total_weighted_score = 0.0
        total_weight = 0.0

        all_rubrics_weights = []
        for rubric in rubrics:
            if isinstance(rubric, dict):
                weight = rubric.get('weight', 1) if 'weight' in rubric else 1
                all_rubrics_weights.append(weight)
            else:
                all_rubrics_weights.append(1)

        for score_item in individual_scores:
            if isinstance(score_item, dict) and 'score' in score_item:
                score_value = score_item['score']
                rubric_id = score_item.get('rubric_id', 0)
                if rubric_id > 0 and rubric_id <= len(all_rubrics_weights):
                    weight = all_rubrics_weights[rubric_id - 1]
                else:
                    weight = 1.0
                total_weighted_score += score_value * weight
                total_weight += weight if weight > 0 else 0

        print(f"[Info] Score: {total_weighted_score} / {total_weight} = "
              f"{total_weighted_score / total_weight if total_weight > 0 else 0}", flush=True)

        overall_score = total_weighted_score / total_weight if total_weight > 0 else None
        return overall_score, individual_scores

    except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
        print(f"[Warning] Error parsing model response: {str(e)}", flush=True)
        return None, None


def paper_rl_reward(score_llm_client: LLMClient, question: str,
                     solution_str: str, rubrics: list, temperature: float = 0):
    """Reward function based on static rubrics."""
    answer = clean_llm_output(solution_str)
    try:
        score, individual_scores = _score_with_rubrics(
            llm_client=score_llm_client,
            question=question,
            answer=answer,
            rubrics=rubrics,
            temperature=temperature
        )
        if score is not None:
            print(f"[Info] Static rubric score: {score}", flush=True)
        else:
            print(f"[Warning] Scoring failed", flush=True)
    except Exception as e:
        print(f"[Warning] Error in paper_rl_reward: {str(e)}", flush=True)
        return None, None
    return score, individual_scores


def llm_score(score_llm_client: LLMClient, question: str,
               solution_str: str, rubrics: list, temperature: float = 0):
    """Use the scoring model to score the generated answer."""
    try:
        score, individual_scores = paper_rl_reward(
            score_llm_client=score_llm_client,
            question=question,
            solution_str=solution_str,
            rubrics=rubrics,
            temperature=temperature
        )
        return score, individual_scores
    except Exception as e:
        print(f"[Error] Error scoring answer: {str(e)}")
        return None, None


# =============================================================================
# Cache utility functions (single-file storage)
# =============================================================================
def load_existing_scores(output_path: str) -> dict:
    """Load an existing scoring cache file; return an empty dict if it does not exist."""
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"  [Score Cache] Loaded existing results, {len(data)} items already scored", flush=True)
        return data
    return {}


def load_answers(answer_path: str) -> dict:
    """Load all cached answers from a single-file JSON."""
    if os.path.exists(answer_path):
        with open(answer_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    print(f"  [Warning] Answer file not found: {answer_path}", flush=True)
    return {}


def save_scores(output_path: str, results: dict) -> None:
    """Write all scoring results into a single JSON file (atomic write)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_path = output_path + ".tmp"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, output_path)
    print(f"  [Save] Saved {len(results)} scoring results to {output_path}", flush=True)


# =============================================================================
# Data extraction utility functions
# =============================================================================
def _extract_question(item: dict) -> str:
    """Extract the question text from a data item.

    Supports multiple data formats:
      - Field names: problem / question / prompt
      - Field types: str / list[dict] / numpy.ndarray[dict]
    """
    prompt = item.get('prompt')
    if prompt is not None:
        if hasattr(prompt, '__iter__') and not isinstance(prompt, (str, bytes)):
            prompt_list = prompt.tolist() if hasattr(prompt, 'tolist') else prompt
            if isinstance(prompt_list, list) and len(prompt_list) > 0:
                msg = prompt_list[0]
                if isinstance(msg, dict):
                    content = msg.get('content', '')
                    if content:
                        return content
        if isinstance(prompt, str):
            return prompt
    question = item.get('problem', '') or item.get('question', '')
    if isinstance(question, str):
        return question
    if isinstance(question, list) and len(question) > 0:
        return question[0].get('content', '') if isinstance(question[0], dict) else str(question[0])
    return str(question) if question else ''


# =============================================================================
# Score a single item
# =============================================================================
def _score_single_item(score_client: LLMClient, item: dict, index: int,
                        total: int, answers: dict,
                        shared_results: dict, lock: threading.Lock,
                        temperature: float = 0.0) -> None:
    """Load the cached answer for a single item and score it, thread-safe.

    If scoring fails, it will retry automatically (up to MAX_SCORE_RETRIES times).
    After retries are exhausted, no result is written so it can be rescored on the next run.
    """
    item_key = f"idx_{index}"

    # Check whether it has already been processed by another thread
    with lock:
        if item_key in shared_results:
            print(f"  [Score Cache HIT] Item {index}: already scored, skipping", flush=True)
            return

    print(f"\n[Progress] Scoring {index + 1}/{total}...", flush=True)

    try:
        question = _extract_question(item)
        extra_info = item.get('extra_info', {})

        if isinstance(extra_info, dict):
            rubrics = extra_info.get('rubrics', [])
        elif isinstance(extra_info, str):
            try:
                extra_info = json.loads(extra_info)
                rubrics = extra_info.get('rubrics', [])
            except (json.JSONDecodeError, TypeError):
                rubrics = []
        else:
            rubrics = []

        if isinstance(rubrics, np.ndarray):
            rubrics = rubrics.tolist()

        if not rubrics:
            print(f"  [Error] Item {index}: rubrics list is empty, skipping", flush=True)
            return

        # Load the already-generated answer from the single file
        cached = answers.get(item_key, {})
        generated_answer = cached.get('answer', '')
        if not generated_answer:
            print(f"  [Warning] Item {index}: no cached answer found, skipping", flush=True)
            return  # do not write a result; it can be retried next time

        print(f"  [Info] Loaded cached answer, length: {len(generated_answer)}", flush=True)

        # Score with retry
        score = None
        individual_scores = None
        for attempt in range(MAX_SCORE_RETRIES):
            print(f"  [Scoring] Item {index}, attempt {attempt + 1}/{MAX_SCORE_RETRIES}...", flush=True)
            score, individual_scores = llm_score(
                score_llm_client=score_client,
                question=question,
                solution_str=generated_answer,
                rubrics=rubrics,
                temperature=temperature
            )
            if score is not None:
                break
            if attempt < MAX_SCORE_RETRIES - 1:
                print(f"  [Retry] Item {index}: scoring failed, retrying...", flush=True)

        # Do not write a result; it will be rescored on the next run
        if score is None:
            print(f"  [Warning] Item {index}: scoring failed after {MAX_SCORE_RETRIES} retries, NOT saved (will retry on next run)", flush=True)
            return

        print(f"  [Scoring] Item {index} score: {score:.4f}", flush=True)
        result = {
            'score': score,
            'individual_scores': individual_scores
        }
        with lock:
            shared_results[item_key] = result
        print(f"  [Score Saved] Item {index}: success", flush=True)

    except Exception as e:
        print(f"  [Error] Failed to score item {index}: {str(e)}, NOT saved", flush=True)


# =============================================================================
# Main function
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Score cached answers using LLM (single-file)")
    parser.add_argument("--parquet_path", type=str, required=True,
                        help="Path to the input parquet file")
    parser.add_argument("--answer_path", type=str, required=True,
                        help="Path to the answers JSON file (single file)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to the output scores JSON file")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of parallel workers")
    parser.add_argument("--max_items", type=int, default=400,
                        help="Max items to process. 0 or negative for all.")

    parser.add_argument("--api_key", type=str, default=os.getenv("OPENSCIENCE_API_KEY", ""),
                        help="API Key for Scorer LLM")
    parser.add_argument("--base_url", type=str, default="http://xxx/",
                        help="Base URL for Scorer LLM")
    parser.add_argument("--model", type=str, default="Qwen3-235B-A22B",
                        help="Model name for Scorer LLM")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80, flush=True)
    print("Score Responses (Single-File Cache)", flush=True)
    print("=" * 80, flush=True)
    print(f"Parquet:  {args.parquet_path}", flush=True)
    print(f"Answers:  {args.answer_path}", flush=True)
    print(f"Output:   {args.output_path}", flush=True)
    print(f"Scorer:   {args.model} @ {args.base_url}", flush=True)
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

    # Load the already-existing answers (single file)
    answers = load_answers(args.answer_path)
    print(f"[Answers] Loaded {len(answers)} cached answers", flush=True)

    # Load the already-existing scoring results (resume support)
    existing_scores = load_existing_scores(args.output_path)

    # Initialize the shared results dictionary
    shared_results = copy.deepcopy(existing_scores)
    lock = threading.Lock()

    # Extract the data that has not yet been processed
    pending_indices = [
        idx for idx in range(total_items)
        if f"idx_{idx}" not in shared_results
        and answers.get(f"idx_{idx}", {}).get('answer', '')
    ]
    print(f"[Cache] Already scored: {len(existing_scores)} items, still to score: {len(pending_indices)} items", flush=True)

    if not pending_indices:
        print("[Cache] All items already scored, nothing to do", flush=True)
        # Print the existing statistics
        results_list = list(shared_results.values())
        if results_list:
            scores = [r['score'] for r in results_list]
            print(f"\nScore Statistics:", flush=True)
            print(f"  Mean:   {np.mean(scores):.4f}", flush=True)
            print(f"  Std:    {np.std(scores):.4f}", flush=True)
            print(f"  Min:    {np.min(scores):.4f}", flush=True)
            print(f"  Max:    {np.max(scores):.4f}", flush=True)
        return

    # Initialize the scoring client
    score_client = LLMClient(
        api_key=args.api_key, base_url=args.base_url, model=args.model
    )
    print("[Init] Scorer client initialized", flush=True)

    # Parallel scoring (only process unfinished items)
    print(f"\n{'=' * 80}", flush=True)
    print(f"Scoring answers (parallel, {args.num_workers} workers)", flush=True)
    print(f"Processing {len(pending_indices)} remaining items", flush=True)
    print(f"{'=' * 80}", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        future_to_index = {
            executor.submit(
                _score_single_item, score_client,
                items[idx], idx, total_items, answers,
                shared_results, lock
            ): idx
            for idx in pending_indices
        }
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                future.result()
            except Exception as e:
                print(f"[Error] Item {index} raised an exception: {str(e)}, NOT saved", flush=True)

    # Save the results
    save_scores(args.output_path, shared_results)

    # Statistics of the results
    print(f"\n{'=' * 80}", flush=True)
    print("Scoring Results Summary", flush=True)
    print(f"{'=' * 80}", flush=True)

    results_list = list(shared_results.values())
    num_scored = len(results_list)
    num_failed = total_items - num_scored

    print(f"\nTotal items: {total_items}", flush=True)
    print(f"Scored: {num_scored}", flush=True)
    print(f"Not scored (will retry): {num_failed}", flush=True)

    if results_list:
        scores = [r['score'] for r in results_list]
        print(f"\nScore Statistics:", flush=True)
        print(f"  Mean:   {np.mean(scores):.4f}", flush=True)
        print(f"  Std:    {np.std(scores):.4f}", flush=True)
        print(f"  Min:    {np.min(scores):.4f}", flush=True)
        print(f"  Max:    {np.max(scores):.4f}", flush=True)
    else:
        print("\n[Warning] No results to compute statistics", flush=True)

    print(f"\n{'=' * 80}", flush=True)
    print("Scoring completed", flush=True)
    print(f"{'=' * 80}", flush=True)


if __name__ == "__main__":
    main()
