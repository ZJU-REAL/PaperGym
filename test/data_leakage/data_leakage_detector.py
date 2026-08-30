"""
Dataset data leakage detection tool

Detection workflow:
1. Data loading phase (Phase 1): read datasets in different formats, unify to (question, rubrics) format
2. Data leakage analysis phase (Phase 2): for each question, batch input all rubrics to LLM, judge whether each rubric content can be directly inferred from the question, output Yes/No label for each rubric
3. Statistics and output leakage ratio

Refer to the batch processing pattern in test/ResearchQA/compute_coverage_ds.py:
- N rubrics for one question merged into a single API call
- Output Yes (leakage exists) / No (no leakage) for each rubric

Supported datasets:
- RubricHub_Science_800.parquet (prompt + Rubrics/reward_model format)
- ResearchPlanGen_ml.parquet / ResearchPlanGen_arxiv.parquet (Goal + Rubric format)
- DuoSciPlan_20k.parquet / DuoSciPlan_design.parquet / DuoSciPlan_innov.parquet (prompt + extra_info.merged_rubrics format)
- ai-co-scientist.parquet (Goal + Rubric format)
- research_qa_test.json (ResearchQA/data/)
"""

import os
import json
import threading
import time
import re
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional, Dict, Any, Set

from openai import OpenAI
from tqdm import tqdm
from openai import (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
)
from retry import retry
import pandas as pd
import numpy as np


# ============================================================
# Phase 1: Data loader (adapted for different dataset formats)
# ============================================================

class DataLoader(ABC):
    """Data loader base class, all concrete loaders must implement load() method"""

    @abstractmethod
    def load(self) -> List[Dict[str, Any]]:
        """
        Load dataset, return unified format list:
        [
            {
                "question": str,          # the question / prompt text
                "rubrics": [str, ...],    # list of rubric descriptions
                "source_id": str          # Source identifier (optional)
            },
            ...
        ]
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """Dataset name, used for output identification"""
        pass


class CSNewParquetLoader(DataLoader):
    """Load cs_new.parquet format dataset"""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def name(self) -> str:
        return os.path.basename(self.file_path)

    def load(self) -> List[Dict[str, Any]]:
        df = pd.read_parquet(self.file_path)
        results = []
        for idx, row in df.iterrows():
            # prompt field may be list[dict] or plain string
            prompt = row["prompt"]
            if isinstance(prompt, list):
                # Extract and concatenate all content fields
                texts = []
                for item in prompt:
                    if isinstance(item, dict) and "content" in item:
                        texts.append(item["content"])
                question = "\n".join(texts)
            else:
                question = str(prompt)

            # rubrics in extra_info
            extra_info = row["extra_info"]
            if isinstance(extra_info, dict) and "rubrics" in extra_info:
                rubrics_raw = extra_info["rubrics"]
            else:
                rubrics_raw = []

            # Extract rubric descriptions (compatible with numpy arrays and plain lists)
            rubrics = []
            for r in rubrics_raw:
                if isinstance(r, dict) and "description" in r:
                    rubrics.append(r["description"])
                elif isinstance(r, str):
                    rubrics.append(r)

            results.append({
                "question": question,
                "rubrics": rubrics,
                "source_id": f"cs_new_row{idx}",
            })

        return results


class ResearchQATestMiniLoader(DataLoader):
    """Load research_qa_test_mini.json format dataset"""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def name(self) -> str:
        return os.path.basename(self.file_path)

    def load(self) -> List[Dict[str, Any]]:
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        results = []
        for item in data:
            question = item.get("query", "")

            raw_rubrics = item.get("rubric", [])
            rubrics = []
            if isinstance(raw_rubrics, list):
                for r in raw_rubrics:
                    if isinstance(r, dict) and "rubric_item" in r:
                        rubrics.append(r["rubric_item"])
                    elif isinstance(r, str):
                        rubrics.append(r)
            elif isinstance(raw_rubrics, str):
                rubrics.append(raw_rubrics)

            results.append({
                "question": question,
                "rubrics": rubrics,
                "source_id": item.get("id", f"qa_row{len(results)}"),
            })

        return results


class HealthBenchJsonlLoader(DataLoader):
    """Load HealthBench.jsonl format dataset

    Column structure:
    - prompt: list of {content, role} dicts (question text)
    - rubrics: list of {criterion, points, tags} dicts
    - prompt_id: unique identifier
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    def name(self) -> str:
        return os.path.basename(self.file_path)

    def load(self) -> List[Dict[str, Any]]:
        results = []
        with open(self.file_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)

                # Extract the prompt text
                prompt = item.get("prompt", [])
                texts = []
                if isinstance(prompt, list):
                    for p in prompt:
                        if isinstance(p, dict) and "content" in p:
                            texts.append(str(p["content"]))
                question = "\n".join(texts)

                # Extract the rubrics
                rubrics = []
                raw_rubrics = item.get("rubrics", [])
                if isinstance(raw_rubrics, list):
                    for r in raw_rubrics:
                        if isinstance(r, dict) and "criterion" in r:
                            rubrics.append(r["criterion"])
                        elif isinstance(r, str):
                            rubrics.append(r)

                # source_id
                source_id = item.get("prompt_id", f"healthbench_row{idx}")

                results.append({
                    "question": question,
                    "rubrics": rubrics,
                    "source_id": source_id,
                })

        return results


class AICoScientistParquetLoader(DataLoader):
    """Load ai-co-scientist.parquet format dataset"""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def name(self) -> str:
        return os.path.basename(self.file_path)

    def load(self) -> List[Dict[str, Any]]:
        df = pd.read_parquet(self.file_path)
        results = []
        for idx, row in df.iterrows():
            # Goal field is the question description
            question = str(row.get("Goal", ""))

            # Rubric field is numpy array or list, normalize to string list
            rubrics_raw = row.get("Rubric", [])
            if isinstance(rubrics_raw, np.ndarray):
                rubrics = rubrics_raw.tolist()
            elif isinstance(rubrics_raw, list):
                rubrics = list(rubrics_raw)
            else:
                rubrics = [str(rubrics_raw)] if rubrics_raw else []

            # Ensure all rubrics are strings
            rubrics = [str(r) for r in rubrics]

            # Use q_id or Identifier as source_id
            source_id = row.get("q_id", "") or row.get("Identifier", "") or f"ai-co_row{idx}"

            results.append({
                "question": question,
                "rubrics": rubrics,
                "source_id": source_id,
            })

        return results


class DuoSciPlanParquetLoader(DataLoader):
    """Load DuoSciPlan_*.parquet format dataset

    Column structure:
    - prompt: numpy array of {content, role} dicts (question text)
    - extra_info: dict, containing multiple rubric lists (merged_rubrics / rubrics / problem_rubrics / solution_rubrics)

    Prefer extra_info["merged_rubrics"] (deduplicated merged rubrics),
    if not exists, fallback to rubrics / problem_rubrics / solution_rubrics in order.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    def name(self) -> str:
        return os.path.basename(self.file_path)

    def load(self) -> List[Dict[str, Any]]:
        df = pd.read_parquet(self.file_path)
        results = []
        for idx, row in df.iterrows():
            # prompt is a numpy array, each element is a {content, role} dict
            prompt = row["prompt"]
            texts = []
            for item in np.atleast_1d(prompt).reshape(-1):
                if isinstance(item, dict) and "content" in item:
                    texts.append(str(item["content"]))
            question = "\n".join(texts)

            # Extract rubrics from extra_info, preferring rubrics
            extra_info = row.get("extra_info", {})
            rubrics_raw = None
            if isinstance(extra_info, dict):
                for key in ["rubrics", "merged_rubrics", "problem_rubrics", "solution_rubrics"]:
                    if key in extra_info and extra_info[key] is not None:
                        val = extra_info[key]
                        # compatible with numpy arrays
                        if hasattr(val, "__len__") and len(val) > 0:
                            rubrics_raw = val
                            break

            # Extract rubric descriptions (compatible with numpy arrays and plain lists)
            rubrics = []
            if rubrics_raw is not None:
                for r in np.atleast_1d(rubrics_raw).reshape(-1):
                    if isinstance(r, dict) and "description" in r:
                        rubrics.append(r["description"])
                    elif isinstance(r, str):
                        rubrics.append(r)

            # Use arxiv_id + index as source_id
            source_id = ""
            if isinstance(extra_info, dict):
                source_id = extra_info.get("arxiv_id", "") or ""
                if "index" in extra_info:
                    source_id = f"{source_id}_{extra_info['index']}" if source_id else str(extra_info["index"])
            if not source_id:
                source_id = f"duosci_row{idx}"

            results.append({
                "question": question,
                "rubrics": rubrics,
                "source_id": source_id,
            })

        return results


class EvalSetScienceParquetLoader(DataLoader):
    """Load eval_set_Science_800.parquet format dataset

    Column structure:
    - prompt: list/numpy array of {content, role} dicts (question text)
    - reward_model: {ground_truth, rubrics, style}, where rubrics is an array of {criterion, points, tags}
    - Rubrics: the top-level Rubrics column, an array of {criterion, points} (same content as reward_model.rubrics, used as a fallback)
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    def name(self) -> str:
        return os.path.basename(self.file_path)

    def load(self) -> List[Dict[str, Any]]:
        df = pd.read_parquet(self.file_path)
        results = []
        for idx, row in df.iterrows():
            # prompt is a numpy array, each element is a {content, role} dict
            prompt = row["prompt"]
            texts = []
            for item in np.atleast_1d(prompt).reshape(-1):
                if isinstance(item, dict) and "content" in item:
                    texts.append(str(item["content"]))
            question = "\n".join(texts)

            # Prefer extracting rubric text from reward_model.rubrics, fallback to top-level Rubrics column
            rubrics_raw = None
            rm = row.get("reward_model")
            if isinstance(rm, dict):
                rubrics_raw = rm.get("rubrics")
            if rubrics_raw is None or len(rubrics_raw) == 0:
                rubrics_raw = row.get("Rubrics", [])

            # Extract rubric descriptions (compatible with numpy arrays and plain lists, take criterion field)
            rubrics = []
            for r in np.atleast_1d(rubrics_raw).reshape(-1):
                if isinstance(r, dict) and "criterion" in r:
                    rubrics.append(r["criterion"])
                elif isinstance(r, str):
                    rubrics.append(r)

            results.append({
                "question": question,
                "rubrics": rubrics,
                "source_id": f"eval_set_row{idx}",
            })

        return results


def get_loader_for_dataset(file_path: str) -> DataLoader:
    """Automatically select appropriate loader based on file extension and column names"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".parquet":
        # Only read schema (metadata) to detect column names, efficient and no data loading needed
        import pyarrow.parquet as pq
        schema = pq.read_schema(file_path)
        cols = set(schema.names)
        if "Goal" in cols and "Rubric" in cols:
            return AICoScientistParquetLoader(file_path)
        if "Rubrics" in cols:  # eval_set format: prompt + Rubrics/reward_model
            return EvalSetScienceParquetLoader(file_path)
        if "extra_info" in cols:  # DuoSciPlan format: prompt + extra_info.merged_rubrics
            return DuoSciPlanParquetLoader(file_path)
        return CSNewParquetLoader(file_path)
    elif ext == ".jsonl":
        return HealthBenchJsonlLoader(file_path)
    elif ext == ".json":
        return ResearchQATestMiniLoader(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


# ============================================================
# Phase 2: LLM data leakage analysis
# ============================================================

# Custom exception for triggering retry
class LeakageAnalysisError(Exception):
    pass


# Thread-level LLM client cache
_thread_local = threading.local()

# Default API configuration (reuse existing config)
DEFAULT_API_KEY = "sk-xxx"
DEFAULT_BASE_URL = "http://xxx/"
DEFAULT_MODEL = "DeepSeek-V4-Flash"

# Root directory of the datasets to analyze; override with LEAKAGE_DATASET_DIR
DEFAULT_DATASET_DIR = os.getenv("LEAKAGE_DATASET_DIR", "dataset")

# DEFAULT_API_KEY = "sk-xxx"
# DEFAULT_BASE_URL = "http://xxx"
# DEFAULT_MODEL = "deepseek-v4-flash"


def get_client(api_key: str = None, base_url: str = None) -> OpenAI:
    """Get current thread OpenAI client (thread-safe)"""
    if not hasattr(_thread_local, "client"):
        _thread_local.client = OpenAI(
            api_key=api_key or DEFAULT_API_KEY,
            base_url=base_url or DEFAULT_BASE_URL,
        )
    return _thread_local.client


LEAKAGE_BATCH_PROMPT_TEMPLATE = """You are an expert in evaluating dataset quality and detecting data leakage.

For each rubric item below, determine whether it can be directly inferred from the question alone.
Output only "Yes" or "No" for each rubric item, one per line, and nothing else.

Question: {question}

Rubrics:
{rubrics_text}

Output:"""

def clean_llm_output(text: str) -> str:
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

def parse_batch_verdicts(raw: str, expected_count: int) -> List[bool]:
    """
    Parse the LLM batch output, extracting the YES/NO verdict for each rubric.

    Reference the verified pattern in compute_coverage_ds.py:
    - Split by \\n -> strip -> filter empty lines -> exact count check
    - No complex regex extraction; each line of the model output should directly be a verdict
    """
    # Remove the <think> reasoning tags (some models output them)
    cleaned = clean_llm_output(raw)

    # Reference compute_coverage_ds.py: split by lines, filter empty lines
    lines = cleaned.split('\n')
    lines = [line.strip() for line in lines if line.strip()]

    # Exact count check
    if len(lines) != expected_count:
        return None

    # Each line should be Yes or No (numeric prefixes allowed, e.g. "1. Yes")
    results = []
    for line in lines:
        # Strip any possible numeric prefix
        line = re.sub(r'^\d+[\.:\)]\s*', '', line).strip()
        if line.upper() == "YES":
            results.append(True)
        elif line.upper() == "NO":
            results.append(False)
        else:
            return None

    return results


@retry(tries=3, delay=1, backoff=2, exceptions=LeakageAnalysisError)
def analyze_question_batch(question: str, rubrics: List[str]) -> List[bool]:
    """
    Perform a single LLM call for all rubrics of a question, batch detect data leakage.
    Returns a boolean list corresponding one-to-one with the rubrics order (True = leakage exists).
    """
    # Build numbered rubric list text
    rubrics_text = "\n".join(f"{i+1}. {r}" for i, r in enumerate(rubrics))
    prompt = LEAKAGE_BATCH_PROMPT_TEMPLATE.format(
        question=question, rubrics_text=rubrics_text
    )

    client = get_client()
    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )

        # Defensive handling of None content (DeepSeek API may return content: null)
        content = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason
        if content is None:
            print(f"  [WARN] API returned empty content, finish_reason={finish_reason}, retrying", flush=True)
            raise LeakageAnalysisError(f"API returned empty content, finish_reason={finish_reason}")

        raw = content.strip()

        results = parse_batch_verdicts(raw, len(rubrics))
        if results is not None:
            return results

        # Retry when it cannot be parsed
        print(f"  [WARN] Batch parse count mismatch: expected {len(rubrics)}, finish_reason={finish_reason}, raw output: '{raw[:300]}'", flush=True)
        raise LeakageAnalysisError(f"Batch parse failed: expected {len(rubrics)} items, finish_reason={finish_reason}, raw output: {raw[:300]}")

    except (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError) as e:
        print(f"  [WARN] API call failed ({type(e).__name__}), retrying: {e}", flush=True)
        raise LeakageAnalysisError(f"API error: {type(e).__name__}: {e}")
    except LeakageAnalysisError:
        raise
    except Exception as e:
        print(f"  [ERROR] API call failed: {e}", flush=True)
        return [False] * len(rubrics)


def analyze_question_parallel(question_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform a single LLM batch call for all rubrics of a question, detect data leakage.
    Reference compute_coverage_ds.py pattern, all rubrics input to LLM at once.

    Args:
        question_data: a dict containing question, rubrics, and source_id

    Returns:
        the leakage detection result and Yes/No label for each rubric
    """
    question = question_data["question"]
    rubrics = question_data["rubrics"]
    source_id = question_data.get("source_id", "unknown")

    if not rubrics:
        return {
            "source_id": source_id,
            "question": question,
            "rubrics": [],
            "leakage_results": [],
            "leakage_labels": [],
            "leaked_count": 0,
            "total_count": 0,
        }

    # Single batch call to analyze all rubrics
    results = analyze_question_batch(question, rubrics)

    leaked_count = sum(1 for r in results if r)
    total_count = len(results)
    # Generate Yes/No labels (corresponding to rubrics order)
    leakage_labels = ["Yes" if r else "No" for r in results]

    return {
        "source_id": source_id,
        "question": question[:100] + "..." if len(question) > 100 else question,
        "rubrics": rubrics,
        "leakage_results": results,       # [True, False, ...] (aligned with rubrics)
        "leakage_labels": leakage_labels,  # ["Yes", "No", ...] (human-readable labels)
        "leaked_count": leaked_count,
        "total_count": total_count,
    }


# ============================================================
# Main flow
# ============================================================

def analyze_dataset(
    file_path: str,
    max_samples: int = 0,
    max_questions_parallel: int = 4,
    output_path: str = None,
    existing_results: Dict[str, Any] = None,
) -> List[Dict[str, Any]]:
    """
    Analyze data leakage for a single dataset.

    Use batch detection mode: all rubrics under each question are input to LLM for judgment at once,
    output Yes (leakage exists) / No (no leakage) label for each rubric.

    Supports incremental save and resume:
    - If output_path is not None, incrementally save results after processing each sample
    - If existing_results is not None, skip already processed source_id

    Args:
        file_path: path to the dataset file
        max_samples: Max samples to analyze (0 means all)
        max_questions_parallel: Parallelism between questions
        output_path: Result save path (optional, for incremental save)
        existing_results: Existing results dict, key is source_id, for resume support

    Returns:
        a list of analysis results per sample, including leakage_labels (Yes/No labels)
    """
    # Phase 1: Load the data
    print(f"\n{'='*70}")
    print(f"[Phase 1] Loading dataset: {file_path}")
    loader = get_loader_for_dataset(file_path)
    all_samples = loader.load()
    print(f"  Loading complete: {len(all_samples)} samples")

    if max_samples > 0:
        all_samples = all_samples[:max_samples]
        print(f"  Limited to first {max_samples} samples")

    # Resume: skip already processed samples
    skipped_ids: Set[str] = set()
    if existing_results:
        processed_ids = set(existing_results.get("results", {}).keys())
        if processed_ids:
            filtered_samples = []
            for s in all_samples:
                sid = s.get("source_id", "")
                if sid in processed_ids:
                    skipped_ids.add(sid)
                else:
                    filtered_samples.append(s)
            all_samples = filtered_samples
            print(f"  Resuming: skipped {len(skipped_ids)} already-processed samples, {len(all_samples)} remaining")
            if not all_samples:
                print("  All samples already processed, return existing results directly")
                # Return the merged result list
                merged = list(existing_results.get("results", {}).values())
                print(f"\n{'='*70}")
                print(f"[Result] Dataset: {loader.name()}")
                total_leaked = sum(r["leaked_count"] for r in merged)
                total_rubrics = sum(r["total_count"] for r in merged)
                print(f"  Analyzed samples: {len(merged)}")
                print(f"  Total rubric count: {total_rubrics}")
                print(f"  Leaked rubric count: {total_leaked}")
                print(f"  Leakage ratio: {total_leaked}/{total_rubrics} = {total_leaked/total_rubrics*100:.2f}%"
                      if total_rubrics > 0 else "  Leakage ratio: N/A (no rubrics)")
                return merged

    # Count the total number of rubrics
    total_rubrics = sum(len(s["rubrics"]) for s in all_samples)
    print(f"  Total rubric count: {total_rubrics}")

    # Phase 2: Leakage analysis
    print(f"\n[Phase 2] Starting data leakage analysis (batch mode: all rubrics of a question are merged into a single API call, parallelism across questions={max_questions_parallel})")

    # Load the existing results for merging
    saved_results = {}
    if existing_results:
        saved_results = existing_results.get("results", {})
    all_results = []
    question_results = [None] * len(all_samples)

    # Get result file lock (need atomicity for multi-threaded incremental writes)
    save_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=max_questions_parallel) as executor:
        future_to_idx = {
            executor.submit(analyze_question_parallel, s): (idx, s.get("source_id", ""))
            for idx, s in enumerate(all_samples)
        }
        # Reference compute_coverage.py tqdm progress bar mode
        pbar = tqdm(total=len(all_samples), desc="Analyzing data leakage", unit="sample")
        for future in as_completed(future_to_idx):
            idx, source_id = future_to_idx[future]
            try:
                question_results[idx] = future.result()
                result = question_results[idx]
                if result:
                    pbar.set_postfix({
                        "leaked": f"{result['leaked_count']}/{result['total_count']}",
                    })
                    # Incremental save to cache
                    saved_results[source_id] = result
                    if output_path:
                        with save_lock:
                            _incremental_save(output_path, loader.name(), saved_results, len(saved_results), in_progress=True)
            except Exception as e:
                print(f"\n  [ERROR] Failed to analyze sample [{idx}] (source_id={source_id}): {e}", flush=True)
                question_results[idx] = None
            pbar.update(1)
        pbar.close()

    # Merge newly processed results with existing results
    new_results = [r for r in question_results if r is not None]
    if existing_results:
        merged_results = list(saved_results.values())
    else:
        merged_results = new_results

    # Final save
    total_leaked = sum(r["leaked_count"] for r in merged_results)
    total_rubrics_analyzed = sum(r["total_count"] for r in merged_results)

    if output_path:
        _incremental_save(output_path, loader.name(), saved_results, len(merged_results), in_progress=False)

        # Also save a complete result file with a summary
        summary_path = output_path.replace(".json", "_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "dataset": file_path,
                "summary": {
                    "samples": len(merged_results),
                    "total_rubrics": total_rubrics_analyzed,
                    "leaked_rubrics": total_leaked,
                    "leakage_ratio": f"{total_leaked}/{total_rubrics_analyzed}",
                    "leakage_percentage": round(total_leaked / total_rubrics_analyzed * 100, 2) if total_rubrics_analyzed > 0 else 0,
                },
            }, f, ensure_ascii=False, indent=2)
        print(f"\n  Results saved:")
        print(f"    - Incremental results: {output_path}")
        print(f"    - Full summary: {summary_path}")

    # Statistics output
    print(f"\n{'='*70}")
    print(f"[Result] Dataset: {loader.name()}")
    print(f"  Analyzed samples: {len(merged_results)}")
    print(f"  Total rubric count: {total_rubrics_analyzed}")
    print(f"  Leaked rubric count: {total_leaked}")
    print(f"  Leakage ratio: {total_leaked}/{total_rubrics_analyzed} = {total_leaked/total_rubrics_analyzed*100:.2f}%"
          if total_rubrics_analyzed > 0 else "  Leakage ratio: N/A (no rubrics)")

    return merged_results


def _incremental_save(
    output_path: str,
    dataset_name: str,
    results_dict: Dict[str, Any],
    num_samples: int,
    in_progress: bool,
):
    """Incrementally save processing results to JSON file (thread-safe)"""
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": dataset_name,
            "status": "in_progress" if in_progress else "completed",
            "num_samples": num_samples,
            "results": results_dict,
        }, f, ensure_ascii=False, indent=2)


def load_existing_results(output_path: str) -> Optional[Dict[str, Any]]:
    """Load saved processing results, for resume support"""
    if os.path.exists(output_path):
        print(f"  [Resume] Detected existing result file: {output_path}")
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            num_existing = len(data.get("results", {}))
            status = data.get("status", "unknown")
            print(f"  [Resume] Found results for {num_existing} samples (status={status})")
            return data
        except (json.JSONDecodeError, IOError) as e:
            print(f"  [Resume] Failed to read existing result file ({e}), reprocessing")
            return None
    return None


def main():
    """Main entry: analyze multiple datasets, supports resume"""
    import argparse

    parser = argparse.ArgumentParser(description="Dataset data leakage detection tool")
    parser.add_argument(
        "--datasets", nargs="+",
        default=[
            # os.path.join(DEFAULT_DATASET_DIR, "RubricHub_Science_800.parquet"),
            # os.path.join(DEFAULT_DATASET_DIR, "ResearchPlanGen_ml.parquet"),
            # os.path.join(DEFAULT_DATASET_DIR, "ResearchPlanGen_arxiv.parquet"),
            # os.path.join(DEFAULT_DATASET_DIR, "DuoSciPlan_20k.parquet"),
            # os.path.join(DEFAULT_DATASET_DIR, "DuoSciPlan_design.parquet"),
            # os.path.join(DEFAULT_DATASET_DIR, "DuoSciPlan_innov.parquet"),
            # os.path.join(DEFAULT_DATASET_DIR, "cs_new.parquet"),
            # os.path.join(DEFAULT_DATASET_DIR, "research_qa_test.json"),
            os.path.join(DEFAULT_DATASET_DIR, "HealthBench_800.jsonl"),
        ],
        help="List of dataset file paths to analyze"
    )
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Maximum number of samples analyzed per dataset (0=all)")
    parser.add_argument("--max_questions_parallel", type=int, default=10,
                        help="Parallel threads between questions")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Result output directory (default ./output)")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Enable resume (default enabled, use --no-resume to disable)")
    parser.add_argument("--no-resume", action="store_false", dest="resume",
                        help="Disable resume, force reprocess all samples")

    args = parser.parse_args()

    summary = []

    for ds_path in args.datasets:
        if not os.path.exists(ds_path):
            print(f"\n[Skip] File does not exist: {ds_path}")
            continue

        # Auto-deduce output path
        ds_basename = os.path.splitext(os.path.basename(ds_path))[0]
        output_path = os.path.join(args.output_dir, f"{ds_basename}_leakage.json")

        # Load the existing results (resume support)
        existing_results = None
        if args.resume:
            existing_results = load_existing_results(output_path)
            if existing_results:
                # If previously completed, check whether reprocessing is needed
                if existing_results.get("status") == "completed":
                    print(f"  [Resume] This dataset is already fully processed, skipping")
                    # Directly extract the details from the existing results
                    merged = list(existing_results.get("results", {}).values())
                    elapsed = 0
                    total_leaked = sum(r["leaked_count"] for r in merged)
                    total_rubrics = sum(r["total_count"] for r in merged)
                    ratio_str = f"{total_leaked}/{total_rubrics}" if total_rubrics > 0 else "N/A"
                    summary.append({
                        "dataset": os.path.basename(ds_path),
                        "samples": len(merged),
                        "total_rubrics": total_rubrics,
                        "leaked": total_leaked,
                        "ratio_str": ratio_str,
                        "time_sec": 0,
                    })
                    continue

        start_time = time.time()
        results = analyze_dataset(
            file_path=ds_path,
            max_samples=args.max_samples,
            max_questions_parallel=args.max_questions_parallel,
            output_path=output_path,
            existing_results=existing_results,
        )
        elapsed = time.time() - start_time

        total_leaked = sum(r["leaked_count"] for r in results)
        total_rubrics = sum(r["total_count"] for r in results)
        ratio_str = f"{total_leaked}/{total_rubrics}" if total_rubrics > 0 else "N/A"

        summary.append({
            "dataset": os.path.basename(ds_path),
            "samples": len(results),
            "total_rubrics": total_rubrics,
            "leaked": total_leaked,
            "ratio_str": ratio_str,
            "time_sec": round(elapsed, 2),
        })

    # Summary output
    print(f"\n{'='*70}")
    print("Summary:")
    print(f"{'Dataset':<40} {'Samples':<8} {'Rubrics':<10} {'Leaked':<8} {'Ratio':<15} {'Time(s)':<8}")
    print("-" * 90)
    for s in summary:
        print(f"{s['dataset']:<40} {s['samples']:<8} {s['total_rubrics']:<10} {s['leaked']:<8} {s['ratio_str']:<15} {s['time_sec']}s")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
