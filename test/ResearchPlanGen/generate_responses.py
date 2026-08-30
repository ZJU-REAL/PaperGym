#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AI Co-Scientist answer generation script answer generation script — generates answers only, no scoring.

Based only on the specified parquet datasets (ResearchPlanGen_ml.parquet) under
the <data-dir>/data directory, generates answers and saves the results with a format
identical to compare/answer/qwen3_1b_opsd_grpo_responses.json:

    {
      "idx_0": {"index": 0, "question": "<goal>\\n---\\n<guidance>", "answer": "<generated>"},
      "idx_1": {...},
      ...
    }

Features:
  - question = the Goal column of the data row + "\\n---\\n" + the "### Important Guidance for Your Answer" prompt text,
    The question field format is fully consistent with the reference file, and this question is the user message sent to the model.
  - Generates answers item by item through an OpenAI-compatible interface (local vLLM); no scoring is called.
  - Supports resume: automatically reuses completed idx when output file exists (--force to override).
  - Multi-threaded parallel processing (--workers).
  - If the model output contains a <think>...</think> reasoning block, it is stripped (consistent with the reference answers).
"""

import argparse
import json
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# The question prompt text from the reference file (keep strictly unchanged, do not modify)
# ---------------------------------------------------------------------------
GUIDANCE_SUFFIX = (
    "### Important Guidance for Your Answer\n"
    "This question should be answered from **two complementary perspectives**: "
    "**Method Innovation** and **Experimental Design**. Please structure your response accordingly:\n\n"
    "- **Method Innovation**: Propose novel methodological ideas with clear technical contributions. "
    "Explain the core insight, why it is innovative, and how it advances the state of the art.\n"
    "- **Experimental Design**: Provide a rigorous and comprehensive experimental plan to validate the "
    "proposed method. Include baselines, metrics, ablation studies, and analysis strategies.\n"
)

SEPARATOR = "\n---\n"


def build_question(goal: str) -> str:
    """Assemble a question consistent with the reference file format."""
    return goal + SEPARATOR + GUIDANCE_SUFFIX


# ---------------------------------------------------------------------------
# Client (thread-level, ensures thread safety)
# ---------------------------------------------------------------------------
_local = threading.local()


def get_client(args):
    if not hasattr(_local, "client"):
        from openai import OpenAI

        _local.client = OpenAI(
            base_url=args.base_url, api_key=args.api_key or "EMPTY"
        )
    return _local.client


def clean_llm_output(text):
    """Strip the <think>...</think> reasoning block (if present), consistent with the reference answers."""
    if not isinstance(text, str):
        return ""
    end_tag = "</think>"
    i = text.find(end_tag)
    if i != -1:
        return text[i + len(end_tag):].strip()
    return text.strip()


def complete(client, model, prompt, temperature, max_tokens, retries=3, timeout=600):
    """Call OpenAI-compatible API once to generate an answer (auto-retry with exponential backoff)."""
    last = None
    for _ in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                # max_tokens=max_tokens,
                timeout=timeout,
            )
            return clean_llm_output(resp.choices[0].message.content or "")
        except Exception as e:  # pylint: disable=broad-except
            last = e
    raise RuntimeError(f"API call failed after {retries} retries: {last}")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def atomic_write(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_questions(args):
    """Read the parquet files in sorted filename order, assign each record a globally incrementing index,
    and return a list of dicts.

    Each entry: {"index": int, "question": str}
    Sorting rule is consistent with rubric_eval.py (sorted filenames).
    """
    questions = []
    for f in sorted(os.listdir(args.data_dir)):
        if not f.endswith(".parquet"):
            continue
        name = f[: -len(".parquet")]
        dataset = name.replace("_data", "")
        if args.datasets and dataset not in args.datasets:
            continue
        path = os.path.join(args.data_dir, f)
        pf = pq.ParquetFile(path)
        if args.limit:
            batches = [next(pf.iter_batches(batch_size=args.limit))]
        else:
            batches = pf.iter_batches(batch_size=128)
        ds_count = 0
        for batch in batches:
            for row in batch.to_pylist():
                goal = (row.get("Goal") or "").strip()
                if not goal:
                    continue
                questions.append(
                    {"index": len(questions), "question": build_question(goal)}
                )
                ds_count += 1
                if args.limit and ds_count >= args.limit:
                    break
            if args.limit and ds_count >= args.limit:
                break
    return questions


# ---------------------------------------------------------------------------
# Single-item generation
# ---------------------------------------------------------------------------
def generate_one(args, index, question):
    client = get_client(args)
    answer = complete(
        client,
        args.model,
        question,
        args.temperature,
        args.max_tokens,
    )
    return index, answer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(
        description="ResearchPlanGen answer generation (no scoring)"
    )
    p.add_argument("--data-dir", default=os.path.join(here, "data"),
                   help="Data directory containing the parquet files (default: <script dir>/data)")
    p.add_argument("--base-url", required=True,
                   help="OpenAI-compatible API base url, e.g. http://127.0.0.1:8300/v1")
    p.add_argument("--api-key", default="EMPTY", help="API key (default: EMPTY)")
    p.add_argument("--model", required=True, help="Model name (vLLM served-model-name)")
    p.add_argument("--output", required=True,
                   help="Path to the output JSON file (same format as *_responses.json)")
    p.add_argument("--datasets", nargs="*", default=None,
                   help="Only process the given dataset (default: all ResearchPlanGen_ml)")
    p.add_argument("--limit", type=int, default=None,
                   help="Maximum number of samples to process per dataset")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Generation temperature")
    p.add_argument("--max-tokens", type=int, default=8192,
                   help="Maximum number of tokens generated per answer")
    p.add_argument("--workers", type=int, default=12,
                   help="Number of parallel threads")
    p.add_argument("--save-interval", type=int, default=1,
                   help="Save every N items (resume support)")
    p.add_argument("--force", action="store_true",
                   help="Ignore existing results and force regeneration")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    questions = load_questions(args)
    print(f"[load] {len(questions)} questions to process.", flush=True)

    # Resume support: load the existing output
    results = {}
    if os.path.exists(args.output) and not args.force:
        try:
            with open(args.output, "r", encoding="utf-8") as fh:
                results = json.load(fh)
            print(f"[resume] Loaded {len(results)} existing results, reusing completed idx.", flush=True)
        except (json.JSONDecodeError, IOError) as e:
            print(f"[warn] Could not read the existing output, regenerating from scratch: {e}", flush=True)

    # Collect the tasks to generate
    pending = []
    for q in questions:
        idx = q["index"]
        key = f"idx_{idx}"
        existing = results.get(key) if not args.force else None
        if existing and isinstance(existing, dict) and str(existing.get("answer") or "").strip():
            continue  # a valid answer already exists -> reuse it
        pending.append((idx, q["question"]))

    print(f"[plan] {len(pending)} to generate ({len(questions) - len(pending)} reused).", flush=True)

    if not pending:
        print("[done] Nothing to generate, output already complete: " + args.output, flush=True)
        return 0

    # Parallel generation
    errors = 0
    completed = 0
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(generate_one, args, idx, q): (idx, q) for idx, q in pending
        }
        for fut in as_completed(futures):
            idx, q = futures[fut]
            try:
                _, answer = fut.result()
            except Exception as e:  # pylint: disable=broad-except
                errors += 1
                print(f"[error] idx_{idx}: {e}", flush=True)
                continue
            with lock:
                results[f"idx_{idx}"] = {"index": idx, "question": q, "answer": answer}
                completed += 1
                if completed % args.save_interval == 0:
                    atomic_write(args.output, results)
            print(f"[done] idx_{idx} ({completed}/{len(pending)})", flush=True)

    # Final save
    atomic_write(args.output, results)
    print(f"[summary] Processed {len(pending)} items: {completed} succeeded, {errors} failed.", flush=True)
    print(f"[output] {args.output} ({len(results)} items in total)", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
