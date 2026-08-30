#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AI Co-Scientist Rubric evaluation script.

Workflow (two LLM calls per sample):
  1. Generate: Call LLM to generate Proposed Research Plan based on Goal.
  2. Grade: Use the GRADING_PROMPT below to score Proposed Plan item by item against Rubric.

Features:
  - Intermediate results stored as JSON (plans/<dataset>/<id>.json and grades/<dataset>/<id>.json),
    Reuses existing scoring results on rerun (resume support), already generated plans are also reused.
  - Supports multithreaded parallel processing (--workers).
  - API configuration (base_url / api-key / model, etc.) all passed via command-line arguments.
  - GRADING_PROMPT is kept strictly unchanged, only placeholder substitution is performed.
"""

import argparse
import ast
import json
import os
import re
import sys
import tempfile
import threading
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Evaluation Prompt (keep strictly unchanged, do not modify; only perform placeholder substitution)
# ---------------------------------------------------------------------------
GRADING_PROMPT = """Evaluate if the Proposed Research Plan satisfies the Research Scenario based on the provided evaluation criteria. # Research Scenario {scenario} You have to evaluate each of the rubric items provided below. # Rubric Item 1: {rubric_item_1} Item 2: {rubric_item_2} . . . Item n: {rubric_item_n} # Reference Solution Here is a reference solution written by an expert: {reference_solution} • It is just meant to demonstrate one possible approach that satisfies the scenario. It is not necessary for the proposed research plan you are grading to match all details in the reference solution. • The Research Plan you have to grade might have different design choices. This is okay, if the choices are valid, and supported with correct rationale. # Proposed Research Plan {proposed_plan} # Instructions First, come up with weaknesses of the proposed plan specific to the scenario. Then, return the following nested XML block for each of the grading items (always close opened XML tags): <rubric> <item num=1> <criteria> Repeat the rubric item string you are checking here. </criteria> 37  <reasoning> Analyze if the proposed plan violates any of the below desiderata with respect to the rubric item. **DESIDERATA** 1. HANDLES ALL CRITERIA: Does the plan satisfy all criteria mentioned in the rubric item? An exception is if the criteria says "such as", "for example", or "including", the response does not have to include the same examples listed to meet the criteria, but whatever is provided must be valid and reasonable. 2. DETAILED, SPECIFIC SOLUTION: Does the part of the plan relevant to satisfying this rubric item include fully specified details on HOW to implement it? There should be no self-proclaimed claims of handling something without doing so. There should be no vague terms, ambiguity, or lack of clarity. It should be described in simple to understand language. 3. NO OVERLOOKED FLAWS OR WEAKNESSES: Are there any important overlooked flaws or weaknesses in the part of the plan addressing this rubric item that invalidate its satisfaction of the rubric item? 4. WELL JUSTIFIED RATIONALE: Is the part of the plan relevant to this grading item well-motivated and justified? For example, are there convincing arguments provided for how the plan handles this grading item is better than simpler solutions or alternate hypotheses? 5. COST AND EFFORT EFFICIENT: Does the plan handle this rubric item cost efficiently, without being unnecessarily complex? Check if a solution requiring less human effort or resources would be equally effective for this rubric item. 6. NO ETHICAL ISSUES: Does this part of the plan have any potential for negative consequences, or is it ethically problematic? 7. CONSISTENT WITH OVERALL PLAN: Is this part of the plan consistent with the rest of the plan? Check if it contradicts any other parts of the plan. - Be skeptical, careful, and come up with valid criticisms. Be as strict as possible, while being unbiased and reasonable. - Note that the plan should not just say it satisfies these desiderata, don't be fooled by that. Check carefully WHETHER, HOW and WHY the proposed plan meets each desiderata for this rubric item one by one. - Based on the above analysis, list the desiderata numbers that are violated. If no part of the plan addresses this rubric item, list all desiderata numbers (1,2,3,4,5,6,7). </reasoning> <errors>[comma-separated list of desiderata numbers that are violated or "none" if no violations]</errors> </item> ... Similarly, for all rubric items... <item num=n> ... </item> </rubric>"""

# Rubric placeholder segment in Prompt (used for replacing with actual items)
RUBRIC_SECTION_TEMPLATE = "Item 1: {rubric_item_1} Item 2: {rubric_item_2} . . . Item n: {rubric_item_n}"

# ---------------------------------------------------------------------------
# Prompt for generating Proposed Research Plan (this part is not subject to "do not modify evaluation prompt" restriction)
# ---------------------------------------------------------------------------
GENERATION_PROMPT_TEMPLATE = """You are an AI co-scientist. Propose a detailed, concrete research plan that addresses the research scenario below.

# Research Scenario
{scenario}

# Requirements
- Propose a novel, feasible research plan (framework / method) that satisfies the scenario.
- Include fully specified details on HOW to implement each component (no vague claims, no ambiguity).
- Justify your design choices with clear, convincing rationale.
- Keep the plan cost / effort efficient and avoid any ethical issues.
- Keep the whole plan internally consistent.

Return ONLY the research plan as plain text, with no extra commentary."""


# ---------------------------------------------------------------------------
# Client (thread-level, ensures thread safety)
# ---------------------------------------------------------------------------
_local = threading.local()


def get_clients(args):
    if not hasattr(_local, "judge"):
        from openai import OpenAI

        _local.judge = OpenAI(
            base_url=args.base_url, api_key=args.api_key or "EMPTY"
        )
        gen_base = args.gen_base_url or args.base_url
        gen_key = args.gen_api_key or args.api_key or "EMPTY"
        _local.gen = OpenAI(base_url=gen_base, api_key=gen_key)
    return _local.gen, _local.judge


def complete(client, model, prompt, temperature, max_tokens, retries=3, timeout=600):
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
            return resp.choices[0].message.content or ""
        except Exception as e:  # pylint: disable=broad-except
            last = e
    raise RuntimeError(f"API call failed after {retries} retries: {last}")


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------
def build_grading_prompt(scenario, rubric_items, reference_solution, proposed_plan):
    rubric_block = " ".join(
        f"Item {i + 1}: {item}" for i, item in enumerate(rubric_items)
    )
    prompt = GRADING_PROMPT
    prompt = prompt.replace(RUBRIC_SECTION_TEMPLATE, rubric_block)
    prompt = prompt.replace("{scenario}", scenario)
    prompt = prompt.replace("{reference_solution}", reference_solution)
    prompt = prompt.replace("{proposed_plan}", proposed_plan)
    return prompt


# ---------------------------------------------------------------------------
# Score result parsing
# ---------------------------------------------------------------------------
def parse_errors(text):
    """Parse <errors> text into a list of violated desiderata numbers."""
    t = (text or "").strip().lower()
    if not t or t == "none":
        return []
    return [int(n) for n in re.findall(r"\d+", t)]


def _extract_items(root_text):
    """Parse each item from <rubric>...</rubric> text."""
    items = []
    for m in re.finditer(
        r"<item[^>]*num=?\"?(\d+)\"?[^>]*>(.*?)</item>", root_text, re.DOTALL
    ):
        num = m.group(1)
        body = m.group(2)
        crit = re.search(r"<criteria>(.*?)</criteria>", body, re.DOTALL)
        reasoning = re.search(r"<reasoning>(.*?)</reasoning>", body, re.DOTALL)
        errors = re.search(r"<errors>(.*?)</errors>", body, re.DOTALL)
        err_text = errors.group(1).strip() if errors else ""
        viol = parse_errors(err_text)
        items.append(
            {
                "num": int(num),
                "criteria": crit.group(1).strip() if crit else "",
                "reasoning": reasoning.group(1).strip() if reasoning else "",
                "errors_raw": err_text,
                "violations": viol,
                "passed": len(viol) == 0,
            }
        )
    return items


def parse_grade_xml(raw):
    """Parse scoring XML returned by model, return item list."""
    # 1) Try to locate the <rubric>...</rubric> block
    m = re.search(r"<rubric>(.*)</rubric>", raw, re.DOTALL)
    rubric_xml = (
        "<rubric>" + m.group(1) + "</rubric>" if m else raw
    )
    # 2) Add quotes to num attribute for ET parsing
    quoted = re.sub(r'<item\s+num=(\d+)>', r'<item num="\1">', rubric_xml)
    try:
        root = ET.fromstring(quoted)
        items = []
        for it in root.findall("item"):
            num = it.get("num")
            crit = it.findtext("criteria") or ""
            reasoning = it.findtext("reasoning") or ""
            err_text = (it.findtext("errors") or "").strip()
            viol = parse_errors(err_text)
            items.append(
                {
                    "num": int(num) if num and num.isdigit() else num,
                    "criteria": crit.strip(),
                    "reasoning": reasoning.strip(),
                    "errors_raw": err_text,
                    "violations": viol,
                    "passed": len(viol) == 0,
                }
            )
        if items:
            return items
    except ET.ParseError:
        pass
    # 3) Fallback: extract items one by one using regex
    return _extract_items(rubric_xml)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def atomic_write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _mock_grade_xml(n_items):
    items = "\n".join(
        f'<item num="{i + 1}"><criteria>item {i + 1}</criteria>'
        f"<reasoning>mock reasoning</reasoning><errors>none</errors></item>"
        for i in range(n_items)
    )
    return f"<rubric>{items}</rubric>"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_records(args):
    records = []
    for f in sorted(os.listdir(args.data_dir)):
        if not f.endswith(".parquet"):
            continue
        name = f[: -len(".parquet")]  # e.g. ResearchPlanGen_ml
        dataset = name.replace("_data", "")
        if args.datasets and dataset not in args.datasets:
            continue
        path = os.path.join(args.data_dir, f)
        # Use iter_batches for on-demand reading to avoid loading the entire large file into memory at once
        pf = pq.ParquetFile(path)
        ds_records = []
        if args.limit:
            batches = [next(pf.iter_batches(batch_size=args.limit))]
        else:
            batches = pf.iter_batches(batch_size=128)
        for batch in batches:
            for i, row in enumerate(batch.to_pylist()):
                goal = row.get("Goal") or ""
                rubric = row.get("Rubric")
                if isinstance(rubric, str):
                    try:
                        rubric = ast.literal_eval(rubric)
                    except Exception:  # pylint: disable=broad-except
                        rubric = [rubric]
                if not isinstance(rubric, list):
                    rubric = [str(rubric)] if rubric is not None else []
                reference = row.get("Reference solution") or ""
                rid = f"{dataset}__{row.get('q_id') or row.get('article_id') or i}"
                ds_records.append(
                    {
                        "id": rid,
                        "dataset": dataset,
                        "goal": goal,
                        "rubric": rubric,
                        "reference": reference,
                        "article_id": row.get("article_id"),
                        "q_id": row.get("q_id"),
                        "identifier": row.get("Identifier"),
                    }
                )
        if args.limit:
            ds_records = ds_records[: args.limit]
        records.extend(ds_records)
    return records


# ---------------------------------------------------------------------------
# Single sample processing
# ---------------------------------------------------------------------------
def process_record(rec, args, output_dir, force):
    dataset = rec["dataset"]
    rid = rec["id"]
    grade_path = os.path.join(output_dir, "grades", dataset, rid + ".json")
    plan_path = os.path.join(output_dir, "plans", dataset, rid + ".json")

    # Scoring result exists -> reuse directly (resume support)
    if os.path.exists(grade_path) and not force:
        with open(grade_path, "r", encoding="utf-8") as fh:
            return json.load(fh), False

    gen_client, judge_client = get_clients(args)

    # Generate or reuse Proposed Plan
    if os.path.exists(plan_path) and not force:
        with open(plan_path, "r", encoding="utf-8") as fh:
            plan = json.load(fh)["plan"]
    else:
        if args.dry_run:
            plan = "(dry-run mock proposed plan)"
        else:
            gen_prompt = GENERATION_PROMPT_TEMPLATE.format(scenario=rec["goal"])
            plan = complete(
                gen_client,
                args.gen_model or args.model,
                gen_prompt,
                args.temperature,
                args.gen_max_tokens,
            )
        atomic_write(
            plan_path,
            {
                "id": rid,
                "dataset": dataset,
                "goal": rec["goal"],
                "plan": plan,
            },
        )

    # Scoring
    grading_prompt = build_grading_prompt(
        rec["goal"], rec["rubric"], rec["reference"], plan
    )
    if args.dry_run:
        raw = _mock_grade_xml(len(rec["rubric"]))
    else:
        raw = complete(
            judge_client,
            args.model,
            grading_prompt,
            args.temperature,
            args.judge_max_tokens,
        )
    items = parse_grade_xml(raw)

    total = len(items)
    passed = sum(1 for it in items if it["passed"])
    desiderata_violations = {str(d): 0 for d in range(1, 8)}
    for it in items:
        for v in it["violations"]:
            if 1 <= v <= 7:
                desiderata_violations[str(v)] += 1

    result = {
        "id": rid,
        "dataset": dataset,
        "article_id": rec.get("article_id"),
        "q_id": rec.get("q_id"),
        "identifier": rec.get("identifier"),
        "goal": rec["goal"],
        "rubric": rec["rubric"],
        "reference_solution": rec["reference"],
        "proposed_plan": plan,
        "grading_prompt": grading_prompt,
        "raw_grade": raw,
        "parsed": {
            "items": items,
            "num_items": total,
            "num_passed": passed,
            "pass_rate": (passed / total) if total else None,
            "desiderata_violation_counts": desiderata_violations,
        },
    }
    atomic_write(grade_path, result)
    return result, True


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def summarize(results, output_dir):
    total_items = sum(r["parsed"]["num_items"] for r in results)
    total_passed = sum(r["parsed"]["num_passed"] for r in results)
    by_dataset = {}
    for r in results:
        d = r["dataset"]
        agg = by_dataset.setdefault(
            d,
            {
                "num_records": 0,
                "num_items": 0,
                "num_passed": 0,
                "desiderata_violation_counts": {str(x): 0 for x in range(1, 8)},
            },
        )
        agg["num_records"] += 1
        agg["num_items"] += r["parsed"]["num_items"]
        agg["num_passed"] += r["parsed"]["num_passed"]
        for k, v in r["parsed"]["desiderata_violation_counts"].items():
            agg["desiderata_violation_counts"][k] += v

    for d, agg in by_dataset.items():
        agg["pass_rate"] = (
            (agg["num_passed"] / agg["num_items"]) if agg["num_items"] else None
        )

    summary = {
        "num_records": len(results),
        "num_items": total_items,
        "num_passed": total_passed,
        "overall_pass_rate": (total_passed / total_items) if total_items else None,
        "by_dataset": by_dataset,
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(
        os.path.join(output_dir, "summary.json"), "w", encoding="utf-8"
    ) as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    with open(
        os.path.join(output_dir, "all_results.json"), "w", encoding="utf-8"
    ) as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    default_data = os.path.join(here, "data")
    default_out = os.path.join(here, "results")

    p = argparse.ArgumentParser(
        description="ResearchPlanGen rubric evaluation (generation + scoring)"
    )
    p.add_argument("--data-dir", default=default_data,
                   help="Directory containing parquet data (default: <script_dir>/data)")
    p.add_argument("--output-dir", default=default_out,
                   help="Result output directory (default: <script_dir>/results)")
    # Scoring (judge) API
    p.add_argument("--base-url", required=True,
                   help="OpenAI-compatible API base url, e.g. http://127.0.0.1:8200/v1")
    p.add_argument("--api-key", default="EMPTY", help="API key (default: EMPTY)")
    p.add_argument("--model", required=True, help="Scoring (judge) model name")
    p.add_argument("--judge-max-tokens", type=int, default=16384,
                   help="Max generation tokens for scoring stage")
    # Generation API
    p.add_argument("--gen-model", default=None,
                   help="Model name for generating proposed plan (default: same as --model)")
    p.add_argument("--gen-base-url", default=None,
                   help="API base url for generation (default: same as --base-url)")
    p.add_argument("--gen-api-key", default=None,
                   help="API key for generation (default: same as --api-key)")
    p.add_argument("--gen-max-tokens", type=int, default=4096,
                   help="Max generation tokens for generation stage")
    # Runtime parameters
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=4,
                   help="Number of parallel threads")
    p.add_argument("--datasets", nargs="*", default=None,
                   help="Only process specified datasets (default: ResearchPlanGen_ml)")
    p.add_argument("--limit", type=int, default=None,
                   help="Max samples per dataset")
    p.add_argument("--force", action="store_true",
                   help="Ignore existing results, force regeneration")
    p.add_argument("--dry-run", action="store_true",
                   help="Do not call API, use mock responses to test workflow")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    records = load_records(args)
    print(f"[load] {len(records)} samples to process.", flush=True)

    results = []
    skipped = 0
    errors = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(process_record, rec, args, args.output_dir, args.force): rec
            for rec in records
        }
        for fut in as_completed(futures):
            rec = futures[fut]
            try:
                res, is_new = fut.result()
                results.append(res)
                if not is_new:
                    skipped += 1
                print(
                    f"[{'done' if is_new else 'skip'}] {rec['dataset']}/{rec['id']} "
                    f"items={res['parsed']['num_items']} "
                    f"passed={res['parsed']['num_passed']}",
                    flush=True,
                )
            except Exception as e:  # pylint: disable=broad-except
                errors += 1
                print(f"[error] {rec['dataset']}/{rec['id']}: {e}", flush=True)

    summary = summarize(results, args.output_dir)
    print(
        f"\n[summary] Processed {len(results)} items ({skipped} reused, {errors} failed)",
        flush=True,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[output] Intermediate results: {args.output_dir}/grades, {args.output_dir}/plans")
    print(f"[output] Summary: {args.output_dir}/summary.json, all_results.json")


if __name__ == "__main__":
    main()
