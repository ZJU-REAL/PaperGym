"""
Query the category of a paper via the arXiv API (e.g., cs.CL / cs.AI / cs.LG, etc.),
then merge the query result with the original JSON and write it to the target directory.

No longer relies on a local cache JSON file: before querying, it first checks whether the
corresponding output file already exists. If it exists, it is skipped (unless --force is used),
so the query results are not cached as JSON either.

Usage:
    python query_category.py                          # Query and merge all papers (skip existing outputs)
    python query_category.py --paper-id 1501.00001     # Query the category of a single paper
    python query_category.py --force                   # Force re-query and overwrite existing outputs
"""

import os
import re
import json
import time
import argparse
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

# ---------- Path Configuration ----------
SOURCE_DIR = "/path/to/method_qa_pairs/econ"
OUTPUT_DIR = "/path/to/data/econ"

# ---------- arXiv API Parameters ----------
ARXIV_API_URL = "https://export.arxiv.org/api/query"
REQUEST_DELAY = 10       # Interval between API requests (seconds), comply with arXiv rules
MAX_RETRIES = 1
RETRY_DELAY = 5

# ---------- Namespaces ----------
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def extract_paper_id(filename: str) -> str | None:
    """Extract the arXiv ID from a filename (e.g., 1501.00001_merged_body.json)"""
    match = re.match(r"(\d{4}\.\d{4,5})", filename)
    return match.group(1) if match else None


def fetch_arxiv_categories(paper_id: str) -> dict | None:
    """
    Query the metadata of a single paper via the arXiv API and extract the primary category
    and all categories.
    Returns dict: {"primary_category": "cs.CL", "categories": ["cs.CL", "cs.AI", ...]}
    """
    url = f"{ARXIV_API_URL}?id_list={paper_id}"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (compatible; paper-rl/1.0; mailto:research@example.com)"
            })
            with urllib.request.urlopen(req, timeout=30) as resp:
                xml_data = resp.read().decode("utf-8")

            root = ET.fromstring(xml_data)
            entries = root.findall("atom:entry", NS)
            if not entries:
                print(f"  [WARN] {paper_id}: API returned no entries", flush=True)
                return None

            entry = entries[0]

            # Extract the primary category
            primary_cat_elem = entry.find("arxiv:primary_category", NS)
            primary_category = primary_cat_elem.get("term") if primary_cat_elem is not None else None

            # Extract all categories
            categories = []
            for cat_elem in entry.findall("atom:category", NS):
                term = cat_elem.get("term")
                if term:
                    categories.append(term)

            # Extract the title for reference (optional)
            title_elem = entry.find("atom:title", NS)
            title = title_elem.text.strip().replace("\n", " ").replace("  ", " ") if title_elem is not None and title_elem.text else ""

            result = {
                "primary_category": primary_category,
                "categories": categories,
                "title": title,
            }
            print(f"  [OK] {paper_id}: primary={primary_category}, categories={categories}", flush=True)
            return result

        except urllib.error.HTTPError as e:
            print(f"  [WARN] {paper_id} HTTP error (attempt {attempt}): {e.code}", flush=True)
        except urllib.error.URLError as e:
            print(f"  [WARN] {paper_id} network error (attempt {attempt}): {e.reason}", flush=True)
        except ET.ParseError as e:
            print(f"  [WARN] {paper_id} XML parse error (attempt {attempt}): {e}", flush=True)
        except Exception as e:
            print(f"  [WARN] {paper_id} unknown error (attempt {attempt}): {e}", flush=True)

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY)

    print(f"  [ERROR] {paper_id}: failed permanently (retried {MAX_RETRIES} times)", flush=True)
    return None


def collect_paper_ids_from_dir(directory: str) -> list[str]:
    """Scan all JSON files in the directory and extract paper IDs (deduplicated and sorted)"""
    paper_ids = set()
    for fname in os.listdir(directory):
        if fname.endswith("_merged_body.json"):
            pid = extract_paper_id(fname)
            if pid:
                paper_ids.add(pid)
    return sorted(paper_ids)


def merge_one_paper(pid: str, cat_info: dict) -> bool:
    """
    Merge the category information of a single paper with the original JSON and write it to OUTPUT_DIR.
    In the merged JSON, primary_category, categories, and title are placed at the beginning of the file.
    Returns whether the merge succeeded.
    """
    src_path = os.path.join(SOURCE_DIR, f"{pid}_merged_body.json")
    dst_path = os.path.join(OUTPUT_DIR, f"{pid}_merged_body.json")

    if not os.path.exists(src_path):
        print(f"  [WARN] {pid}: source file does not exist, skipping", flush=True)
        return False

    with open(src_path, "r", encoding="utf-8") as f:
        src_data = json.load(f)

    # Build a new JSON with category information placed at the beginning
    merged = {
        "primary_category": cat_info.get("primary_category"),
        "categories": cat_info.get("categories", []),
        "title": cat_info.get("title", ""),
        "arxiv_id": pid,
    }
    # Append the other fields of the original JSON in their original order (skip processed keys)
    for key, value in src_data.items():
        if key == "filename":
            continue
        merged[key] = value

    with open(dst_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"  [OK] {pid}: merged and written to {dst_path}", flush=True)
    return True


def batch_query_and_merge(paper_ids: list[str], force: bool = False):
    """
    Batch query paper categories and merge them into the output.
    Decides whether to call the arXiv API by checking whether the corresponding output file exists.
    :param paper_ids: List of paper IDs
    :param force: Whether to force a re-query (ignore existing output files)
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total = len(paper_ids)
    success = 0
    skipped = 0
    fail = 0
    query_count = 0

    for idx, pid in enumerate(paper_ids, 1):
        dst_path = os.path.join(OUTPUT_DIR, f"{pid}_merged_body.json")

        # Decide whether to query based on whether the output file exists
        if not force and os.path.exists(dst_path):
            print(f"  [SKIP] [{idx}/{total}] {pid} output already exists, skipping", flush=True)
            skipped += 1
            continue

        # Comply with the rate-limiting rules before requesting the arXiv API
        if query_count > 0:
            time.sleep(REQUEST_DELAY)
        query_count += 1

        print(f"  [QUERY] [{idx}/{total}] querying {pid} ...", flush=True)
        result = fetch_arxiv_categories(pid)
        # When the result is unavailable (None), do not store the corresponding JSON file; leave it for a later retry
        if not result:
            print(f"  [ERROR] [{idx}/{total}] {pid}: no category information retrieved, skipping write (not stored)", flush=True)
            fail += 1
            continue

        if merge_one_paper(pid, result):
            success += 1
        else:
            fail += 1

    print(f"\n[SUMMARY] Done: {success} succeeded, {skipped} skipped, {fail} not retrieved (not stored)", flush=True)
    print(f"[OUTPUT] Output directory: {OUTPUT_DIR}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Query paper categories via the arXiv API and merge them into the output directory")
    parser.add_argument("--paper-id", type=str, help="ID of a single paper to query (e.g. 1501.00001)")
    parser.add_argument("--force", action="store_true", help="Force re-query and overwrite existing output files")
    parser.add_argument("--no-delay", action="store_true", help="Disable the delay between API requests (not recommended)")
    args = parser.parse_args()

    global REQUEST_DELAY

    if args.no_delay:
        REQUEST_DELAY = 0

    # Single-paper query mode
    if args.paper_id:
        print(f"[QUERY] Querying a single paper: {args.paper_id}", flush=True)
        result = fetch_arxiv_categories(args.paper_id)
        if result:
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            merge_one_paper(args.paper_id, result)
        return

    # Batch query mode
    print(f"[SCAN] Scanning directory: {SOURCE_DIR}", flush=True)
    paper_ids = collect_paper_ids_from_dir(SOURCE_DIR)
    print(f"[SCAN] Found {len(paper_ids)} papers in total", flush=True)

    if not args.force:
        pending = [
            pid for pid in paper_ids
            if not os.path.exists(os.path.join(OUTPUT_DIR, f"{pid}_merged_body.json"))
        ]
        print(f"[PENDING] {len(pending)} to process ({len(paper_ids) - len(pending)} already have output)", flush=True)
    else:
        pending = paper_ids
        print(f"[PENDING] Force mode: {len(pending)} to process", flush=True)

    if pending:
        print("[START] Starting batch query and merge...", flush=True)
        batch_query_and_merge(paper_ids, force=args.force)
    else:
        print("[DONE] All papers already have output, nothing to query", flush=True)


if __name__ == "__main__":
    main()
