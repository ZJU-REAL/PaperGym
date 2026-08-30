import os
import re
import time
import requests
from bs4 import BeautifulSoup

# -------------------------- Configuration Parameters --------------------------
# Target list page URL template ({year} and {month} will be replaced)
LIST_URL_TEMPLATE = "https://arxiv.org/list/cs/{year}-{month:02d}?skip=0&show=250"
# Local save directory
SAVE_DIR = "/path/to/arxiv_latex_sources/cs"
# Request headers (mimic a browser to avoid being blocked)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
# Request delay (seconds): comply with arXiv crawler rules to avoid frequent requests
DELAY = 1
# Download time range: start year-month ~ end year-month (inclusive)
START_YEAR, START_MONTH = 2025, 6
END_YEAR, END_MONTH = 2025, 12
# Network request timeout (seconds)
REQUEST_TIMEOUT = 20
# Maximum retry count
MAX_RETRIES = 1
# Retry interval (seconds)
RETRY_DELAY = 5
# -------------------------------------------------------------

def get_arxiv_paper_ids(list_url):
    """
    Extract all paper arXiv IDs from the list page (with timeout and retry mechanism)
    :param list_url: Paper list page URL
    :return: List of paper IDs
    """
    for attempt in range(1, MAX_RETRIES + 4):
        try:
            print(f"  Requesting the listing page (attempt {attempt})...", flush=True)
            response = requests.get(list_url, headers=HEADERS, timeout=REQUEST_TIMEOUT+30)
            response.raise_for_status()  # Raise an HTTP exception if an error occurred
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Use regex to match all paper abs links and extract IDs
            id_pattern = re.compile(r"^/abs/(\d+\.\d+)")
            paper_ids = []
            
            # Iterate over all a tags and filter paper IDs
            for a_tag in soup.find_all("a", href=True):
                match = id_pattern.match(a_tag["href"])
                if match:
                    paper_id = match.group(1)
                    if paper_id not in paper_ids:  # deduplicate
                        paper_ids.append(paper_id)
            
            print(f"[OK] Extracted {len(paper_ids)} papers", flush=True)
            return paper_ids
        
        except requests.exceptions.Timeout as e:
            print(f"[WARN] Request timed out (attempt {attempt}/{MAX_RETRIES}): {str(e)}", flush=True)
        except requests.exceptions.ConnectionError as e:
            print(f"[WARN] Connection error (attempt {attempt}/{MAX_RETRIES}): {str(e)}", flush=True)
        except requests.exceptions.HTTPError as e:
            print(f"[WARN] HTTP error (attempt {attempt}/{MAX_RETRIES}): {str(e)}", flush=True)
        except Exception as e:
            print(f"[WARN] Unknown error (attempt {attempt}/{MAX_RETRIES}): {str(e)}", flush=True)
        
        if attempt < MAX_RETRIES:
            print(f"  Waiting {RETRY_DELAY}s before retrying...", flush=True)
            time.sleep(RETRY_DELAY)
    
    print(f"[ERROR] Failed permanently to fetch the paper list (retried {MAX_RETRIES} times)", flush=True)
    return []

def download_tex_source(paper_id, save_dir):
    """
    Download the TeX source of a single paper (with timeout and retry mechanism)
    :param paper_id: arXiv paper ID
    :param save_dir: Save directory
    """
    # arXiv TeX source code fixed download link
    download_url = f"https://arxiv.org/e-print/{paper_id}"
    save_path = os.path.join(save_dir, f"{paper_id}.tar.gz")

    # Skip files already downloaded
    if os.path.exists(save_path):
        print(f"[SKIP] {paper_id} already exists, skipping download", flush=True)
        return

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Streaming download (suitable for large files)
            with requests.get(download_url, headers=HEADERS, stream=True, timeout=REQUEST_TIMEOUT) as r:
                r.raise_for_status()
                with open(save_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            print(f"[OK] Downloaded {paper_id}.tar.gz", flush=True)
            return
        
        except requests.exceptions.Timeout as e:
            print(f"[WARN] Download timed out for {paper_id} (attempt {attempt}/{MAX_RETRIES}): {str(e)}", flush=True)
        except requests.exceptions.ConnectionError as e:
            print(f"[WARN] Connection error for {paper_id} (attempt {attempt}/{MAX_RETRIES}): {str(e)}", flush=True)
        except Exception as e:
            print(f"[WARN] Download failed for {paper_id} (attempt {attempt}/{MAX_RETRIES}): {str(e)}", flush=True)
        
        if attempt < MAX_RETRIES:
            print(f"  Waiting {RETRY_DELAY}s before retrying...", flush=True)
            time.sleep(RETRY_DELAY)
    
    print(f"[ERROR] Download failed permanently for {paper_id} (retried {MAX_RETRIES} times)", flush=True)

def generate_year_months(start_year, start_month, end_year, end_month):
    """
    Generate all (year, month) tuples from the start year-month to the end year-month
    """
    result = []
    year, month = start_year, start_month
    while (year < end_year) or (year == end_year and month <= end_month):
        result.append((year, month))
        month += 1
        if month > 12:
            month = 1
            year += 1
    return result


def main():
    # Create the save directory
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 1. Iterate over all year-months and collect all paper IDs
    all_paper_ids = []
    year_months = generate_year_months(START_YEAR, START_MONTH, END_YEAR, END_MONTH)
    
    print(f"Iterating paper listings from {START_YEAR}-{START_MONTH:02d} to {END_YEAR}-{END_MONTH:02d}...", flush=True)
    for year, month in year_months:
        list_url = LIST_URL_TEMPLATE.format(year=year, month=month)
        print(f"\n[FETCH] Fetching the paper list for {year}-{month:02d}...", flush=True)
        paper_ids = get_arxiv_paper_ids(list_url)
        for pid in paper_ids:
            if pid not in all_paper_ids:
                all_paper_ids.append(pid)
        print(f"   Paper IDs collected so far: {len(all_paper_ids)}", flush=True)
        time.sleep(DELAY)  # delay between page requests
    
    if not all_paper_ids:
        print("No papers found, exiting", flush=True)
        return
    
    print(f"\n[SUMMARY] Collected {len(all_paper_ids)} papers in total", flush=True)
    
    # 2. Batch download TeX source code
    print("Downloading TeX sources...", flush=True)
    for idx, paper_id in enumerate(all_paper_ids, 1):
        print(f"\n[{idx}/{len(all_paper_ids)}]", flush=True)
        download_tex_source(paper_id, SAVE_DIR)
        time.sleep(DELAY)  # enforce a delay to comply with crawler rules
    
    print(f"\n[DONE] All tasks completed. Sources saved to: {os.path.abspath(SAVE_DIR)}", flush=True)

if __name__ == "__main__":
    main()