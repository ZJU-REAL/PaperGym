import logging
logger = logging.getLogger(__name__)

import os
from pathlib import Path
import pandas as pd
import json
from slugify import slugify
from tqdm import tqdm
import shutil
import re
import argparse

# Import LLM utilities
from utils import Paths, LLM

# Global variable for LLM instance
judge_relative = None

def extract_judgement_and_justification(output_str):
    """
    Extracts the judgement and justification from a formatted output string.

    Parameters:
        output_str (str): The full model output string.

    Returns:
        dict: A dictionary with keys 'judgement' and 'justification'.
              If parsing fails, returns {'judgement': None, 'justification': None}.
    """
    # Extract judgement using regex
    judgement_match = re.search(r'\*\*output:\s*\{"judgement":\s*"([^"]+)"\s*\}\*\*', output_str)
    
    # Extract justification: assume it starts right after 'Justification:' and continues to the end
    justification_match = re.search(r'Justification:\s*(.+)', output_str, re.DOTALL)

    if judgement_match and justification_match:
        score = judgement_match.group(1).strip()
        feedback = justification_match.group(1).strip()
        return score, feedback
    
    return None, None


def remove_markdown_links(text):
    """
    Remove markdown links from text where the link text is specifically "link".
    
    Args:
        text: Input text that may contain markdown links
        
    Returns:
        Text with markdown links removed where link text is "link"
    """
    if not isinstance(text, str):
        return text
    
    # Pattern to match markdown links where the link text is specifically "link": [link](url)
    # This will replace [link](url) with empty string
    pattern = r'\[link\]\([^)]+\)'
    cleaned_text = re.sub(pattern, '', text)
    
    return cleaned_text

def initialize_gpt_judge(judge_model: str = "gpt-4.1-mini"):
    """Initialize GPT judge for relative comparison."""
    global judge_relative
    if judge_relative is None:
        logger.info(f"Initializing GPT-based judge with model: {judge_model}")
        judge_relative = LLM(
            prompt=Paths.prompt_folder / 'response_preference' / 'response_pairwise_judge_binary.prompt', # Only A or B 
            markers=['[QUERY]', '[RESPONSE_1]', '[RESPONSE_2]'],
            provider='gpt',
            model=judge_model
        )
        logger.info("GPT-based judge initialization successful")

def get_pairwise_score(query, response_A, response_B):
    """Get pairwise comparison score for a single query-response pair using GPT."""
    # Clean responses by removing markdown links
    cleaned_response_A = remove_markdown_links(response_A)
    cleaned_response_B = remove_markdown_links(response_B)
    
    # Call the relative judge
    result = judge_relative.call([query, cleaned_response_A, cleaned_response_B])
    result = result.replace('Response 1', 'Response A').replace('Response 2', 'Response B')
    
    # Parse the result to extract preference
    score, feedback = extract_judgement_and_justification(result)
    score = score.replace('Response ', '')
    
    return score, feedback

def get_answer_by_id_and_implementation(id, field, implementation):
    """Get answer from response file based on ID, field, and implementation."""
    field_folder = Path("/nlp/data/academic_pdfs/multi_domain_testbed/systems_comparison") / slugify(field)
    mode = implementation.split('-')[0]
    provider = '-'.join(implementation.split('-')[1:])
    response_path = field_folder / mode / f"{provider}.json"
    with open(response_path, "r", encoding="utf-8") as f:
        response_results = json.load(f)
    response_map = {item['corpusid_sectionid']: item for item in response_results}
    return response_map[id]['answer']

def run_judge_relative_by_gpt(general_domain, tournament_csv_path, output_dir, judge_model="gpt-4.1-mini"):
    """
    Run relative judging of responses using GPT pairwise comparison.
    
    Args:
        general_domain: The general domain to process (e.g., "Economics & Business")
        tournament_csv_path: Path to the tournament CSV file
        output_dir: Directory to save the output results
        judge_model: GPT model to use for judging
    """
    # Initialize GPT judge
    initialize_gpt_judge(judge_model)
    
    test_split_path = Path("/nlp/data/academic_pdfs/multi_domain_testbed/data_release/researchqa/test.json")
    tournament_csv_path = Path(tournament_csv_path)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Define output file path
    out_path = output_dir / f"{slugify(general_domain)}.json"
    logger.info(f"Will save relative judge results to {out_path}...")

    with open(test_split_path, "r", encoding="utf-8") as f:
        test_split = json.load(f)

    with open(tournament_csv_path, "r", encoding="utf-8") as f:
        tournament_csv = pd.read_csv(f)

    id_to_data = {item['id']: item for item in test_split}    
    tournament_data = tournament_csv[tournament_csv['general_domain'] == general_domain]
    
    # Load existing results if any
    results = []
    existing_comparisons = set()
    if out_path.exists():
        # Create backup of existing file
        backup_path = out_path.with_name(f"{out_path.stem}-backup.json")
        shutil.copy2(out_path, backup_path)
        with open(backup_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        # Create set of existing comparisons to avoid duplicates
        existing_comparisons = {(item['session_id'], item['implementation_A'], item['implementation_B']) 
                               for item in results}
    
    # Process each comparison
    for index, row in tqdm(tournament_data.iterrows(), total=len(tournament_data), desc=f"Processing {general_domain} comparisons"):
        session_id = row['session_id'] 
        implementation_A = row['implementation_A']
        implementation_B = row['implementation_B']
        
        # Skip if already processed
        if (session_id, implementation_A, implementation_B) in existing_comparisons:
            continue
            
        data = id_to_data[session_id]
        query = data['query']
        field = data['field']
        
        try:
            response_A = get_answer_by_id_and_implementation(session_id, field, implementation_A)
            response_B = get_answer_by_id_and_implementation(session_id, field, implementation_B)
            
            # Get pairwise scores
            score, feedback = get_pairwise_score(query, response_A, response_B)
            score_swapped, feedback_swapped = get_pairwise_score(query, response_B, response_A)
            
            # Handle swapping logic for swapped score
            swap_r1r2_score = lambda x: 'B' if x == 'A' else 'A' if x == 'B' else None
            swap_r1r2_feedback = lambda x: x.replace('Response A', 'TEMP_PLACEHOLDER').replace('Response B', 'Response A').replace('TEMP_PLACEHOLDER', 'Response B')
            score_swapped = swap_r1r2_score(score_swapped)
            feedback_swapped = swap_r1r2_feedback(feedback_swapped)
            
            new_entry = {
                "session_id": session_id,
                "implementation_A": implementation_A,
                "implementation_B": implementation_B,
                "query": query,
                "response_A": response_A,
                "response_B": response_B,
                "judge": score,
                "judge_feedback": feedback,
                "judge_swapped": score_swapped,
                "judge_swapped_feedback": feedback_swapped
            }
            results.append(new_entry)
            existing_comparisons.add((session_id, implementation_A, implementation_B))
            
            # Save after each comparison
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.warning(f"Failed to get responses for session {session_id}, implementation_A {implementation_A}, implementation_B {implementation_B}")
            logger.warning(f"Error: {str(e)}")
            continue
    
    logger.info(f"Completed relative judging for {general_domain}. Total comparisons: {len(results)}")

def run_judge_relative_by_gpt_main():
    """Main function to handle command-line arguments and run GPT relative judging."""
    parser = argparse.ArgumentParser(description="Run GPT judging for systems comparison")
    parser.add_argument('--general_domains', type=str, nargs='+', required=True,
                       help='List of general domains to process for relative judging')
    parser.add_argument('--tournament_csv_path', type=str, required=True,
                       help='Path to the tournament CSV file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save the output results')
    parser.add_argument('--judge_model', type=str, default='gpt-4.1-mini',
                       help='GPT model to use for judging (default: gpt-4.1-mini)')
    
    args = parser.parse_args()
    
    print(f"Running GPT relative judging for {len(args.general_domains)} general domains")
    print(f"Tournament CSV path: {args.tournament_csv_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Judge model: {args.judge_model}")
    print("=" * 80)
    
    for i, domain in enumerate(args.general_domains, 1):
        print(f"\nProcessing domain {i}/{len(args.general_domains)}: {domain}")
        print("-" * 60)
        try:
            run_judge_relative_by_gpt(
                general_domain=domain,
                tournament_csv_path=args.tournament_csv_path,
                output_dir=args.output_dir,
                judge_model=args.judge_model,
            )
            print(f"✓ Completed relative judging for {domain}")
        except Exception as e:
            print(f"✗ Error processing {domain}: {str(e)}")
            logger.error(f"Error processing domain {domain}: {str(e)}")
            continue
    
    print("\n" + "=" * 80)
    print("Completed relative judging for all domains")


if __name__ == "__main__":
    run_judge_relative_by_gpt_main() 