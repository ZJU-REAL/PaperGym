import os
import json
import re
import sys
import argparse
from openai import OpenAI
from typing import List, Dict, Optional, Tuple
from multiprocessing import Pool, cpu_count
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
import threading


# Thread-local storage for LLM clients
_thread_local = threading.local()


def get_llm_client(model: str = "qwen3-32b-1"):
    """
    Get the thread-specific LLM client instance, using threading.local() to ensure thread safety.
    Each thread will have its own client connection to avoid concurrent access conflicts.
    
    Args:
        model (str): The LLM model name to use
        
    Returns:
        LLMClient: The client instance for the current thread
    """
    # Use thread-local storage so that each thread has its own client
    if not hasattr(_thread_local, 'clients'):
        _thread_local.clients = {}
    
    if model not in _thread_local.clients:
        _thread_local.clients[model] = LLMClient(model=model)
    
    return _thread_local.clients[model]


# LLM API Client
class LLMClient:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = "qwen3-32b-1", timeout: int = 600):
        """
        Initialize the LLM client with API key and base URL
        """
        if api_key is None:
            api_key = os.getenv("OPENSCIENCE_API_KEY", "")
        if base_url is None:
            base_url = os.getenv("OPENSCIENCE_BASE_URL", "http://your-api/v1")
        
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model = model

    def generate_response(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7, timeout: int = None) -> str:
        """
        Generate response from the LLM based on the given prompt
        """
        try:
            # Use the passed-in timeout parameter; fall back to the client's default timeout if not provided
            request_timeout = timeout if timeout is not None else self.client.timeout
            
            chat_completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": f"{prompt}"}],
                # max_tokens=max_tokens,
                temperature=temperature,
                # extra_body={
                #     "chat_template_kwargs": {"enable_thinking": False},
                # },
                timeout=request_timeout,
            )
            raw_content = chat_completion.choices[0].message.content
            cleaned_content = clean_llm_output(raw_content)
            return cleaned_content
        except Exception as e:
            print(f"[Process: {os.getpid()}] Error generating response: {str(e)}", flush=True)
            return ""


# def clean_llm_output(text: str) -> str:
#     """
#     Remove specific tags and their content from LLM output.
#     Specifically removes content within <think>...<think> and <think>...</think> tags
#     from the LLM output, which are often used as processing markers but should not
#     be included in the final response. The function preserves any text outside these tags.

#     Args:
#         text (str): The raw output text from the LLM

#     Returns:
#         str: Cleaned text with specified tags and their content removed
#     """
#     # Remove <think>...</think> tags and their content
#     text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
#     text = re.sub(r'<think>.*?<think>', '', text, flags=re.DOTALL)
#     return text.strip()


def clean_llm_output(text: str) -> str:
    if not isinstance(text, str):
        return ""
    
    # Find the position of the </think> tag
    think_end_tag = "</think>"
    tag_index = text.find(think_end_tag)
    
    if tag_index != -1:
        # Extract the content after the tag and strip leading/trailing whitespace
        cleaned_text = text[tag_index + len(think_end_tag):].strip()
        return cleaned_text
    else:
        # If the tag is not found, return the original text (stripped of leading/trailing whitespace)
        return text.strip()


def read_paper_bodies(paper_dir: str) -> List[Dict[str, str]]:
    """
    Read all paper body files from the given directory
    """
    papers = []
    for filename in os.listdir(paper_dir):
        if filename.endswith('.tex'):
            filepath = os.path.join(paper_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                papers.append({
                    'filename': filename,
                    'content': content
                })
    return papers


def _filter_and_combine_extracted_info(extracted_list: List[Optional[str]]) -> str:
    """Filter out empty/None entries and combine the valid extracted information into a formatted string."""
    valid_infos = [info for info in extracted_list if info is not None and info.strip()]
    if not valid_infos:
        return ""
    return "\n\n".join([f"Extracted Part {i+1}:\n{info}" for i, info in enumerate(valid_infos)])


def process_single_section_all_types(section_content: str, llm_client: LLMClient, section_idx: int, paper_filename: str) -> Optional[Dict[str, str]]:
    """Process a single section, extracting four types of information at once: research field,
    research background, solution, and experimental design.
    Introduces a JSON fault-tolerant parsing mechanism specifically for LaTeX math symbols."""
    if not section_content.strip():
        print(f"[Thread: {threading.get_ident()}] Empty section {section_idx} in {paper_filename}", flush=True)
        return None
        
    # Combine the prompts for the four types of information extraction into one call
    combined_extraction_prompt = f"""You are an expert academic researcher.
Analyze the following paper section and extract information according to four categories.

[Paper Section Content]:
{section_content}

[Four Categories to Extract]:
1. **research_field**: The specific academic discipline, sub-field, and focused research direction of the paper.
2. **research_background**: Current state of the field, historical/related work, and limitations/gaps of prior works.
3. **solution**: Core idea, motivation, methodology, technical means, model architecture, and key components proposed by the authors of THIS paper.
4. **experiment**: Experimental design methodology, datasets, baselines, evaluation metrics, implementation details, and analysis strategies.

[Critical Rules]:
1. **STRICTLY extract from original text**: You MUST only extract information that appears in the provided section content. DO NOT fabricate, infer, or add any content not explicitly present in the text. If you cannot find sentence-level evidence in the section, do not extract it.
2. For each category, if relevant information exists in the section, extract and concisely summarize the key points. If NO relevant information exists, set the field to an empty string "".
3. The extracted information should be in the original language of the paper.
4. You MUST output your response in STRICT, valid JSON format.
5. CRITICAL: If your output contains LaTeX mathematical symbols (e.g., \\sum, \\mathbb), you MUST double-escape the backslashes in the JSON string (e.g., use \\\\sum, \\\\mathbb).
6. Do not include any conversational text, reasoning, or thinking process.
7. ABSOLUTE PROHIBITION: DO NOT output `<think>` tags, step-by-step explanations, or any reasoning.

[Output Format]:
Please wrap your output in ```json ``` blocks exactly like this:
```json
{{
    "research_field": "extracted research field info or empty string",
    "research_background": "extracted research background info or empty string",
    "solution": "extracted solution info or empty string",
    "experiment": "extracted experiment info or empty string"
}}
```"""
    try:
        response = llm_client.generate_response(combined_extraction_prompt, max_tokens=4096, temperature=0.1)
        if not response:
            print(f"[Thread: {threading.get_ident()}] No response for section {section_idx} in {paper_filename}", flush=True)
            return None
            
        json_str = response.strip()
        
        # Extract the JSON string
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', json_str, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            generic_match = re.search(r'```\s*(\{.*?\})\s*```', json_str, re.DOTALL)
            if generic_match:
                json_str = generic_match.group(1)
            else:
                bracket_match = re.search(r'\{.*?\}', json_str, re.DOTALL)
                if bracket_match:
                    json_str = bracket_match.group(0)
                    
        # Regex preprocessing to fix LaTeX escaping issues
        json_str = re.sub(r'(?<!\\)\\(?![\\nrtbf"u/])', r'\\\\', json_str)
                    
        result_dict = json.loads(json_str)
        return {
            "research_field": result_dict.get("research_field") or "",
            "research_background": result_dict.get("research_background") or "",
            "solution": result_dict.get("solution") or "",
            "experiment": result_dict.get("experiment") or "",
        }
    except json.JSONDecodeError as e:
        print(f"[Error] JSON parse failed for section {section_idx} in {paper_filename}. Error: {str(e)}\nRaw Output: {response}", flush=True)
        return None
    except Exception as e:
        print(f"[Error] Failed to process section {section_idx} in {paper_filename}: {str(e)}", flush=True)
        return None


def extract_all_info_from_sections(
    paper_filename: str, 
    llm_client: LLMClient,
    base_dir: str = "/path/to/paper_sections/cs",
) -> Optional[Dict[str, List[str]]]:
    """
    Sequentially process all sections of a paper, extracting four types of information at once
    (research field, research background, solution, experimental design).
    Each section calls the LLM only once, extracting all four types of information simultaneously.
    
    Returns:
        Optional[Dict[str, List[str]]]: A dict with four keys, each corresponding to a list
        whose length equals the number of sections, with each element being an extracted string.
        Keys: "research_field", "research_background", "solution", "experiment"
        Returns None if any section fails to be processed.
    """
    try:
        json_filename = os.path.splitext(paper_filename)[0] + "_divided.json"
        json_path = os.path.join(base_dir, json_filename)
        
        if not os.path.exists(json_path):
            print(f"[Thread: {threading.get_ident()}] JSON file not found: {json_path}", flush=True)
            return None
            
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            
        if 'sections' not in json_data or not isinstance(json_data['sections'], list):
            print(f"[Thread: {threading.get_ident()}] Invalid JSON structure in: {json_path}", flush=True)
            return None
            
        sections = json_data['sections']
        num_sections = len(sections)
        
        # Initialize lists for the four types of information, each with length equal to the number of sections
        research_field_list = [""] * num_sections
        research_background_list = [""] * num_sections
        solution_list = [""] * num_sections
        experiment_list = [""] * num_sections
        
        # Process each section sequentially in a single thread
        for i, content in enumerate(sections):
            result_dict = process_single_section_all_types(content, llm_client, i, paper_filename)
            if result_dict is None:
                # A section failed to be processed; abort the entire paper
                print(f"[Thread: {threading.get_ident()}] Section {i} for {paper_filename} returned None. Aborting entire paper.", flush=True)
                return None
            
            # Extract the four types of information from the result dict
            research_field_list[i] = result_dict.get("research_field") or ""
            research_background_list[i] = result_dict.get("research_background") or ""
            solution_list[i] = result_dict.get("solution") or ""
            experiment_list[i] = result_dict.get("experiment") or ""
                    
        return {
            "research_field": research_field_list,
            "research_background": research_background_list,
            "solution": solution_list,
            "experiment": experiment_list
        }
        
    except Exception as e:
        print(f"[Thread: {threading.get_ident()}] Fatal Error processing {paper_filename}: {str(e)}", flush=True)
        return None


def generate_research_field(llm_client: LLMClient, paper_filename: str, extracted_research_field_list: List[str]) -> str:
    """
    Integrate the extracted research field information (Reduce stage) to generate the paper's research field description.
    
    Args:
        llm_client: LLM client
        paper_filename: Paper filename (for logging)
        extracted_research_field_list: List of research field information extracted from each section
    """
    if not paper_filename:
        print(f"[Thread: {threading.get_ident()}] No paper filename provided.", flush=True)
        return ""
    
    # Filter and combine valid information
    combined_info = _filter_and_combine_extracted_info(extracted_research_field_list)
    
    if not combined_info:
        print(f"[Thread: {threading.get_ident()}] No relevant info found for {paper_filename}", flush=True)
        return ""
    
    # Reduce stage - structured summarization
    field_prompt = f"""
    You are an expert academic taxonomy specialist.
    Based on the extracted information from various sections of a paper, synthesize a highly concise description of its Research Field.
    Extracted Information:
    {combined_info}
    Task Requirements:
    1. Define the Broad Discipline (e.g., Computer Science, Biology).
    2. Define the Specific Sub-field (e.g., Natural Language Processing, Reinforcement Learning).
    3. Define the Focused Research Direction.
    4. EXCLUDE any background context, proposed methods, or experimental details.
    
    Please output a concise summary in 1-3 sentences directly. Do not use filler words like "The research field of this paper is...".
    """
    try:
        research_field = llm_client.generate_response(field_prompt, max_tokens=1024, temperature=0.1)
        return research_field.strip()
    except Exception as e:
        print(f"[Thread: {threading.get_ident()}] Error generating research field for {paper_filename}: {str(e)}", flush=True)
        return ""


def generate_research_background(llm_client: LLMClient, paper_filename: str, extracted_background_list: List[str]) -> str:
    """
    Integrate the extracted research background information (Reduce stage) to generate the paper's research background description.
    
    Args:
        llm_client: LLM client
        paper_filename: Paper filename (for logging)
        extracted_background_list: List of research background information extracted from each section
    """
    if not paper_filename:
        print(f"[Thread: {threading.get_ident()}] No paper filename provided.", flush=True)
        return ""
    
    # Filter and combine valid information
    combined_info = _filter_and_combine_extracted_info(extracted_background_list)
    
    if not combined_info:
        print(f"[Thread: {threading.get_ident()}] Warning: No relevant info found for {paper_filename}. Returning empty string.", flush=True)
        return ""
    
    # Reduce stage - structured summarization
    initial_prompt = f"""
    You are an expert academic paper reviewer.
    Based on the following extracted information from various sections of a paper, comprehensively summarize the 'Research Background'.
    Extracted Information:
    {combined_info}
    Task Requirements:
    Please synthesize the information and structure your response strictly into the following three parts:
    1. **Field Overview**: Briefly introduce the current state and background of the research field.
    2. **Historical Work**: Summarize the existing methods, related works, and their characteristics.
    3. **Core Problems (Gaps)**: Clearly identify the limitations of existing works and the specific problems this paper aims to solve.
    Note: Focus ONLY on the background. Do not describe the specific novel solutions or experimental results proposed in this paper. Make the summary academic, coherent, and concise.
    """
    try:
        initial_background = llm_client.generate_response(initial_prompt, max_tokens=4096, temperature=0.2)
        if not initial_background or not initial_background.strip():
            print(f"[Thread: {threading.get_ident()}] Warning: No valid background found for {paper_filename}. Returning empty string.", flush=True)
            return ""
        return initial_background.strip()
    except Exception as e:
        print(f"[Thread: {threading.get_ident()}] Error generating research background for {paper_filename}: {str(e)}", flush=True)
        return ""


def generate_solution(llm_client: LLMClient, paper_filename: str, extracted_solution_list: List[str]) -> str:
    """
    Integrate the extracted solution information (Reduce stage) to generate the paper's solution description.
    
    Args:
        llm_client: LLM client
        paper_filename: Paper filename (for logging)
        extracted_solution_list: List of solution information extracted from each section
    """
    if not paper_filename:
        print(f"[Thread: {threading.get_ident()}] Warning: No paper filename provided. Returning empty string.", flush=True)
        return ""
    
    # Filter and combine valid information
    combined_info = _filter_and_combine_extracted_info(extracted_solution_list)
    
    if not combined_info:
        print(f"[Thread: {threading.get_ident()}] Warning: No valid info found for {paper_filename}. Returning empty string.", flush=True)
        return ""
    
    # Reduce stage - structured output
    solution_prompt = f"""
    You are an expert academic paper reviewer.
    Based on the following extracted methodological details from a paper, comprehensively describe the proposed 'Solution'.
    Extracted Information:
    {combined_info}
    Task Requirements:
    Please organize your detailed description strictly using the following 4 dimensions (Use Markdown formatting):
    
    1. **Core Idea and Motivation**: Core problems addressed, basic idea of the method, and the primary motivation.
    2. **Method Architecture and Technical Details**: Overall architecture, key technical means, algorithm flow, or mathematical principles.
    3. **Key Technical Components**: Core components, their responsibilities, and how they collaborate.
    4. **Core Innovation Points**: How this method innovates compared to existing/traditional approaches.
    Notes:
    - Faithfully summarize based ONLY on the provided content.
    - Prioritize preserving specific technical details and key terminology.
    - Exclude overly detailed numerical settings (e.g., learning rates, batch sizes, epochs) and focus on method-level design.
    - If information is insufficient for a specific dimension, summarize what is available but ensure the core architecture is clear.
    """
    try:
        solution = llm_client.generate_response(solution_prompt, max_tokens=8192, temperature=0.2)
        if not solution or not solution.strip():
            print(f"[Thread: {threading.get_ident()}] Warning: No solution found for {paper_filename}", flush=True)
            return ""
        return solution.strip()
    except Exception as e:
        print(f"[Thread: {threading.get_ident()}] Error generating solution for {paper_filename}: {str(e)}", flush=True)
        return ""


def generate_experiment(llm_client: LLMClient, paper_filename: str, extracted_experiment_list: List[str]) -> str:
    """
    Integrate the extracted experimental design information (Reduce stage) to generate the paper's experimental design description.
    
    Args:
        llm_client: LLM client
        paper_filename: Paper filename (for logging)
        extracted_experiment_list: List of experimental design information extracted from each section
    """
    if not paper_filename:
        print(f"[Thread: {threading.get_ident()}] Warning: No paper filename provided. Returning empty string.", flush=True)
        return ""
    
    # Filter and combine valid information
    combined_info = _filter_and_combine_extracted_info(extracted_experiment_list)
    
    if not combined_info:
        print(f"[Thread: {threading.get_ident()}] Warning: No valid info found for {paper_filename}. Returning empty string.", flush=True)
        return ""
    
    # Reduce stage - structured output
    experiment_prompt = f"""
    You are an expert academic paper reviewer.
    Based on the following extracted experimental details from a paper, comprehensively describe the 'Experiment Design, Process, and Analysis Strategy'.
    Extracted Information:
    {combined_info}
    Task Requirements:
    Please organize your detailed description strictly using the following 5 dimensions (Use Markdown formatting):
    
    1. **Experimental Design and Framework**: Overall experimental setup, methodology, and structural organization.
    2. **Data and Baseline Configuration**: Dataset selection, preprocessing steps, baseline methods, and comparison protocols.
    3. **Evaluation and Metrics**: Performance measures, evaluation criteria, and validation procedures.
    4. **Implementation and Settings**: Parameter configurations, environment setup, and implementation details.
    5. **Analysis and Validation Strategy**: Ablation studies, statistical validation, and component contribution analysis.
    Notes:
    - Faithfully summarize based ONLY on the provided content.
    - Prioritize preserving specific experimental details and methodological information.
    - Exclude raw numerical results, focusing on the experimental methodology and process.
    - If information is insufficient for a specific dimension, summarize what is available but ensure the experimental approach is clear.
    """
    try:
        experiment = llm_client.generate_response(experiment_prompt, max_tokens=8192, temperature=0.2)
        if not experiment or not experiment.strip():
            print(f"[Thread: {threading.get_ident()}] Warning: No experiment found for {paper_filename}", flush=True)
            return ""
        return experiment.strip()
    except Exception as e:
        print(f"[Thread: {threading.get_ident()}] Error generating experiment for {paper_filename}: {str(e)}", flush=True)
        return ""


def process_paper_batch(args):
    """
    Process a batch of papers to obtain background, solution and other information
    """
    paper_batch, output_dir, model, batch_idx, total_batches, base_dir = args
    
    # Initialize a new LLM client for this thread
    llm_client = get_llm_client(model=model)
    
    processed_count = 0
    batch_size = len(paper_batch)
    
    thread_id = threading.get_ident()
    print(f"[Thread: {thread_id}] Starting batch {batch_idx+1}/{total_batches} with {batch_size} papers", flush=True)
    
    for i, paper in enumerate(paper_batch):
        filename = paper['filename']
        safe_filename = os.path.splitext(filename)[0] + ".json"
        output_path = os.path.join(output_dir, safe_filename)
        
        print(f"[Thread: {thread_id}] Processing paper {i+1}/{batch_size} ({filename}) in batch {batch_idx+1}/{total_batches}", flush=True)
        
        # Check if already processed
        if os.path.exists(output_path):
            print(f"[Thread: {thread_id}] Skipped {filename} - already processed", flush=True)
            continue
        
        try:
            # Phase 1: Extract the four types of information from all sections at once (Map stage)
            all_extracted = extract_all_info_from_sections(paper['filename'], llm_client, base_dir=base_dir)
            
            if all_extracted is None:
                print(f"[Thread: {thread_id}] Skipped {filename} - extraction failed (section too long or API error)", flush=True)
                continue
            
            # Phase 2: Integrate the four types of information separately (Reduce stage)
            research_field = generate_research_field(llm_client, paper['filename'], all_extracted["research_field"])
            if not research_field.strip():
                print(f"[Thread: {thread_id}] Skipped {filename} - no research field info found", flush=True)
                continue
            
            research_background = generate_research_background(llm_client, paper['filename'], all_extracted["research_background"])
            if not research_background.strip():
                print(f"[Thread: {thread_id}] Skipped {filename} - no research background info found", flush=True)
                continue
            
            solution = generate_solution(llm_client, paper['filename'], all_extracted["solution"])
            if not solution.strip():
                print(f"[Thread: {thread_id}] Skipped {filename} - no solution info found", flush=True)
                continue
            
            experiment = generate_experiment(llm_client, paper['filename'], all_extracted["experiment"])
            if not experiment.strip():
                print(f"[Thread: {thread_id}] Skipped {filename} - no experiment info found", flush=True)
                continue
            
            # Store the research field, research background, solution and experiment data
            background_data = {
                "filename": paper['filename'],
                "research_field": research_field,
                "research_background": research_background,
                "solution": solution,
                "experiment": experiment
            }
            
            # Save individual file immediately
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(background_data, f, ensure_ascii=False, indent=2)
                
            print(f"[Thread: {thread_id}] Successfully processed {filename}", flush=True)
            processed_count += 1
            
        except Exception as e:
            print(f"[Thread: {threading.get_ident()}] Error processing {filename}: {str(e)}", flush=True)
            continue
    
    print(f"[Thread: {threading.get_ident()}] Completed batch {batch_idx+1}/{total_batches}, processed {processed_count} papers", flush=True)
    return processed_count


def process_papers_to_research_background(data_dir: str = "/path/to/paper_bodies_latex",
                                        output_dir: str = "/path/to/research_background",
                                        models: List[str] = ["qwen3-32b-1"],
                                        num_threads_per_model: int = 8,
                                        start_idx: int = 0,
                                        end_idx: int = None,
                                        base_dir: str = "/path/to/paper_sections/cs"):
    """
    Main function to process papers and generate research background using multiple models and threads.
    Implements dual-level parallelization: first split data by model, then split into batches 
    within each model for parallel processing.
    
    Args:
        data_dir (str): Directory containing input paper .tex files
        output_dir (str): Directory where individual JSON files will be saved
        models (List[str]): List of LLM model names to use
        num_threads_per_model (int): Number of parallel threads to use per model (default: 8)
        start_idx (int): Starting index for processing (for dataset splitting)
        end_idx (int): Ending index for processing (None for all remaining)
        base_dir (str): Directory containing paper sections JSON files (divided sections)
    
    Returns:
        int: Number of successfully processed papers
    """
    # Initialize
    os.makedirs(output_dir, exist_ok=True)
    
    # Read paper bodies
    papers = read_paper_bodies(data_dir)
    
    # Apply dataset splitting
    total_original = len(papers)
    papers = papers[start_idx:end_idx] if end_idx else papers[start_idx:]
    
    print(f"[Main process] Total papers available: {total_original}", flush=True)
    print(f"[Main process] Processing papers from index {start_idx}" + (f" to {end_idx}" if end_idx else " to end"), flush=True)
    print(f"[Main process] Papers to process (before filtering): {len(papers)}", flush=True)
    
    if not papers:
        print("No papers found to process.", flush=True)
        return 0
    
    # Pre-filter: exclude papers that are already processed or missing corresponding paper_section files
    filtered_papers = []
    skipped_processed = 0
    skipped_no_section = 0
    for paper in papers:
        filename = paper['filename']
        safe_filename = os.path.splitext(filename)[0] + ".json"
        output_path = os.path.join(output_dir, safe_filename)
        
        # Check if already processed
        if os.path.exists(output_path):
            skipped_processed += 1
            continue
        
        # Check if corresponding paper_section JSON exists
        json_filename = os.path.splitext(filename)[0] + "_divided.json"
        json_path = os.path.join(base_dir, json_filename)
        if not os.path.exists(json_path):
            skipped_no_section += 1
            continue
        
        filtered_papers.append(paper)
    
    print(f"[Main process] Filtered out {skipped_processed} already-processed papers, {skipped_no_section} papers missing section JSONs", flush=True)
    print(f"[Main process] Remaining papers to process: {len(filtered_papers)}", flush=True)
    
    papers = filtered_papers
    
    if not papers:
        print("No papers remaining after pre-filtering.", flush=True)
        return 0

    # Calculate papers per model
    num_models = len(models)
    papers_per_model = len(papers) // num_models
    remainder_papers = len(papers) % num_models
    
    # Prepare arguments for each model
    model_args = []
    for model_idx, model in enumerate(models):
        # Divide papers for each model, distributing remainder papers
        start_paper_idx = model_idx * papers_per_model + min(model_idx, remainder_papers)
        end_paper_idx = start_paper_idx + papers_per_model + (1 if model_idx < remainder_papers else 0)
        papers_for_model = papers[start_paper_idx:end_paper_idx]
        
        print(f"[Main process] Model {model} will process {len(papers_for_model)} papers", flush=True)
        
        model_args.append((papers_for_model, output_dir, model, num_threads_per_model, base_dir))

    print(f"Starting parallel processing across {num_models} models for {len(papers)} papers...", flush=True)
    
    # Use ThreadPoolExecutor to process different models in parallel
    successful_results = 0
    with ThreadPoolExecutor(max_workers=num_models) as executor:
        # Submit all model tasks to the thread pool
        futures = [executor.submit(process_papers_for_single_model, arg) for arg in model_args]
        
        # Collect results as they complete
        results = [future.result() for future in futures]
        successful_results = sum(results)
    
    print(f"All models processing completed. Successfully processed {successful_results} papers and saved individual files to {output_dir}", flush=True)
    return successful_results


def process_papers_for_single_model(args):
    """
    Process papers for a single model with multiple threads
    """
    papers_for_model, output_dir, model, num_threads_per_model, base_dir = args
    
    # Determine actual number of threads to use based on number of papers for this model
    actual_num_threads = min(num_threads_per_model, len(papers_for_model))
    
    if actual_num_threads == 0:
        return 0
    
    # Split papers into exactly 'actual_num_threads' batches for parallel processing within model
    # Each thread will handle one batch
    num_papers_per_batch = len(papers_for_model) // actual_num_threads
    remainder = len(papers_for_model) % actual_num_threads
    
    paper_batches = []
    start_idx_batch = 0
    for i in range(actual_num_threads):
        # Distribute remainder papers evenly among the first 'remainder' batches
        end_idx_batch = start_idx_batch + num_papers_per_batch + (1 if i < remainder else 0)
        paper_batches.append(papers_for_model[start_idx_batch:end_idx_batch])
        start_idx_batch = end_idx_batch
    
    # Prepare arguments for each thread within this model
    thread_args = [
        (batch, output_dir, model, idx, len(paper_batches), base_dir)
        for idx, batch in enumerate(paper_batches)
    ]
    
    print(f"Starting parallel processing for model {model} with {actual_num_threads} threads for {len(papers_for_model)} papers in {len(paper_batches)} batches...", flush=True)
    
    # Use ThreadPoolExecutor to process batches in parallel within this model
    with ThreadPoolExecutor(max_workers=actual_num_threads) as executor:
        # Submit all tasks to the thread pool
        futures = [executor.submit(process_paper_batch, arg) for arg in thread_args]
        
        # Collect results as they complete
        results = [future.result() for future in futures]
        model_processed_count = sum(results)
    
    print(f"Model {model} processing completed. Processed {model_processed_count} papers", flush=True)
    return model_processed_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate research background from academic papers")
    parser.add_argument("--data_dir", type=str, default="/path/to/paper_bodies/cs",
                       help="Directory containing input paper .tex files")
    parser.add_argument("--output_dir", type=str, default="/path/to/research_background/cs",
                       help="Directory where individual JSON files will be saved")
    parser.add_argument("--models", nargs='+', default=["DeepSeek-V4-Flash"],
                       help="List of LLM model names to use")
    parser.add_argument("--num_threads_per_model", type=int, default=8,
                       help="Number of parallel threads per model")
    parser.add_argument("--start_idx", type=int, default=0,
                       help="Starting index for processing (for dataset splitting)")
    parser.add_argument("--end_idx", type=int, default=None,
                       help="Ending index for processing (None for all remaining)")
    parser.add_argument("--base_dir", type=str, default="/path/to/paper_sections/cs",
                       help="Directory containing paper sections JSON files (divided sections)")
    
    args = parser.parse_args()
    
    # Process papers
    background_data = process_papers_to_research_background(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        models=args.models,
        num_threads_per_model=args.num_threads_per_model,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        base_dir=args.base_dir,
    )
