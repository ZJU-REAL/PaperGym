# Remove dynamic scoring criteria, rely only on static scoring criteria and general scoring criteria
import re
import numpy as np
import json
from openai import OpenAI
import os
from typing import Optional

# Judge model used for scoring. Override with the JUDGE_MODEL_PATH env var.
DEFAULT_JUDGE_MODEL = os.path.expanduser("~/.cache/modelscope/hub/models/Qwen/Qwen3-4B")


# General scoring criteria list
GENERAL_RUBRICS = [
    "HANDLES ALL CRITERIA: Does the plan satisfy all criteria mentioned in the rubric item? An exception is if the criteria says \"such as\", \"for example\", or \"including\", the response does not have to include the same examples listed to meet the criteria, but whatever is provided must be valid and reasonable.",
    "DETAILED, SPECIFIC SOLUTION: Does the part of the plan relevant to satisfying this rubric item include fully specified details on HOW to implement it? There should be no self-proclaimed claims of handling something without doing so. There should be no vague terms, ambiguity, or lack of clarity. It should be described in simple to understand language.",
    "NO OVERLOOKED FLAWS OR WEAKNESSES: Are there any important overlooked flaws or weaknesses in the part of the plan addressing this rubric item that invalidate its satisfaction of the rubric item?",
    "WELL JUSTIFIED RATIONALE: Is the part of the plan relevant to this grading item well-motivated and justified? For example, are there convincing arguments provided for how the plan handles this grading item is better than simpler solutions or alternate hypotheses?",
    "COST AND EFFORT EFFICIENT: Does the plan handle this rubric item cost efficiently, without being unnecessarily complex? Check if a solution requiring less human effort or resources would be equally effective for this rubric item.",
    "NO ETHICAL ISSUES: Does this part of the plan have any potential for negative consequences, or is it ethically problematic?",
    "CONSISTENT WITH OVERALL PLAN: Is this part of the plan consistent with the overall approach?"
]


def preprocess_json_string(json_str: str) -> str:
    """
    Preprocess a JSON string and repair possible escaping issues.
    """
    # Replace single quotes with double quotes
    json_str = json_str.replace("'", '"')
    
    # First try direct parsing
    try:
        json.loads(json_str)
        return json_str  # Parses cleanly as-is, no repair needed
    except json.JSONDecodeError:
        pass

    # Simple repair strategy - check and fix common issues character by character
    repaired = ""
    i = 0
    while i < len(json_str):
        char = json_str[i]
        if char == '\\':
            # Check next character
            if i + 1 < len(json_str):
                next_char = json_str[i + 1]
                # If valid escape sequence, keep it
                if next_char in ['"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u']:
                    repaired += char + next_char
                    i += 2
                else:
                    # If not a valid escape character, may be redundant backslash
                    # Only remove this backslash if in string value and may cause problems
                    repaired += next_char  # Skip the backslash
                    i += 2
            else:
                # Trailing backslash, remove it
                break
        else:
            repaired += char
            i += 1
    
    return repaired


def _parse_llm_response(response: str) -> list:
    """
    Parse the JSON response returned by the LLM in layers, following the
    multi-level parsing strategy in solve_background.py, with extra tolerance
    for LaTeX math symbols.
    """
    import json
    import re

    # Multi-level JSON parsing strategy
    parsed_data = None

    # Preprocess response, handle LaTeX escaping issues
    processed_response = response.strip()
    
    # 1. Try extracting JSON from code blocks (prioritize json format)
    json_match = re.search(r'```json\s*(\[?.*?\]?)\s*```', processed_response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        # Try generic code block
        generic_match = re.search(r'```\s*(\[?.*?\]?)\s*```', processed_response, re.DOTALL)
        if generic_match:
            json_str = generic_match.group(1).strip()
        else:
            # Try directly extracting JSON array
            bracket_match = re.search(r'\[.*\]', processed_response, re.DOTALL)
            if bracket_match:
                json_str = bracket_match.group(0)
            else:
                # Last attempt: find content between first [ and last ]
                start_idx = processed_response.find('[')
                end_idx = processed_response.rfind(']')
                if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                    json_str = processed_response[start_idx:end_idx+1]
    
    if 'json_str' in locals():
        # Preprocess and fix JSON strings that may contain LaTeX math symbols
        # Process incorrectly escaped backslashes
        json_str = re.sub(r'(?<!\\)\\(?![\\nrtbf"u/])', r'\\\\', json_str)
        
        try:
            parsed_data = json.loads(json_str)
        except json.JSONDecodeError as e:
            # If direct parsing fails, try using preprocess_json_string function to fix
            try:
                fixed_json = preprocess_json_string(json_str)
                parsed_data = json.loads(fixed_json)
            except json.JSONDecodeError:
                print(f"[Error] JSON parse failed: {str(e)}", flush=True)
                pass

    # If parsing succeeds and is list type, return result
    if parsed_data and isinstance(parsed_data, list):
        return parsed_data

    # If all parsing methods fail, return default score
    print(f"[WARNING] Failed to parse valid response, returning default", flush=True)
    # print(f"[DEBUG] Raw output: {response}", flush=True)
    # Return an empty list in default format so external functions can continue processing
    return []


# LLM API Client (same as in extract_info.py)
class LLMClient:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        """
        Initialize the LLM client with API key and base URL
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def generate_response(self, prompt: str, max_tokens: int = 10240, temperature: float = 0.7, n: int = 1) -> str:
        """
        Generate response from the LLM based on the given prompt
        """
        try:
            chat_completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": f"{prompt} /no_think"}],
                # max_tokens=max_tokens,
                temperature=temperature,
                n=n
            )
            
            if n == 1:
                raw_content = chat_completion.choices[0].message.content
                cleaned_content = clean_llm_output(raw_content)
                return cleaned_content
            else:
                responses = []
                for choice in chat_completion.choices:
                    raw_content = choice.message.content
                    cleaned_content = clean_llm_output(raw_content)
                    responses.append(cleaned_content)
                return responses
        except Exception as e:
            print(f"[Error] Error generating response: {str(e)}")
            if n == 1:
                return ""
            else:
                return [""] * n


def clean_llm_output(text: str) -> str:
    """
    Clean the LLM output by removing specific internal processing tags and their content.

    It removes non-user-visible markers from the model output, such as <think>
    and <extra_vision>. These markers belong to the model's internal reasoning
    process and must not appear in the final answer presented to the user.

    Args:
        text (str): Raw output text produced by the LLM.

    Returns:
        str: Clean text with the specified tags and their content removed.
    """
    if not isinstance(text, str):
        return ""
        
    # Remove the <think>...</think> block (non-greedy match)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Remove any unclosed or self-contained <think> tags
    text = re.sub(r'<think>.*?(?=<\/think>|$)', '', text, flags=re.DOTALL)
    
    return text.strip()


def _score_with_rubrics(llm_client: LLMClient, question: str, answer: str, reference_answer: str, rubrics: list) -> float:
    """
    General scoring function: score an answer against the given rubrics.
    
    Args:
        llm_client: LLM client instance
        question: The research question
        answer: The answer generated by the model
        reference_response: The reference answer
        rubrics: List of rubric items
        
    Returns:
        float: Normalized overall score in [0, 1]
    """
    print(f"[Debug] Answer is: {answer[:20]}", flush=True)

    # Convert scoring criteria to readable format
    rubrics_text = "\n".join([
        f"{i+1}. {rubric.get('description', rubric) if isinstance(rubric, dict) else rubric}"
        for i, rubric in enumerate(rubrics)
    ])
    
    num_rubrics = len(rubrics)
    
    # Construct prompt
    prompt = f"""You are an exceptionally rigorous and professional academic peer reviewer and domain expert. Your task is to conduct an objective, impartial, and high-standard evaluation of complex scientific research tasks, which may include methodology design, experimental setup, or data analysis pipelines.

### Evaluation Task
You must evaluate the provided answer against the given scoring rubrics. For each rubric, you must assign a strict binary score (0 or 1) indicating whether the answer completely satisfies the criterion.

### Input Data
<Question>
{question}
</Question>

<Rubrics>
{rubrics_text}
</Rubrics>

<Answer>
{answer}
</Answer>

<Reference_Answer>
{reference_answer}
</Reference_Answer>

### Strict Scoring Guidelines (MUST Follow):
1. **Score 1 (Completely Satisfies)**: The answer fully, explicitly, and comprehensively meets all requirements of the specific rubric, demonstrating quality comparable to or better than the reference answer.
2. **Score 0 (Does Not Satisfy)**: Assign 0 if ANY of the following apply:
   - The answer only partially meets the criterion or exhibits obvious methodological, experimental, or analytical flaws compared to the reference answer.
   - The content is vague, generic, or lacks specific technical details that are present in the reference answer.
   - Important elements required by the rubric are omitted.
   - The answer contains factual errors, logical fallacies, or scientific inaccuracies not present in the reference answer.
   - The response is too brief to demonstrate a complete research methodology compared to the reference answer.
   - The answer fails to address aspects of the rubric that the reference answer addresses properly.
3. **Comparison with Reference**: Use the reference answer as a benchmark for quality and completeness. The answer should meet the rubric at least as well as the reference answer does.
4. **ZERO Leniency**: Do NOT award a 1 simply because the answer is "on the right track" or "mostly correct". Strict precision is required for a score of 1.

### Output Requirements:
- You MUST output ONLY a valid JSON array. Do not output any thinking process or conversational filler.
- **CRITICAL**: You must provide the `reason` FIRST, and then the `score`. 
- **IMPORTANT**: The number of output items in the JSON array MUST match the number of input rubrics EXACTLY. You have {num_rubrics} rubrics, so you must output exactly {num_rubrics} items. Do NOT use a fixed number from the example below - the example shows only 2-3 items for illustration, but you must output one item for EACH rubric listed above.

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
    # Use Qwen3 model for inference
    try:
        response = llm_client.generate_response(prompt, temperature=0.0)
        # Clean output
        response = clean_llm_output(response)
        
        # Use encapsulated parsing function to process response
        individual_scores = _parse_llm_response(response)
        
    except Exception as e:
        print(f"[Error] Error during model inference: {str(e)}", flush=True)
        return 0.5

    # Parse scoring results
    try:
        # Check if scoring items exist
        if len(individual_scores) == 0:
            print(f"[Error] No individual scores found in the response", flush=True)
            return 0.5
        
        # Check if each element contains necessary fields
        required_fields = {'rubric_id', 'score', 'reason'}
        for idx, score_item in enumerate(individual_scores):
            if not isinstance(score_item, dict):
                print(f"[Error] Item at index {idx} is not a dictionary: {score_item}")
                return 0.5
            
            missing_fields = required_fields - set(score_item.keys())
            if missing_fields:
                print(f"[Error] Missing fields {missing_fields} in item at index {idx}: {score_item}")
                return 0.5

        total_weighted_score = 0.0
        total_weight = 0.0
        
        # Get weights for all rubrics (default to 1 if no weight field)
        all_rubrics_weights = []
        for rubric in rubrics:
            if isinstance(rubric, dict):
                # Check if weight field exists, otherwise default to 1
                weight = rubric.get('weight', 1) if 'weight' in rubric else 1
                all_rubrics_weights.append(weight)
            else:
                all_rubrics_weights.append(1)
        
        for score_item in individual_scores:
            if isinstance(score_item, dict) and 'score' in score_item:
                score_value = score_item['score']
                rubric_id = score_item.get('rubric_id', 0)
                
                # Get weight for this rubric
                if rubric_id > 0 and rubric_id <= len(all_rubrics_weights):
                    weight = all_rubrics_weights[rubric_id - 1]
                else:
                    weight = 1.0
                
                total_weighted_score += score_value * weight
                total_weight += weight if weight > 0 else 0
            else:
                # If format is incorrect, use default values
                total_weighted_score += 0.0
                total_weight += 1.0
        
        print(f"[Info] Score: {total_weighted_score} / {total_weight} = {total_weighted_score / total_weight if total_weight > 0 else 0}", flush=True)
        
        # Calculate normalized weighted total score (between 0 and 1)
        if total_weight > 0:
            overall_score = total_weighted_score / total_weight
        else:
            print("[Warning] total_weight <= 0")
            overall_score = 0.5
        
        return overall_score
        
    except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
        print(f"[Warning] Error parsing model response: {str(e)}", flush=True)
        # Return default score directly when parsing fails
        return 0.5


def score_answer_with_rubrics(llm_client: LLMClient, question: str, answer: str, reference_answer: str, general_rubrics: list, combined_rubrics: list) -> float:
    """
    Compute a score from the question, the answer and the rubrics using a
    two-stage strategy:
    1. Score the general rubrics on their own
    2. Score the combined rubrics (static + dynamic) on their own  
    3. Combine both parts by weight to obtain the final score
    
    Args:
        llm_client: LLM client instance
        question: The research question  
        answer: The answer generated by the model
        reference_answer: The reference answer
        general_rubrics: List of general rubric items
        combined_rubrics: List of combined rubric items (static + dynamic)
        
    Returns:
        float: Normalized overall score in [0, 1]
    """
    # Perform two scoring separately
    general_score = _score_with_rubrics(llm_client, question, answer, reference_answer, general_rubrics)
    print(f"[Info] General rubric score: {general_score}", flush=True)
    
    combined_score = _score_with_rubrics(llm_client, question, answer, reference_answer, combined_rubrics)
    print(f"[Info] Combined rubric score: {combined_score}", flush=True)
    
    # Combine proportionally (general 20%, combined 80%)
    final_score = 0.2 * general_score + 0.8 * combined_score
    print(f"[Info] Final combined score: 0.2*{general_score} + 0.8*{combined_score} = {final_score}", flush=True)
    
    return final_score


def paper_rl_reward(data_source, solution_str, ground_truth, extra_info=None):
    """
    Reward function based on rule checks and model-based scoring.
    Args:
        data_source: Data source containing the question and the rubrics
        solution_str: Answer generated by the model (may contain <answer> tags)
        ground_truth: Ground-truth answer (optional)
        extra_info: Extra information (e.g. rubric items)

    Returns:
        float: Normalized overall score in [0, 1]
    """
    score = 0.0

    answer = clean_llm_output(solution_str)

    # Extract questions and scoring criteria from data_source
    try:
        question = extra_info.get("content", "")
        static_rubrics = extra_info.get("rubrics", [])
        # print("STATIC RUBRIC: ")
        # print(static_rubrics, flush=True)
        # print(static_rubrics[:10], flush=True)
        reference_answer = ground_truth
        
        # Ensure model is loaded
        llm_client = LLMClient(
            api_key=os.getenv("PAPERGYM_API_KEY", ""),
            base_url=os.getenv("PAPERGYM_BASE_URL", "http://your-api/v1"),
            model=os.getenv("JUDGE_MODEL_PATH", DEFAULT_JUDGE_MODEL)
        )
        
        # Directly use static scoring criteria, no longer generate dynamic scoring rules
        combined_rubrics = static_rubrics
        
        # General scoring criteria
        general_rubrics = [
            {"id": i+1, "description": rubric, "weight": 1}
            for i, rubric in enumerate(GENERAL_RUBRICS)
        ]
        
        # Use scoring model to score answers - pass general and static scoring criteria separately
        score = score_answer_with_rubrics(
            llm_client=llm_client,
            question=question,
            answer=answer,
            reference_answer=reference_answer,
            general_rubrics=general_rubrics,
            combined_rubrics=combined_rubrics
        )
        
    except Exception as e:
        print(f"[Warning] Error in paper_rl_reward: {str(e)}", flush=True)
        return 0.5  # The framework requires a numeric value; use a neutral score
    
    return score
