from pathlib import Path
import re
import spacy
from wtpsplit import SaT
from typing import Union, List, Optional
from openai import OpenAI
import torch


# Get API keys from environment variables
from dotenv import load_dotenv
import os
load_dotenv(override=True)
S2_API_KEY = os.environ.get('S2_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
ANTHORPIC_API_KEY = os.environ.get('ANTHORPIC_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')

import openai
import anthropic
from google import genai
# openai.organization = OPENAI_ORG
openai.api_key = OPENAI_API_KEY
client_claude = anthropic.Client(api_key=ANTHORPIC_API_KEY)
client_gemini = genai.Client(api_key=GOOGLE_API_KEY)
openai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

class Paths:

    ## PDF folder: for the pdf files
    pdf_folder = Path('/nlp/data/academic_pdfs')

    ## Prompt foleder: for the prompt txt files
    prompt_folder = Path('prompts')  

    ## Viz folder: the visualization folder 
    viz_folder = Path('viz')  


# Provider to default model and model name patterns mapping
PROVIDER_MODEL_MAP = {
    "gpt": {
        "default": "gpt-4.1",
        "patterns": ["gpt-"]
    },
    "claude": {
        "default": "claude-sonnet-4-20250514",
        "patterns": ["claude-"]
    },
    "gemini": {
        "default": "gemini-2.5-pro-preview-06-05",
        "patterns": ["gemini-"]
    },
    "llama": {
        "default": "meta-llama/llama-3.3-70b-instruct",
        "patterns": ["llama-"]
    },
    "qwen": {
        "default": "qwen/qwen3-32b",
        "patterns": ["qwen-"]
    },
    "openscholar": {
        "default": "OpenScholar/Llama-3.1_OpenScholar-8B",
        "patterns": ["OpenScholar/"]
    }
}

class LLM:
    def __init__(self, prompt=None, markers=['[CONTENTS]'], model=None, temperature=0, provider=None):
        """
        Initializes the LLM class with a model and/or provider.
        """
        # If neither provider nor model is specified, use default provider
        if provider is None and model is None:
            provider = "gpt"
            model = PROVIDER_MODEL_MAP[provider]["default"]
        # If only provider is specified, use its default model
        elif provider is not None and model is None:
            provider = provider.lower()
            if provider not in PROVIDER_MODEL_MAP:
                raise ValueError(f"Unknown provider: {provider}")
            model = PROVIDER_MODEL_MAP[provider]["default"]
        # If only model is specified, infer provider from model name
        elif provider is None and model is not None:
            for prov, info in PROVIDER_MODEL_MAP.items():
                if any(model.startswith(pat) for pat in info["patterns"]):
                    provider = prov
                    break
            else:
                raise ValueError(f"Cannot infer provider from model name: {model}")
        elif provider is not None and model is not None:
            provider = provider.lower()
            if provider not in PROVIDER_MODEL_MAP:
                raise ValueError(f"Unknown provider: {provider}")
            if not any(model.startswith(pat) for pat in PROVIDER_MODEL_MAP[provider]["patterns"]):
                raise ValueError(f"Model '{model}' does not match provider '{provider}'")

        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.prompt = prompt
        self.markers = markers
        
        # Load OpenScholar model and tokenizer if using openscholar provider
        if self.provider == "openscholar":
            self._load_openscholar_model()

    def _load_openscholar_model(self):
        """Load OpenScholar model and tokenizer."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load tokenizer and model
            self.openscholar_tokenizer = AutoTokenizer.from_pretrained(self.model)
            if self.openscholar_tokenizer.pad_token is None:
                self.openscholar_tokenizer.pad_token = self.openscholar_tokenizer.eos_token
                self.openscholar_tokenizer.pad_token_id = self.openscholar_tokenizer.eos_token_id
            self.openscholar_tokenizer.padding_side = "left"
            
            self.openscholar_model = AutoModelForCausalLM.from_pretrained(
                self.model, 
                torch_dtype=torch.float16
            )
            self.openscholar_model = self.openscholar_model.to(device)
            self.openscholar_model.eval()
            self.openscholar_device = device
            
        except ImportError as e:
            raise ImportError(f"OpenScholar requires transformers and torch: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load OpenScholar model: {e}")

    def _load_prompt(self):
        """
        Loads the prompt text, whether from a file or directly from a string.

        Returns:
            str: The loaded prompt text
        """
        if isinstance(self.prompt, str) or isinstance(self.prompt, Path):
            # Read the prompt from a file
            if isinstance(self.prompt, Path) or self.prompt.endswith('.txt') or self.prompt.endswith('.md') or self.prompt.endswith('.prompt'):
                with open(self.prompt, 'r') as f:
                    return f.read()
            # Treat the input as raw text for the prompt
            else:
                # If not a file, assume it's the actual prompt text
                return self.prompt
        raise ValueError("Prompt must be a valid string (file path or raw text).")

    def call(self, replacements, verbose=False, debug=False):
        """
        Generate the LM output using a specific prompt file after replacing the markers with the replacements one-by-one in the prompt.

        Args:
            - replacements (str / list of str): A single string or a list of replacement strings that will replace the markers in the prompt text
            - debug (bool): If True, print the dummy text without calling LLM to save the API usage when debugging

        Returns:
            str: The content generated by the language model after applying the replacements and evaluating the modified prompt
        """
        if debug:
            return "This is a dummy text for debugging."
        prompt = self._load_prompt()
        if isinstance(replacements, str):
            replacements = [replacements]
        replacements = [str(r) for r in replacements]
        assert len(replacements) == len(self.markers), "The number of replacement pairs must match the number of markers."
        assert all(marker in prompt for marker in self.markers), "All markers must be present in the prompt."
        for marker, replacement in zip(self.markers, replacements):
            prompt = prompt.replace(marker, replacement)
        
        try:
            if self.provider == "gpt":
                try:
                    chat_completion = openai.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=self.model,
                        temperature=self.temperature,
                        top_p=1,
                    )
                    result = chat_completion.choices[0].message.content
                except Exception as e:
                    print(f"OpenAI API Error: {str(e)}")
                    return f"Error: {str(e)}"
            elif self.provider == "claude":
                try:
                    chat_completion = client_claude.messages.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=self.model,
                        temperature=self.temperature,
                        max_tokens=8192,
                    )
                    result = chat_completion.content[0].text
                except Exception as e:
                    print(f"Claude API Error: {str(e)}")
                    return f"Error: {str(e)}"
            elif self.provider == "gemini":
                try:
                    response = client_gemini.models.generate_content(
                        model=self.model,
                        contents=prompt,
                        config=genai.types.GenerateContentConfig(
                            temperature=self.temperature
                        )
                    )
                    result = response.text
                except Exception as e:
                    print(f"Gemini API Error: {str(e)}")
                    return f"Error: {str(e)}"
            elif self.provider in ["llama", "qwen"]:
                try:
                    chat_completion = openai_client.chat.completions.create(
                        extra_headers={
                            "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
                            "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
                        },
                        extra_body={},
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                    )
                    result = chat_completion.choices[0].message.content
                except Exception as e:
                    print(f"OpenRouter API Error: {str(e)}")
                    return f"Error: {str(e)}"
            elif self.provider == "openscholar":
                try:
                    # Tokenize input
                    inputs = self.openscholar_tokenizer(prompt, return_tensors="pt", padding=True).to(self.openscholar_device)
                    
                    # Generate output
                    with torch.no_grad():
                        output_ids = self.openscholar_model.generate(
                            **inputs,
                            max_new_tokens=2048,
                            temperature=self.temperature,
                            do_sample=True if self.temperature > 0 else False,
                            pad_token_id=self.openscholar_tokenizer.pad_token_id,
                            eos_token_id=self.openscholar_tokenizer.eos_token_id,
                        )
                    
                    # Decode result (remove the original prompt)
                    response = self.openscholar_tokenizer.decode(output_ids[0], skip_special_tokens=True)
                    result = response[len(prompt):].strip()
                    
                except Exception as e:
                    print(f"OpenScholar Error: {str(e)}")
                    return f"Error: {str(e)}"
        except Exception as e:
            print(f"Unexpected error in LLM call: {str(e)}")
            return f"Error: {str(e)}"

        if verbose:
            print("# Prompt:\n", prompt, "\n# Result:\n", result)
        return result


class ContentSegmenter:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def segment(self, content, with_bullet_point=False):
        '''
        Segment the content into a list of sentence. It is free from the interruption of citation format, paragraph title, and special contents.
        Args:
            - content (str): the string content
            - with_bullet_point (Boolean): if True, put bullet points 1. 2. 3. ... before each sentence
        The nltk package fails to segment sentence with citations in "()" format, while spacy package fails to segment sentence with citations in "[]" format.
        Here we use spacy package and replace bracket content, predefined special contents and paragraph titles temporarily with reserved tokens 
        to avoid interrupting the sentence segmentation, then replace them back in the end.
        '''
        ##TODO: use free LLM to segment the very complicated content, e.g. the math papers
        # Special item list - predefined by heuristic
        special_items = [
            'et al.',
            # 'etc.',
            'e.g.',
            'resp.',
            'Fig.',
            'e.g.', 
            ' + ',
            ': ',
            '%',
            '. . .', # math equation
            '++', # chemistry
            '• ', # bullet point
        ]
        
        content = str(content)

        # Remove paragraph index in the format (pX.X)
        pattern = r'\(p[0-9]+\.[0-9]+\)' 
        content = re.sub(pattern, '', content)

        # Replace paragraph titles that are strictly preceded by a newline (may followed by space), in title case, ending with a period followed by a space.
        title_items = re.findall(r'(?<=\n)\s*([A-Z][\w-]*(?:\s+[A-Z][\w-]*)*\.) ', content)
        title_replacements = {item: f"title_replacement_{i}" for i, item in enumerate(title_items)}
        for original, marker in title_replacements.items():
            content = content.replace(original, marker)
        
        # Replace each special content
        special_replacements = {item: f"speical_replacement_{i}" for i, item in enumerate(special_items)}
        for original, marker in special_replacements.items():
            content = content.replace(original, marker)

        # Replace each bracket content
        bracket_items = re.findall(r'\[.*?\]|\(.*?\)', content)
        bracket_replacements = {item: f"bracket_replacement_{i}" for i, item in enumerate(bracket_items)}
        for original, marker in bracket_replacements.items():
            content = content.replace(original, marker)

        # Print the intermidate replaced content for debug
        # print(content)

        # Segment sentences using spacy
        sentences = [sent.text for sent in self.nlp(content).sents]

        # Restore bracket items and special contents - FIFO 
        bracket_replacements_reverse_sorted = sorted(bracket_replacements.items(), key=lambda item: int(item[1].strip('#').split('_')[-1]), reverse=True)
        for original, marker in bracket_replacements_reverse_sorted:
            sentences = [sent.replace(marker, original) for sent in sentences]

        special_replacements_reverse_sorted = sorted(special_replacements.items(), key=lambda item: int(item[1].strip('#').split('_')[-1]), reverse=True)
        for original, marker in special_replacements_reverse_sorted:
            sentences = [sent.replace(marker, original) for sent in sentences]

        title_replacements_reverse_sorted = sorted(title_replacements.items(), key=lambda item: int(item[1].strip('#').split('_')[-1]), reverse=True)
        for original, marker in title_replacements_reverse_sorted:
            sentences = [sent.replace(marker, original) for sent in sentences]
        
        # Process sentences to remove newlines and strip whitespace
        sentences = [sent.replace('\n', '').strip() for sent in sentences]
        sentences = [sent for sent in sentences if sent not in ['', ' ']]

        # Concatenate the too-short sentence to the end of the previous sentence if it contains inline citation in the end like "[1]." otherwise to the start of the next sentence
        # new_sentences = [sentences[i] if len(sentences[i]) > 30 else sentences[i] + ' ' + sentences[i+1] for i in range(len(sentences)-1)]
        new_sentences = []
        for i, sent in enumerate(sentences):
            if len(sent) <= 30:
                if re.match(r'\[\d+\]\.', sent) and i > 0:
                    new_sentences[-1] += sent
                elif i < len(sentences) - 1:
                    sentences[i+1] = sent + ' ' + sentences[i+1]
                else:
                    new_sentences.append(sent)
            else:
                new_sentences.append(sent)

        # Format and output the sentences
        if with_bullet_point:
            return [f"{i+1}. {sent}" for i, sent in enumerate(new_sentences)]
        else:
            return new_sentences

if __name__ == "__main__":
    # print("Testing API Keys:")
    # print("GPT API Key:", OPENAI_API_KEY)
    # print("Claude API Key:", ANTHORPIC_API_KEY)
    # print("Gemini API Key:", GOOGLE_API_KEY)
    # print("OpenRouter API Key:", OPENROUTER_API_KEY)
    
    # print("\nTesting LLM class:")
    # prompt = "Hello [NAME], I am an alian. "
    # for provider in ["gpt", "gemini"]:
    #     llm = LLM(prompt=prompt, markers=['[NAME]'], provider=provider)
    #     print(f"Provider: {provider}, Response: {llm.call(['world'])}")
    
    print("\nTesting Prometheus class:")
    prometheus = Prometheus()
    
    # Test case 1: Absolute grading
    print("\nTest Case 1: Absolute Grading")
    instruction = "Explain the concept of machine learning in simple terms."
    response = """Machine learning is a type of artificial intelligence that allows computers to learn from data without being explicitly programmed. It's like teaching a computer to recognize patterns and make decisions based on examples, similar to how humans learn from experience."""
    reference = """Machine learning is a branch of artificial intelligence that enables computers to learn from data and improve their performance over time without being explicitly programmed. It works by identifying patterns in data and using these patterns to make predictions or decisions, much like how humans learn from experience."""
    rubric = "Evaluate based on clarity, accuracy, and completeness of the explanation"
    
    print("\nInstruction:", instruction)
    print("Response:", response)
    print("Reference:", reference)
    print("Rubric:", rubric)
    print("\nEvaluation Result:")
    print(prometheus.absolute_grade(instruction, response, reference, rubric))
    
    # Test case 2: Relative grading
    print("\nTest Case 2: Relative Grading")
    instruction = "Write a short summary of climate change."
    response1 = """Climate change refers to long-term shifts in temperatures and weather patterns. Since the 1800s, human activities have been the main driver of climate change, primarily due to burning fossil fuels like coal, oil and gas."""
    response2 = """Climate change is when the Earth's temperature changes over a long time. People burning things like coal and oil make it happen. This can make the weather different and cause problems for animals and plants."""
    rubric = "Evaluate based on accuracy, completeness, and clarity of the explanation"
    
    print("\nInstruction:", instruction)
    print("Response 1:", response1)
    print("Response 2:", response2)
    print("Rubric:", rubric)
    print("\nEvaluation Result:")
    print(prometheus.relative_grade(instruction, response1, response2, rubric))
    
    # Test case 3: Content segmentation
    print("\nTest Case 3: Content Segmentation")
    segmenter = ContentSegmenter()
    content = """Machine learning is a type of AI. It helps computers learn from data. Deep learning is a subset of ML. It uses neural networks. These networks are inspired by human brains."""
    print("\nOriginal content:", content)
    print("\nSegmented content:")
    print(segmenter.segment(content, with_bullet_point=True))