# merge_paper_latex.py: Merge multiple LaTeX source files into one
import os
import re
from pathlib import Path
import tarfile


def extract_latex_sources(latex_sources_dir="/path/to/arxiv_latex_sources/physics_new"):
    """
    Extract all LaTeX source packages
    
    Args:
        latex_sources_dir: LaTeX source directory path
    """
    latex_sources_dir = os.path.abspath(latex_sources_dir)
    if not os.path.exists(latex_sources_dir):
        print(f"Directory {latex_sources_dir} does not exist")
        return
    
    # Create the extraction directory
    extracted_dir = os.path.join(latex_sources_dir, "extracted")
    os.makedirs(extracted_dir, exist_ok=True)
    
    # Find all tar.gz files
    for file in os.listdir(latex_sources_dir):
        if file.endswith(".tar.gz"):
            paper_id = file.replace(".tar.gz", "")
            extract_path = os.path.join(extracted_dir, paper_id)
            
            if not os.path.exists(extract_path):
                os.makedirs(extract_path)
                tar_path = os.path.join(latex_sources_dir, file)
                
                try:
                    with tarfile.open(tar_path, "r:gz") as tar:
                        tar.extractall(path=extract_path)
                    print(f"Successfully extracted {file} to {extract_path}")
                except Exception as e:
                    print(f"Failed to extract {file}: {str(e)}")


def find_main_tex_file(directory):
    """
    Find the main .tex file in a directory
    
    Args:
        directory: The directory to search
        
    Returns:
        The path of the main .tex file, or None if not found
    """
    # Common main filenames
    main_filenames = [
        "main.tex", "master.tex", "document.tex", "paper.tex", 
        "article.tex", "thesis.tex", "dissertation.tex"
    ]
    
    # First search for common main filenames
    for filename in main_filenames:
        filepath = os.path.join(directory, filename)
        if os.path.exists(filepath):
            return filepath
    
    # Find all .tex files
    tex_files = list(Path(directory).rglob("*.tex"))
    
    # If there is only one .tex file, it is the main file
    if len(tex_files) == 1:
        return str(tex_files[0])
    
    # Heuristic method: find a file containing \documentclass, which is usually the main file
    for tex_file in tex_files:
        try:
            with open(tex_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                if '\\documentclass' in content:
                    return str(tex_file)
        except Exception as e:
            print(f"Error reading file {tex_file}: {str(e)}")
    
    # If still not found, return the largest .tex file
    if tex_files:
        largest_file = max(tex_files, key=lambda f: f.stat().st_size)
        return str(largest_file)
    
    return None


def collect_input_files(main_tex_content, base_directory):
    """
    Collect all files referenced via \input or \include
    
    Args:
        main_tex_content: The content of the main .tex file
        base_directory: The directory containing the main file
        
    Returns:
        A list of all referenced file paths
    """
    # Find the \input and \include commands
    input_pattern = r'\\(input|include)\{([^}]*)\}'
    matches = re.findall(input_pattern, main_tex_content)
    
    input_files = []
    for command, filename in matches:
        # Handle relative paths
        if not filename.endswith('.tex'):
            filename += '.tex'
            
        file_path = os.path.join(base_directory, filename)
        
        # Check whether the file exists
        if os.path.exists(file_path):
            input_files.append(file_path)
        else:
            # Try to find it in subdirectories
            for root, dirs, files in os.walk(base_directory):
                candidate_path = os.path.join(root, filename)
                if os.path.exists(candidate_path):
                    input_files.append(candidate_path)
                    break
    
    return input_files


def merge_latex_files(main_tex_path, output_path=None):
    """
    Merge LaTeX files into the main file
    
    Args:
        main_tex_path: Path of the main .tex file
        output_path: Output file path, defaults to the original filename plus the '_merged' suffix
    """
    if not os.path.exists(main_tex_path):
        print(f"File {main_tex_path} does not exist")
        return False
    
    # Default output path
    if output_path is None:
        base_name = os.path.splitext(main_tex_path)[0]
        output_path = f"{base_name}_merged.tex"
    
    # Read the main file content
    try:
        with open(main_tex_path, 'r', encoding='utf-8', errors='ignore') as f:
            main_content = f.read()
    except Exception as e:
        print(f"Error reading the main file: {str(e)}")
        return False
    
    # Get the base directory
    base_directory = os.path.dirname(main_tex_path)
    
    # Replace the \input and \include commands
    def replace_input_command(match):
        command, filename = match.groups()
        if not filename.endswith('.tex'):
            filename += '.tex'
            
        # Find the actual file path
        file_path = os.path.join(base_directory, filename)
        if not os.path.exists(file_path):
            # Find it in subdirectories
            for root, dirs, files in os.walk(base_directory):
                candidate_path = os.path.join(root, filename)
                if os.path.exists(candidate_path):
                    file_path = candidate_path
                    break
        
        # Read and return the file content
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    # Remove declarations such as \documentclass that should only appear in the main file
                    content = re.sub(r'\\documentclass.*$', '', content, flags=re.MULTILINE)
                    content = re.sub(r'\\begin\{document\}', '', content, flags=re.MULTILINE)
                    content = re.sub(r'\\end\{document\}', '', content, flags=re.MULTILINE)
                    return f"% Content from {filename}\n{content}\n"
            except Exception as e:
                print(f"Error reading input file {file_path}: {str(e)}")
                return match.group(0)
        else:
            print(f"Input file not found: {filename}")
            return match.group(0)
    
    # Replace all \input and \include commands
    merged_content = re.sub(r'\\(input|include)\{([^}]*)\}', replace_input_command, main_content)
    
    # Write the merged file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(merged_content)
        print(f"Merge completed, output file: {output_path}")
        return True
    except Exception as e:
        print(f"Error writing the output file: {str(e)}")
        return False


def batch_merge_latex_files(latex_sources_dir="/path/to/arxiv_latex_sources/physics_new"):
    """
    Batch process all extracted LaTeX files
    
    Args:
        latex_sources_dir: LaTeX source directory path
    """
    latex_sources_dir = os.path.abspath(latex_sources_dir)
    extracted_dir = os.path.join(latex_sources_dir, "extracted")
    if not os.path.exists(extracted_dir):
        print(f"Extraction directory {extracted_dir} does not exist, run the extraction step first")
        return
    
    merged_dir = os.path.join(latex_sources_dir, "merged")
    os.makedirs(merged_dir, exist_ok=True)
    
    # Iterate over all extracted paper directories
    for paper_dir in os.listdir(extracted_dir):
        paper_path = os.path.join(extracted_dir, paper_dir)
        if os.path.isdir(paper_path):
            print(f"Processing paper: {paper_dir}")
            
            # Find the main .tex file
            main_tex_path = find_main_tex_file(paper_path)
            if main_tex_path:
                print(f"  Found main file: {os.path.basename(main_tex_path)}")
                
                # Set the output path
                output_path = os.path.join(merged_dir, f"{paper_dir}_merged.tex")
                
                # Merge the files
                merge_latex_files(main_tex_path, output_path)
            else:
                print(f"  No main .tex file found")


if __name__ == "__main__":
    # Example usage
    # 1. First extract the source packages
    extract_latex_sources()
    
    # 2. Batch merge the LaTeX files
    batch_merge_latex_files()