import os
import json
import re
from typing import List, Dict


def split_paper_by_section(content: str) -> List[str]:
    """
    Split the paper body by \\section
    The title and abstract before the first section are treated as a separate item
    """
    # Use a regex to find all positions of \section
    # Matches the \section{...} form, but excludes \subsection or \subsubsection
    section_pattern = r'\\section\{'
    
    # Find the starting positions of all sections
    section_starts = []
    for match in re.finditer(section_pattern, content):
        section_starts.append(match.start())
    
    if not section_starts:
        # If no sections are found, return the entire content as one part
        return [content] if content.strip() else []
    
    # Split the content
    sections = []
    
    # Add the content before the first section (usually the title and abstract)
    if section_starts[0] > 0:
        sections.append(content[:section_starts[0]].strip())
    
    # Process each section and its content one by one
    for i in range(len(section_starts)):
        start_pos = section_starts[i]
        
        # Determine the end position (the start of the next section or the end of the document)
        if i + 1 < len(section_starts):
            end_pos = section_starts[i + 1]
        else:
            end_pos = len(content)
        
        # Extract the section content
        section_content = content[start_pos:end_pos]
        sections.append(section_content.strip())
    
    # Filter out empty parts
    sections = [section for section in sections if section]
    
    return sections


def process_tex_file(tex_path: str, output_dir: str):
    """
    Process a single .tex file, split it by sections and save it as a JSON file
    """
    # Read the .tex file content
    with open(tex_path, 'r', encoding='utf-8') as f:
        paper_content = f.read()
    
    # Split the paper by sections
    sections = split_paper_by_section(paper_content)
    
    # Skip the paper if any section length exceeds 35000 characters
    for section in sections:
        if len(section) > 35000:
            print(f"Skipped {tex_path}: section exceeds 20480 chars ({len(section)} chars)")
            return

    # Create a list item for each section
    result_data = {
        "filename": os.path.basename(tex_path),
        "sections": sections
    }
    
    # Generate the output filename
    base_name = os.path.splitext(os.path.basename(tex_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_divided.json")
    
    # Write the new JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    
    print(f"Processed {tex_path} -> {output_path}")
    print(f"  - Found {len(sections)} sections")


def process_all_tex_files(input_dir: str, output_dir: str):
    """
    Process all .tex files in the input directory
    """
    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all .tex files
    tex_files = [f for f in os.listdir(input_dir) if f.endswith('.tex')]
    
    print(f"Found {len(tex_files)} .tex files to process")
    
    for i, filename in enumerate(tex_files):
        input_path = os.path.join(input_dir, filename)
        try:
            print(f"Processing file {i+1}/{len(tex_files)}: {filename}")
            process_tex_file(input_path, output_dir)
        except Exception as e:
            print(f"Error processing {input_path}: {str(e)}")


if __name__ == "__main__":
    input_directory = "/path/to/paper_bodies/physics_new"
    output_directory = "/path/to/paper_sections/physics_new"
    
    process_all_tex_files(input_directory, output_directory)