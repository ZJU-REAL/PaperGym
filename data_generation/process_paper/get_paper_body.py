# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# get_paper_body: Extract the body content from LaTeX files, while preserving the title and abstract parts

import os
import re
from pathlib import Path


def remove_comments(content):
    """
    Remove comments from a LaTeX file
    
    Args:
        content: LaTeX file content
        
    Returns:
        The content with comments removed
    """
    # Remove line comments, but preserve the percent signs that may belong to the comments themselves
    # Be careful to avoid accidentally removing percent signs in URLs
    lines = content.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Find comment symbols that are not part of a URL
        comment_pos = -1
        for i, char in enumerate(line):
            if char == '%' and (i == 0 or line[i-1] != '\\'):
                # Check whether it is part of a URL
                if 'http' not in line[max(0, i-10):i+10]:
                    comment_pos = i
                    break
        
        if comment_pos != -1:
            # Keep the part before the comment and strip the trailing whitespace
            cleaned_line = line[:comment_pos].rstrip()
        else:
            cleaned_line = line
            
        cleaned_lines.append(cleaned_line)
    
    # Handle comments within and between paragraphs
    processed_lines = []
    i = 0
    
    while i < len(cleaned_lines):
        line = cleaned_lines[i]
        
        # If the current line has content, add it directly
        if line.strip():
            processed_lines.append(line)
            i += 1
        else:
            # The current line is empty, or became empty after comment removal
            
            # Count the number of consecutive empty lines
            empty_count = 0
            j = i
            while j < len(cleaned_lines) and not cleaned_lines[j].strip():
                empty_count += 1
                j += 1
            
            # Determine the nature of the empty lines:
            # 1. If there is only one empty line and both sides have content, it may be a separator caused
            #    by comments, so the paragraphs should be merged
            # 2. If there are multiple consecutive empty lines, it is a real paragraph separator
            
            # Check whether there is content before
            has_prev_content = False
            prev_idx = len(processed_lines) - 1
            while prev_idx >= 0:
                if processed_lines[prev_idx].strip():
                    has_prev_content = True
                    break
                prev_idx -= 1
            
            # Check whether there is content after
            has_next_content = False
            next_idx = j
            while next_idx < len(cleaned_lines):
                if cleaned_lines[next_idx].strip():
                    has_next_content = True
                    break
                next_idx += 1
            
            if empty_count == 1 and has_prev_content and has_next_content:
                # A single empty line with content on both sides is likely a separator caused by comments;
                # do not add an empty line
                pass
            elif empty_count > 1:
                # Multiple consecutive empty lines indicate a real paragraph separator; keep one empty line
                processed_lines.append("")
            
            # Process the next non-empty line
            if j < len(cleaned_lines):
                i = j
                # Continue processing the next non-empty line
                if cleaned_lines[i].strip():
                    processed_lines.append(cleaned_lines[i])
                    i += 1
            else:
                i = j
    
    # Remove possible empty lines at the beginning
    while processed_lines and not processed_lines[0].strip():
        processed_lines.pop(0)
        
    # Remove possible empty lines at the end
    while processed_lines and not processed_lines[-1].strip():
        processed_lines.pop()
    
    return '\n'.join(processed_lines)


def remove_header_footer(content):
    """
    Remove header, footer and other formatting settings
    
    Args:
        content: LaTeX content
        
    Returns:
        The content with headers and footers removed
    """
    # Remove page style settings
    content = re.sub(r'\\pagestyle\{[^}]*\}', '', content)
    content = re.sub(r'\\thispagestyle\{[^}]*\}', '', content)
    
    # Remove header/footer-related parameters that may appear in geometry settings
    content = re.sub(r'\\fancyhf\{[^}]*\}', '', content)
    content = re.sub(r'\\fancyhead\[[^\]]*\]\{[^}]*\}', '', content)
    content = re.sub(r'\\fancyfoot\[[^\]]*\]\{[^}]*\}', '', content)
    content = re.sub(r'\\renewcommand\\headrulewidth\{[^}]*\}', '', content)
    content = re.sub(r'\\renewcommand\\footrulewidth\{[^}]*\}', '', content)
    
    return content


def extract_title(content):
    """
    Extract the paper title
    
    Args:
        content: LaTeX content
        
    Returns:
        The title content or an empty string
    """
    # Find the \title{...} command
    title_match = re.search(r'\\title(?:\[[^\]]*\])?\{((?:[^{}]|\{[^{}]*\})*)\}', content, re.DOTALL)
    if title_match:
        title = title_match.group(1)
        # Simply clean up the commands in the title (remove common formatting commands)
        title = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', title)  # Remove simple nested commands
        title = re.sub(r'\\[a-zA-Z]+', '', title)  # Remove other commands
        return title.strip()
    return ""


def extract_abstract(content):
    """
    Extract the abstract content
    
    Args:
        content: LaTeX content
        
    Returns:
        The abstract content or an empty string
    """
    # Find the \begin{abstract}...\end{abstract} environment
    abstract_match = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', content, re.DOTALL)
    if abstract_match:
        abstract = abstract_match.group(1)
        # Simply clean up the commands in the abstract
        abstract = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', abstract)  # Remove simple nested commands
        abstract = re.sub(r'\\[a-zA-Z]+', '', abstract)  # Remove other commands
        return abstract.strip()
    return ""


def extract_body_content(content):
    """
    Extract the body content, preserving the title and abstract parts
    
    Args:
        content: LaTeX content
        
    Returns:
        The content containing the title, abstract and body
    """
    # Extract the title and abstract
    title = extract_title(content)
    # abstract = extract_abstract(content)

    # print("[abstract is]: ", abstract)
    
    # Find the position of begin{document}
    doc_begin = re.search(r'\\begin\{document\}', content)
    if not doc_begin:
        return content
    
    # Start processing from after \begin{document}
    body_start = doc_begin.end()
    body_content = content[body_start:]
    
    # Remove the \maketitle command, because we have already extracted the title
    body_content = re.sub(r'\\maketitle', '', body_content)
    
    # Remove the references section (including \bibliography and \begin{thebibliography})
    body_content = re.sub(r'\\bibliography\{[^}]*\}', '', body_content)
    body_content = re.sub(r'\\begin\{thebibliography\}.*?\\end\{thebibliography\}', '', body_content, flags=re.DOTALL)
    
    # Remove the appendix section
    body_content = re.sub(r'\\appendix.*', '', body_content, flags=re.DOTALL)
    
    # Remove \end{document} and everything after it
    body_content = re.sub(r'\\end\{document\}.*', '', body_content, flags=re.DOTALL)
    
    # Prepend the title (if it exists) before the body
    header_content = ""
    if title:
        header_content += f"\\title{{{title}}}\n"
    body_content = header_content + body_content
    
    return body_content


def clean_latex_content(content):
    """
    Clean up LaTeX content, removing excess whitespace and empty commands, but preserving paragraph structure, title and abstract
    
    Args:
        content: The content to be cleaned
        
    Returns:
        The cleaned content
    """
    # 1. Basic whitespace character cleanup
    # First remove consecutive spaces within lines
    content = re.sub(r' +', ' ', content)
    
    # 2. Key step for preserving paragraph structure
    # Split the content by lines and determine whether each line is valid content
    lines = content.split('\n')
    preserved_lines = []
    for line in lines:
        stripped_line = line.strip()
        # If the line is empty after stripping whitespace, treat it as a potential paragraph separator
        # We keep an empty string to mark this position
        if not stripped_line:
            preserved_lines.append('')
        else:
            # Otherwise, preserve the processed line content
            preserved_lines.append(line)
    
    # Reassemble the list into a string
    content = '\n'.join(preserved_lines)
    
    # 2. Formatting command cleanup
    # Remove empty LaTeX commands (e.g., \quad, \qquad, etc.)
    content = re.sub(r'\\quad\s*', ' ', content)
    content = re.sub(r'\\qquad\s*', ' ', content)
    
    # Remove empty environment commands (e.g., empty emph, textbf, etc.)
    content = re.sub(r'\\emph\{\s*\}', '', content)
    content = re.sub(r'\\textbf\{\s*\}', '', content)
    content = re.sub(r'\\textit\{\s*\}', '', content)
    content = re.sub(r'\\texttt\{\s*\}', '', content)
    content = re.sub(r'\\textsc\{\s*\}', '', content)
    content = re.sub(r'\\textsl\{\s*\}', '', content)
    content = re.sub(r'\\textrm\{\s*\}', '', content)
    
    # Remove useless commands
    content = re.sub(r'\\hfill\s*', '', content)
    content = re.sub(r'\\vspace\{[^}]*\}', '', content)
    content = re.sub(r'\\hspace\{[^}]*\}', '', content)
    
    # 3. Image-related content cleanup
    # Remove the includegraphics command
    content = re.sub(r'\\includegraphics(?:\[[^\]]*\])?\{[^\}]*\}', '', content)
    
    # Remove the figure environment (including figure*, figureH and other variants)
    content = re.sub(r'\\begin\{figure\*?\}.*?\\end\{figure\*?\}', '', content, flags=re.DOTALL)
    
    # Remove the wrapfigure environment
    content = re.sub(r'\\begin\{wrapfigure\}.*?\\end\{wrapfigure\}', '', content, flags=re.DOTALL)
    
    # Remove the subfigure environment
    content = re.sub(r'\\begin\{subfigure\}.*?\\end\{subfigure\}', '', content, flags=re.DOTALL)
    
    # Remove commands related to the graphicx package
    content = re.sub(r'\\graphicspath\{[^\}]*\}', '', content)
    
    # Remove image-related caption and related commands
    content = re.sub(r'\\caption(?:\[[^\]]*\])?\{[^\}]*\}', '', content)
    content = re.sub(r'\\captionsetup\{[^\}]*\}', '', content)
    
    # Remove float position control commands
    content = re.sub(r'\\centering\s*', '', content)
    content = re.sub(r'\\raggedleft\s*', '', content)
    content = re.sub(r'\\raggedright\s*', '', content)
    
    # Remove label commands (especially image-related ones)
    content = re.sub(r'\\label\{fig:[^\}]*\}', '', content)
    content = re.sub(r'\\label\{Figure:[^\}]*\}', '', content)
    content = re.sub(r'\\label\{figure:[^\}]*\}', '', content)
    
    # 4. Table-related content cleanup
    # Remove table-related environments
    content = re.sub(r'\\begin\{table\*?\}.*?\\end\{table\*?\}', '', content, flags=re.DOTALL)
    content = re.sub(r'\\begin\{tabular\}.*?\\end\{tabular\}', '', content, flags=re.DOTALL)
    content = re.sub(r'\\begin\{adjustbox\}.*?\\end\{adjustbox\}', '', content, flags=re.DOTALL)
    # Keep the center environment; the removal code is commented out
    # content = re.sub(r'\\begin\{center\}.*?\\end\{center\}', '', content, flags=re.DOTALL)
    
    # Remove table-related commands
    content = re.sub(r'\\toprule\s*', '', content)
    content = re.sub(r'\\midrule\s*', '', content)
    content = re.sub(r'\\bottomrule\s*', '', content)
    content = re.sub(r'\\cmidrule\{[^\}]*\}', '', content)
    content = re.sub(r'\\cline\{[^\}]*\}', '', content)
    content = re.sub(r'\\multicolumn\{[^\}]*\}', '', content)
    content = re.sub(r'\\multirow\{[^\}]*\}', '', content)
    
    # 5. List-related content cleanup
    # Remove list environments
    content = re.sub(r'\\begin\{enumerate\}.*?\\end\{enumerate\}', '', content, flags=re.DOTALL)
    content = re.sub(r'\\begin\{itemize\}.*?\\end\{itemize\}', '', content, flags=re.DOTALL)
    content = re.sub(r'\\begin\{list\}.*?\\end\{list\}', '', content, flags=re.DOTALL)
    content = re.sub(r'\\begin\{description\}.*?\\end\{description\}', '', content, flags=re.DOTALL)
    
    # Remove list-related commands
    content = re.sub(r'\\item\s*', '', content)
    content = re.sub(r'\\item\[.*?\]\s*', '', content)
    
    # 6. Math-related content cleanup
    # Comment out the code that removes math environments to preserve formula content
    # content = re.sub(r'\\begin\{align\*?\}.*?\\end\{align\*?\}', '', content, flags=re.DOTALL)
    # content = re.sub(r'\\begin\{equation\*?\}.*?\\end\{equation\*?\}', '', content, flags=re.DOTALL)
    # content = re.sub(r'\\begin\{eqnarray\*?\}.*?\\end\{eqnarray\*?\}', '', content, flags=re.DOTALL)
    # content = re.sub(r'\\begin\{multline\*?\}.*?\\end\{multline\*?\}', '', content, flags=re.DOTALL)
    # content = re.sub(r'\\begin\{gather\*?\}.*?\\end\{gather\*?\}', '', content, flags=re.DOTALL)
    # content = re.sub(r'\\begin\{alignat\*?\}.*?\\end\{alignat\*?\}', '', content, flags=re.DOTALL)
    # content = re.sub(r'\\begin\{flalign\*?\}.*?\\end\{flalign\*?\}', '', content, flags=re.DOTALL)
    
    # Keep the displaymath environment, but clean up the commands inside it
    # content = re.sub(r'\\begin\{displaymath\}.*?\\end\{displaymath\}', '', content, flags=re.DOTALL)
    
    # Remove math-related commands, but keep the environments
    content = re.sub(r'\\\$', '', content)  # Remove the math mode toggle symbol
    # content = re.sub(r'\\begin\{math\}.*?\\end\{math\}', '', content, flags=re.DOTALL)
    # content = re.sub(r'\\begin\{align\*?\}.*?\\end\{align\*?\}', '', content, flags=re.DOTALL)
    
    # 7. Reference/citation-related content cleanup
    # Remove cite commands, including various variants
    content = re.sub(r'\\cite\{[^\}]*\}', '', content)
    content = re.sub(r'\\cite\[.*?\]\{[^\}]*\}', '', content)  # \cite[p.10]{key} with optional argument
    content = re.sub(r'\\newcite\{[^\}]*\}', '', content)
    content = re.sub(r'\\newcite\[.*?\]\{[^\}]*\}', '', content)  # \newcite[p.10]{key} with optional argument
    content = re.sub(r'\\cite[tp]\{[^\}]*\}', '', content)  # \citet, \citep
    content = re.sub(r'\\citealp\{[^\}]*\}', '', content)  # \citealp
    content = re.sub(r'\\citealt\{[^\}]*\}', '', content)  # \citealt
    content = re.sub(r'\\Citet\{[^\}]*\}', '', content)  # \Citet
    content = re.sub(r'\\Citep\{[^\}]*\}', '', content)  # \Citep
    
    # Remove the bibitem environment
    content = re.sub(r'\\begin\{bibitem\}.*?\\end\{bibitem\}', '', content, flags=re.DOTALL)
    
    # Remove bibstyle and related commands
    content = re.sub(r'\\bibliographystyle\{[^\}]*\}', '', content)
    content = re.sub(r'\\bibstyle\{[^\}]*\}', '', content)
    
    # Remove the nocite command
    content = re.sub(r'\\nocite\{[^\}]*\}', '', content)
    
    # Remove reference-related labels
    content = re.sub(r'\\label\{bib:[^\}]*\}', '', content)
    content = re.sub(r'\\label\{cite:[^\}]*\}', '', content)
    
    # Remove reference database citations
    content = re.sub(r'\\bibliography\{[^\}]*\}', '', content)
    
    # Remove the thebibliography environment
    content = re.sub(r'\\begin\{thebibliography\}.*?\\end\{thebibliography\}', '', content, flags=re.DOTALL)
    
    # Remove citation-related packages
    content = re.sub(r'\\usepackage\{natbib\}', '', content)
    content = re.sub(r'\\usepackage\{biblatex\}', '', content)
    
    # Remove citation-related setup commands
    content = re.sub(r'\\bibliographystyle\{[^\}]*\}', '', content)
    content = re.sub(r'\\setcitestyle\{[^\}]*\}', '', content)
    
    # 8. Other environment and command cleanup
    # Remove other environments
    content = re.sub(r'\\begin\{quote\}.*?\\end\{quote\}', '', content, flags=re.DOTALL)
    content = re.sub(r'\\begin\{quotation\}.*?\\end\{quotation\}', '', content, flags=re.DOTALL)
    content = re.sub(r'\\begin\{verse\}.*?\\end\{verse\}', '', content, flags=re.DOTALL)
    content = re.sub(r'\\begin\{verbatim\}.*?\\end\{verbatim\}', '', content, flags=re.DOTALL)
    content = re.sub(r'\\begin\{code\}.*?\\end\{code\}', '', content, flags=re.DOTALL)
    
    # Remove other commands
    content = re.sub(r'\\noindent\s*', '', content)
    content = re.sub(r'\\small\s*', '', content)
    content = re.sub(r'\\footnotesize\s*', '', content)
    content = re.sub(r'\\scriptsize\s*', '', content)
    content = re.sub(r'\\tiny\s*', '', content)
    content = re.sub(r'\\large\s*', '', content)
    content = re.sub(r'\\Large\s*', '', content)
    content = re.sub(r'\\LARGE\s*', '', content)
    content = re.sub(r'\\huge\s*', '', content)
    content = re.sub(r'\\Huge\s*', '', content)
    
    # Remove url commands
    content = re.sub(r'\\url\{[^\}]*\}', '', content)
    content = re.sub(r'\\href\{[^\}]*\}\{[^\}]*\}', '', content)
    
    # Remove footnote commands
    content = re.sub(r'\\footnote\{[^\}]*\}', '', content)
    
    # Remove some special commands
    content = re.sub(r'\\FloatBarrier\s*', '', content)
    content = re.sub(r'\\newpage\s*', '', content)
    content = re.sub(r'\\clearpage\s*', '', content)
    content = re.sub(r'\\pagebreak\s*', '', content)
    content = re.sub(r'\\nobreak\s*', '', content)
    
    # Remove custom commands (e.g., \se)
    # Fix the regex error and correctly extract the content in \se{...}
    content = re.sub(r'\\se\{([^\}]*)\}', r'\1', content)
    
    # Add code to remove all label tags
    # Remove all \label{...} tags, including various types (e.g., label{sec:...}, label{eq:...}, etc.)
    content = re.sub(r'\\label\{[^}]*\}', '', content)
    
    # 9. Final cleanup
    # Compress multiple consecutive empty lines into a single empty line
    content = re.sub(r'\n\s*\n\s*\n(\s*\n)*', '\n\n', content)
    
    # Finally, remove any extra empty lines at the end of the file that may result from compression
    content = re.sub(r'\n\s*$', '\n', content)
    
    return content


def process_single_file(input_path, output_path):
    """
    Process a single LaTeX file
    
    Args:
        input_path: Input file path
        output_path: Output file path
    """
    try:
        # Use absolute paths
        input_path = os.path.abspath(input_path)
        output_path = os.path.abspath(output_path)
        
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Apply the processing steps
        content = remove_comments(content)
        content = remove_header_footer(content)
        content = extract_body_content(content)
        content = clean_latex_content(content)
        
        # Write the output file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        print(f"Processing completed: {input_path} -> {output_path}")
        return True
    except Exception as e:
        print(f"Error processing file {input_path}: {str(e)}")
        return False


def batch_process_files(input_dir="/path/to/arxiv_latex_sources/physics_new/merged", 
                       output_dir="/path/to/paper_bodies/physics_new"):
    """
    Batch process LaTeX files
    
    Args:
        input_dir: Input directory path
        output_dir: Output directory path
    """
    # Use absolute paths
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    
    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all merged .tex files
    input_path = Path(input_dir)
    tex_files = list(input_path.glob("*.tex"))
    
    if not tex_files:
        print(f"No .tex files found in directory {input_dir}")
        return
    
    print(f"Found {len(tex_files)} files to process")
    
    # Process each file
    success_count = 0
    for tex_file in tex_files:
        output_file = os.path.join(output_dir, f"{tex_file.stem}_body.tex")
        if process_single_file(str(tex_file), output_file):
            success_count += 1
    
    print(f"Batch processing completed: {success_count}/{len(tex_files)} files succeeded")


if __name__ == "__main__":
    batch_process_files()