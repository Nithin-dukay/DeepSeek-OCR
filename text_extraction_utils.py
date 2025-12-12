"""
Text Extraction Utilities for DeepSeek-OCR
Addresses GitHub Issue #151: Provides clean text extraction from structured
outputs, eliminating need for stdout capture.
"""

import re
from typing import Dict, List, Tuple, Optional, Union
from enum import Enum


class OutputFormat(Enum):
    """Supported output formats."""
    MARKDOWN = "markdown"
    PLAIN_TEXT = "plain_text"
    STRUCTURED = "structured"
    RAW = "raw"


def extract_text_from_output(
    output: str,
    format: OutputFormat = OutputFormat.MARKDOWN,
    remove_tags: bool = True,
    remove_headers: bool = False
) -> str:
    """
    Extract clean text from DeepSeek-OCR output.
    
    Handles:
    - Structured <|ref|>/<|det|> blocks
    - Special tokens and markers
    - Format conversion
    
    Args:
        output: Raw model output
        format: Desired output format
        remove_tags: Whether to remove special tags
        remove_headers: Whether to remove header markers
        
    Returns:
        Cleaned text
    """
    text = output
    
    if remove_tags:
        text = remove_special_tags(text)
    
    if remove_headers:
        text = remove_header_markers(text)
    
    if format == OutputFormat.PLAIN_TEXT:
        text = convert_to_plain_text(text)
    elif format == OutputFormat.MARKDOWN:
        text = clean_markdown(text)
    
    return text.strip()


def remove_special_tags(text: str) -> str:
    """
    Remove DeepSeek-OCR special tags.
    
    Tags include:
    - <|ref|>...<|/ref|>: Reference tags
    - <|det|>...<|/det|>: Detection tags
    - <|grounding|>: Grounding marker
    - <image>: Image token
    """
    # Remove ref/det blocks entirely
    text = re.sub(r'<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>', '', text, flags=re.DOTALL)
    
    # Remove individual tags
    text = re.sub(r'<\|grounding\|>', '', text)
    text = re.sub(r'<image>', '', text)
    text = re.sub(r'<\|ref\|>', '', text)
    text = re.sub(r'<\|/ref\|>', '', text)
    text = re.sub(r'<\|det\|>', '', text)
    text = re.sub(r'<\|/det\|>', '', text)
    
    return text


def remove_header_markers(text: str) -> str:
    """Remove common header markers and artifacts."""
    # Remove BOS/EOS tokens if present
    text = re.sub(r'<\|begin_of_text\|>', '', text)
    text = re.sub(r'<\|end_of_text\|>', '', text)
    text = re.sub(r'<s>', '', text)
    text = re.sub(r'</s>', '', text)
    
    return text


def convert_to_plain_text(text: str) -> str:
    """
    Convert markdown/structured text to plain text.
    
    Removes:
    - Markdown formatting
    - HTML tags
    - Special characters
    """
    # Remove markdown headers
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    
    # Remove markdown bold/italic
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    text = re.sub(r'_(.+?)_', r'\1', text)
    
    # Remove markdown links
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    # Remove markdown images
    text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove markdown code blocks
    text = re.sub(r'```[^\n]*\n.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    # Remove markdown tables (convert to plain text)
    text = re.sub(r'\|', ' ', text)
    text = re.sub(r'^[-\s]+$', '', text, flags=re.MULTILINE)
    
    return text


def clean_markdown(text: str) -> str:
    """
    Clean markdown output while preserving structure.
    """
    # Remove excessive newlines
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    
    # Remove empty markdown elements
    text = re.sub(r'^\s*[-*+]\s*$', '', text, flags=re.MULTILINE)
    
    # Clean up table formatting
    text = re.sub(r'\|\s*\|', '|', text)
    
    return text


def parse_structured_output(output: str) -> Dict[str, List[Dict]]:
    """
    Parse structured output with ref/det blocks.
    
    Returns:
        Dictionary with parsed elements:
        - images: List of image references
        - detections: List of detection blocks
        - text: Plain text content
    """
    result = {
        'images': [],
        'detections': [],
        'text': ''
    }
    
    # Extract ref/det blocks
    pattern = r'<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>'
    matches = re.findall(pattern, output, re.DOTALL)
    
    for ref, det in matches:
        ref = ref.strip()
        det = det.strip()
        
        if ref.lower() == 'image':
            result['images'].append({
                'type': 'image',
                'reference': ref,
                'detection': det
            })
        else:
            result['detections'].append({
                'type': ref,
                'reference': ref,
                'detection': det
            })
    
    # Extract plain text (remove all ref/det blocks)
    text = re.sub(pattern, '', output, flags=re.DOTALL)
    text = remove_special_tags(text)
    result['text'] = text.strip()
    
    return result


def extract_images_from_output(
    output: str,
    replace_with_placeholder: bool = True,
    placeholder_format: str = "![](images/{index}.jpg)"
) -> Tuple[str, List[Dict]]:
    """
    Extract image references and optionally replace with placeholders.
    
    Args:
        output: Raw model output
        replace_with_placeholder: Whether to replace with markdown placeholders
        placeholder_format: Format string for placeholders (use {index})
        
    Returns:
        Tuple of (cleaned_text, list of image references)
    """
    images = []
    text = output
    
    # Find image ref/det blocks
    pattern = r'<\|ref\|>image<\|/ref\|><\|det\|>(.*?)<\|/det\|>'
    matches = re.finditer(pattern, text, re.DOTALL | re.IGNORECASE)
    
    for idx, match in enumerate(matches):
        detection = match.group(1).strip()
        
        images.append({
            'index': idx,
            'detection': detection,
            'original': match.group(0)
        })
        
        if replace_with_placeholder:
            placeholder = placeholder_format.format(index=idx)
            text = text.replace(match.group(0), placeholder + '\n', 1)
        else:
            text = text.replace(match.group(0), '', 1)
    
    return text, images


def clean_formula_output(text: str) -> str:
    """
    Clean mathematical formula output.
    
    Removes annotations and cleans LaTeX formatting.
    """
    # Remove quad annotations in formulas
    formula_pattern = r'\\[\[\(](.*?)\\[\]\)]'
    
    def process_formula(match):
        formula = match.group(1)
        # Remove \quad annotations
        formula = re.sub(r'\\quad\s*\([^)]*\)', '', formula)
        formula = formula.strip()
        return match.group(0)[0] + formula + match.group(0)[-1]
    
    text = re.sub(formula_pattern, process_formula, text)
    
    # Replace special LaTeX symbols
    text = text.replace('\\coloneqq', ':=')
    text = text.replace('\\eqqcolon', '=:')
    
    return text


def extract_tables_from_output(output: str) -> List[Dict]:
    """
    Extract tables from markdown output.
    
    Returns:
        List of table dictionaries with headers and rows
    """
    tables = []
    
    # Find markdown tables
    table_pattern = r'(\|.+\|[\r\n]+\|[-:\s|]+\|[\r\n]+(?:\|.+\|[\r\n]*)+)'
    matches = re.finditer(table_pattern, output, re.MULTILINE)
    
    for match in matches:
        table_text = match.group(1)
        lines = [line.strip() for line in table_text.split('\n') if line.strip()]
        
        if len(lines) < 3:
            continue
        
        # Parse header
        header = [cell.strip() for cell in lines[0].split('|')[1:-1]]
        
        # Parse rows (skip separator line)
        rows = []
        for line in lines[2:]:
            row = [cell.strip() for cell in line.split('|')[1:-1]]
            if row:
                rows.append(row)
        
        tables.append({
            'header': header,
            'rows': rows,
            'original': table_text
        })
    
    return tables


def get_text_only_output(
    output: str,
    preserve_structure: bool = True
) -> str:
    """
    Get text-only output, removing all special elements.
    
    This is the recommended method for users who want plain text OCR
    without structured tags or markdown.
    
    Args:
        output: Raw model output
        preserve_structure: Whether to preserve line breaks and paragraphs
        
    Returns:
        Plain text output
    """
    # Remove all special tags
    text = remove_special_tags(output)
    text = remove_header_markers(text)
    
    # Convert to plain text
    text = convert_to_plain_text(text)
    
    if preserve_structure:
        # Clean up excessive whitespace but preserve paragraphs
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
    else:
        # Collapse all whitespace
        text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def example_usage():
    """Example usage of text extraction utilities."""
    
    # Example 1: Raw output with ref/det blocks
    print("=== Example 1: Extract from structured output ===")
    raw_output = """
    <|ref|>title<|/ref|><|det|>[[100, 100, 500, 150]]<|/det|>
    
    # Historical Newspaper Article
    
    This is the main text of the article from the 1800s.
    
    <|ref|>image<|/ref|><|det|>[[50, 200, 600, 400]]<|/det|>
    
    The article continues with more historical information.
    """
    
    clean_text = extract_text_from_output(
        raw_output,
        format=OutputFormat.MARKDOWN,
        remove_tags=True
    )
    print(f"Cleaned markdown:\n{clean_text}\n")
    
    plain_text = extract_text_from_output(
        raw_output,
        format=OutputFormat.PLAIN_TEXT,
        remove_tags=True
    )
    print(f"Plain text:\n{plain_text}\n")
    
    # Example 2: Parse structured output
    print("\n=== Example 2: Parse structured output ===")
    parsed = parse_structured_output(raw_output)
    print(f"Images: {len(parsed['images'])}")
    print(f"Detections: {len(parsed['detections'])}")
    print(f"Text length: {len(parsed['text'])}")
    
    # Example 3: Extract images
    print("\n=== Example 3: Extract images ===")
    text_with_placeholders, images = extract_images_from_output(raw_output)
    print(f"Extracted {len(images)} image(s)")
    print(f"Text with placeholders:\n{text_with_placeholders}")
    
    # Example 4: Get text-only output
    print("\n=== Example 4: Text-only output ===")
    text_only = get_text_only_output(raw_output, preserve_structure=True)
    print(f"Text only:\n{text_only}")
    
    # Example 5: Clean formula output
    print("\n=== Example 5: Clean formula output ===")
    formula_output = r"The equation is \[E = mc^2 \quad (Einstein's equation)\]"
    cleaned_formula = clean_formula_output(formula_output)
    print(f"Original: {formula_output}")
    print(f"Cleaned: {cleaned_formula}")
    
    # Example 6: Extract tables
    print("\n=== Example 6: Extract tables ===")
    table_output = """
    | Year | Event |
    |------|-------|
    | 1800 | First newspaper |
    | 1850 | Second newspaper |
    """
    tables = extract_tables_from_output(table_output)
    print(f"Extracted {len(tables)} table(s)")
    if tables:
        print(f"Headers: {tables[0]['header']}")
        print(f"Rows: {tables[0]['rows']}")


if __name__ == "__main__":
    example_usage()
