"""
LaTeX Formula Processing Utilities

This module provides functions to convert LaTeX formulas from the model's output format
to markdown-compatible formats or special tag formats.

Addresses GitHub Issue #219: Adding special tags for LaTeX formulas
"""

import re
from typing import Literal


def convert_latex_to_markdown(text: str, format_type: Literal['markdown', 'tags'] = 'markdown') -> str:
    """
    Convert LaTeX formulas in text to markdown-compatible format or special tags.
    
    Args:
        text: Input text containing LaTeX formulas in \(...\) and \[...\] format
        format_type: Output format - 'markdown' for $...$ and $$...$$ or 'tags' for <|math|>...<|/math|>
    
    Returns:
        Text with converted LaTeX formulas
    
    Examples:
        >>> convert_latex_to_markdown(r'\(x^2\)', 'markdown')
        '$x^2$'
        >>> convert_latex_to_markdown(r'\[E=mc^2\]', 'markdown')
        '$$E=mc^2$$'
        >>> convert_latex_to_markdown(r'\(x^2\)', 'tags')
        '<|math|>x^2<|/math|>'
    """
    if format_type == 'markdown':
        return _convert_to_markdown_delimiters(text)
    elif format_type == 'tags':
        return _convert_to_special_tags(text)
    else:
        raise ValueError(f"Invalid format_type: {format_type}. Must be 'markdown' or 'tags'")


def _convert_to_markdown_delimiters(text: str) -> str:
    """
    Convert LaTeX formulas to standard markdown format.
    
    Converts:
    - \(...\) to $...$  (inline math)
    - \[...\] to $$...$$ (display math)
    """
    # Convert display math first (to avoid conflicts with inline math)
    # Pattern: \[ ... \] -> $$ ... $$
    text = re.sub(
        r'\\?\\\[(.+?)\\?\\\]',
        r'$$\1$$',
        text,
        flags=re.DOTALL
    )
    
    # Convert inline math
    # Pattern: \( ... \) -> $ ... $
    text = re.sub(
        r'\\?\\\((.+?)\\?\\\)',
        r'$\1$',
        text,
        flags=re.DOTALL
    )
    
    return text


def _convert_to_special_tags(text: str) -> str:
    """
    Convert LaTeX formulas to special tag format.
    
    Converts:
    - \(...\) to <|math|>...<|/math|>  (inline math)
    - \[...\] to <|math|>...<|/math|>  (display math)
    """
    # Convert display math first
    # Pattern: \[ ... \] -> <|math|> ... <|/math|>
    text = re.sub(
        r'\\?\\\[(.+?)\\?\\\]',
        r'<|math|>\1<|/math|>',
        text,
        flags=re.DOTALL
    )
    
    # Convert inline math
    # Pattern: \( ... \) -> <|math|> ... <|/math|>
    text = re.sub(
        r'\\?\\\((.+?)\\?\\\)',
        r'<|math|>\1<|/math|>',
        text,
        flags=re.DOTALL
    )
    
    return text


def clean_formula(text: str) -> str:
    """
    Clean formula text by removing unnecessary annotations.
    
    This function is maintained for backward compatibility with existing code.
    It removes \quad annotations from display math formulas.
    
    Args:
        text: Input text containing LaTeX formulas
    
    Returns:
        Cleaned text
    """
    formula_pattern = r'\\?\\\[(.*?)\\?\\\]'
    
    def process_formula(match):
        formula = match.group(1)
        # Remove \quad annotations
        formula = re.sub(r'\\quad\s*\([^)]*\)', '', formula)
        formula = formula.strip()
        return r'\[' + formula + r'\]'
    
    cleaned_text = re.sub(formula_pattern, process_formula, text, flags=re.DOTALL)
    return cleaned_text


def process_model_output(text: str, latex_format: Literal['markdown', 'tags'] = 'markdown', 
                        clean_formulas: bool = True) -> str:
    """
    Process model output text to handle LaTeX formulas.
    
    This is the main function to use in processing scripts.
    
    Args:
        text: Raw model output text
        latex_format: Output format for LaTeX - 'markdown' or 'tags'
        clean_formulas: Whether to clean formulas (remove \quad annotations)
    
    Returns:
        Processed text with properly formatted LaTeX
    
    Example:
        >>> output = model.generate(...)
        >>> processed = process_model_output(output, latex_format='markdown')
    """
    # First clean formulas if requested
    if clean_formulas:
        text = clean_formula(text)
    
    # Then convert to desired format
    text = convert_latex_to_markdown(text, format_type=latex_format)
    
    return text
