"""
Utility functions for prompt validation and optimization for DeepSeek-OCR.

This module provides helper functions to ensure prompts work correctly with the model
and the n-gram repetition prevention logic.
"""

import re
from typing import Tuple, Optional


def validate_prompt(prompt: str, image_token: str = "<image>") -> Tuple[bool, Optional[str]]:
    """
    Validate a prompt for use with DeepSeek-OCR.
    
    Args:
        prompt: The prompt string to validate
        image_token: The image token used in the prompt (default: "<image>")
    
    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if prompt is valid, False otherwise
        - error_message: None if valid, error description if invalid
    """
    if not prompt:
        return False, "Prompt cannot be empty"
    
    if image_token not in prompt:
        return False, f"Prompt must contain the image token '{image_token}'"
    
    # Check if prompt starts with image token
    if not prompt.startswith(image_token):
        return False, f"Prompt should start with '{image_token}'"
    
    # Count image tokens
    image_token_count = prompt.count(image_token)
    if image_token_count > 1:
        return False, f"Prompt should contain exactly one '{image_token}' token, found {image_token_count}"
    
    return True, None


def optimize_prompt(prompt: str, max_length: int = 200) -> str:
    """
    Optimize a prompt for better model performance.
    
    This function:
    - Removes excessive whitespace
    - Ensures proper formatting
    - Truncates if too long (while preserving image token)
    
    Args:
        prompt: The prompt to optimize
        max_length: Maximum recommended prompt length (default: 200 characters)
    
    Returns:
        Optimized prompt string
    """
    # Remove excessive whitespace while preserving single spaces and newlines
    prompt = re.sub(r' +', ' ', prompt)  # Multiple spaces to single space
    prompt = re.sub(r'\n+', '\n', prompt)  # Multiple newlines to single newline
    prompt = prompt.strip()
    
    # If prompt is too long, truncate while keeping image token
    if len(prompt) > max_length:
        image_token = "<image>"
        if image_token in prompt:
            parts = prompt.split(image_token, 1)
            if len(parts) == 2:
                # Keep image token and truncate the instruction part
                instruction = parts[1].strip()
                max_instruction_len = max_length - len(image_token) - 2  # -2 for newline
                if len(instruction) > max_instruction_len:
                    instruction = instruction[:max_instruction_len].rsplit(' ', 1)[0] + "."
                prompt = f"{image_token}\n{instruction}"
    
    return prompt


def get_recommended_prompts() -> dict:
    """
    Get a dictionary of recommended prompts for different use cases.
    
    Returns:
        Dictionary mapping use case names to recommended prompt strings
    """
    return {
        "document_to_markdown": "<image>\n<|grounding|>Convert the document to markdown.",
        "document_to_markdown_simple": "<image>\n<|grounding|>Convert to markdown.",
        "ocr_image": "<image>\n<|grounding|>OCR this image.",
        "ocr_simple": "<image>\n<|grounding|>Extract text.",
        "free_ocr": "<image>\nFree OCR.",
        "parse_figure": "<image>\nParse the figure.",
        "describe_image": "<image>\nDescribe this image in detail.",
        "describe_simple": "<image>\nDescribe this image.",
    }


def suggest_prompt_fix(original_prompt: str) -> str:
    """
    Suggest a fixed version of a problematic prompt.
    
    This function analyzes common issues and suggests corrections:
    - Too long prompts are shortened
    - Redundant instructions are simplified
    - Complex phrasing is simplified
    
    Args:
        original_prompt: The original prompt that may have issues
    
    Returns:
        Suggested improved prompt
    """
    prompt = original_prompt.strip()
    
    # Extract the instruction part (after image token)
    image_token = "<image>"
    if image_token not in prompt:
        return prompt
    
    parts = prompt.split(image_token, 1)
    if len(parts) != 2:
        return prompt
    
    instruction = parts[1].strip()
    
    # Common simplifications
    simplifications = {
        # Long instructions -> Short instructions
        r"convert the document to markdown,?\s*don'?t add any extra space between letters\.?": 
            "<|grounding|>Convert the document to markdown.",
        r"convert the document to markdown,?\s*without\s+extra\s+spaces?\.?": 
            "<|grounding|>Convert the document to markdown.",
        r"convert the document to markdown,?\s*keep\s+original\s+formatting\.?": 
            "<|grounding|>Convert the document to markdown.",
        r"convert the document to markdown,?\s*preserve\s+layout\.?": 
            "<|grounding|>Convert the document to markdown.",
        # OCR variations
        r"ocr this image,?\s*extract all text\.?": 
            "<|grounding|>OCR this image.",
        r"extract all text from this image\.?": 
            "<|grounding|>OCR this image.",
        r"read all text in this image\.?": 
            "<|grounding|>OCR this image.",
    }
    
    instruction_lower = instruction.lower()
    for pattern, replacement in simplifications.items():
        if re.search(pattern, instruction_lower, re.IGNORECASE):
            return f"{image_token}\n{replacement}"
    
    # If no specific pattern matched but instruction is long, suggest shortening
    if len(instruction) > 100:
        # Try to extract the core task
        if "markdown" in instruction_lower:
            return f"{image_token}\n<|grounding|>Convert the document to markdown."
        elif "ocr" in instruction_lower or "text" in instruction_lower:
            return f"{image_token}\n<|grounding|>OCR this image."
        elif "describe" in instruction_lower:
            return f"{image_token}\nDescribe this image."
    
    return prompt


def explain_prompt_issue(prompt: str) -> str:
    """
    Explain potential issues with a given prompt.
    
    Args:
        prompt: The prompt to analyze
    
    Returns:
        Explanation of potential issues, or empty string if no issues found
    """
    issues = []
    
    # Check length
    if len(prompt) > 150:
        issues.append(
            f"Prompt is quite long ({len(prompt)} characters). "
            "Shorter, more concise prompts often work better. "
            "Consider simplifying to the core instruction."
        )
    
    # Check for redundant instructions
    redundant_patterns = [
        (r"don'?t\s+add\s+extra\s+space", "The model handles spacing automatically"),
        (r"without\s+extra\s+space", "The model handles spacing automatically"),
        (r"keep\s+original\s+format", "The model preserves formatting by default"),
        (r"preserve\s+layout", "The model preserves layout by default with <|grounding|>"),
        (r"extract\s+all\s+text", "OCR instruction already implies extracting all text"),
    ]
    
    prompt_lower = prompt.lower()
    for pattern, explanation in redundant_patterns:
        if re.search(pattern, prompt_lower):
            issues.append(
                f"Instruction contains redundant guidance: '{pattern}'. {explanation}."
            )
    
    # Check for multiple instructions
    instruction_keywords = ["convert", "extract", "ocr", "describe", "parse", "read"]
    keyword_count = sum(1 for kw in instruction_keywords if kw in prompt_lower)
    if keyword_count > 2:
        issues.append(
            "Prompt contains multiple different instructions. "
            "Focus on one primary task for best results."
        )
    
    if issues:
        return "\n".join(f"- {issue}" for issue in issues)
    
    return ""


# Example usage and testing
if __name__ == "__main__":
    # Test cases
    test_prompts = [
        "<image>\n<|grounding|>Convert the document to markdown.",
        "<image>\n<|grounding|>Convert the document to markdown, dont add any extra space between letters.",
        "<image>\n<|grounding|>OCR this image and extract all text without adding extra spaces.",
        "<image>\nFree OCR.",
    ]
    
    print("=== Prompt Validation and Optimization ===\n")
    
    for prompt in test_prompts:
        print(f"Original: {prompt}")
        
        # Validate
        is_valid, error = validate_prompt(prompt)
        print(f"Valid: {is_valid}")
        if error:
            print(f"Error: {error}")
        
        # Check for issues
        issues = explain_prompt_issue(prompt)
        if issues:
            print(f"Issues:\n{issues}")
        
        # Suggest fix
        suggested = suggest_prompt_fix(prompt)
        if suggested != prompt:
            print(f"Suggested: {suggested}")
        
        print("-" * 80)
        print()
