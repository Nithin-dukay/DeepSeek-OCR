"""
Utility functions for prompt validation and preprocessing.
This module helps prevent issues with prompt modifications that can cause model misbehavior.
"""

import re
from typing import Tuple, Optional


def validate_prompt(prompt: str, image_token: str = "<image>") -> Tuple[bool, Optional[str]]:
    """
    Validate a prompt to ensure it follows the expected format.
    
    Args:
        prompt: The prompt string to validate
        image_token: The image token to look for (default: "<image>")
    
    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if prompt is valid, False otherwise
        - error_message: None if valid, error description if invalid
    """
    
    # Check if prompt contains image token
    if image_token not in prompt:
        return False, f"Prompt must contain '{image_token}' token"
    
    # Check if prompt is not empty after image token
    parts = prompt.split(image_token)
    if len(parts) < 2:
        return False, "Prompt must have content after image token"
    
    after_image = parts[1].strip()
    if not after_image:
        return False, "Prompt must have non-empty content after image token"
    
    # Check for excessive length (very long prompts can cause issues)
    if len(prompt) > 1000:
        return False, "Prompt is too long (max 1000 characters recommended)"
    
    return True, None


def normalize_prompt(prompt: str) -> str:
    """
    Normalize a prompt to ensure consistent formatting.
    
    Args:
        prompt: The prompt string to normalize
    
    Returns:
        Normalized prompt string
    """
    
    # Remove excessive whitespace
    prompt = re.sub(r'\s+', ' ', prompt)
    
    # Ensure proper spacing around special tokens
    prompt = re.sub(r'<image>\s*', '<image>\n', prompt)
    prompt = re.sub(r'<\|grounding\|>\s*', '<|grounding|>', prompt)
    
    # Ensure prompt ends with proper punctuation or space
    if prompt and not prompt[-1] in '.!? \n':
        prompt += ' '
    
    return prompt


def get_recommended_ngram_params(prompt: str, base_ngram_size: int = 30) -> dict:
    """
    Get recommended n-gram parameters based on prompt characteristics.
    
    Args:
        prompt: The prompt string
        base_ngram_size: Base n-gram size to use (default: 30)
    
    Returns:
        Dictionary with recommended parameters for NoRepeatNGramLogitsProcessor
    """
    
    # Calculate prompt complexity - count words after the image token
    if '<image>' in prompt:
        # Only count the instruction part, not the image token
        instruction_part = prompt.split('<image>')[-1]
        prompt_length = len(instruction_part.split())
    else:
        prompt_length = len(prompt.split())
    
    # For longer or more complex prompts, adjust parameters
    if prompt_length > 10:
        # Longer prompts need more conservative n-gram blocking
        return {
            'ngram_size': base_ngram_size,
            'window_size': 90,
            'min_generated_tokens': 20,  # Wait longer before applying blocking
        }
    else:
        # Shorter prompts can use standard settings
        return {
            'ngram_size': base_ngram_size,
            'window_size': 90,
            'min_generated_tokens': 10,
        }


def sanitize_prompt(prompt: str, image_token: str = "<image>") -> str:
    """
    Sanitize a prompt to remove potentially problematic patterns.
    
    Args:
        prompt: The prompt string to sanitize
        image_token: The image token (default: "<image>")
    
    Returns:
        Sanitized prompt string
    """
    
    # First normalize
    prompt = normalize_prompt(prompt)
    
    # Remove any control characters except newlines
    prompt = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', prompt)
    
    # Ensure image token is properly formatted
    if image_token in prompt:
        # Make sure there's a newline after image token
        prompt = prompt.replace(image_token, f"{image_token}\n")
        # Remove duplicate newlines
        prompt = re.sub(r'\n+', '\n', prompt)
    
    return prompt


def create_safe_prompt(base_instruction: str, image_token: str = "<image>", 
                       task_type: str = "grounding") -> str:
    """
    Create a safe, well-formatted prompt from a base instruction.
    
    Args:
        base_instruction: The instruction text (e.g., "Convert the document to markdown")
        image_token: The image token to use (default: "<image>")
        task_type: The task type - "grounding", "free_ocr", or "description" (default: "grounding")
    
    Returns:
        A properly formatted prompt string
    """
    
    # Sanitize the instruction
    instruction = base_instruction.strip()
    
    # Ensure proper punctuation
    if instruction and not instruction[-1] in '.!?':
        instruction += '.'
    
    # Build prompt based on task type
    if task_type == "grounding":
        prompt = f"{image_token}\n<|grounding|>{instruction} "
    elif task_type == "free_ocr":
        prompt = f"{image_token}\nFree OCR. "
    elif task_type == "description":
        prompt = f"{image_token}\n{instruction} "
    else:
        raise ValueError(f"Unknown task_type: {task_type}. Must be 'grounding', 'free_ocr', or 'description'")
    
    # Validate the created prompt
    is_valid, error = validate_prompt(prompt, image_token)
    if not is_valid:
        raise ValueError(f"Generated prompt is invalid: {error}")
    
    return prompt


# Example usage and recommended prompts
RECOMMENDED_PROMPTS = {
    "document_to_markdown": "<image>\n<|grounding|>Convert the document to markdown.",
    "ocr_image": "<image>\n<|grounding|>OCR this image.",
    "free_ocr": "<image>\nFree OCR.",
    "parse_figure": "<image>\nParse the figure.",
    "describe_image": "<image>\nDescribe this image in detail.",
}


def get_prompt_template(task: str) -> Optional[str]:
    """
    Get a recommended prompt template for a specific task.
    
    Args:
        task: Task name (e.g., "document_to_markdown", "ocr_image", etc.)
    
    Returns:
        Prompt template string, or None if task not found
    """
    return RECOMMENDED_PROMPTS.get(task)
