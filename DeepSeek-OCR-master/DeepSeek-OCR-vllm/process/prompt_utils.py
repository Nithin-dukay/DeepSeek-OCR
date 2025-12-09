"""
Utility functions for prompt validation and optimization for DeepSeek-OCR.

This module provides helper functions to ensure prompts work correctly with the model
and avoid common issues like n-gram repetition blocking.
"""

import re
from typing import Tuple, Optional


class PromptValidator:
    """Validates and optimizes prompts for DeepSeek-OCR model."""
    
    # Recommended prompt templates
    RECOMMENDED_PROMPTS = {
        "document_markdown": "<image>\n<|grounding|>Convert the document to markdown.",
        "document_markdown_detailed": "<image>\n<|grounding|>Convert the document to markdown with all formatting preserved.",
        "ocr_image": "<image>\n<|grounding|>OCR this image.",
        "free_ocr": "<image>\nFree OCR.",
        "parse_figure": "<image>\nParse the figure.",
        "describe_image": "<image>\nDescribe this image in detail.",
        "locate_text": "<image>\nLocate <|ref|>{text}<|/ref|> in the image.",
    }
    
    # Known problematic patterns
    PROBLEMATIC_PATTERNS = [
        (r"don't|dont", "Avoid contractions - use 'do not' instead"),
        (r"[,;]\s*[a-z]+\s+[a-z]+\s+[a-z]+\s+[a-z]+", "Long comma-separated instructions may cause issues"),
    ]
    
    @staticmethod
    def validate_prompt(prompt: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a prompt for potential issues.
        
        Args:
            prompt: The prompt string to validate
            
        Returns:
            Tuple of (is_valid, warning_message)
            - is_valid: True if prompt is likely to work well
            - warning_message: None if valid, otherwise a warning message
        """
        if not prompt:
            return False, "Prompt cannot be empty"
        
        if "<image>" not in prompt:
            return False, "Prompt must contain '<image>' token"
        
        # Check for multiple image tokens (currently only single image supported in basic usage)
        image_count = prompt.count("<image>")
        if image_count > 1:
            return True, f"Prompt contains {image_count} image tokens - ensure you provide {image_count} images"
        
        # Check for problematic patterns
        for pattern, message in PromptValidator.PROBLEMATIC_PATTERNS:
            if re.search(pattern, prompt, re.IGNORECASE):
                return True, f"Warning: {message}. Pattern found: '{pattern}'"
        
        # Check prompt length - very long prompts after <|grounding|> may cause issues
        if "<|grounding|>" in prompt:
            grounding_part = prompt.split("<|grounding|>", 1)[1]
            if len(grounding_part.split()) > 20:
                return True, "Warning: Very long instruction after <|grounding|> may affect model performance"
        
        return True, None
    
    @staticmethod
    def optimize_prompt(prompt: str) -> str:
        """
        Optimize a prompt for better model performance.
        
        Args:
            prompt: The original prompt
            
        Returns:
            Optimized prompt string
        """
        # Remove extra whitespace
        prompt = " ".join(prompt.split())
        
        # Ensure proper spacing around special tokens
        prompt = re.sub(r'\s*<image>\s*', '<image>\n', prompt)
        prompt = re.sub(r'\s*<\|grounding\|>\s*', '<|grounding|>', prompt)
        
        # Ensure prompt ends with proper punctuation or space
        if "<|grounding|>" in prompt:
            parts = prompt.split("<|grounding|>")
            if len(parts) == 2 and parts[1]:
                instruction = parts[1].strip()
                if not instruction.endswith(('.', '!', '?', ' ')):
                    instruction += '.'
                prompt = parts[0] + "<|grounding|>" + instruction
        
        return prompt
    
    @staticmethod
    def get_recommended_prompt(task: str) -> Optional[str]:
        """
        Get a recommended prompt for a specific task.
        
        Args:
            task: Task name (e.g., 'document_markdown', 'ocr_image')
            
        Returns:
            Recommended prompt string or None if task not found
        """
        return PromptValidator.RECOMMENDED_PROMPTS.get(task)
    
    @staticmethod
    def suggest_alternative(prompt: str) -> Optional[str]:
        """
        Suggest an alternative prompt if the current one might cause issues.
        
        Args:
            prompt: The current prompt
            
        Returns:
            Suggested alternative prompt or None
        """
        is_valid, warning = PromptValidator.validate_prompt(prompt)
        
        if not is_valid or warning:
            # Try to find a similar recommended prompt
            if "markdown" in prompt.lower():
                return PromptValidator.RECOMMENDED_PROMPTS["document_markdown"]
            elif "ocr" in prompt.lower():
                if "<|grounding|>" in prompt:
                    return PromptValidator.RECOMMENDED_PROMPTS["ocr_image"]
                else:
                    return PromptValidator.RECOMMENDED_PROMPTS["free_ocr"]
            elif "figure" in prompt.lower():
                return PromptValidator.RECOMMENDED_PROMPTS["parse_figure"]
            elif "describe" in prompt.lower():
                return PromptValidator.RECOMMENDED_PROMPTS["describe_image"]
        
        return None


def validate_and_optimize_prompt(prompt: str, verbose: bool = True) -> str:
    """
    Convenience function to validate and optimize a prompt.
    
    Args:
        prompt: The prompt to validate and optimize
        verbose: If True, print warnings and suggestions
        
    Returns:
        Optimized prompt string
    """
    validator = PromptValidator()
    
    # Validate
    is_valid, warning = validator.validate_prompt(prompt)
    
    if not is_valid:
        raise ValueError(f"Invalid prompt: {warning}")
    
    if warning and verbose:
        print(f"⚠️  Prompt Warning: {warning}")
        
        # Suggest alternative
        alternative = validator.suggest_alternative(prompt)
        if alternative:
            print(f"💡 Suggested alternative: {alternative}")
    
    # Optimize
    optimized = validator.optimize_prompt(prompt)
    
    if optimized != prompt and verbose:
        print(f"✓ Prompt optimized")
        print(f"  Original:  {repr(prompt)}")
        print(f"  Optimized: {repr(optimized)}")
    
    return optimized


# Example usage and testing
if __name__ == "__main__":
    print("=== DeepSeek-OCR Prompt Validator ===\n")
    
    # Test cases
    test_prompts = [
        "<image>\n<|grounding|>Convert the document to markdown.",
        "<image>\n<|grounding|>Convert the document to markdown, dont add any extra space between letters.",
        "<image>\n<|grounding|>Convert the document to markdown with proper formatting.",
        "<image>\nFree OCR.",
        "Convert to markdown",  # Invalid - no <image>
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"Test {i}: {repr(prompt)}")
        try:
            optimized = validate_and_optimize_prompt(prompt, verbose=True)
            print(f"✓ Final prompt: {repr(optimized)}\n")
        except ValueError as e:
            print(f"✗ Error: {e}\n")
