"""
Prompt validation and sanitization module for DeepSeek-OCR.

This module helps prevent repetitive generation issues by validating and 
normalizing prompts before they are sent to the model.
"""

import re
from typing import Tuple, Optional
import warnings


class PromptValidator:
    """Validates and sanitizes prompts for DeepSeek-OCR model."""
    
    # Special tokens that must be present
    IMAGE_TOKEN = "<image>"
    GROUNDING_TOKEN = "<|grounding|>"
    REF_START_TOKEN = "<|ref|>"
    REF_END_TOKEN = "<|/ref|>"
    
    # Recommended prompt patterns
    RECOMMENDED_PROMPTS = {
        "document": "<image>\n<|grounding|>Convert the document to markdown.",
        "image_ocr": "<image>\n<|grounding|>OCR this image.",
        "free_ocr": "<image>\nFree OCR.",
        "figure": "<image>\nParse the figure.",
        "description": "<image>\nDescribe this image in detail.",
    }
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize the prompt validator.
        
        Args:
            strict_mode: If True, raises exceptions for invalid prompts.
                        If False, issues warnings and attempts to fix prompts.
        """
        self.strict_mode = strict_mode
    
    def validate_and_sanitize(self, prompt: str) -> Tuple[str, bool]:
        """
        Validate and sanitize a prompt.
        
        Args:
            prompt: The input prompt to validate
            
        Returns:
            Tuple of (sanitized_prompt, is_valid)
            - sanitized_prompt: The cleaned/fixed prompt
            - is_valid: Whether the original prompt was valid
        """
        original_prompt = prompt
        is_valid = True
        
        # Check for image token
        if self.IMAGE_TOKEN not in prompt:
            msg = f"Prompt must contain '{self.IMAGE_TOKEN}' token"
            if self.strict_mode:
                raise ValueError(msg)
            warnings.warn(msg)
            is_valid = False
            prompt = f"{self.IMAGE_TOKEN}\n{prompt}"
        
        # Normalize whitespace and punctuation
        prompt = self._normalize_prompt(prompt)
        
        # Check for overly complex instructions
        if self._is_overly_complex(prompt):
            msg = (
                "Prompt appears overly complex. Consider simplifying to avoid "
                "generation issues. Recommended: Keep instructions concise and direct."
            )
            warnings.warn(msg)
            is_valid = False
            # Attempt to simplify
            prompt = self._simplify_prompt(prompt)
        
        # Validate special token placement
        if not self._validate_token_placement(prompt):
            msg = "Special tokens may not be in optimal positions"
            warnings.warn(msg)
            is_valid = False
        
        # Check for problematic patterns
        if self._has_problematic_patterns(prompt):
            msg = (
                "Prompt contains patterns that may cause repetitive generation. "
                "Consider using one of the recommended prompt formats."
            )
            warnings.warn(msg)
            is_valid = False
        
        if not is_valid and original_prompt != prompt:
            print(f"\n[Prompt Validator] Original prompt modified:")
            print(f"  Original: {original_prompt}")
            print(f"  Sanitized: {prompt}")
            print(f"  Suggestion: Use one of the recommended prompts for best results.\n")
        
        return prompt, is_valid
    
    def _normalize_prompt(self, prompt: str) -> str:
        """Normalize whitespace and punctuation in the prompt."""
        # Remove extra spaces
        prompt = re.sub(r' +', ' ', prompt)
        
        # Normalize newlines (keep single newlines, remove multiple)
        prompt = re.sub(r'\n\n+', '\n', prompt)
        
        # Ensure proper spacing after special tokens
        prompt = re.sub(r'(<image>)([^\n])', r'\1\n\2', prompt)
        prompt = re.sub(r'(<\|grounding\|>)([^\s])', r'\1 \2', prompt)
        
        # Remove trailing/leading whitespace
        prompt = prompt.strip()
        
        return prompt
    
    def _is_overly_complex(self, prompt: str) -> bool:
        """Check if the prompt is overly complex."""
        # Extract the instruction part (after special tokens)
        instruction = prompt
        for token in [self.IMAGE_TOKEN, self.GROUNDING_TOKEN]:
            instruction = instruction.replace(token, '')
        instruction = instruction.strip()
        
        # Check length
        if len(instruction) > 150:
            return True
        
        # Check for multiple sentences with complex punctuation
        sentences = re.split(r'[.!?]+', instruction)
        if len(sentences) > 3:
            return True
        
        # Check for multiple clauses (commas)
        if instruction.count(',') > 2:
            return True
        
        return False
    
    def _simplify_prompt(self, prompt: str) -> str:
        """Attempt to simplify an overly complex prompt."""
        # Extract the main instruction (first sentence)
        parts = prompt.split('\n')
        
        # Keep special tokens and first instruction
        simplified_parts = []
        for part in parts:
            if self.IMAGE_TOKEN in part or self.GROUNDING_TOKEN in part:
                simplified_parts.append(part)
            elif part.strip():
                # Take only the first sentence
                first_sentence = re.split(r'[.!?]+', part)[0].strip()
                if first_sentence:
                    # Remove complex clauses after commas
                    if ',' in first_sentence:
                        first_sentence = first_sentence.split(',')[0].strip()
                    simplified_parts.append(first_sentence + '.')
                break
        
        return '\n'.join(simplified_parts)
    
    def _validate_token_placement(self, prompt: str) -> bool:
        """Validate that special tokens are in correct positions."""
        # Image token should be at the start
        if not prompt.startswith(self.IMAGE_TOKEN):
            return False
        
        # If grounding token is present, it should be after image token
        if self.GROUNDING_TOKEN in prompt:
            image_pos = prompt.find(self.IMAGE_TOKEN)
            grounding_pos = prompt.find(self.GROUNDING_TOKEN)
            if grounding_pos < image_pos:
                return False
        
        return True
    
    def _has_problematic_patterns(self, prompt: str) -> bool:
        """Check for patterns known to cause issues."""
        # Extract instruction part
        instruction = prompt
        for token in [self.IMAGE_TOKEN, self.GROUNDING_TOKEN]:
            instruction = instruction.replace(token, '')
        instruction = instruction.strip().lower()
        
        # Patterns that may cause issues
        problematic_patterns = [
            r'don\'?t\s+add',  # Negative instructions
            r'without\s+\w+',  # Without clauses
            r'no\s+extra',     # No extra...
            r'avoid\s+\w+',    # Avoid instructions
        ]
        
        for pattern in problematic_patterns:
            if re.search(pattern, instruction):
                return True
        
        return False
    
    def get_recommended_prompt(self, task_type: str = "document") -> str:
        """
        Get a recommended prompt for a specific task type.
        
        Args:
            task_type: One of "document", "image_ocr", "free_ocr", "figure", "description"
            
        Returns:
            Recommended prompt string
        """
        return self.RECOMMENDED_PROMPTS.get(
            task_type, 
            self.RECOMMENDED_PROMPTS["document"]
        )
    
    def suggest_alternative(self, prompt: str) -> Optional[str]:
        """
        Suggest an alternative prompt if the current one is problematic.
        
        Args:
            prompt: The current prompt
            
        Returns:
            Suggested alternative prompt or None
        """
        instruction = prompt.lower()
        
        # Detect task type from prompt
        if "markdown" in instruction or "document" in instruction:
            return self.RECOMMENDED_PROMPTS["document"]
        elif "figure" in instruction or "chart" in instruction:
            return self.RECOMMENDED_PROMPTS["figure"]
        elif "describe" in instruction:
            return self.RECOMMENDED_PROMPTS["description"]
        elif "ocr" in instruction:
            if self.GROUNDING_TOKEN in prompt:
                return self.RECOMMENDED_PROMPTS["image_ocr"]
            else:
                return self.RECOMMENDED_PROMPTS["free_ocr"]
        
        return None


def validate_prompt(prompt: str, strict_mode: bool = False) -> Tuple[str, bool]:
    """
    Convenience function to validate and sanitize a prompt.
    
    Args:
        prompt: The input prompt to validate
        strict_mode: If True, raises exceptions for invalid prompts
        
    Returns:
        Tuple of (sanitized_prompt, is_valid)
    """
    validator = PromptValidator(strict_mode=strict_mode)
    return validator.validate_and_sanitize(prompt)


def get_recommended_prompt(task_type: str = "document") -> str:
    """
    Convenience function to get a recommended prompt.
    
    Args:
        task_type: One of "document", "image_ocr", "free_ocr", "figure", "description"
        
    Returns:
        Recommended prompt string
    """
    validator = PromptValidator()
    return validator.get_recommended_prompt(task_type)
