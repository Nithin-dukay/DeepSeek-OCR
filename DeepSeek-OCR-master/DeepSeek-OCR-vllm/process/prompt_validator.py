"""
Prompt Validator for DeepSeek-OCR

This module validates user prompts and provides suggestions for safe modifications.
The DeepSeek-OCR model is sensitive to prompt format, and deviations from
recommended formats can cause repetitive or incorrect outputs.
"""

import re
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class PromptValidationResult:
    """Result of prompt validation"""
    is_valid: bool
    warning_level: str  # 'none', 'low', 'medium', 'high'
    message: str
    suggested_prompt: Optional[str] = None
    issues: List[str] = None
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []


class PromptValidator:
    """
    Validates prompts for DeepSeek-OCR model.
    
    The model works best with specific prompt formats:
    - Document OCR: <image>\n<|grounding|>Convert the document to markdown.
    - Image OCR: <image>\n<|grounding|>OCR this image.
    - Free OCR: <image>\nFree OCR.
    - Figure parsing: <image>\nParse the figure.
    - General description: <image>\nDescribe this image in detail.
    """
    
    # Recommended prompt templates
    RECOMMENDED_PROMPTS = {
        'document_markdown': '<image>\n<|grounding|>Convert the document to markdown.',
        'document_ocr': '<image>\n<|grounding|>OCR this image.',
        'free_ocr': '<image>\nFree OCR.',
        'figure_parse': '<image>\nParse the figure.',
        'general_description': '<image>\nDescribe this image in detail.',
        'locate_text': '<image>\nLocate <|ref|>TEXT<|/ref|> in the image.',
    }
    
    # Patterns that commonly cause issues
    PROBLEMATIC_PATTERNS = [
        (r'don\'?t\s+add', 'Negative instructions (e.g., "dont add") can confuse the model'),
        (r'without\s+\w+', 'Negative constraints (e.g., "without X") may not work as expected'),
        (r'no\s+extra\s+space', 'Spacing instructions are not reliably followed'),
        (r'make\s+sure', 'Meta-instructions about output quality may cause issues'),
        (r'please\s+\w+', 'Politeness tokens are unnecessary and may affect output'),
        (r'you\s+should', 'Direct model instructions may interfere with generation'),
        (r'always\s+\w+', 'Absolute constraints may cause repetitive behavior'),
        (r'never\s+\w+', 'Negative absolute constraints are problematic'),
    ]
    
    # Safe modification patterns
    SAFE_MODIFICATIONS = {
        'format': ['markdown', 'json', 'plain text', 'structured'],
        'focus': ['text only', 'tables', 'figures', 'layout'],
        'language': ['in English', 'in Chinese', 'preserve language'],
    }
    
    def __init__(self):
        self.validation_cache = {}
    
    def validate(self, prompt: str) -> PromptValidationResult:
        """
        Validate a prompt and provide feedback.
        
        Args:
            prompt: The user's prompt string
            
        Returns:
            PromptValidationResult with validation details
        """
        # Check cache
        if prompt in self.validation_cache:
            return self.validation_cache[prompt]
        
        issues = []
        warning_level = 'none'
        suggested_prompt = None
        
        # Check if prompt contains <image> token
        if '<image>' not in prompt:
            issues.append('Missing <image> token - required for image input')
            warning_level = 'high'
        
        # Check if prompt is one of the recommended formats
        is_recommended = prompt in self.RECOMMENDED_PROMPTS.values()
        
        if is_recommended:
            result = PromptValidationResult(
                is_valid=True,
                warning_level='none',
                message='✓ Using recommended prompt format',
                issues=[]
            )
            self.validation_cache[prompt] = result
            return result
        
        # Check for problematic patterns
        for pattern, issue_desc in self.PROBLEMATIC_PATTERNS:
            if re.search(pattern, prompt, re.IGNORECASE):
                issues.append(issue_desc)
                warning_level = 'high'
        
        # Check prompt length
        if len(prompt) > 200:
            issues.append('Prompt is very long - shorter prompts work better')
            warning_level = max(warning_level, 'medium', key=lambda x: ['none', 'low', 'medium', 'high'].index(x))
        
        # Check for multiple sentences after the main instruction
        if '<|grounding|>' in prompt:
            parts = prompt.split('<|grounding|>')
            if len(parts) > 1:
                instruction = parts[1]
                sentences = [s.strip() for s in re.split(r'[.!?]+', instruction) if s.strip()]
                if len(sentences) > 2:
                    issues.append('Multiple sentences after <|grounding|> may cause confusion')
                    warning_level = max(warning_level, 'medium', key=lambda x: ['none', 'low', 'medium', 'high'].index(x))
        
        # Generate suggestion
        if issues:
            suggested_prompt = self._suggest_alternative(prompt)
        
        # Determine if valid
        is_valid = warning_level in ['none', 'low']
        
        # Create message
        if warning_level == 'high':
            message = '⚠️  WARNING: This prompt may cause repetitive or incorrect output!'
        elif warning_level == 'medium':
            message = '⚠️  CAUTION: This prompt format may not work optimally'
        elif warning_level == 'low':
            message = 'ℹ️  Note: Minor prompt format deviation detected'
        else:
            message = '✓ Prompt format looks acceptable'
        
        result = PromptValidationResult(
            is_valid=is_valid,
            warning_level=warning_level,
            message=message,
            suggested_prompt=suggested_prompt,
            issues=issues
        )
        
        self.validation_cache[prompt] = result
        return result
    
    def _suggest_alternative(self, prompt: str) -> str:
        """
        Suggest an alternative prompt based on the user's intent.
        
        Args:
            prompt: The problematic prompt
            
        Returns:
            A suggested alternative prompt
        """
        # Try to infer user intent
        prompt_lower = prompt.lower()
        
        # Document to markdown conversion
        if 'markdown' in prompt_lower or 'document' in prompt_lower:
            return self.RECOMMENDED_PROMPTS['document_markdown']
        
        # General OCR
        if 'ocr' in prompt_lower:
            return self.RECOMMENDED_PROMPTS['document_ocr']
        
        # Figure/chart parsing
        if 'figure' in prompt_lower or 'chart' in prompt_lower or 'graph' in prompt_lower:
            return self.RECOMMENDED_PROMPTS['figure_parse']
        
        # Default to document markdown
        return self.RECOMMENDED_PROMPTS['document_markdown']
    
    def get_recommended_prompts(self) -> dict:
        """Get all recommended prompt templates"""
        return self.RECOMMENDED_PROMPTS.copy()
    
    def print_validation_report(self, result: PromptValidationResult):
        """
        Print a formatted validation report.
        
        Args:
            result: The validation result to print
        """
        print("\n" + "="*70)
        print("PROMPT VALIDATION REPORT")
        print("="*70)
        print(f"\nStatus: {result.message}")
        print(f"Warning Level: {result.warning_level.upper()}")
        
        if result.issues:
            print(f"\nIssues Detected ({len(result.issues)}):")
            for i, issue in enumerate(result.issues, 1):
                print(f"  {i}. {issue}")
        
        if result.suggested_prompt:
            print(f"\n💡 Suggested Alternative:")
            print(f"   {result.suggested_prompt}")
        
        if not result.is_valid:
            print("\n⚠️  RECOMMENDATION: Use one of the standard prompt formats")
            print("   to avoid repetitive or incorrect outputs.")
            print("\n   See PROMPT_GUIDELINES.md for more information.")
        
        print("="*70 + "\n")


def validate_prompt(prompt: str, verbose: bool = True) -> PromptValidationResult:
    """
    Convenience function to validate a prompt.
    
    Args:
        prompt: The prompt to validate
        verbose: Whether to print the validation report
        
    Returns:
        PromptValidationResult
    """
    validator = PromptValidator()
    result = validator.validate(prompt)
    
    if verbose and result.warning_level in ['medium', 'high']:
        validator.print_validation_report(result)
    
    return result


# Example usage
if __name__ == '__main__':
    # Test cases
    test_prompts = [
        '<image>\n<|grounding|>Convert the document to markdown.',
        '<image>\n<|grounding|>Convert the document to markdown, dont add any extra space between letters.',
        '<image>\nFree OCR.',
        '<image>\n<|grounding|>Please OCR this image without any extra spaces.',
    ]
    
    validator = PromptValidator()
    
    print("\n" + "="*70)
    print("PROMPT VALIDATOR TEST")
    print("="*70)
    
    for prompt in test_prompts:
        print(f"\nTesting prompt: {prompt[:60]}...")
        result = validator.validate(prompt)
        validator.print_validation_report(result)
