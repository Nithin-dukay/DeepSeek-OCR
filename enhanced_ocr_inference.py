"""
Enhanced DeepSeek-OCR Inference Script
Addresses GitHub Issue: Repetition loops, None returns, missing chat_template

Features:
- Returns text instead of stdout-only
- Anti-repetition guardrails with configurable parameters
- Chat template support
- Retry logic for failed extractions
- Column splitting for wide images
"""

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, Any, List
import re
import warnings


class DeepSeekOCRInference:
    """Enhanced OCR inference with anti-repetition guardrails"""
    
    def __init__(
        self,
        model_path: str = "deepseek-ai/deepseek-ocr",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize the OCR model with enhanced configuration
        
        Args:
            model_path: Path or HuggingFace model ID
            device: Device to run inference on
            torch_dtype: Torch data type for model weights
        """
        self.device = device
        self.torch_dtype = torch_dtype
        
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        
        # Add chat template if missing
        self._setup_chat_template()
        
        print(f"Model loaded successfully on {device}")
    
    def _setup_chat_template(self):
        """Setup chat template for tokenizer.apply_chat_template() support"""
        if self.tokenizer.chat_template is None:
            # DeepSeek-style chat template
            self.tokenizer.chat_template = (
                "{% for message in messages %}"
                "{% if message['role'] == 'user' %}"
                "User: {{ message['content'] }}\n\n"
                "{% elif message['role'] == 'assistant' %}"
                "Assistant: {{ message['content'] }}\n\n"
                "{% endif %}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "Assistant: "
                "{% endif %}"
            )
            print("Chat template added to tokenizer")
    
    def infer(
        self,
        image_path: str,
        prompt: Optional[str] = None,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.2,
        no_repeat_ngram_size: int = 10,
        length_penalty: float = 1.0,
        num_beams: int = 1,
        early_stopping: bool = True,
        detect_repetition: bool = True,
        retry_on_failure: bool = True,
        max_retries: int = 2,
        **kwargs
    ) -> str:
        """
        Perform OCR inference on an image with anti-repetition guardrails
        
        Args:
            image_path: Path to the image file
            prompt: Optional custom prompt (default: "Extract all text from the image")
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = greedy)
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repeating tokens (>1.0 discourages repetition)
            no_repeat_ngram_size: Size of n-grams that cannot repeat
            length_penalty: Penalty for length (>1.0 encourages longer sequences)
            num_beams: Number of beams for beam search
            early_stopping: Stop when all beams finish
            detect_repetition: Check output for repetition and retry if found
            retry_on_failure: Retry with stricter parameters if repetition detected
            max_retries: Maximum number of retry attempts
            **kwargs: Additional generation parameters
            
        Returns:
            Extracted text as string
        """
        # Load and prepare image
        image = Image.open(image_path).convert("RGB")
        
        # Default prompt optimized for OCR
        if prompt is None:
            prompt = "Extract all text from the image, preserving layout and structure."
        
        # Prepare conversation
        conversation = [
            {
                "role": "user",
                "content": prompt,
                "images": [image],
            }
        ]
        
        # Apply chat template
        prompt_text = self.tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )
        
        # Tokenize
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Add image if model expects it
        if hasattr(self.model, 'prepare_inputs_for_generation'):
            # Some models need special image handling
            inputs['images'] = [image]
        
        # Generation parameters with anti-repetition guardrails
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "length_penalty": length_penalty,
            "num_beams": num_beams,
            "early_stopping": early_stopping,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            **kwargs
        }
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **generation_config
            )
        
        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # Check for repetition
        if detect_repetition and self._has_repetition(generated_text):
            warnings.warn(
                "Repetition detected in output. "
                f"Retry {max_retries - retry_on_failure + 1}/{max_retries}"
            )
            
            if retry_on_failure and max_retries > 0:
                # Retry with stricter parameters
                return self.infer(
                    image_path=image_path,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=max(0.0, temperature - 0.1),
                    repetition_penalty=min(2.0, repetition_penalty + 0.3),
                    no_repeat_ngram_size=min(20, no_repeat_ngram_size + 5),
                    detect_repetition=True,
                    retry_on_failure=True,
                    max_retries=max_retries - 1,
                    **kwargs
                )
        
        return generated_text.strip()
    
    def infer_batch(
        self,
        image_paths: List[str],
        prompts: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
        """
        Perform OCR on multiple images
        
        Args:
            image_paths: List of image file paths
            prompts: Optional list of prompts (one per image)
            **kwargs: Generation parameters
            
        Returns:
            List of extracted texts
        """
        if prompts is None:
            prompts = [None] * len(image_paths)
        
        results = []
        for img_path, prompt in zip(image_paths, prompts):
            try:
                text = self.infer(img_path, prompt=prompt, **kwargs)
                results.append(text)
            except Exception as e:
                warnings.warn(f"Failed to process {img_path}: {e}")
                results.append("")
        
        return results
    
    def _has_repetition(
        self,
        text: str,
        min_repeat_length: int = 50,
        max_repeat_ratio: float = 0.3
    ) -> bool:
        """
        Detect if text contains significant repetition
        
        Args:
            text: Text to check
            min_repeat_length: Minimum length of repeated substring to flag
            max_repeat_ratio: Maximum ratio of repeated content allowed
            
        Returns:
            True if significant repetition detected
        """
        if len(text) < min_repeat_length * 2:
            return False
        
        # Check for exact substring repetition
        for length in range(min_repeat_length, len(text) // 2):
            for i in range(len(text) - length * 2):
                substring = text[i:i + length]
                # Check if this substring repeats immediately after
                if text[i + length:i + length * 2] == substring:
                    repeat_ratio = length / len(text)
                    if repeat_ratio > max_repeat_ratio:
                        return True
        
        # Check for line-level repetition (common in OCR failures)
        lines = text.split('\n')
        if len(lines) > 10:
            unique_lines = set(lines)
            if len(unique_lines) / len(lines) < 0.5:
                # More than 50% duplicate lines
                return True
        
        return False
    
    def infer_with_column_split(
        self,
        image_path: str,
        num_columns: int = 2,
        **kwargs
    ) -> str:
        """
        Process wide images by splitting into columns
        Useful for newspaper layouts with multiple columns
        
        Args:
            image_path: Path to image
            num_columns: Number of columns to split into
            **kwargs: Generation parameters
            
        Returns:
            Combined text from all columns
        """
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        column_width = width // num_columns
        
        results = []
        for i in range(num_columns):
            left = i * column_width
            right = (i + 1) * column_width if i < num_columns - 1 else width
            
            # Crop column
            column_img = image.crop((left, 0, right, height))
            
            # Save temporarily
            temp_path = f"/tmp/column_{i}.png"
            column_img.save(temp_path)
            
            # Process column
            text = self.infer(temp_path, **kwargs)
            results.append(text)
        
        return "\n\n--- Column Break ---\n\n".join(results)


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced DeepSeek-OCR Inference")
    parser.add_argument("image_path", type=str, help="Path to image file")
    parser.add_argument("--model_path", type=str, default="deepseek-ai/deepseek-ocr")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=10)
    parser.add_argument("--num_columns", type=int, default=None, 
                       help="Split image into columns (for newspapers)")
    parser.add_argument("--output", type=str, default=None, 
                       help="Output file path (default: print to stdout)")
    
    args = parser.parse_args()
    
    # Initialize model
    ocr = DeepSeekOCRInference(model_path=args.model_path)
    
    # Run inference
    if args.num_columns:
        text = ocr.infer_with_column_split(
            args.image_path,
            num_columns=args.num_columns,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
    else:
        text = ocr.infer(
            args.image_path,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
    
    # Output
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Output saved to {args.output}")
    else:
        print("\n=== Extracted Text ===\n")
        print(text)


if __name__ == "__main__":
    main()
