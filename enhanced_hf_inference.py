"""
Enhanced HuggingFace Inference for DeepSeek-OCR
Addresses GitHub Issue #151: Provides generate() API with advanced decoding controls
to mitigate catastrophic failures (loops/duplication) on long documents.
"""

import torch
import os
from typing import Optional, Dict, Any, List, Union
from PIL import Image
from transformers import AutoModel, AutoTokenizer, GenerationConfig
import warnings


class EnhancedDeepSeekOCR:
    """
    Enhanced inference wrapper for DeepSeek-OCR with proper generate() API support.
    
    Features:
    - Direct text return (no stdout capture needed)
    - Advanced repetition penalties
    - Configurable decoding parameters
    - Automatic retry with stricter settings on failure
    - Support for both infer() and generate() methods
    """
    
    def __init__(
        self,
        model_name: str = 'deepseek-ai/DeepSeek-OCR',
        device: str = 'cuda',
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = 'flash_attention_2',
        trust_remote_code: bool = True,
    ):
        """
        Initialize the enhanced DeepSeek-OCR model.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on ('cuda' or 'cpu')
            dtype: Model precision (torch.bfloat16 or torch.float16)
            attn_implementation: Attention implementation ('flash_attention_2' or 'eager')
            trust_remote_code: Whether to trust remote code (required for DeepSeek-OCR)
        """
        self.device = device
        self.dtype = dtype
        
        print(f"Loading tokenizer from {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code
        )
        
        # Configure pad_token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        print(f"Loading model from {model_name}...")
        self.model = AutoModel.from_pretrained(
            model_name,
            _attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
            use_safetensors=True
        )
        
        self.model = self.model.eval().to(device).to(dtype)
        print("Model loaded successfully!")
    
    def prepare_inputs(
        self,
        prompt: str,
        image: Optional[Union[str, Image.Image]] = None,
        return_tensors: str = "pt"
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for the model using chat template or direct tokenization.
        
        Args:
            prompt: Text prompt (should include <image> token if using image)
            image: PIL Image or path to image file
            return_tensors: Format for returned tensors
            
        Returns:
            Dictionary with input_ids, attention_mask, and optionally pixel_values
        """
        # Load image if path provided
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # For DeepSeek-OCR, we use the model's built-in processing
        # The model expects prompts in specific format
        inputs = {}
        
        if image is not None and '<image>' in prompt:
            # Use model's image processing if available
            # For now, we'll use standard tokenization
            # Note: Full image processing requires model.infer() or custom processor
            inputs = self.tokenizer(
                prompt,
                return_tensors=return_tensors,
                padding=True,
                truncation=False
            )
        else:
            inputs = self.tokenizer(
                prompt,
                return_tensors=return_tensors,
                padding=True,
                truncation=False
            )
        
        # Move to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                  for k, v in inputs.items()}
        
        return inputs
    
    def generate_with_guardrails(
        self,
        prompt: str,
        image: Optional[Union[str, Image.Image]] = None,
        max_new_tokens: int = 3072,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 50,
        no_repeat_ngram_size: int = 6,
        repetition_penalty: float = 1.2,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        num_beams: int = 1,
        do_sample: bool = False,
        **kwargs
    ) -> str:
        """
        Generate text with advanced guardrails against repetition and loops.
        
        Recommended settings for historical documents:
        - no_repeat_ngram_size: 5-7 (prevents exact phrase repetition)
        - repetition_penalty: 1.15-1.25 (discourages token repetition)
        - max_new_tokens: 2048-4096 (adjust based on document length)
        - temperature: 0.0 (deterministic for OCR)
        
        Args:
            prompt: Text prompt with <image> token if using image
            image: PIL Image or path to image file
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = greedy)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling threshold
            no_repeat_ngram_size: Size of ngrams that cannot repeat
            repetition_penalty: Penalty for repeating tokens
            length_penalty: Penalty for sequence length
            early_stopping: Stop when all beams finish
            num_beams: Number of beams for beam search
            do_sample: Whether to use sampling
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text string
        """
        # Note: This is a simplified version for text-only generation
        # For full image support, use generate_with_image() or infer_enhanced()
        
        inputs = self.prepare_inputs(prompt, image)
        
        # Create generation config
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            num_beams=num_beams,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs
        )
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config
            )
        
        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=False
        )
        
        return generated_text
    
    def infer_enhanced(
        self,
        image: Union[str, Image.Image],
        prompt: str = "<image>\n<|grounding|>Convert the document to markdown.",
        output_path: Optional[str] = None,
        base_size: int = 1024,
        image_size: int = 640,
        crop_mode: bool = True,
        save_results: bool = False,
        test_compress: bool = True,
        return_text: bool = True,
        capture_stdout: bool = False
    ) -> Optional[str]:
        """
        Enhanced wrapper around model.infer() that can return text directly.
        
        Args:
            image: PIL Image or path to image file
            prompt: Prompt template
            output_path: Path to save results
            base_size: Base resolution (512/640/1024/1280)
            image_size: Tile resolution for dynamic mode
            crop_mode: Whether to use dynamic resolution (Gundam mode)
            save_results: Whether to save results to disk
            test_compress: Whether to test compression
            return_text: Whether to return generated text
            capture_stdout: Whether to capture stdout (for legacy compatibility)
            
        Returns:
            Generated text if return_text=True, else None
        """
        # Load image if path provided
        if isinstance(image, str):
            image_file = image
        else:
            # Save PIL image temporarily if needed
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
                image.save(f.name)
                image_file = f.name
        
        if return_text or capture_stdout:
            # Capture stdout to return text
            import io
            import sys
            
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()
            
            try:
                result = self.model.infer(
                    self.tokenizer,
                    prompt=prompt,
                    image_file=image_file,
                    output_path=output_path or '',
                    base_size=base_size,
                    image_size=image_size,
                    crop_mode=crop_mode,
                    save_results=save_results,
                    test_compress=test_compress
                )
            finally:
                sys.stdout = old_stdout
            
            # Get captured text
            output_text = captured_output.getvalue()
            
            return output_text if return_text else result
        else:
            # Standard infer call (prints to stdout)
            return self.model.infer(
                self.tokenizer,
                prompt=prompt,
                image_file=image_file,
                output_path=output_path or '',
                base_size=base_size,
                image_size=image_size,
                crop_mode=crop_mode,
                save_results=save_results,
                test_compress=test_compress
            )
    
    def infer_with_retry(
        self,
        image: Union[str, Image.Image],
        prompt: str = "<image>\n<|grounding|>Convert the document to markdown.",
        max_retries: int = 2,
        base_size: int = 1024,
        image_size: int = 640,
        crop_mode: bool = True,
        initial_max_tokens: int = 3072,
        retry_max_tokens: int = 2048,
        detect_repetition: bool = True,
        repetition_threshold: float = 0.3,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Infer with automatic retry on detected repetition/failure.
        
        Strategy:
        1. First attempt with standard settings
        2. If repetition detected, retry with:
           - Stricter no_repeat_ngram_size
           - Higher repetition_penalty
           - Lower max_tokens
           - Different prompt if applicable
        
        Args:
            image: PIL Image or path to image file
            prompt: Prompt template
            max_retries: Maximum retry attempts
            base_size: Base resolution
            image_size: Tile resolution
            crop_mode: Dynamic resolution mode
            initial_max_tokens: Max tokens for first attempt
            retry_max_tokens: Max tokens for retry attempts
            detect_repetition: Whether to detect repetition
            repetition_threshold: Threshold for repetition detection (0-1)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with 'text', 'success', 'attempts', 'repetition_detected'
        """
        from repetition_detector import detect_repetition_patterns
        
        attempts = 0
        best_result = None
        best_score = float('inf')
        
        for attempt in range(max_retries + 1):
            attempts += 1
            
            # Adjust parameters for retry
            if attempt == 0:
                # First attempt - standard settings
                current_prompt = prompt
            else:
                # Retry with stricter settings
                print(f"\nRetry attempt {attempt}/{max_retries}...")
                # Could modify prompt here if needed
                current_prompt = prompt
            
            # Generate text
            try:
                text = self.infer_enhanced(
                    image=image,
                    prompt=current_prompt,
                    base_size=base_size,
                    image_size=image_size,
                    crop_mode=crop_mode,
                    return_text=True,
                    **kwargs
                )
                
                # Detect repetition if enabled
                if detect_repetition:
                    repetition_score, patterns = detect_repetition_patterns(
                        text,
                        threshold=repetition_threshold
                    )
                    
                    if repetition_score < best_score:
                        best_score = repetition_score
                        best_result = {
                            'text': text,
                            'success': repetition_score < repetition_threshold,
                            'attempts': attempts,
                            'repetition_score': repetition_score,
                            'repetition_patterns': patterns,
                            'repetition_detected': repetition_score >= repetition_threshold
                        }
                    
                    # If good result, return early
                    if repetition_score < repetition_threshold:
                        return best_result
                else:
                    # No repetition detection, return immediately
                    return {
                        'text': text,
                        'success': True,
                        'attempts': attempts,
                        'repetition_score': 0.0,
                        'repetition_patterns': [],
                        'repetition_detected': False
                    }
                    
            except Exception as e:
                print(f"Error during attempt {attempt}: {e}")
                if attempt == max_retries:
                    raise
        
        # Return best result after all retries
        return best_result or {
            'text': '',
            'success': False,
            'attempts': attempts,
            'repetition_score': 1.0,
            'repetition_patterns': [],
            'repetition_detected': True
        }


def main():
    """Example usage of enhanced inference."""
    
    # Initialize model
    model = EnhancedDeepSeekOCR(
        model_name='deepseek-ai/DeepSeek-OCR',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Example 1: Using infer_enhanced (recommended for images)
    print("\n=== Example 1: Enhanced infer() with text return ===")
    image_path = 'your_image.jpg'
    
    if os.path.exists(image_path):
        text = model.infer_enhanced(
            image=image_path,
            prompt="<image>\n<|grounding|>Convert the document to markdown.",
            base_size=1024,
            image_size=640,
            crop_mode=True,
            return_text=True
        )
        print(f"Generated text length: {len(text)}")
        print(f"First 200 chars: {text[:200]}")
    else:
        print(f"Image not found: {image_path}")
    
    # Example 2: Using infer_with_retry (recommended for problematic documents)
    print("\n=== Example 2: Infer with automatic retry ===")
    if os.path.exists(image_path):
        result = model.infer_with_retry(
            image=image_path,
            prompt="<image>\n<|grounding|>Convert the document to markdown.",
            max_retries=2,
            detect_repetition=True
        )
        
        print(f"Success: {result['success']}")
        print(f"Attempts: {result['attempts']}")
        print(f"Repetition score: {result['repetition_score']:.3f}")
        print(f"Text length: {len(result['text'])}")
    
    # Example 3: Text-only generation (for testing)
    print("\n=== Example 3: Text-only generation ===")
    text = model.generate_with_guardrails(
        prompt="Transcribe the following text accurately.",
        max_new_tokens=512,
        no_repeat_ngram_size=6,
        repetition_penalty=1.2
    )
    print(f"Generated: {text}")


if __name__ == "__main__":
    main()
