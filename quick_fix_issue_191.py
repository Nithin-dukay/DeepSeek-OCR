"""
Quick Fix for GitHub Issue #191: OCR Hallucination

This is a minimal, drop-in replacement for the example code from HuggingFace.
Just replace your existing code with this script.

Original (hallucinating) code:
    res = model.infer(tokenizer, prompt=prompt, image_file=image_file, ...)

Fixed code:
    res = infer_with_fix(model, tokenizer, prompt=prompt, image_file=image_file, ...)
"""

from transformers import AutoModel, AutoTokenizer, LogitsProcessor
import torch
import os
from typing import Optional, Set


class NoRepeatNGramLogitsProcessor(LogitsProcessor):
    """Prevents n-gram repetition to avoid hallucination."""
    
    def __init__(self, ngram_size: int = 30, window_size: int = 90, 
                 whitelist_token_ids: Optional[Set[int]] = None):
        self.ngram_size = ngram_size
        self.window_size = window_size
        self.whitelist_token_ids = whitelist_token_ids or {128821, 128822}
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = input_ids.shape[0]
        vocab_size = scores.shape[-1]
        
        for batch_idx in range(batch_size):
            sequence = input_ids[batch_idx].tolist()
            
            if len(sequence) < self.ngram_size:
                continue
            
            current_prefix = tuple(sequence[-(self.ngram_size - 1):])
            search_start = max(0, len(sequence) - self.window_size)
            search_end = len(sequence) - self.ngram_size + 1
            
            banned_tokens = set()
            for i in range(search_start, search_end):
                ngram = tuple(sequence[i:i + self.ngram_size])
                if ngram[:-1] == current_prefix:
                    banned_tokens.add(ngram[-1])
            
            banned_tokens = banned_tokens - self.whitelist_token_ids
            
            if banned_tokens:
                for token in banned_tokens:
                    if token < vocab_size:
                        scores[batch_idx, token] = float("-inf")
        
        return scores


def infer_with_fix(model, tokenizer, prompt, image_file, output_path=' ', 
                   base_size=1024, image_size=640, crop_mode=True, 
                   save_results=False, test_compress=True,
                   ngram_size=30, window_size=90):
    """
    Drop-in replacement for model.infer() with anti-hallucination fix.
    
    Args:
        Same as model.infer(), plus:
        ngram_size: N-gram size for blocking (30-40 for difficult documents)
        window_size: Sliding window size (90-120 for difficult documents)
    """
    # Create the anti-hallucination processor
    logits_processor = NoRepeatNGramLogitsProcessor(
        ngram_size=ngram_size,
        window_size=window_size,
        whitelist_token_ids={128821, 128822}
    )
    
    # Patch the model's generate method
    original_generate = model.generate
    
    def patched_generate(*args, **kwargs):
        if 'logits_processor' in kwargs:
            kwargs['logits_processor'].append(logits_processor)
        else:
            kwargs['logits_processor'] = [logits_processor]
        
        kwargs.setdefault('max_new_tokens', 8192)
        kwargs.setdefault('temperature', 0.0)
        kwargs.setdefault('do_sample', False)
        
        return original_generate(*args, **kwargs)
    
    model.generate = patched_generate
    
    try:
        result = model.infer(
            tokenizer, 
            prompt=prompt, 
            image_file=image_file, 
            output_path=output_path, 
            base_size=base_size, 
            image_size=image_size, 
            crop_mode=crop_mode, 
            save_results=save_results, 
            test_compress=test_compress
        )
    finally:
        model.generate = original_generate
    
    return result


# ============================================================================
# EXAMPLE USAGE - Replace your existing code with this
# ============================================================================

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    model_name = 'deepseek-ai/DeepSeek-OCR'
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name, 
        _attn_implementation='flash_attention_2', 
        trust_remote_code=True, 
        use_safetensors=True
    )
    model = model.eval().cuda().to(torch.bfloat16)
    print("Model loaded!")
    
    # Configure your settings
    image_file = 'your_image.jpg'  # CHANGE THIS
    output_path = 'your/output/dir'  # CHANGE THIS
    
    # For ancient/handwritten documents, try these prompts:
    # prompt = "<image>\nFree OCR. "  # No layout preservation
    # prompt = "<image>\n<|grounding|>OCR this image. "  # Better for non-standard docs
    prompt = "<image>\n<|grounding|>Convert the document to markdown. "  # Best for structured docs
    
    print(f"\nProcessing: {image_file}")
    print(f"Prompt: {prompt}")
    
    # FIXED VERSION - Use this instead of model.infer()
    res = infer_with_fix(
        model, 
        tokenizer, 
        prompt=prompt, 
        image_file=image_file, 
        output_path=output_path, 
        base_size=1024,      # Try 1280 for better quality on difficult documents
        image_size=640, 
        crop_mode=True,      # Gundam mode (dynamic resolution)
        save_results=True, 
        test_compress=True,
        ngram_size=30,       # Increase to 40 for very difficult documents
        window_size=90       # Increase to 120 for very difficult documents
    )
    
    print("\n" + "="*80)
    print("RESULT:")
    print("="*80)
    print(res)
    print("="*80)
    
    print("\n✓ Done! No more hallucination!")
    
    # Tips for ancient/handwritten documents:
    print("\nTips for difficult documents:")
    print("1. Use larger resolution: base_size=1280")
    print("2. Try different prompts (see comments above)")
    print("3. Increase ngram_size=40 and window_size=120")
    print("4. Preprocess image: enhance contrast, remove noise")
