"""
Fix for GitHub Issue #191: Consistently hallucinationing during Free OCR

This script demonstrates how to fix hallucination issues when using DeepSeek-OCR
with the Transformers/HuggingFace API, especially for handwritten or ancient documents.

The key fix is adding the NoRepeatNGramLogitsProcessor to prevent the model from
generating repetitive patterns that lead to hallucinations.
"""

from transformers import AutoModel, AutoTokenizer
import torch
import os
import sys

# Add the DeepSeek-OCR-hf directory to path to import the processor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'DeepSeek-OCR-master/DeepSeek-OCR-hf'))
from ngram_norepeat_hf import NoRepeatNGramLogitsProcessor, get_recommended_params


def setup_model(device='cuda:0'):
    """
    Setup the DeepSeek-OCR model with proper configuration.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = device.split(':')[-1] if ':' in device else '0'
    
    model_name = 'deepseek-ai/DeepSeek-OCR'
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    print("Loading model...")
    model = AutoModel.from_pretrained(
        model_name, 
        _attn_implementation='flash_attention_2', 
        trust_remote_code=True, 
        use_safetensors=True
    )
    model = model.eval().cuda().to(torch.bfloat16)
    
    print("Model loaded successfully!")
    return model, tokenizer


def run_ocr_with_hallucination_fix(
    model,
    tokenizer,
    image_file,
    output_path='./output',
    document_type='handwritten',
    prompt_type='free_ocr',
    base_size=1024,
    image_size=640,
    crop_mode=True
):
    """
    Run OCR with hallucination prevention.
    
    Args:
        model: DeepSeek-OCR model
        tokenizer: Model tokenizer
        image_file: Path to image file
        output_path: Directory to save results
        document_type: Type of document - "handwritten", "printed", "table", or "general"
        prompt_type: Type of prompt - "free_ocr", "markdown", "grounding", or "custom"
        base_size: Base resolution (512, 640, 1024, or 1280)
        image_size: Image size for cropping (typically 640)
        crop_mode: Whether to use dynamic cropping (Gundam mode)
    
    Returns:
        OCR result text
    """
    
    # Select appropriate prompt
    prompts = {
        'free_ocr': "<image>\nFree OCR. ",
        'markdown': "<image>\n<|grounding|>Convert the document to markdown. ",
        'grounding': "<image>\n<|grounding|>OCR this image.",
        'detail': "<image>\nDescribe this image in detail.",
    }
    prompt = prompts.get(prompt_type, prompts['free_ocr'])
    
    print(f"\nConfiguration:")
    print(f"  Document type: {document_type}")
    print(f"  Prompt type: {prompt_type}")
    print(f"  Image: {image_file}")
    print(f"  Base size: {base_size}, Image size: {image_size}, Crop mode: {crop_mode}")
    
    # Get recommended parameters for the document type
    params = get_recommended_params(document_type)
    print(f"\nHallucination Prevention Parameters:")
    print(f"  N-gram size: {params['ngram_size']}")
    print(f"  Window size: {params['window_size']}")
    print(f"  Whitelist tokens: {params['whitelist_token_ids']}")
    
    # Create logits processor to prevent hallucinations
    logits_processor = NoRepeatNGramLogitsProcessor(
        ngram_size=params["ngram_size"],
        window_size=params["window_size"],
        whitelist_token_ids=params["whitelist_token_ids"]
    )
    
    print("\nRunning OCR...")
    
    # Run inference with hallucination prevention
    res = model.infer(
        tokenizer, 
        prompt=prompt, 
        image_file=image_file, 
        output_path=output_path, 
        base_size=base_size, 
        image_size=image_size, 
        crop_mode=crop_mode, 
        save_results=True, 
        test_compress=True,
        logits_processor=logits_processor  # This is the key fix!
    )
    
    print("OCR completed!")
    return res


def main():
    """
    Example usage demonstrating the fix for Issue #191
    """
    
    print("="*80)
    print("DeepSeek-OCR Hallucination Fix - GitHub Issue #191")
    print("="*80)
    
    # Setup model
    model, tokenizer = setup_model(device='cuda:0')
    
    # Example 1: Ancient Portuguese handwritten document (from the issue)
    print("\n" + "="*80)
    print("Example 1: Handwritten Document (Ancient Portuguese)")
    print("="*80)
    
    result = run_ocr_with_hallucination_fix(
        model=model,
        tokenizer=tokenizer,
        image_file='your_image.jpg',  # Replace with your image path
        output_path='./output_handwritten',
        document_type='handwritten',  # Use handwritten parameters
        prompt_type='free_ocr',
        base_size=1024,
        image_size=640,
        crop_mode=True
    )
    
    print(f"\nResult preview (first 500 chars):")
    print(result[:500] if result else "No result")
    
    # Example 2: Printed document
    print("\n" + "="*80)
    print("Example 2: Printed Document")
    print("="*80)
    
    result = run_ocr_with_hallucination_fix(
        model=model,
        tokenizer=tokenizer,
        image_file='your_printed_doc.jpg',  # Replace with your image path
        output_path='./output_printed',
        document_type='printed',  # Use printed document parameters
        prompt_type='markdown',
        base_size=1024,
        image_size=640,
        crop_mode=True
    )
    
    print(f"\nResult preview (first 500 chars):")
    print(result[:500] if result else "No result")
    
    # Example 3: Document with tables
    print("\n" + "="*80)
    print("Example 3: Document with Tables")
    print("="*80)
    
    result = run_ocr_with_hallucination_fix(
        model=model,
        tokenizer=tokenizer,
        image_file='your_table_doc.jpg',  # Replace with your image path
        output_path='./output_table',
        document_type='table',  # Use table parameters (allows <td> repetition)
        prompt_type='markdown',
        base_size=1024,
        image_size=640,
        crop_mode=True
    )
    
    print(f"\nResult preview (first 500 chars):")
    print(result[:500] if result else "No result")
    
    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80)


if __name__ == "__main__":
    # Quick usage example
    print("""
    QUICK USAGE:
    
    from transformers import AutoModel, AutoTokenizer
    import torch
    import os
    from ngram_norepeat_hf import NoRepeatNGramLogitsProcessor, get_recommended_params
    
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    model_name = 'deepseek-ai/DeepSeek-OCR'
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, _attn_implementation='flash_attention_2', 
                                     trust_remote_code=True, use_safetensors=True)
    model = model.eval().cuda().to(torch.bfloat16)
    
    # THE FIX: Add logits processor to prevent hallucinations
    params = get_recommended_params("handwritten")  # or "printed", "table", "general"
    logits_processor = NoRepeatNGramLogitsProcessor(
        ngram_size=params["ngram_size"],
        window_size=params["window_size"],
        whitelist_token_ids=params["whitelist_token_ids"]
    )
    
    prompt = "<image>\\nFree OCR. "
    image_file = 'your_image.jpg'
    output_path = 'your/output/dir'
    
    res = model.infer(tokenizer, prompt=prompt, image_file=image_file, 
                     output_path=output_path, base_size=1024, image_size=640, 
                     crop_mode=True, save_results=True, test_compress=True,
                     logits_processor=logits_processor)  # <-- Add this parameter!
    """)
    
    # Uncomment to run the full examples
    # main()
