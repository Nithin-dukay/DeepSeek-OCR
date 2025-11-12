"""
Example: OCR for Ancient Portuguese Handwritten Documents

This example demonstrates how to process ancient Portuguese handwritten documents
to minimize hallucinations and maximize accuracy.

This addresses GitHub Issue #191 where users experienced hallucinations
when using "Free OCR" mode on ancient Portuguese handwritten images.
"""

from transformers import AutoModel, AutoTokenizer
from PIL import Image, ImageEnhance
import torch
import os


def preprocess_historical_document(image_path: str) -> Image.Image:
    """
    Preprocess historical Portuguese document for better OCR results
    """
    img = Image.open(image_path)
    
    # Convert to RGB
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize if too large (max 2048px)
    max_size = 2048
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = tuple(int(dim * ratio) for dim in img.size)
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    # Enhance contrast for faded historical documents
    contrast = ImageEnhance.Contrast(img)
    img = contrast.enhance(1.8)  # Aggressive enhancement for ancient documents
    
    # Enhance sharpness
    sharpness = ImageEnhance.Sharpness(img)
    img = sharpness.enhance(1.5)
    
    return img


def main():
    # Configuration
    model_name = 'deepseek-ai/DeepSeek-OCR'
    image_path = 'ancient_portuguese_document.jpg'  # Replace with your image
    output_dir = 'output_portuguese'
    
    print("Loading DeepSeek-OCR model...")
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        _attn_implementation='flash_attention_2',
        trust_remote_code=True,
        use_safetensors=True
    )
    model = model.eval().cuda().to(torch.bfloat16)
    
    print("Preprocessing image...")
    
    # Preprocess the image
    preprocessed_img = preprocess_historical_document(image_path)
    
    # Save preprocessed image for reference
    os.makedirs(output_dir, exist_ok=True)
    preprocessed_path = os.path.join(output_dir, 'preprocessed.jpg')
    preprocessed_img.save(preprocessed_path, quality=95)
    
    print("Running OCR with optimized parameters...")
    
    # CRITICAL: Use grounded OCR mode, NOT "Free OCR"
    # This is the key to avoiding hallucinations
    prompt = "<image>\n<|grounding|>Extract all Portuguese text from this historical handwritten document, preserving original spelling and diacritics."
    
    # Run OCR with conservative parameters
    result = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=preprocessed_path,
        output_path=output_dir,
        base_size=1024,      # Good balance for handwritten documents
        image_size=640,      # Gundam mode for better accuracy
        crop_mode=True,      # Enable crop mode
        save_results=True,
        test_compress=True
    )
    
    print("\n" + "=" * 70)
    print("OCR RESULT - PRIMARY PASS")
    print("=" * 70)
    print(result)
    
    # Optional: Run validation pass with slightly different parameters
    print("\n" + "=" * 70)
    print("Running validation pass...")
    print("=" * 70)
    
    prompt_validation = "<image>\n<|grounding|>Extract all text from this ancient Portuguese manuscript."
    
    result_validation = model.infer(
        tokenizer,
        prompt=prompt_validation,
        image_file=preprocessed_path,
        output_path=output_dir,
        base_size=1024,
        image_size=640,
        crop_mode=True,
        save_results=False,
        test_compress=True
    )
    
    print("\n" + "=" * 70)
    print("OCR RESULT - VALIDATION PASS")
    print("=" * 70)
    print(result_validation)
    
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print("Compare the two results above to identify any inconsistencies.")
    print("The most consistent text is likely the most accurate.")
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
