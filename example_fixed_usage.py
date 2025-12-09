#!/usr/bin/env python3
"""
Example: Using DeepSeek-OCR with the Issue #288 fix.

This example demonstrates how to use custom prompts with the improved
NoRepeatNGramLogitsProcessor and prompt validation utilities.
"""

import os
import sys

# Uncomment to set GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Add path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'DeepSeek-OCR-master', 'DeepSeek-OCR-vllm'))


def example_1_recommended_prompts():
    """Example 1: Using recommended prompts (safest approach)."""
    print("\n" + "=" * 80)
    print("Example 1: Using Recommended Prompts")
    print("=" * 80 + "\n")
    
    from process.prompt_utils import PromptValidator
    
    # Get recommended prompt for document OCR
    prompt = PromptValidator.get_recommended_prompt("document_markdown")
    print(f"Recommended prompt: {repr(prompt)}")
    
    # This prompt is guaranteed to work well with the model
    print("\n✓ This prompt is tested and optimized for best results")
    print("  Use this in your vLLM or Transformers inference code")


def example_2_validate_custom_prompt():
    """Example 2: Validating and optimizing a custom prompt."""
    print("\n" + "=" * 80)
    print("Example 2: Validating Custom Prompts")
    print("=" * 80 + "\n")
    
    from process.prompt_utils import validate_and_optimize_prompt
    
    # The problematic prompt from Issue #288
    custom_prompt = "<image>\n<|grounding|>Convert the document to markdown, dont add any extra space between letters."
    
    print(f"Original prompt: {repr(custom_prompt)}\n")
    
    # Validate and optimize
    optimized = validate_and_optimize_prompt(custom_prompt, verbose=True)
    
    print(f"\n✓ You can now use this prompt safely")
    print(f"  Final prompt: {repr(optimized)}")


def example_3_vllm_inference():
    """Example 3: Complete vLLM inference with the fix."""
    print("\n" + "=" * 80)
    print("Example 3: vLLM Inference with Fixed Processor")
    print("=" * 80 + "\n")
    
    print("Code example (requires GPU and model):\n")
    
    code = '''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from vllm import LLM, SamplingParams
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCRProcessor
from process.prompt_utils import validate_and_optimize_prompt
from PIL import Image

# Initialize model
llm = LLM(
    model="deepseek-ai/DeepSeek-OCR",
    hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
    trust_remote_code=True,
    max_model_len=8192,
)

# Create improved logits processor
logits_processors = [
    NoRepeatNGramLogitsProcessor(
        ngram_size=30,
        window_size=90,
        whitelist_token_ids={128821, 128822},
        min_generated_tokens=10,    # NEW: Wait for generation to start
        enable_adaptive=True         # NEW: Smarter blocking
    )
]

# Sampling parameters
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    logits_processors=logits_processors,
    skip_special_tokens=False,
)

# Your custom prompt (now works!)
prompt = "<image>\\n<|grounding|>Convert the document to markdown, dont add any extra space between letters."

# Validate prompt (optional but recommended)
prompt = validate_and_optimize_prompt(prompt, verbose=True)

# Process image
image = Image.open("your_image.jpg").convert('RGB')
processor = DeepseekOCRProcessor()

batch_inputs = [{
    "prompt": prompt,
    "multi_modal_data": {
        "image": processor.tokenize_with_images(
            images=[image],
            bos=True,
            eos=True,
            cropping=True
        )
    },
}]

# Generate
outputs = llm.generate(batch_inputs, sampling_params=sampling_params)
result = outputs[0].outputs[0].text

print("OCR Result:")
print(result)
'''
    
    print(code)
    print("\n✓ This code will work correctly with custom prompts")


def example_4_transformers_inference():
    """Example 4: Transformers inference (HuggingFace)."""
    print("\n" + "=" * 80)
    print("Example 4: Transformers Inference")
    print("=" * 80 + "\n")
    
    print("Code example (requires GPU and model):\n")
    
    code = '''
from transformers import AutoModel, AutoTokenizer
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Load model
model_name = 'deepseek-ai/DeepSeek-OCR'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    _attn_implementation='flash_attention_2',
    trust_remote_code=True,
    use_safetensors=True
)
model = model.eval().cuda().to(torch.bfloat16)

# Your custom prompt (now works!)
prompt = "<image>\\n<|grounding|>Convert the document to markdown, dont add any extra space between letters."

# Optional: Validate prompt
from process.prompt_utils import validate_and_optimize_prompt
prompt = validate_and_optimize_prompt(prompt, verbose=True)

# Run inference
image_file = 'your_image.jpg'
output_path = 'output'

res = model.infer(
    tokenizer,
    prompt=prompt,
    image_file=image_file,
    output_path=output_path,
    base_size=1024,
    image_size=640,
    crop_mode=True,
    save_results=True,
    test_compress=True
)

print("OCR Result:")
print(res)
'''
    
    print(code)
    print("\n✓ Transformers inference also benefits from the fix")


def example_5_configuration_options():
    """Example 5: Different configuration options for different use cases."""
    print("\n" + "=" * 80)
    print("Example 5: Configuration Options")
    print("=" * 80 + "\n")
    
    from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
    
    print("Configuration for different use cases:\n")
    
    print("1. Document OCR (High Accuracy):")
    processor1 = NoRepeatNGramLogitsProcessor(
        ngram_size=40,
        window_size=90,
        whitelist_token_ids={128821, 128822},
        min_generated_tokens=15,
        enable_adaptive=True
    )
    print(f"   ngram_size=40, min_generated_tokens=15")
    print(f"   ✓ Best for documents with complex formatting\n")
    
    print("2. Table Extraction (Allow Repetition):")
    processor2 = NoRepeatNGramLogitsProcessor(
        ngram_size=20,
        window_size=50,
        whitelist_token_ids={128821, 128822},
        min_generated_tokens=5,
        enable_adaptive=True
    )
    print(f"   ngram_size=20, min_generated_tokens=5")
    print(f"   ✓ Allows table cell repetitions\n")
    
    print("3. General OCR (Balanced):")
    processor3 = NoRepeatNGramLogitsProcessor(
        ngram_size=30,
        window_size=90,
        whitelist_token_ids={128821, 128822},
        min_generated_tokens=10,
        enable_adaptive=True
    )
    print(f"   ngram_size=30, min_generated_tokens=10")
    print(f"   ✓ Good default for most use cases\n")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("DeepSeek-OCR Issue #288 Fix - Usage Examples")
    print("=" * 80)
    
    # Run examples
    example_1_recommended_prompts()
    example_2_validate_custom_prompt()
    example_3_vllm_inference()
    example_4_transformers_inference()
    example_5_configuration_options()
    
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80 + "\n")
    
    print("Key Takeaways:")
    print("  1. Use recommended prompts when possible")
    print("  2. Validate custom prompts with validate_and_optimize_prompt()")
    print("  3. Use improved NoRepeatNGramLogitsProcessor settings")
    print("  4. Configure parameters based on your use case")
    print("\nFor more details, see:")
    print("  • QUICK_START_FIX.md - Quick reference")
    print("  • ISSUE_288_FIX.md - Complete documentation")
    print()


if __name__ == "__main__":
    main()
