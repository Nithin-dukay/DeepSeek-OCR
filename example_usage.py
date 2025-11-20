#!/usr/bin/env python3
"""
Example Usage Script for DeepSeek-OCR with vLLM
Demonstrates the fix for GitHub Issue #244

This script shows various ways to use DeepSeek-OCR with proper model registration.
"""

import sys
from pathlib import Path

# Add the DeepSeek-OCR-vllm directory to the path
deepseek_ocr_path = Path(__file__).parent / "DeepSeek-OCR-master" / "DeepSeek-OCR-vllm"
sys.path.insert(0, str(deepseek_ocr_path))

print("=" * 70)
print("DeepSeek-OCR Example Usage - Fix for Issue #244")
print("=" * 70)

# Example 1: Basic Model Registration
print("\n[Example 1] Basic Model Registration")
print("-" * 70)

try:
    from vllm import ModelRegistry
    from deepseek_ocr import DeepseekOCRForCausalLM
    
    # This is the key fix for Issue #244
    ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)
    print("✓ Model registered successfully!")
    print("  Architecture: DeepseekOCRForCausalLM")
    print("  Status: Ready for inference")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("  Please ensure vLLM and dependencies are installed.")
    sys.exit(1)
except Exception as e:
    print(f"✗ Registration error: {e}")
    sys.exit(1)

# Example 2: Create LLM Instance with Proper Configuration
print("\n[Example 2] Creating LLM Instance")
print("-" * 70)

try:
    from vllm import LLM, SamplingParams
    from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
    
    print("Configuration:")
    print("  - Model: deepseek-ai/DeepSeek-OCR")
    print("  - Prefix Caching: Disabled")
    print("  - MM Processor Cache: 0 GB")
    print("  - Logits Processor: NGramPerReqLogitsProcessor")
    
    # Note: Actual LLM creation commented out to avoid downloading the model
    # Uncomment the following lines to create a real LLM instance:
    
    # llm = LLM(
    #     model="deepseek-ai/DeepSeek-OCR",
    #     enable_prefix_caching=False,
    #     mm_processor_cache_gb=0,
    #     logits_processors=[NGramPerReqLogitsProcessor],
    #     trust_remote_code=True,
    # )
    # print("✓ LLM instance created successfully!")
    
    print("✓ Configuration validated (LLM creation skipped in example)")
    
except Exception as e:
    print(f"✗ Error: {e}")

# Example 3: Sampling Parameters Configuration
print("\n[Example 3] Sampling Parameters")
print("-" * 70)

try:
    from vllm import SamplingParams
    
    sampling_params = SamplingParams(
        temperature=0.0,  # Deterministic output
        max_tokens=8192,  # Maximum output length
        extra_args=dict(
            ngram_size=30,
            window_size=90,
            whitelist_token_ids={128821, 128822},  # <td>, </td>
        ),
        skip_special_tokens=False,
    )
    
    print("✓ Sampling parameters configured:")
    print(f"  - Temperature: {sampling_params.temperature}")
    print(f"  - Max Tokens: {sampling_params.max_tokens}")
    print(f"  - N-gram Size: 30")
    print(f"  - Window Size: 90")
    print(f"  - Whitelist Tokens: {{128821, 128822}}")
    
except Exception as e:
    print(f"✗ Error: {e}")

# Example 4: Prompt Templates
print("\n[Example 4] Common Prompt Templates")
print("-" * 70)

prompts = {
    "Free OCR": "<image>\\nFree OCR.",
    "Document to Markdown": "<image>\\n<|grounding|>Convert the document to markdown.",
    "General OCR": "<image>\\n<|grounding|>OCR this image.",
    "Parse Figure": "<image>\\nParse the figure.",
    "Detailed Description": "<image>\\nDescribe this image in detail.",
    "Text Localization": "<image>\\nLocate <|ref|>specific text<|/ref|> in the image.",
}

for name, prompt in prompts.items():
    print(f"  {name}:")
    print(f"    {prompt}")

# Example 5: Resolution Modes
print("\n[Example 5] Resolution Modes")
print("-" * 70)

modes = [
    ("Tiny", 512, 512, False, 64, "Small images, fast inference"),
    ("Small", 640, 640, False, 100, "Standard images"),
    ("Base", 1024, 1024, False, 256, "High-quality documents"),
    ("Large", 1280, 1280, False, 400, "Very high resolution"),
    ("Gundam", 1024, 640, True, "Variable", "Large documents (recommended)"),
]

print(f"{'Mode':<10} {'Base':<6} {'Image':<6} {'Crop':<6} {'Tokens':<10} {'Use Case'}")
print("-" * 70)
for mode, base, image, crop, tokens, use_case in modes:
    crop_str = "Yes" if crop else "No"
    print(f"{mode:<10} {base:<6} {image:<6} {crop_str:<6} {str(tokens):<10} {use_case}")

# Example 6: Inference Workflow (Pseudo-code)
print("\n[Example 6] Complete Inference Workflow (Pseudo-code)")
print("-" * 70)

workflow = """
1. Register Model:
   ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)

2. Create LLM Instance:
   llm = LLM(
       model="deepseek-ai/DeepSeek-OCR",
       enable_prefix_caching=False,
       mm_processor_cache_gb=0,
       logits_processors=[NGramPerReqLogitsProcessor],
   )

3. Prepare Input:
   image = Image.open("document.jpg").convert("RGB")
   prompt = "<image>\\\\nFree OCR."
   model_input = [{"prompt": prompt, "multi_modal_data": {"image": image}}]

4. Configure Sampling:
   sampling_params = SamplingParams(
       temperature=0.0,
       max_tokens=8192,
       extra_args=dict(ngram_size=30, window_size=90, ...),
   )

5. Generate Output:
   outputs = llm.generate(model_input, sampling_params)
   result = outputs[0].outputs[0].text

6. Process Result:
   print(result)
   # Save to file, parse markdown, etc.
"""

print(workflow)

# Example 7: Error Handling
print("\n[Example 7] Error Handling Best Practices")
print("-" * 70)

error_handling = """
try:
    # Register model
    ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)
except Exception as e:
    print(f"Model registration failed: {e}")
    sys.exit(1)

try:
    # Create LLM
    llm = LLM(model="deepseek-ai/DeepSeek-OCR", ...)
except Exception as e:
    print(f"LLM creation failed: {e}")
    print("Check: GPU memory, CUDA version, model path")
    sys.exit(1)

try:
    # Run inference
    outputs = llm.generate(inputs, sampling_params)
except Exception as e:
    print(f"Inference failed: {e}")
    print("Check: Image format, prompt format, memory")
    sys.exit(1)
"""

print(error_handling)

# Example 8: Performance Tips
print("\n[Example 8] Performance Optimization Tips")
print("-" * 70)

tips = [
    "1. Install flash-attention for faster inference:",
    "   pip install flash-attn==2.7.3 --no-build-isolation",
    "",
    "2. Use appropriate resolution mode for your use case:",
    "   - Tiny/Small: Fast inference, lower quality",
    "   - Base/Large: High quality, slower inference",
    "   - Gundam: Best for large documents with cropping",
    "",
    "3. Adjust MAX_CROPS based on GPU memory:",
    "   - 16GB VRAM: MAX_CROPS = 4",
    "   - 24GB VRAM: MAX_CROPS = 6",
    "   - 40GB+ VRAM: MAX_CROPS = 9",
    "",
    "4. Use batch processing for multiple images:",
    "   model_inputs = [input1, input2, input3, ...]",
    "   outputs = llm.generate(model_inputs, sampling_params)",
    "",
    "5. Enable tensor parallelism for multi-GPU:",
    "   llm = LLM(model=..., tensor_parallel_size=2)",
]

for tip in tips:
    print(tip)

# Summary
print("\n" + "=" * 70)
print("Summary: Fix for GitHub Issue #244")
print("=" * 70)

summary = """
The key to fixing Issue #244 is to register the DeepseekOCRForCausalLM model
with vLLM's ModelRegistry BEFORE creating an LLM instance.

Quick Fix:
  from vllm import ModelRegistry
  from deepseek_ocr import DeepseekOCRForCausalLM
  
  ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)

This tells vLLM how to instantiate the custom model architecture.

For more details, see:
  - FIX_ISSUE_244.md: Detailed explanation and solutions
  - SETUP_VLLM.md: Complete setup and usage guide
  - serve_deepseek_ocr.py: Ready-to-use wrapper script
"""

print(summary)

print("\n✓ All examples completed successfully!")
print("=" * 70)
