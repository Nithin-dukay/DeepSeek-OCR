"""
DeepSeek-OCR Table Recognition Example
======================================

This script demonstrates how to use DeepSeek-OCR for table recognition
and conversion to HTML format.

Usage:
    python table_to_html_example.py --image path/to/table_image.jpg --output output.html

Requirements:
    - DeepSeek-OCR model installed
    - vLLM or transformers library
"""

import argparse
import os
from pathlib import Path


def table_to_html_vllm(image_path: str, output_path: str = None):
    """
    Convert a table image to HTML using vLLM inference.
    
    Args:
        image_path: Path to the input image containing a table
        output_path: Optional path to save the HTML output
    
    Returns:
        str: HTML representation of the table
    """
    import torch
    from vllm import LLM, SamplingParams
    from vllm.model_executor.models.registry import ModelRegistry
    from PIL import Image
    from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
    from process.image_process import DeepseekOCRProcessor
    
    # Import and register the model
    try:
        from deepseek_ocr import DeepseekOCRForCausalLM
        ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)
    except ImportError:
        # Model might already be registered
        pass
    
    # Model configuration
    MODEL_PATH = 'deepseek-ai/DeepSeek-OCR'
    
    # Table-to-HTML prompt
    PROMPT = '<image>\n<|grounding|>Convert the table to HTML.'
    
    print(f"Loading image from: {image_path}")
    image = Image.open(image_path).convert('RGB')
    
    # Initialize the model
    print("Initializing DeepSeek-OCR model...")
    llm = LLM(
        model=MODEL_PATH,
        hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
        block_size=256,
        enforce_eager=False,
        trust_remote_code=True,
        max_model_len=8192,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.75,
    )
    
    # Configure logits processor with whitelist for table tokens
    # Token IDs: 128821 = <td>, 128822 = </td>
    logits_processors = [
        NoRepeatNGramLogitsProcessor(
            ngram_size=30,
            window_size=90,
            whitelist_token_ids={128821, 128822}  # Allow repetition of <td>, </td>
        )
    ]
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        logits_processors=logits_processors,
        skip_special_tokens=False,
    )
    
    # Process the image
    print("Processing image...")
    image_features = DeepseekOCRProcessor().tokenize_with_images(
        images=[image],
        bos=True,
        eos=True,
        cropping=True  # Use dynamic cropping for better results
    )
    
    # Prepare the request
    request = {
        "prompt": PROMPT,
        "multi_modal_data": {"image": image_features}
    }
    
    # Generate HTML output
    print("Generating HTML table...")
    outputs = llm.generate([request], sampling_params=sampling_params)
    
    html_output = outputs[0].outputs[0].text
    
    # Clean up the output
    html_output = html_output.strip()
    
    # Save to file if output path is provided
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create a complete HTML document
        full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Table Recognition Result</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
    </style>
</head>
<body>
    <h1>Table Recognition Result</h1>
    <p>Source: {image_path}</p>
    {html_output}
</body>
</html>"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_html)
        
        print(f"HTML saved to: {output_path}")
    
    return html_output


def table_to_html_transformers(image_path: str, output_path: str = None):
    """
    Convert a table image to HTML using Transformers inference.
    
    Args:
        image_path: Path to the input image containing a table
        output_path: Optional path to save the HTML output
    
    Returns:
        str: HTML representation of the table
    """
    from transformers import AutoModel, AutoTokenizer
    import torch
    from pathlib import Path
    
    MODEL_NAME = 'deepseek-ai/DeepSeek-OCR'
    
    print("Loading DeepSeek-OCR model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        _attn_implementation='flash_attention_2',
        trust_remote_code=True,
        use_safetensors=True
    )
    model = model.eval().cuda().to(torch.bfloat16)
    
    # Table-to-HTML prompt
    prompt = "<image>\n<|grounding|>Convert the table to HTML."
    
    print(f"Processing image: {image_path}")
    
    # Create temporary output directory if needed
    temp_output = output_path if output_path else "temp_output"
    os.makedirs(temp_output, exist_ok=True)
    
    # Run inference
    result = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=image_path,
        output_path=temp_output,
        base_size=1024,
        image_size=640,
        crop_mode=True,
        save_results=True,
        test_compress=True
    )
    
    print("Table conversion complete!")
    
    if output_path:
        print(f"Results saved to: {output_path}")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Convert table images to HTML using DeepSeek-OCR'
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to the input image containing a table'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='table_output.html',
        help='Path to save the HTML output (default: table_output.html)'
    )
    parser.add_argument(
        '--backend',
        type=str,
        choices=['vllm', 'transformers'],
        default='vllm',
        help='Inference backend to use (default: vllm)'
    )
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return
    
    print("="*60)
    print("DeepSeek-OCR Table Recognition")
    print("="*60)
    print(f"Input image: {args.image}")
    print(f"Output file: {args.output}")
    print(f"Backend: {args.backend}")
    print("="*60)
    
    try:
        if args.backend == 'vllm':
            html_output = table_to_html_vllm(args.image, args.output)
        else:
            html_output = table_to_html_transformers(args.image, args.output)
        
        print("\n" + "="*60)
        print("HTML Output Preview:")
        print("="*60)
        print(html_output[:500] + "..." if len(html_output) > 500 else html_output)
        print("="*60)
        print("\n✓ Table conversion completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error during table conversion: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
