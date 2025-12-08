#!/usr/bin/env python3
"""
Example client for DeepSeek-OCR vLLM server.

This script demonstrates how to use the vLLM server with different modes
for OCR tasks.

Usage:
    python example_client.py --image path/to/image.jpg
    python example_client.py --image document.pdf --prompt "Convert to markdown"
"""

import argparse
import base64
import sys
from io import BytesIO
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not installed. Install with: pip install openai")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("Error: PIL package not installed. Install with: pip install Pillow")
    sys.exit(1)


def encode_image(image_path: str) -> str:
    """Encode image to base64 string."""
    image = Image.open(image_path)
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Encode to base64
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return f"data:image/jpeg;base64,{img_str}"


def ocr_document(
    image_path: str,
    prompt: str = "<image>\n<|grounding|>Convert the document to markdown.",
    base_url: str = "http://localhost:8000/v1",
    api_key: str = "123",
    model: str = "ocr",
    max_tokens: int = 8192,
    temperature: float = 0.0,
    stream: bool = False
) -> str:
    """
    Perform OCR on a document using the vLLM server.
    
    Args:
        image_path: Path to the image file
        prompt: Prompt template (must include <image> token)
        base_url: Base URL of the vLLM server
        api_key: API key for authentication
        model: Model name
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        stream: Whether to stream the response
    
    Returns:
        OCR result text
    """
    # Encode image
    print(f"Loading image: {image_path}")
    image_url = encode_image(image_path)
    
    # Create client
    client = OpenAI(base_url=base_url, api_key=api_key)
    
    # Prepare messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }
    ]
    
    print(f"Sending request to {base_url}")
    print(f"Prompt: {prompt}")
    print(f"Streaming: {stream}")
    print("-" * 60)
    
    if stream:
        # Streaming response
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True
        )
        
        full_text = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end='', flush=True)
                full_text += content
        print()  # New line at the end
        return full_text
    else:
        # Non-streaming response
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        result = response.choices[0].message.content
        print(result)
        return result


def main():
    parser = argparse.ArgumentParser(
        description="Example client for DeepSeek-OCR vLLM server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic OCR
  python example_client.py --image document.jpg
  
  # Convert to markdown with grounding
  python example_client.py --image document.jpg --prompt "<image>\\n<|grounding|>Convert the document to markdown."
  
  # Free OCR without layout
  python example_client.py --image document.jpg --prompt "<image>\\nFree OCR."
  
  # Describe image
  python example_client.py --image photo.jpg --prompt "<image>\\nDescribe this image in detail."
  
  # Stream output
  python example_client.py --image document.jpg --stream
  
  # Custom server
  python example_client.py --image document.jpg --base-url http://localhost:8002/v1 --api-key my-key

Common Prompts:
  Document OCR with layout:
    <image>\\n<|grounding|>Convert the document to markdown.
  
  Free OCR (no layout):
    <image>\\nFree OCR.
  
  Parse figure/chart:
    <image>\\nParse the figure.
  
  General description:
    <image>\\nDescribe this image in detail.
  
  Locate text:
    <image>\\nLocate <|ref|>specific text<|/ref|> in the image.
        """
    )
    
    parser.add_argument(
        '--image',
        required=True,
        help='Path to the image file'
    )
    parser.add_argument(
        '--prompt',
        default='<image>\n<|grounding|>Convert the document to markdown.',
        help='Prompt template (must include <image> token)'
    )
    parser.add_argument(
        '--base-url',
        default='http://localhost:8000/v1',
        help='Base URL of the vLLM server'
    )
    parser.add_argument(
        '--api-key',
        default='123',
        help='API key for authentication'
    )
    parser.add_argument(
        '--model',
        default='ocr',
        help='Model name'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=8192,
        help='Maximum tokens to generate'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='Sampling temperature'
    )
    parser.add_argument(
        '--stream',
        action='store_true',
        help='Stream the response'
    )
    parser.add_argument(
        '--output',
        help='Output file to save the result'
    )
    
    args = parser.parse_args()
    
    # Check if image exists
    if not Path(args.image).exists():
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)
    
    try:
        # Perform OCR
        result = ocr_document(
            image_path=args.image,
            prompt=args.prompt,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            stream=args.stream
        )
        
        # Save to file if specified
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(result)
            print(f"\n{'-' * 60}")
            print(f"Result saved to: {args.output}")
        
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
