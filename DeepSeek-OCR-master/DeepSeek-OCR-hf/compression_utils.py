"""
Compression Utilities for DeepSeek-OCR

This module provides utilities to fix and enhance the compression evaluation
functionality in DeepSeek-OCR. It addresses the issue where model.infer() 
returns None when test_compress=True.

GitHub Issue #285: Query about Compression Study Evaluation
"""

import types
import json
import csv
from pathlib import Path
from typing import Dict, Any, Optional, List


def patch_infer_method(model):
    """
    Monkey-patch the model's infer method to fix the return value issue
    when test_compress=True.
    
    This fixes the bug where the original infer method prints compression
    statistics but doesn't return the output text.
    
    Args:
        model: The DeepSeek-OCR model instance
        
    Returns:
        The patched model
    """
    original_infer = model.infer
    
    def patched_infer(tokenizer, prompt='', image_file='', output_path='', 
                     base_size=1024, image_size=640, crop_mode=True, 
                     test_compress=False, save_results=False, eval_mode=False,
                     return_stats=False):
        """
        Patched infer method that returns output even when test_compress=True.
        
        Additional parameter:
            return_stats (bool): If True, returns a dict with output and compression stats
        """
        # Store original stdout to capture print statements
        import sys
        from io import StringIO
        
        if test_compress and not eval_mode:
            # Temporarily redirect stdout to capture compression stats
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()
            
            # Call original method with eval_mode=True to get output
            output = original_infer(
                tokenizer, 
                prompt=prompt, 
                image_file=image_file, 
                output_path=output_path,
                base_size=base_size, 
                image_size=image_size, 
                crop_mode=crop_mode,
                test_compress=False,  # Disable to avoid None return
                save_results=save_results,
                eval_mode=True  # Enable to get output
            )
            
            # Now call again with test_compress to print stats
            original_infer(
                tokenizer, 
                prompt=prompt, 
                image_file=image_file, 
                output_path=output_path,
                base_size=base_size, 
                image_size=image_size, 
                crop_mode=crop_mode,
                test_compress=True,
                save_results=False,  # Don't save twice
                eval_mode=False
            )
            
            # Restore stdout
            sys.stdout = old_stdout
            
            if return_stats:
                # Parse compression stats from captured output
                stats_text = captured_output.getvalue()
                stats = parse_compression_stats(stats_text)
                return {
                    'output': output,
                    'stats': stats
                }
            
            return output
        else:
            # Call original method for other cases
            return original_infer(
                tokenizer, 
                prompt=prompt, 
                image_file=image_file, 
                output_path=output_path,
                base_size=base_size, 
                image_size=image_size, 
                crop_mode=crop_mode,
                test_compress=test_compress,
                save_results=save_results,
                eval_mode=eval_mode
            )
    
    # Replace the method
    model.infer = types.MethodType(patched_infer, model)
    return model


def parse_compression_stats(stats_text: str) -> Dict[str, Any]:
    """
    Parse compression statistics from the printed output.
    
    Args:
        stats_text: The captured stdout text containing compression stats
        
    Returns:
        Dictionary with parsed statistics
    """
    stats = {}
    
    lines = stats_text.strip().split('\n')
    for line in lines:
        if 'image size:' in line:
            # Extract (width, height)
            size_str = line.split('image size:')[1].strip()
            stats['image_size'] = size_str
        elif 'valid image tokens:' in line:
            tokens = line.split('valid image tokens:')[1].strip()
            stats['image_tokens'] = int(tokens)
        elif 'output texts tokens (valid):' in line:
            tokens = line.split('output texts tokens (valid):')[1].strip()
            stats['output_tokens'] = int(tokens)
        elif 'compression ratio:' in line:
            ratio = line.split('compression ratio:')[1].strip()
            stats['compression_ratio'] = float(ratio)
    
    return stats


def evaluate_compression_batch(model, tokenizer, image_files: List[str], 
                               prompt: str = "<image>\n<|grounding|>Convert the document to markdown.",
                               output_dir: str = "./compression_results",
                               base_size: int = 1024,
                               image_size: int = 640,
                               crop_mode: bool = True) -> List[Dict[str, Any]]:
    """
    Evaluate compression on a batch of images and collect statistics.
    
    Args:
        model: The DeepSeek-OCR model instance (should be patched)
        tokenizer: The tokenizer instance
        image_files: List of image file paths
        prompt: The prompt to use for inference
        output_dir: Directory to save results
        base_size: Base size parameter for inference
        image_size: Image size parameter for inference
        crop_mode: Whether to use crop mode
        
    Returns:
        List of dictionaries containing results and statistics for each image
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for idx, image_file in enumerate(image_files):
        print(f"\nProcessing {idx+1}/{len(image_files)}: {image_file}")
        print("-" * 80)
        
        try:
            # Run inference with compression stats
            result = model.infer(
                tokenizer,
                prompt=prompt,
                image_file=image_file,
                output_path=str(output_path / f"image_{idx}"),
                base_size=base_size,
                image_size=image_size,
                crop_mode=crop_mode,
                test_compress=True,
                save_results=True,
                return_stats=True
            )
            
            if isinstance(result, dict):
                result['image_file'] = image_file
                result['index'] = idx
                results.append(result)
            else:
                # Fallback if return_stats not supported
                results.append({
                    'image_file': image_file,
                    'index': idx,
                    'output': result,
                    'stats': {}
                })
                
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            results.append({
                'image_file': image_file,
                'index': idx,
                'error': str(e)
            })
    
    return results


def save_compression_results(results: List[Dict[str, Any]], 
                            output_file: str = "compression_results.json",
                            csv_file: Optional[str] = "compression_results.csv"):
    """
    Save compression evaluation results to JSON and optionally CSV.
    
    Args:
        results: List of result dictionaries from evaluate_compression_batch
        output_file: Path to save JSON results
        csv_file: Optional path to save CSV summary
    """
    # Save full results as JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {output_file}")
    
    # Save summary as CSV if requested
    if csv_file:
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Index', 'Image File', 'Image Size', 'Image Tokens', 
                'Output Tokens', 'Compression Ratio', 'Error'
            ])
            
            for result in results:
                stats = result.get('stats', {})
                writer.writerow([
                    result.get('index', ''),
                    result.get('image_file', ''),
                    stats.get('image_size', ''),
                    stats.get('image_tokens', ''),
                    stats.get('output_tokens', ''),
                    stats.get('compression_ratio', ''),
                    result.get('error', '')
                ])
        
        print(f"CSV summary saved to {csv_file}")


def calculate_average_compression(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate average compression statistics from results.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Dictionary with average statistics
    """
    valid_results = [r for r in results if 'stats' in r and 'compression_ratio' in r['stats']]
    
    if not valid_results:
        return {}
    
    total_image_tokens = sum(r['stats'].get('image_tokens', 0) for r in valid_results)
    total_output_tokens = sum(r['stats'].get('output_tokens', 0) for r in valid_results)
    avg_compression_ratio = sum(r['stats'].get('compression_ratio', 0) for r in valid_results) / len(valid_results)
    
    return {
        'num_images': len(valid_results),
        'total_image_tokens': total_image_tokens,
        'total_output_tokens': total_output_tokens,
        'avg_image_tokens': total_image_tokens / len(valid_results),
        'avg_output_tokens': total_output_tokens / len(valid_results),
        'avg_compression_ratio': avg_compression_ratio,
        'overall_compression_ratio': total_output_tokens / total_image_tokens if total_image_tokens > 0 else 0
    }


def print_compression_summary(results: List[Dict[str, Any]]):
    """
    Print a summary of compression evaluation results.
    
    Args:
        results: List of result dictionaries
    """
    avg_stats = calculate_average_compression(results)
    
    if not avg_stats:
        print("\nNo valid compression statistics found.")
        return
    
    print("\n" + "=" * 80)
    print("COMPRESSION EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Number of images processed: {avg_stats['num_images']}")
    print(f"Average image tokens: {avg_stats['avg_image_tokens']:.2f}")
    print(f"Average output tokens: {avg_stats['avg_output_tokens']:.2f}")
    print(f"Average compression ratio: {avg_stats['avg_compression_ratio']:.2f}")
    print(f"Overall compression ratio: {avg_stats['overall_compression_ratio']:.2f}")
    print("=" * 80)


# Example usage
if __name__ == "__main__":
    print("Compression Utilities for DeepSeek-OCR")
    print("=" * 80)
    print("\nThis module provides utilities to fix the compression evaluation issue.")
    print("\nUsage example:")
    print("""
    from transformers import AutoModel, AutoTokenizer
    from compression_utils import patch_infer_method
    
    # Load model
    model = AutoModel.from_pretrained('deepseek-ai/DeepSeek-OCR', trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-OCR', trust_remote_code=True)
    
    # Patch the model
    model = patch_infer_method(model)
    
    # Now test_compress=True will return the output
    result = model.infer(
        tokenizer,
        prompt="<image>\\n<|grounding|>Convert the document to markdown.",
        image_file="your_image.jpg",
        output_path="./output",
        test_compress=True,
        save_results=True
    )
    
    print(result)  # Will now contain the output text instead of None
    """)
