"""
Example: Batch Processing Multiple Handwritten Documents

This example shows how to process multiple handwritten documents efficiently
while maintaining high accuracy and minimizing hallucinations.
"""

from transformers import AutoModel, AutoTokenizer
from PIL import Image, ImageEnhance
import torch
import os
from pathlib import Path
import json
from datetime import datetime


class BatchOCRProcessor:
    """
    Batch processor for handwritten documents
    """
    
    def __init__(self, model_name='deepseek-ai/DeepSeek-OCR'):
        print(f"Loading model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        self.model = AutoModel.from_pretrained(
            model_name,
            _attn_implementation='flash_attention_2',
            trust_remote_code=True,
            use_safetensors=True
        )
        
        self.model = self.model.eval().cuda().to(torch.bfloat16)
        print("Model loaded successfully")
    
    def preprocess_image(self, image_path: str, output_path: str) -> str:
        """
        Preprocess image for better OCR
        """
        img = Image.open(image_path)
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize if needed
        max_size = 2048
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Enhance for handwritten documents
        contrast = ImageEnhance.Contrast(img)
        img = contrast.enhance(1.5)
        
        sharpness = ImageEnhance.Sharpness(img)
        img = sharpness.enhance(1.3)
        
        img.save(output_path, quality=95)
        return output_path
    
    def process_single_image(
        self,
        image_path: str,
        output_dir: str,
        doc_type: str = 'handwritten',
        language: str = None
    ) -> dict:
        """
        Process a single image
        """
        print(f"\nProcessing: {Path(image_path).name}")
        
        # Create output directory for this image
        image_name = Path(image_path).stem
        image_output_dir = os.path.join(output_dir, image_name)
        os.makedirs(image_output_dir, exist_ok=True)
        
        # Preprocess
        preprocessed_path = os.path.join(image_output_dir, 'preprocessed.jpg')
        self.preprocess_image(image_path, preprocessed_path)
        
        # Prepare prompt
        if doc_type == 'handwritten':
            prompt = "<image>\n<|grounding|>Extract all text from this handwritten document with precise positioning."
        elif doc_type == 'historical':
            prompt = "<image>\n<|grounding|>Extract all text from this historical document, preserving original spelling."
        else:
            prompt = "<image>\n<|grounding|>Convert the document to markdown."
        
        if language:
            prompt = prompt.replace("Extract all text", f"Extract all {language} text")
        
        # Run OCR
        try:
            result = self.model.infer(
                self.tokenizer,
                prompt=prompt,
                image_file=preprocessed_path,
                output_path=image_output_dir,
                base_size=1024,
                image_size=640,
                crop_mode=True,
                save_results=True,
                test_compress=True
            )
            
            return {
                'image_path': image_path,
                'status': 'success',
                'result': result,
                'output_dir': image_output_dir
            }
        
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return {
                'image_path': image_path,
                'status': 'error',
                'error': str(e),
                'output_dir': image_output_dir
            }
    
    def process_batch(
        self,
        input_dir: str,
        output_dir: str,
        doc_type: str = 'handwritten',
        language: str = None,
        extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    ) -> list:
        """
        Process all images in a directory
        """
        print(f"\n{'=' * 70}")
        print(f"BATCH PROCESSING")
        print(f"{'=' * 70}")
        print(f"Input Directory: {input_dir}")
        print(f"Output Directory: {output_dir}")
        print(f"Document Type: {doc_type}")
        print(f"Language: {language or 'Auto-detect'}")
        
        # Find all images
        image_files = [
            f for f in Path(input_dir).iterdir()
            if f.suffix.lower() in extensions and f.is_file()
        ]
        
        print(f"\nFound {len(image_files)} images to process")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each image
        results = []
        for i, image_file in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] Processing: {image_file.name}")
            
            result = self.process_single_image(
                str(image_file),
                output_dir,
                doc_type=doc_type,
                language=language
            )
            
            results.append(result)
            
            # Print result summary
            if result['status'] == 'success':
                print(f"✓ Success - Output saved to: {result['output_dir']}")
            else:
                print(f"✗ Error: {result['error']}")
        
        # Save batch summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'input_dir': input_dir,
            'output_dir': output_dir,
            'doc_type': doc_type,
            'language': language,
            'total_images': len(image_files),
            'successful': sum(1 for r in results if r['status'] == 'success'),
            'failed': sum(1 for r in results if r['status'] == 'error'),
            'results': results
        }
        
        summary_path = os.path.join(output_dir, 'batch_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'=' * 70}")
        print(f"BATCH PROCESSING COMPLETE")
        print(f"{'=' * 70}")
        print(f"Total Images: {summary['total_images']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        print(f"Summary saved to: {summary_path}")
        
        return results


def main():
    """
    Example usage
    """
    # Configuration
    input_directory = 'input_images'  # Directory containing your images
    output_directory = 'batch_output'  # Where to save results
    document_type = 'handwritten'  # or 'historical', 'printed', etc.
    language = 'Portuguese'  # Optional: specify language
    
    # Initialize processor
    processor = BatchOCRProcessor()
    
    # Process all images in the directory
    results = processor.process_batch(
        input_dir=input_directory,
        output_dir=output_directory,
        doc_type=document_type,
        language=language
    )
    
    # Print detailed results
    print("\n" + "=" * 70)
    print("DETAILED RESULTS")
    print("=" * 70)
    
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] {Path(result['image_path']).name}")
        print(f"Status: {result['status']}")
        
        if result['status'] == 'success':
            print(f"Output: {result['output_dir']}")
            print("\nExtracted Text (first 200 chars):")
            print(result['result'][:200] + "..." if len(result['result']) > 200 else result['result'])
        else:
            print(f"Error: {result['error']}")


if __name__ == '__main__':
    main()
