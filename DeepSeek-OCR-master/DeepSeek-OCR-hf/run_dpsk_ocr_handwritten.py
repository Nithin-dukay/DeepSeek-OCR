"""
DeepSeek-OCR Optimized Script for Handwritten Documents

This script is specifically optimized for handwritten and historical documents
to minimize hallucinations and maximize accuracy.

Usage:
    python run_dpsk_ocr_handwritten.py --image path/to/image.jpg --output output_dir/

Features:
    - Optimized parameters for handwritten text
    - Image preprocessing for historical documents
    - Multiple OCR passes for validation
    - Support for different document types
    - Detailed logging and error handling
"""

import argparse
import os
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image, ImageEnhance, ImageFilter
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HandwrittenOCRProcessor:
    """
    Optimized OCR processor for handwritten and historical documents
    """
    
    def __init__(
        self,
        model_name: str = 'deepseek-ai/DeepSeek-OCR',
        device: str = 'cuda',
        use_flash_attention: bool = True
    ):
        """
        Initialize the OCR processor
        
        Args:
            model_name: HuggingFace model name
            device: Device to run on ('cuda' or 'cpu')
            use_flash_attention: Whether to use flash attention (requires compatible GPU)
        """
        logger.info(f"Loading model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        model_kwargs = {
            'trust_remote_code': True,
            'use_safetensors': True
        }
        
        if use_flash_attention and device == 'cuda':
            try:
                model_kwargs['_attn_implementation'] = 'flash_attention_2'
                logger.info("Using Flash Attention 2")
            except Exception as e:
                logger.warning(f"Flash Attention not available: {e}")
        
        self.model = AutoModel.from_pretrained(model_name, **model_kwargs)
        self.model = self.model.eval()
        
        if device == 'cuda' and torch.cuda.is_available():
            self.model = self.model.cuda().to(torch.bfloat16)
            logger.info("Model loaded on CUDA with bfloat16")
        else:
            logger.info("Model loaded on CPU")
        
        self.device = device
    
    def preprocess_image(
        self,
        image_path: str,
        enhance_contrast: float = 1.5,
        enhance_sharpness: float = 1.3,
        denoise: bool = False,
        max_size: int = 2048
    ) -> Image.Image:
        """
        Preprocess image for better OCR results on handwritten documents
        
        Args:
            image_path: Path to input image
            enhance_contrast: Contrast enhancement factor (1.0 = no change)
            enhance_sharpness: Sharpness enhancement factor (1.0 = no change)
            denoise: Whether to apply denoising filter
            max_size: Maximum size for longest dimension
            
        Returns:
            Preprocessed PIL Image
        """
        logger.info(f"Preprocessing image: {image_path}")
        
        img = Image.open(image_path)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
            logger.info(f"Converted image from {img.mode} to RGB")
        
        # Resize if too large
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = tuple(int(dim * ratio) for dim in img.size)
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            logger.info(f"Resized image to {new_size}")
        
        # Enhance contrast (helps with faded historical documents)
        if enhance_contrast != 1.0:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(enhance_contrast)
            logger.info(f"Enhanced contrast by factor {enhance_contrast}")
        
        # Enhance sharpness
        if enhance_sharpness != 1.0:
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(enhance_sharpness)
            logger.info(f"Enhanced sharpness by factor {enhance_sharpness}")
        
        # Optional denoising (use carefully with historical documents)
        if denoise:
            img = img.filter(ImageFilter.MedianFilter(size=3))
            logger.info("Applied median filter for denoising")
        
        return img
    
    def get_prompt_for_document_type(
        self,
        doc_type: str = 'handwritten',
        language: Optional[str] = None,
        preserve_layout: bool = True
    ) -> str:
        """
        Get optimized prompt based on document type
        
        Args:
            doc_type: Type of document ('handwritten', 'historical', 'printed', 'mixed')
            language: Specific language (e.g., 'Portuguese', 'Latin')
            preserve_layout: Whether to preserve layout information
            
        Returns:
            Optimized prompt string
        """
        prompts = {
            'handwritten': "<image>\n<|grounding|>Extract all text from this handwritten document with precise positioning.",
            'historical': "<image>\n<|grounding|>Extract all text from this historical document, preserving original spelling and diacritics.",
            'printed': "<image>\n<|grounding|>Convert the document to markdown.",
            'mixed': "<image>\n<|grounding|>Extract all text from this document containing both printed and handwritten content.",
            'table': "<image>\n<|grounding|>Extract text preserving table structure and convert to markdown.",
            'form': "<image>\n<|grounding|>Extract all text from this form, preserving field labels and values."
        }
        
        base_prompt = prompts.get(doc_type, prompts['handwritten'])
        
        # Add language specification if provided
        if language:
            base_prompt = base_prompt.replace(
                "Extract all text",
                f"Extract all {language} text"
            )
        
        # Remove grounding if layout preservation not needed
        if not preserve_layout:
            base_prompt = base_prompt.replace("<|grounding|>", "")
        
        logger.info(f"Using prompt for {doc_type} document: {base_prompt[:50]}...")
        return base_prompt
    
    def run_ocr(
        self,
        image_path: str,
        output_path: str,
        doc_type: str = 'handwritten',
        language: Optional[str] = None,
        preprocess: bool = True,
        base_size: int = 1024,
        image_size: int = 640,
        crop_mode: bool = True,
        multi_pass: bool = False,
        save_results: bool = True
    ) -> Dict:
        """
        Run OCR on the image with optimized parameters
        
        Args:
            image_path: Path to input image
            output_path: Directory to save results
            doc_type: Type of document
            language: Specific language
            preprocess: Whether to preprocess the image
            base_size: Base size for model (512, 640, 1024, 1280)
            image_size: Image size for processing
            crop_mode: Whether to use crop mode (Gundam mode)
            multi_pass: Run multiple passes for validation
            save_results: Whether to save results to file
            
        Returns:
            Dictionary containing OCR results and metadata
        """
        logger.info(f"Starting OCR for: {image_path}")
        
        # Preprocess image if requested
        if preprocess:
            processed_img = self.preprocess_image(image_path)
            # Save preprocessed image temporarily
            temp_path = os.path.join(output_path, 'preprocessed_temp.jpg')
            os.makedirs(output_path, exist_ok=True)
            processed_img.save(temp_path, quality=95)
            image_to_process = temp_path
        else:
            image_to_process = image_path
        
        # Get optimized prompt
        prompt = self.get_prompt_for_document_type(
            doc_type=doc_type,
            language=language,
            preserve_layout=True
        )
        
        # Run primary OCR pass
        logger.info("Running primary OCR pass...")
        result = self.model.infer(
            self.tokenizer,
            prompt=prompt,
            image_file=image_to_process,
            output_path=output_path,
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode,
            save_results=save_results,
            test_compress=True
        )
        
        results = {
            'primary': result,
            'image_path': image_path,
            'doc_type': doc_type,
            'language': language,
            'parameters': {
                'base_size': base_size,
                'image_size': image_size,
                'crop_mode': crop_mode,
                'preprocessed': preprocess
            }
        }
        
        # Run validation passes if requested
        if multi_pass:
            logger.info("Running validation passes...")
            
            # Pass 2: Slightly different prompt
            prompt2 = self.get_prompt_for_document_type(
                doc_type='historical' if doc_type == 'handwritten' else doc_type,
                language=language,
                preserve_layout=True
            )
            result2 = self.model.infer(
                self.tokenizer,
                prompt=prompt2,
                image_file=image_to_process,
                output_path=output_path,
                base_size=base_size,
                image_size=image_size,
                crop_mode=crop_mode,
                save_results=False,
                test_compress=True
            )
            results['validation_pass'] = result2
            
            logger.info("Multi-pass OCR completed")
        
        # Clean up temporary preprocessed image
        if preprocess and os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Save metadata
        if save_results:
            metadata_path = os.path.join(output_path, 'ocr_metadata.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Metadata saved to: {metadata_path}")
        
        logger.info("OCR processing completed successfully")
        return results
    
    def batch_process(
        self,
        image_dir: str,
        output_dir: str,
        doc_type: str = 'handwritten',
        **kwargs
    ) -> List[Dict]:
        """
        Process multiple images in a directory
        
        Args:
            image_dir: Directory containing images
            output_dir: Directory to save results
            doc_type: Type of documents
            **kwargs: Additional arguments for run_ocr
            
        Returns:
            List of result dictionaries
        """
        logger.info(f"Starting batch processing: {image_dir}")
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [
            f for f in Path(image_dir).iterdir()
            if f.suffix.lower() in image_extensions
        ]
        
        logger.info(f"Found {len(image_files)} images to process")
        
        results = []
        for i, image_file in enumerate(image_files, 1):
            logger.info(f"Processing image {i}/{len(image_files)}: {image_file.name}")
            
            # Create output subdirectory for this image
            image_output_dir = os.path.join(output_dir, image_file.stem)
            
            try:
                result = self.run_ocr(
                    image_path=str(image_file),
                    output_path=image_output_dir,
                    doc_type=doc_type,
                    **kwargs
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {image_file.name}: {e}")
                results.append({
                    'image_path': str(image_file),
                    'error': str(e)
                })
        
        # Save batch summary
        summary_path = os.path.join(output_dir, 'batch_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Batch processing completed. Summary saved to: {summary_path}")
        return results


def main():
    parser = argparse.ArgumentParser(
        description='DeepSeek-OCR optimized for handwritten documents',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage for handwritten document
  python run_dpsk_ocr_handwritten.py --image document.jpg --output results/
  
  # Historical Portuguese document with preprocessing
  python run_dpsk_ocr_handwritten.py --image ancient.jpg --output results/ \\
      --doc-type historical --language Portuguese --preprocess
  
  # Multi-pass validation for critical documents
  python run_dpsk_ocr_handwritten.py --image important.jpg --output results/ \\
      --multi-pass --preprocess
  
  # Batch processing
  python run_dpsk_ocr_handwritten.py --batch-dir images/ --output results/ \\
      --doc-type handwritten
  
  # High-resolution document (Large model size)
  python run_dpsk_ocr_handwritten.py --image large_doc.jpg --output results/ \\
      --base-size 1280 --image-size 1280 --no-crop
        """
    )
    
    # Input/Output arguments
    parser.add_argument(
        '--image',
        type=str,
        help='Path to input image file'
    )
    parser.add_argument(
        '--batch-dir',
        type=str,
        help='Directory containing multiple images for batch processing'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for results'
    )
    
    # Document type arguments
    parser.add_argument(
        '--doc-type',
        type=str,
        default='handwritten',
        choices=['handwritten', 'historical', 'printed', 'mixed', 'table', 'form'],
        help='Type of document (default: handwritten)'
    )
    parser.add_argument(
        '--language',
        type=str,
        help='Specific language (e.g., Portuguese, Latin, English)'
    )
    
    # Processing arguments
    parser.add_argument(
        '--preprocess',
        action='store_true',
        help='Enable image preprocessing (recommended for historical documents)'
    )
    parser.add_argument(
        '--no-preprocess',
        action='store_true',
        help='Disable image preprocessing'
    )
    parser.add_argument(
        '--multi-pass',
        action='store_true',
        help='Run multiple OCR passes for validation'
    )
    
    # Model size arguments
    parser.add_argument(
        '--base-size',
        type=int,
        default=1024,
        choices=[512, 640, 1024, 1280],
        help='Base size for model (default: 1024)'
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=640,
        help='Image size for processing (default: 640)'
    )
    parser.add_argument(
        '--no-crop',
        action='store_true',
        help='Disable crop mode (use full image)'
    )
    
    # Model arguments
    parser.add_argument(
        '--model-name',
        type=str,
        default='deepseek-ai/DeepSeek-OCR',
        help='HuggingFace model name'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run on (default: cuda)'
    )
    parser.add_argument(
        '--no-flash-attention',
        action='store_true',
        help='Disable Flash Attention 2'
    )
    
    args = parser.parse_args()
    
    # Validate input arguments
    if not args.image and not args.batch_dir:
        parser.error("Either --image or --batch-dir must be specified")
    
    if args.image and args.batch_dir:
        parser.error("Cannot specify both --image and --batch-dir")
    
    # Determine preprocessing
    preprocess = True
    if args.no_preprocess:
        preprocess = False
    elif args.preprocess:
        preprocess = True
    
    # Initialize processor
    try:
        processor = HandwrittenOCRProcessor(
            model_name=args.model_name,
            device=args.device,
            use_flash_attention=not args.no_flash_attention
        )
    except Exception as e:
        logger.error(f"Failed to initialize OCR processor: {e}")
        return 1
    
    # Run OCR
    try:
        if args.image:
            # Single image processing
            results = processor.run_ocr(
                image_path=args.image,
                output_path=args.output,
                doc_type=args.doc_type,
                language=args.language,
                preprocess=preprocess,
                base_size=args.base_size,
                image_size=args.image_size,
                crop_mode=not args.no_crop,
                multi_pass=args.multi_pass,
                save_results=True
            )
            
            logger.info("=" * 60)
            logger.info("OCR RESULT:")
            logger.info("=" * 60)
            print(results['primary'])
            
            if args.multi_pass:
                logger.info("\n" + "=" * 60)
                logger.info("VALIDATION PASS:")
                logger.info("=" * 60)
                print(results['validation_pass'])
        
        else:
            # Batch processing
            results = processor.batch_process(
                image_dir=args.batch_dir,
                output_dir=args.output,
                doc_type=args.doc_type,
                language=args.language,
                preprocess=preprocess,
                base_size=args.base_size,
                image_size=args.image_size,
                crop_mode=not args.no_crop,
                multi_pass=args.multi_pass,
                save_results=True
            )
            
            logger.info(f"Processed {len(results)} images")
            successful = sum(1 for r in results if 'error' not in r)
            logger.info(f"Successful: {successful}/{len(results)}")
        
        logger.info(f"\nResults saved to: {args.output}")
        return 0
    
    except Exception as e:
        logger.error(f"OCR processing failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
