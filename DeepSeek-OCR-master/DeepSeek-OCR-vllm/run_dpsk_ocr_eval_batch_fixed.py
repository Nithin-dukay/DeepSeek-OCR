"""
Fixed batch evaluation script for DeepSeek OCR (Issue #299)

This script processes multiple images with proper error handling for Triton/CUDA errors.
Key improvements:
1. Process images one at a time to avoid batch-related MoE kernel issues
2. Retry logic for failed images
3. Skip problematic images and continue processing
4. Detailed error logging
5. Progress tracking and statistics
"""

import asyncio
import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Optional
import traceback

import torch

# Critical environment variables
if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"

os.environ['VLLM_USE_V1'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.model_executor.models.registry import ModelRegistry
import time
from deepseek_ocr import DeepseekOCRForCausalLM
from PIL import Image, ImageOps
from tqdm import tqdm
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCRProcessor
from config import MODEL_PATH, INPUT_PATH, OUTPUT_PATH, PROMPT, CROP_MODE

ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)

class BatchProcessor:
    def __init__(self, model_path: str, output_path: str):
        self.model_path = model_path
        self.output_path = output_path
        self.engine = None
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'errors': []
        }
        
    async def initialize_engine(self, enforce_eager: bool = True, 
                               gpu_memory_util: float = 0.75,
                               max_model_len: int = 8192):
        """Initialize the vLLM engine with safe parameters."""
        print(f"\nInitializing engine with:")
        print(f"  - enforce_eager: {enforce_eager}")
        print(f"  - gpu_memory_utilization: {gpu_memory_util}")
        print(f"  - max_model_len: {max_model_len}")
        
        engine_args = AsyncEngineArgs(
            model=self.model_path,
            hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
            block_size=256,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_util,
            enable_prefix_caching=False,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        print("Engine initialized successfully\n")
        
    def load_image(self, image_path: str) -> Optional[Image.Image]:
        """Load and preprocess an image."""
        try:
            image = Image.open(image_path)
            corrected_image = ImageOps.exif_transpose(image)
            return corrected_image.convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
            
    async def process_single_image(self, image_path: str, prompt: str, 
                                   retry_count: int = 0, max_retries: int = 2) -> Optional[str]:
        """Process a single image with retry logic."""
        
        # Load image
        image = self.load_image(image_path)
        if image is None:
            return None
            
        # Preprocess image
        try:
            if '<image>' in prompt:
                image_features = DeepseekOCRProcessor().tokenize_with_images(
                    images=[image], 
                    bos=True, 
                    eos=True, 
                    cropping=CROP_MODE
                )
            else:
                image_features = ''
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {e}")
            return None
            
        # Generate output
        try:
            logits_processors = [NoRepeatNGramLogitsProcessor(
                ngram_size=30, 
                window_size=90, 
                whitelist_token_ids={128821, 128822}
            )]

            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=8192,
                logits_processors=logits_processors,
                skip_special_tokens=False,
            )
            
            request_id = f"request-{Path(image_path).stem}-{int(time.time())}"

            if image_features and '<image>' in prompt:
                request = {
                    "prompt": prompt,
                    "multi_modal_data": {"image": image_features}
                }
            else:
                request = {"prompt": prompt}
                
            full_text = ""
            async for request_output in self.engine.generate(
                request, sampling_params, request_id
            ):
                if request_output.outputs:
                    full_text = request_output.outputs[0].text
                    
            return full_text
            
        except RuntimeError as e:
            error_msg = str(e)
            if ("illegal memory access" in error_msg or 
                "CUBLAS_STATUS_EXECUTION_FAILED" in error_msg or
                "Triton Error" in error_msg):
                
                print(f"\nTriton/CUDA error on {Path(image_path).name} (attempt {retry_count + 1})")
                
                if retry_count < max_retries:
                    print(f"Retrying... (attempt {retry_count + 2}/{max_retries + 1})")
                    # Clear CUDA cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    await asyncio.sleep(2)
                    return await self.process_single_image(image_path, prompt, retry_count + 1, max_retries)
                else:
                    print(f"Failed after {max_retries + 1} attempts")
                    return None
            else:
                print(f"Unexpected error: {error_msg}")
                return None
                
    async def process_batch(self, image_paths: List[str], prompt: str):
        """Process a batch of images sequentially."""
        
        if self.engine is None:
            await self.initialize_engine()
            
        self.stats['total'] = len(image_paths)
        results = {}
        
        print(f"\nProcessing {len(image_paths)} images...")
        print("="*60 + "\n")
        
        for idx, image_path in enumerate(tqdm(image_paths, desc="Processing images")):
            image_name = Path(image_path).name
            print(f"\n[{idx+1}/{len(image_paths)}] Processing: {image_name}")
            
            try:
                result = await self.process_single_image(image_path, prompt)
                
                if result:
                    # Save result
                    output_file = Path(self.output_path) / f"{Path(image_path).stem}.txt"
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(result)
                    
                    results[image_name] = {
                        'status': 'success',
                        'output_file': str(output_file)
                    }
                    self.stats['success'] += 1
                    print(f"✓ Success: {image_name}")
                else:
                    results[image_name] = {
                        'status': 'failed',
                        'error': 'Processing returned None'
                    }
                    self.stats['failed'] += 1
                    self.stats['errors'].append({
                        'image': image_name,
                        'error': 'Processing returned None'
                    })
                    print(f"✗ Failed: {image_name}")
                    
            except Exception as e:
                error_msg = str(e)
                results[image_name] = {
                    'status': 'error',
                    'error': error_msg
                }
                self.stats['failed'] += 1
                self.stats['errors'].append({
                    'image': image_name,
                    'error': error_msg,
                    'traceback': traceback.format_exc()
                })
                print(f"✗ Error: {image_name} - {error_msg}")
                
            # Clear cache after each image
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        return results
        
    def save_results_summary(self, results: Dict):
        """Save processing results and statistics."""
        summary_file = Path(self.output_path) / "processing_summary.json"
        
        summary = {
            'statistics': self.stats,
            'results': results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            
        print("\n" + "="*60)
        print("PROCESSING SUMMARY")
        print("="*60)
        print(f"Total images: {self.stats['total']}")
        print(f"Successful: {self.stats['success']} ({self.stats['success']/self.stats['total']*100:.1f}%)")
        print(f"Failed: {self.stats['failed']} ({self.stats['failed']/self.stats['total']*100:.1f}%)")
        print(f"\nSummary saved to: {summary_file}")
        print("="*60 + "\n")
        
        if self.stats['errors']:
            print("\nFailed images:")
            for error in self.stats['errors']:
                print(f"  - {error['image']}: {error['error']}")

async def main():
    print("="*60)
    print("DeepSeek OCR - Fixed Batch Evaluation (Issue #299)")
    print("="*60 + "\n")
    
    # Validate paths
    if not INPUT_PATH or not os.path.exists(INPUT_PATH):
        print(f"ERROR: INPUT_PATH not set or doesn't exist: {INPUT_PATH}")
        print("Please set INPUT_PATH in config.py to a directory containing images")
        sys.exit(1)
        
    if not OUTPUT_PATH:
        print("ERROR: OUTPUT_PATH not set in config.py")
        sys.exit(1)
        
    # Find all images
    input_dir = Path(INPUT_PATH)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_paths = [
        str(p) for p in input_dir.rglob('*') 
        if p.suffix.lower() in image_extensions
    ]
    
    if not image_paths:
        print(f"ERROR: No images found in {INPUT_PATH}")
        sys.exit(1)
        
    print(f"Found {len(image_paths)} images in {INPUT_PATH}")
    
    # Process images
    processor = BatchProcessor(MODEL_PATH, OUTPUT_PATH)
    results = await processor.process_batch(image_paths, PROMPT)
    processor.save_results_summary(results)
    
    # Exit with appropriate code
    if processor.stats['failed'] > 0:
        print("\nWARNING: Some images failed to process")
        print("Check processing_summary.json for details")
        sys.exit(1)
    else:
        print("\nAll images processed successfully!")
        sys.exit(0)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
