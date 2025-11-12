"""
Example: Region-Based OCR for Complex Documents

This example demonstrates how to process complex documents by dividing them
into regions, which can help reduce hallucinations on large or complex layouts.
"""

from transformers import AutoModel, AutoTokenizer
from PIL import Image, ImageDraw, ImageEnhance
import torch
import os
from typing import List, Tuple


class RegionBasedOCR:
    """
    Process documents by regions to improve accuracy
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
    
    def preprocess_image(self, img: Image.Image) -> Image.Image:
        """
        Preprocess image for better OCR
        """
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Enhance for handwritten documents
        contrast = ImageEnhance.Contrast(img)
        img = contrast.enhance(1.5)
        
        sharpness = ImageEnhance.Sharpness(img)
        img = sharpness.enhance(1.3)
        
        return img
    
    def create_regions(
        self,
        image_size: Tuple[int, int],
        strategy: str = 'grid'
    ) -> List[Tuple[float, float, float, float]]:
        """
        Create regions for processing
        
        Args:
            image_size: (width, height) of the image
            strategy: 'grid', 'horizontal', 'vertical', or 'custom'
            
        Returns:
            List of regions as (x1, y1, x2, y2) in normalized coordinates [0-1]
        """
        if strategy == 'grid':
            # Divide into 4 quadrants
            return [
                (0.0, 0.0, 0.5, 0.5),  # Top-left
                (0.5, 0.0, 1.0, 0.5),  # Top-right
                (0.0, 0.5, 0.5, 1.0),  # Bottom-left
                (0.5, 0.5, 1.0, 1.0),  # Bottom-right
            ]
        
        elif strategy == 'horizontal':
            # Divide into horizontal strips
            return [
                (0.0, 0.0, 1.0, 0.33),   # Top third
                (0.0, 0.33, 1.0, 0.67),  # Middle third
                (0.0, 0.67, 1.0, 1.0),   # Bottom third
            ]
        
        elif strategy == 'vertical':
            # Divide into vertical strips
            return [
                (0.0, 0.0, 0.33, 1.0),   # Left third
                (0.33, 0.0, 0.67, 1.0),  # Middle third
                (0.67, 0.0, 1.0, 1.0),   # Right third
            ]
        
        elif strategy == 'fine_grid':
            # Divide into 9 regions (3x3 grid)
            regions = []
            for row in range(3):
                for col in range(3):
                    x1 = col / 3.0
                    y1 = row / 3.0
                    x2 = (col + 1) / 3.0
                    y2 = (row + 1) / 3.0
                    regions.append((x1, y1, x2, y2))
            return regions
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def visualize_regions(
        self,
        image_path: str,
        regions: List[Tuple[float, float, float, float]],
        output_path: str
    ):
        """
        Visualize regions on the image
        """
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta', 'lime']
        
        for i, (x1, y1, x2, y2) in enumerate(regions):
            # Convert normalized coordinates to pixels
            box = (
                int(x1 * width),
                int(y1 * height),
                int(x2 * width),
                int(y2 * height)
            )
            
            # Draw rectangle
            color = colors[i % len(colors)]
            draw.rectangle(box, outline=color, width=3)
            
            # Draw region number
            draw.text((box[0] + 10, box[1] + 10), f"Region {i+1}", fill=color)
        
        img.save(output_path)
        print(f"Region visualization saved to: {output_path}")
    
    def process_region(
        self,
        image: Image.Image,
        region: Tuple[float, float, float, float],
        region_id: int,
        output_dir: str,
        doc_type: str = 'handwritten'
    ) -> dict:
        """
        Process a single region
        """
        x1, y1, x2, y2 = region
        width, height = image.size
        
        # Crop region
        box = (
            int(x1 * width),
            int(y1 * height),
            int(x2 * width),
            int(y2 * height)
        )
        region_img = image.crop(box)
        
        # Preprocess
        region_img = self.preprocess_image(region_img)
        
        # Save region image
        region_path = os.path.join(output_dir, f'region_{region_id}.jpg')
        region_img.save(region_path, quality=95)
        
        # Prepare prompt
        prompt = f"<image>\n<|grounding|>Extract all text from this region of a handwritten document."
        
        # Run OCR
        try:
            result = self.model.infer(
                self.tokenizer,
                prompt=prompt,
                image_file=region_path,
                output_path=output_dir,
                base_size=1024,
                image_size=640,
                crop_mode=True,
                save_results=False,
                test_compress=True
            )
            
            return {
                'region_id': region_id,
                'coordinates': region,
                'pixel_box': box,
                'status': 'success',
                'text': result
            }
        
        except Exception as e:
            print(f"Error processing region {region_id}: {e}")
            return {
                'region_id': region_id,
                'coordinates': region,
                'pixel_box': box,
                'status': 'error',
                'error': str(e)
            }
    
    def process_document(
        self,
        image_path: str,
        output_dir: str,
        strategy: str = 'grid',
        doc_type: str = 'handwritten'
    ) -> dict:
        """
        Process entire document using region-based approach
        """
        print(f"\n{'=' * 70}")
        print(f"REGION-BASED OCR")
        print(f"{'=' * 70}")
        print(f"Image: {image_path}")
        print(f"Strategy: {strategy}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load image
        img = Image.open(image_path)
        print(f"Image size: {img.size[0]} x {img.size[1]}")
        
        # Create regions
        regions = self.create_regions(img.size, strategy)
        print(f"Created {len(regions)} regions")
        
        # Visualize regions
        viz_path = os.path.join(output_dir, 'regions_visualization.jpg')
        self.visualize_regions(image_path, regions, viz_path)
        
        # Process each region
        results = []
        for i, region in enumerate(regions, 1):
            print(f"\nProcessing region {i}/{len(regions)}...")
            
            result = self.process_region(
                img,
                region,
                i,
                output_dir,
                doc_type
            )
            
            results.append(result)
            
            if result['status'] == 'success':
                print(f"✓ Region {i} processed successfully")
                print(f"  Text length: {len(result['text'])} characters")
            else:
                print(f"✗ Region {i} failed: {result['error']}")
        
        # Combine results
        combined_text = self.combine_results(results, strategy)
        
        # Save combined result
        combined_path = os.path.join(output_dir, 'combined_result.txt')
        with open(combined_path, 'w', encoding='utf-8') as f:
            f.write(combined_text)
        
        print(f"\n{'=' * 70}")
        print(f"PROCESSING COMPLETE")
        print(f"{'=' * 70}")
        print(f"Regions processed: {len(results)}")
        print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}")
        print(f"Combined result saved to: {combined_path}")
        
        return {
            'image_path': image_path,
            'strategy': strategy,
            'regions': results,
            'combined_text': combined_text,
            'output_dir': output_dir
        }
    
    def combine_results(
        self,
        results: List[dict],
        strategy: str
    ) -> str:
        """
        Combine results from multiple regions
        """
        combined = []
        
        for result in results:
            if result['status'] == 'success':
                combined.append(f"=== Region {result['region_id']} ===")
                combined.append(result['text'])
                combined.append("")  # Empty line between regions
        
        return "\n".join(combined)


def main():
    """
    Example usage
    """
    # Configuration
    image_path = 'complex_document.jpg'  # Replace with your image
    output_dir = 'region_based_output'
    
    # Choose strategy:
    # - 'grid': 2x2 grid (4 regions)
    # - 'horizontal': 3 horizontal strips
    # - 'vertical': 3 vertical strips
    # - 'fine_grid': 3x3 grid (9 regions)
    strategy = 'grid'
    
    # Initialize processor
    processor = RegionBasedOCR()
    
    # Process document
    result = processor.process_document(
        image_path=image_path,
        output_dir=output_dir,
        strategy=strategy,
        doc_type='handwritten'
    )
    
    # Print combined result
    print("\n" + "=" * 70)
    print("COMBINED TEXT FROM ALL REGIONS")
    print("=" * 70)
    print(result['combined_text'])
    
    # Print individual region results
    print("\n" + "=" * 70)
    print("INDIVIDUAL REGION RESULTS")
    print("=" * 70)
    
    for region_result in result['regions']:
        if region_result['status'] == 'success':
            print(f"\nRegion {region_result['region_id']}:")
            print(f"Coordinates: {region_result['coordinates']}")
            print(f"Text (first 100 chars): {region_result['text'][:100]}...")


if __name__ == '__main__':
    main()
