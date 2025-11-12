"""
OCR Configuration Helper

Interactive tool to help users select optimal parameters for their OCR tasks.
Analyzes document characteristics and recommends best settings.

Usage:
    python ocr_config_helper.py --image path/to/image.jpg
    python ocr_config_helper.py --interactive
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
from PIL import Image
import numpy as np


class OCRConfigHelper:
    """
    Helper class to recommend optimal OCR configurations
    """
    
    def __init__(self):
        self.config_presets = {
            'handwritten_low_quality': {
                'doc_type': 'handwritten',
                'preprocess': True,
                'enhance_contrast': 1.8,
                'enhance_sharpness': 1.5,
                'denoise': True,
                'base_size': 1024,
                'image_size': 640,
                'crop_mode': True,
                'multi_pass': True,
                'description': 'For low-quality handwritten documents with faded text'
            },
            'handwritten_high_quality': {
                'doc_type': 'handwritten',
                'preprocess': True,
                'enhance_contrast': 1.3,
                'enhance_sharpness': 1.2,
                'denoise': False,
                'base_size': 1024,
                'image_size': 640,
                'crop_mode': True,
                'multi_pass': False,
                'description': 'For clear, high-quality handwritten documents'
            },
            'historical_document': {
                'doc_type': 'historical',
                'preprocess': True,
                'enhance_contrast': 2.0,
                'enhance_sharpness': 1.5,
                'denoise': True,
                'base_size': 1280,
                'image_size': 1280,
                'crop_mode': False,
                'multi_pass': True,
                'description': 'For ancient/historical documents with degraded quality'
            },
            'printed_document': {
                'doc_type': 'printed',
                'preprocess': False,
                'enhance_contrast': 1.0,
                'enhance_sharpness': 1.0,
                'denoise': False,
                'base_size': 1024,
                'image_size': 1024,
                'crop_mode': False,
                'multi_pass': False,
                'description': 'For modern printed documents with clear text'
            },
            'mixed_content': {
                'doc_type': 'mixed',
                'preprocess': True,
                'enhance_contrast': 1.4,
                'enhance_sharpness': 1.3,
                'denoise': False,
                'base_size': 1024,
                'image_size': 640,
                'crop_mode': True,
                'multi_pass': True,
                'description': 'For documents with both printed and handwritten text'
            },
            'table_form': {
                'doc_type': 'table',
                'preprocess': False,
                'enhance_contrast': 1.2,
                'enhance_sharpness': 1.1,
                'denoise': False,
                'base_size': 1280,
                'image_size': 1280,
                'crop_mode': False,
                'multi_pass': False,
                'description': 'For tables and forms with structured layout'
            },
            'large_document': {
                'doc_type': 'handwritten',
                'preprocess': True,
                'enhance_contrast': 1.5,
                'enhance_sharpness': 1.3,
                'denoise': False,
                'base_size': 1280,
                'image_size': 1280,
                'crop_mode': False,
                'multi_pass': False,
                'description': 'For large, high-resolution documents'
            }
        }
    
    def analyze_image(self, image_path: str) -> Dict:
        """
        Analyze image characteristics to recommend optimal settings
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with image analysis results
        """
        img = Image.open(image_path)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Get basic properties
        width, height = img.size
        aspect_ratio = width / height
        total_pixels = width * height
        
        # Convert to numpy for analysis
        img_array = np.array(img)
        
        # Calculate brightness and contrast metrics
        grayscale = np.mean(img_array, axis=2)
        mean_brightness = np.mean(grayscale)
        std_brightness = np.std(grayscale)
        
        # Estimate contrast (normalized standard deviation)
        contrast_score = std_brightness / 128.0  # Normalize to 0-2 range
        
        # Estimate sharpness using Laplacian variance
        from scipy import ndimage
        laplacian = ndimage.laplace(grayscale)
        sharpness_score = np.var(laplacian) / 1000.0  # Normalize
        
        # Detect if image is likely handwritten vs printed
        # Handwritten typically has more variation and less uniformity
        edge_density = np.sum(np.abs(laplacian) > 10) / total_pixels
        
        # Estimate quality
        quality_score = (contrast_score + min(sharpness_score, 1.0)) / 2
        
        analysis = {
            'dimensions': {
                'width': width,
                'height': height,
                'aspect_ratio': round(aspect_ratio, 2),
                'total_pixels': total_pixels,
                'megapixels': round(total_pixels / 1_000_000, 2)
            },
            'quality_metrics': {
                'mean_brightness': round(mean_brightness, 2),
                'contrast_score': round(contrast_score, 2),
                'sharpness_score': round(min(sharpness_score, 1.0), 2),
                'edge_density': round(edge_density, 4),
                'overall_quality': round(quality_score, 2)
            },
            'characteristics': {
                'is_high_resolution': total_pixels > 2_000_000,
                'is_low_contrast': contrast_score < 0.5,
                'is_blurry': sharpness_score < 0.3,
                'likely_handwritten': edge_density > 0.15
            }
        }
        
        return analysis
    
    def recommend_config(self, image_path: str) -> Dict:
        """
        Recommend optimal configuration based on image analysis
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Recommended configuration dictionary
        """
        analysis = self.analyze_image(image_path)
        
        # Start with base recommendations
        recommendations = {
            'image_analysis': analysis,
            'recommended_preset': None,
            'custom_config': {},
            'reasoning': []
        }
        
        chars = analysis['characteristics']
        quality = analysis['quality_metrics']
        dims = analysis['dimensions']
        
        # Determine document type and preset
        if chars['likely_handwritten']:
            if quality['overall_quality'] < 0.4:
                preset = 'handwritten_low_quality'
                recommendations['reasoning'].append(
                    "Low quality handwritten document detected - using aggressive preprocessing"
                )
            elif quality['is_low_contrast'] or quality['is_blurry']:
                preset = 'historical_document'
                recommendations['reasoning'].append(
                    "Historical/degraded handwritten document - using maximum enhancement"
                )
            else:
                preset = 'handwritten_high_quality'
                recommendations['reasoning'].append(
                    "High quality handwritten document - using moderate preprocessing"
                )
        else:
            if chars['is_high_resolution']:
                preset = 'large_document'
                recommendations['reasoning'].append(
                    "Large high-resolution document - using full resolution processing"
                )
            else:
                preset = 'printed_document'
                recommendations['reasoning'].append(
                    "Printed document detected - minimal preprocessing needed"
                )
        
        recommendations['recommended_preset'] = preset
        recommendations['custom_config'] = self.config_presets[preset].copy()
        
        # Fine-tune based on specific characteristics
        if quality['is_low_contrast']:
            recommendations['custom_config']['enhance_contrast'] = min(
                recommendations['custom_config']['enhance_contrast'] + 0.3,
                2.5
            )
            recommendations['reasoning'].append(
                "Increased contrast enhancement due to low contrast"
            )
        
        if quality['is_blurry']:
            recommendations['custom_config']['enhance_sharpness'] = min(
                recommendations['custom_config']['enhance_sharpness'] + 0.2,
                2.0
            )
            recommendations['reasoning'].append(
                "Increased sharpness enhancement due to blur"
            )
        
        if dims['megapixels'] > 5:
            recommendations['custom_config']['base_size'] = 1280
            recommendations['custom_config']['image_size'] = 1280
            recommendations['custom_config']['crop_mode'] = False
            recommendations['reasoning'].append(
                "Using large model size for high-resolution image"
            )
        
        if quality['overall_quality'] < 0.3:
            recommendations['custom_config']['multi_pass'] = True
            recommendations['reasoning'].append(
                "Enabling multi-pass validation due to low quality"
            )
        
        return recommendations
    
    def generate_command(self, image_path: str, config: Dict, output_dir: str = 'output') -> str:
        """
        Generate command line for run_dpsk_ocr_handwritten.py
        
        Args:
            image_path: Path to the image
            config: Configuration dictionary
            output_dir: Output directory
            
        Returns:
            Command line string
        """
        cmd_parts = [
            'python run_dpsk_ocr_handwritten.py',
            f'--image "{image_path}"',
            f'--output "{output_dir}"',
            f'--doc-type {config["doc_type"]}'
        ]
        
        if config.get('preprocess', False):
            cmd_parts.append('--preprocess')
        
        if config.get('multi_pass', False):
            cmd_parts.append('--multi-pass')
        
        if config.get('base_size'):
            cmd_parts.append(f'--base-size {config["base_size"]}')
        
        if config.get('image_size'):
            cmd_parts.append(f'--image-size {config["image_size"]}')
        
        if not config.get('crop_mode', True):
            cmd_parts.append('--no-crop')
        
        return ' \\\n    '.join(cmd_parts)
    
    def interactive_mode(self):
        """
        Interactive mode to help users configure OCR
        """
        print("=" * 70)
        print("DeepSeek-OCR Configuration Helper")
        print("=" * 70)
        print()
        
        # Get image path
        while True:
            image_path = input("Enter path to your image file: ").strip().strip('"\'')
            if Path(image_path).exists():
                break
            print(f"Error: File not found: {image_path}")
        
        print("\nAnalyzing image...")
        recommendations = self.recommend_config(image_path)
        
        # Display analysis
        print("\n" + "=" * 70)
        print("IMAGE ANALYSIS")
        print("=" * 70)
        
        analysis = recommendations['image_analysis']
        dims = analysis['dimensions']
        quality = analysis['quality_metrics']
        chars = analysis['characteristics']
        
        print(f"\nDimensions: {dims['width']} x {dims['height']} ({dims['megapixels']} MP)")
        print(f"Aspect Ratio: {dims['aspect_ratio']}")
        print(f"\nQuality Metrics:")
        print(f"  - Overall Quality: {quality['overall_quality']:.2f} / 1.0")
        print(f"  - Contrast Score: {quality['contrast_score']:.2f}")
        print(f"  - Sharpness Score: {quality['sharpness_score']:.2f}")
        print(f"  - Mean Brightness: {quality['mean_brightness']:.1f} / 255")
        
        print(f"\nCharacteristics:")
        print(f"  - High Resolution: {'Yes' if chars['is_high_resolution'] else 'No'}")
        print(f"  - Low Contrast: {'Yes' if chars['is_low_contrast'] else 'No'}")
        print(f"  - Blurry: {'Yes' if chars['is_blurry'] else 'No'}")
        print(f"  - Likely Handwritten: {'Yes' if chars['likely_handwritten'] else 'No'}")
        
        # Display recommendations
        print("\n" + "=" * 70)
        print("RECOMMENDATIONS")
        print("=" * 70)
        
        preset = recommendations['recommended_preset']
        config = recommendations['custom_config']
        
        print(f"\nRecommended Preset: {preset}")
        print(f"Description: {self.config_presets[preset]['description']}")
        
        print("\nReasoning:")
        for reason in recommendations['reasoning']:
            print(f"  • {reason}")
        
        print("\nRecommended Configuration:")
        print(f"  - Document Type: {config['doc_type']}")
        print(f"  - Preprocessing: {'Enabled' if config['preprocess'] else 'Disabled'}")
        if config['preprocess']:
            print(f"    • Contrast Enhancement: {config['enhance_contrast']:.1f}x")
            print(f"    • Sharpness Enhancement: {config['enhance_sharpness']:.1f}x")
            print(f"    • Denoising: {'Enabled' if config['denoise'] else 'Disabled'}")
        print(f"  - Model Size: Base={config['base_size']}, Image={config['image_size']}")
        print(f"  - Crop Mode: {'Enabled' if config['crop_mode'] else 'Disabled'}")
        print(f"  - Multi-Pass Validation: {'Enabled' if config['multi_pass'] else 'Disabled'}")
        
        # Ask about language
        print("\n" + "=" * 70)
        language = input("Enter document language (or press Enter to skip): ").strip()
        
        # Ask about output directory
        output_dir = input("Enter output directory (default: output): ").strip() or 'output'
        
        # Generate command
        print("\n" + "=" * 70)
        print("RECOMMENDED COMMAND")
        print("=" * 70)
        print()
        
        command = self.generate_command(image_path, config, output_dir)
        if language:
            command += f' \\\n    --language "{language}"'
        
        print(command)
        
        # Ask if user wants to save config
        print("\n" + "=" * 70)
        save = input("Save configuration to file? (y/n): ").strip().lower()
        
        if save == 'y':
            config_file = input("Enter config filename (default: ocr_config.json): ").strip()
            config_file = config_file or 'ocr_config.json'
            
            save_data = {
                'image_path': image_path,
                'analysis': analysis,
                'recommendations': recommendations,
                'command': command,
                'language': language,
                'output_dir': output_dir
            }
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            print(f"\nConfiguration saved to: {config_file}")
        
        print("\n" + "=" * 70)
        print("You can now run the recommended command to process your document.")
        print("=" * 70)
    
    def list_presets(self):
        """
        List all available configuration presets
        """
        print("=" * 70)
        print("AVAILABLE CONFIGURATION PRESETS")
        print("=" * 70)
        print()
        
        for name, preset in self.config_presets.items():
            print(f"{name}:")
            print(f"  Description: {preset['description']}")
            print(f"  Document Type: {preset['doc_type']}")
            print(f"  Preprocessing: {'Yes' if preset['preprocess'] else 'No'}")
            print(f"  Model Size: {preset['base_size']} / {preset['image_size']}")
            print(f"  Multi-Pass: {'Yes' if preset['multi_pass'] else 'No'}")
            print()


def main():
    parser = argparse.ArgumentParser(
        description='OCR Configuration Helper - Recommend optimal settings for your documents',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (recommended for first-time users)
  python ocr_config_helper.py --interactive
  
  # Analyze specific image and get recommendations
  python ocr_config_helper.py --image document.jpg
  
  # Analyze and save configuration
  python ocr_config_helper.py --image document.jpg --save-config config.json
  
  # List all available presets
  python ocr_config_helper.py --list-presets
        """
    )
    
    parser.add_argument(
        '--image',
        type=str,
        help='Path to image file to analyze'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    parser.add_argument(
        '--list-presets',
        action='store_true',
        help='List all available configuration presets'
    )
    parser.add_argument(
        '--save-config',
        type=str,
        help='Save configuration to JSON file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Output directory for generated command (default: output)'
    )
    parser.add_argument(
        '--language',
        type=str,
        help='Document language'
    )
    
    args = parser.parse_args()
    
    helper = OCRConfigHelper()
    
    # List presets mode
    if args.list_presets:
        helper.list_presets()
        return 0
    
    # Interactive mode
    if args.interactive:
        helper.interactive_mode()
        return 0
    
    # Image analysis mode
    if args.image:
        if not Path(args.image).exists():
            print(f"Error: Image file not found: {args.image}")
            return 1
        
        print("Analyzing image...")
        recommendations = helper.recommend_config(args.image)
        
        # Display results
        print("\n" + "=" * 70)
        print("ANALYSIS RESULTS")
        print("=" * 70)
        
        analysis = recommendations['image_analysis']
        print(f"\nImage: {args.image}")
        print(f"Dimensions: {analysis['dimensions']['width']} x {analysis['dimensions']['height']}")
        print(f"Quality Score: {analysis['quality_metrics']['overall_quality']:.2f} / 1.0")
        
        print("\n" + "=" * 70)
        print("RECOMMENDATIONS")
        print("=" * 70)
        
        preset = recommendations['recommended_preset']
        print(f"\nRecommended Preset: {preset}")
        print(f"Description: {helper.config_presets[preset]['description']}")
        
        print("\nReasoning:")
        for reason in recommendations['reasoning']:
            print(f"  • {reason}")
        
        config = recommendations['custom_config']
        command = helper.generate_command(args.image, config, args.output_dir)
        if args.language:
            command += f' \\\n    --language "{args.language}"'
        
        print("\n" + "=" * 70)
        print("RECOMMENDED COMMAND")
        print("=" * 70)
        print()
        print(command)
        print()
        
        # Save config if requested
        if args.save_config:
            save_data = {
                'image_path': args.image,
                'analysis': analysis,
                'recommendations': recommendations,
                'command': command,
                'language': args.language,
                'output_dir': args.output_dir
            }
            
            with open(args.save_config, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            print(f"Configuration saved to: {args.save_config}")
        
        return 0
    
    # No arguments provided
    parser.print_help()
    return 0


if __name__ == '__main__':
    exit(main())
