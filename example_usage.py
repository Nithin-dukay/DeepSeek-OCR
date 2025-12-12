"""
Example Usage Script for DeepSeek-OCR Enhanced Features
Demonstrates solutions to GitHub Issue #151
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))


def example_1_basic_inference():
    """Example 1: Basic inference with text return."""
    print("\n" + "="*60)
    print("Example 1: Basic Inference with Text Return")
    print("="*60)
    
    try:
        from enhanced_hf_inference import EnhancedDeepSeekOCR
        import torch
        
        # Check if CUDA is available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        if device == 'cpu':
            print("WARNING: Running on CPU. This will be slow.")
            print("For production use, GPU is strongly recommended.")
        
        # Initialize model
        print("\nInitializing model...")
        model = EnhancedDeepSeekOCR(
            model_name='deepseek-ai/DeepSeek-OCR',
            device=device
        )
        
        print("✓ Model initialized successfully!")
        print("\nNOTE: To use this example with an actual image:")
        print("  text = model.infer_enhanced(")
        print("      image='your_image.jpg',")
        print("      prompt='<image>\\n<|grounding|>Convert the document to markdown.',")
        print("      return_text=True")
        print("  )")
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Make sure all dependencies are installed:")
        print("  pip install torch transformers pillow")
    except Exception as e:
        print(f"✗ Error: {e}")


def example_2_chat_template():
    """Example 2: Chat template configuration."""
    print("\n" + "="*60)
    print("Example 2: Chat Template Configuration")
    print("="*60)
    
    try:
        from chat_template_utils import (
            load_tokenizer_with_chat_template,
            format_ocr_prompt,
            apply_chat_template_for_ocr
        )
        
        print("\nLoading tokenizer with chat template...")
        
        # This will work even without the model downloaded
        # (tokenizer is smaller and faster to download)
        try:
            tokenizer = load_tokenizer_with_chat_template(
                model_name='deepseek-ai/DeepSeek-OCR',
                template_type='default'
            )
            print("✓ Tokenizer loaded with chat template!")
            print(f"  Chat template configured: {tokenizer.chat_template is not None}")
            print(f"  Pad token: {tokenizer.pad_token}")
            
            # Format a prompt
            messages = format_ocr_prompt(
                instruction="Convert the document to markdown.",
                use_grounding=True
            )
            print(f"\n✓ Formatted prompt messages:")
            for msg in messages:
                print(f"  Role: {msg['role']}")
                print(f"  Content: {msg['content'][:100]}...")
            
        except Exception as e:
            print(f"✗ Could not load tokenizer: {e}")
            print("This is expected if you don't have internet or the model isn't accessible.")
            print("\nShowing example format instead:")
            
            messages = format_ocr_prompt(
                instruction="Convert the document to markdown.",
                use_grounding=True
            )
            print(f"✓ Example formatted prompt:")
            for msg in messages:
                print(f"  Role: {msg['role']}")
                print(f"  Content: {msg['content']}")
        
    except ImportError as e:
        print(f"✗ Import error: {e}")


def example_3_repetition_detection():
    """Example 3: Repetition detection."""
    print("\n" + "="*60)
    print("Example 3: Repetition Detection")
    print("="*60)
    
    try:
        from repetition_detector import (
            analyze_ocr_output,
            detect_repetition_patterns
        )
        
        # Test with good text
        print("\nTest 1: Good quality text")
        good_text = """
        The British Library contains historical newspapers from the 1800s and 1900s.
        These documents provide valuable insights into the social and political climate
        of the era. Researchers can access digitized versions of these newspapers online.
        The collection includes articles, advertisements, and public notices.
        """
        
        result = analyze_ocr_output(good_text, expected_length=200)
        print(f"  Status: {result['status']}")
        print(f"  Quality score: {result['quality_score']:.3f}")
        print(f"  Repetition score: {result['repetition_score']:.3f}")
        print(f"  ✓ {result['recommendations'][0]}")
        
        # Test with repetitive text (stuck loop)
        print("\nTest 2: Stuck loop (catastrophic failure)")
        bad_text = "and the " * 50 + "newspaper article continues..."
        
        result = analyze_ocr_output(bad_text, expected_length=100)
        print(f"  Status: {result['status']}")
        print(f"  Quality score: {result['quality_score']:.3f}")
        print(f"  Repetition score: {result['repetition_score']:.3f}")
        print(f"  Patterns detected: {len(result['patterns'])}")
        
        if result['patterns']:
            pattern = result['patterns'][0]
            print(f"  Pattern type: {pattern['type']}")
            if 'consecutive_count' in pattern:
                print(f"  Consecutive repetitions: {pattern['consecutive_count']}")
        
        print(f"  ✗ Recommendation: {result['recommendations'][0][:80]}...")
        
        # Test with length anomaly
        print("\nTest 3: Length anomaly (3x expected)")
        long_text = "This is a newspaper article. " * 100
        
        result = analyze_ocr_output(long_text, expected_length=100)
        print(f"  Status: {result['status']}")
        print(f"  Quality score: {result['quality_score']:.3f}")
        print(f"  Length ratio: {result['length_ratio']:.2f}x")
        print(f"  Length anomaly: {result['length_anomaly']:.3f}")
        print(f"  ✗ Recommendation: {result['recommendations'][0][:80]}...")
        
        print("\n✓ Repetition detection working correctly!")
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
    except Exception as e:
        print(f"✗ Error: {e}")


def example_4_text_extraction():
    """Example 4: Text extraction utilities."""
    print("\n" + "="*60)
    print("Example 4: Text Extraction Utilities")
    print("="*60)
    
    try:
        from text_extraction_utils import (
            extract_text_from_output,
            get_text_only_output,
            parse_structured_output,
            extract_images_from_output,
            OutputFormat
        )
        
        # Sample output with structured tags
        raw_output = """
<|ref|>title<|/ref|><|det|>[[100, 100, 500, 150]]<|/det|>

# Historical Newspaper Article

This is the main text of the article from the 1800s.
The article discusses important events of the era.

<|ref|>image<|/ref|><|det|>[[50, 200, 600, 400]]<|/det|>

The article continues with more historical information.
Multiple paragraphs provide detailed context.
        """
        
        print("\nTest 1: Extract clean markdown")
        clean_md = extract_text_from_output(
            raw_output,
            format=OutputFormat.MARKDOWN,
            remove_tags=True
        )
        print(f"  Length: {len(clean_md)} chars")
        print(f"  Preview: {clean_md[:100]}...")
        print("  ✓ Markdown extracted")
        
        print("\nTest 2: Extract plain text")
        plain_text = extract_text_from_output(
            raw_output,
            format=OutputFormat.PLAIN_TEXT,
            remove_tags=True
        )
        print(f"  Length: {len(plain_text)} chars")
        print(f"  Preview: {plain_text[:100]}...")
        print("  ✓ Plain text extracted")
        
        print("\nTest 3: Parse structured output")
        parsed = parse_structured_output(raw_output)
        print(f"  Images found: {len(parsed['images'])}")
        print(f"  Detections found: {len(parsed['detections'])}")
        print(f"  Text length: {len(parsed['text'])} chars")
        print("  ✓ Structured parsing successful")
        
        print("\nTest 4: Extract images with placeholders")
        text_with_placeholders, images = extract_images_from_output(raw_output)
        print(f"  Images extracted: {len(images)}")
        if images:
            print(f"  First image detection: {images[0]['detection'][:50]}...")
        print("  ✓ Image extraction successful")
        
        print("\nTest 5: Get text-only output")
        text_only = get_text_only_output(raw_output, preserve_structure=True)
        print(f"  Length: {len(text_only)} chars")
        print(f"  Preview: {text_only[:100]}...")
        print("  ✓ Text-only extraction successful")
        
        print("\n✓ All text extraction utilities working correctly!")
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
    except Exception as e:
        print(f"✗ Error: {e}")


def example_5_complete_workflow():
    """Example 5: Complete workflow demonstration."""
    print("\n" + "="*60)
    print("Example 5: Complete Workflow (Conceptual)")
    print("="*60)
    
    print("""
This example shows the complete workflow for processing a historical document:

1. Load image
   >>> from PIL import Image
   >>> image = Image.open('historical_newspaper.jpg')

2. Initialize model with enhanced features
   >>> from enhanced_hf_inference import EnhancedDeepSeekOCR
   >>> model = EnhancedDeepSeekOCR()

3. Process with automatic retry and repetition detection
   >>> result = model.infer_with_retry(
   ...     image=image,
   ...     prompt="<image>\\n<|grounding|>Convert the document to markdown.",
   ...     max_retries=2,
   ...     detect_repetition=True,
   ...     repetition_threshold=0.3
   ... )

4. Check result quality
   >>> if result['success']:
   ...     print(f"Success! Attempts: {result['attempts']}")
   ...     raw_text = result['text']
   ... else:
   ...     print(f"Failed. Repetition score: {result['repetition_score']:.3f}")

5. Extract clean text
   >>> from text_extraction_utils import get_text_only_output
   >>> clean_text = get_text_only_output(raw_text, preserve_structure=True)

6. Analyze quality (optional)
   >>> from repetition_detector import analyze_ocr_output
   >>> analysis = analyze_ocr_output(clean_text, expected_length=1000)
   >>> print(f"Quality: {analysis['status']}")

7. Save results
   >>> with open('output.txt', 'w', encoding='utf-8') as f:
   ...     f.write(clean_text)

✓ This workflow addresses all issues from GitHub Issue #151:
  - Returns text directly (no stdout capture)
  - Automatic repetition detection and retry
  - Clean text extraction
  - Quality analysis and recommendations
    """)


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("DeepSeek-OCR Enhanced Features - Example Usage")
    print("Solutions for GitHub Issue #151")
    print("="*60)
    
    examples = [
        ("Basic Inference", example_1_basic_inference),
        ("Chat Template", example_2_chat_template),
        ("Repetition Detection", example_3_repetition_detection),
        ("Text Extraction", example_4_text_extraction),
        ("Complete Workflow", example_5_complete_workflow),
    ]
    
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nRunning all examples...\n")
    
    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n✗ Error in {name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60)
    print("\nFor more information, see:")
    print("  - BEST_PRACTICES.md: Comprehensive guide")
    print("  - enhanced_hf_inference.py: Enhanced inference API")
    print("  - chat_template_utils.py: Chat template configuration")
    print("  - repetition_detector.py: Repetition detection")
    print("  - text_extraction_utils.py: Text extraction utilities")
    print("\n")


if __name__ == "__main__":
    main()
