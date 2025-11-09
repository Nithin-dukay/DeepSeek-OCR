# Usage Examples for Improved DeepSeek-OCR Inference

This document provides practical examples for using the improved inference script that fixes hallucination issues (GitHub Issue #191).

## Quick Start

### Basic Usage
```bash
python run_dpsk_ocr_improved.py \
    --image_file /path/to/your/image.jpg \
    --output_path ./output
```

## Document Type Specific Examples

### 1. Ancient Handwritten Documents (Portuguese, Latin, etc.)

For challenging handwritten documents, use more aggressive anti-repetition settings:

```bash
python run_dpsk_ocr_improved.py \
    --image_file ancient_portuguese_manuscript.jpg \
    --output_path ./output/ancient_docs \
    --prompt_type free_ocr \
    --ngram_size 20 \
    --window_size 60 \
    --base_size 1024 \
    --image_size 640 \
    --crop_mode
```

**Why these settings?**
- `ngram_size 20`: Catches shorter repetitive patterns common in hallucinations
- `window_size 60`: Shorter memory window for faster processing
- `crop_mode`: Enables dynamic cropping for better quality on large images

### 2. Modern Printed Documents

For clean printed documents, use standard settings:

```bash
python run_dpsk_ocr_improved.py \
    --image_file modern_document.jpg \
    --output_path ./output/printed \
    --prompt_type grounding \
    --ngram_size 30 \
    --window_size 90
```

**Why these settings?**
- `prompt_type grounding`: Preserves document structure (headings, tables, lists)
- `ngram_size 30`: Standard setting for printed text
- `window_size 90`: Balanced memory window

### 3. Documents with Tables

For documents containing tables, use grounding mode:

```bash
python run_dpsk_ocr_improved.py \
    --image_file document_with_tables.jpg \
    --output_path ./output/tables \
    --prompt_type grounding \
    --ngram_size 40 \
    --window_size 100
```

**Why these settings?**
- `prompt_type grounding`: Converts to markdown with table structure
- `ngram_size 40`: Allows natural table patterns to repeat
- Whitelist tokens (`<td>`, `</td>`) are automatically allowed to repeat

### 4. Low Quality or Degraded Documents

For poor quality scans or degraded documents:

```bash
python run_dpsk_ocr_improved.py \
    --image_file degraded_scan.jpg \
    --output_path ./output/degraded \
    --prompt_type free_ocr \
    --ngram_size 15 \
    --window_size 50 \
    --base_size 1280 \
    --image_size 640 \
    --crop_mode
```

**Why these settings?**
- `ngram_size 15`: Very aggressive anti-repetition
- `base_size 1280`: Larger base size for better quality
- `crop_mode`: Dynamic cropping helps with quality

### 5. Small Images or Notes

For small images that don't need cropping:

```bash
python run_dpsk_ocr_improved.py \
    --image_file small_note.jpg \
    --output_path ./output/notes \
    --prompt_type free_ocr \
    --base_size 640 \
    --image_size 640 \
    --no_crop_mode
```

**Why these settings?**
- `no_crop_mode`: Disables cropping for small images
- `base_size 640`: Smaller size for faster processing

## Advanced Examples

### Custom Prompt

```bash
python run_dpsk_ocr_improved.py \
    --image_file scientific_paper.jpg \
    --output_path ./output/custom \
    --prompt_type custom \
    --custom_prompt "<image>\n<|grounding|>Extract all mathematical equations and convert to LaTeX. "
```

### Batch Processing Script

Create a bash script to process multiple images:

```bash
#!/bin/bash
# process_batch.sh

INPUT_DIR="/path/to/images"
OUTPUT_DIR="/path/to/output"

for img in "$INPUT_DIR"/*.jpg; do
    filename=$(basename "$img" .jpg)
    echo "Processing: $filename"
    
    python run_dpsk_ocr_improved.py \
        --image_file "$img" \
        --output_path "$OUTPUT_DIR/$filename" \
        --prompt_type free_ocr \
        --ngram_size 25 \
        --window_size 70
done

echo "Batch processing complete!"
```

Make it executable and run:
```bash
chmod +x process_batch.sh
./process_batch.sh
```

### Python Integration

Use the improved script from your Python code:

```python
import subprocess
import json

def run_ocr(image_path, output_path, document_type='handwritten'):
    """
    Run OCR with appropriate settings based on document type.
    
    Args:
        image_path: Path to input image
        output_path: Directory for output
        document_type: 'handwritten', 'printed', or 'table'
    """
    # Settings based on document type
    settings = {
        'handwritten': {
            'ngram_size': 20,
            'window_size': 60,
            'prompt_type': 'free_ocr'
        },
        'printed': {
            'ngram_size': 30,
            'window_size': 90,
            'prompt_type': 'grounding'
        },
        'table': {
            'ngram_size': 40,
            'window_size': 100,
            'prompt_type': 'grounding'
        }
    }
    
    config = settings.get(document_type, settings['printed'])
    
    cmd = [
        'python', 'run_dpsk_ocr_improved.py',
        '--image_file', image_path,
        '--output_path', output_path,
        '--prompt_type', config['prompt_type'],
        '--ngram_size', str(config['ngram_size']),
        '--window_size', str(config['window_size'])
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"OCR completed successfully for {image_path}")
        # Read the output
        with open(f"{output_path}/result.txt", 'r', encoding='utf-8') as f:
            return f.read()
    else:
        print(f"Error: {result.stderr}")
        return None

# Example usage
text = run_ocr(
    image_path='ancient_manuscript.jpg',
    output_path='./output/manuscript',
    document_type='handwritten'
)
print(text)
```

## Troubleshooting Examples

### Problem: Still seeing repetitions

Try more aggressive settings:
```bash
python run_dpsk_ocr_improved.py \
    --image_file problematic_image.jpg \
    --output_path ./output/aggressive \
    --ngram_size 10 \
    --window_size 40 \
    --repetition_penalty 1.3
```

### Problem: Missing valid repeated text

Try less aggressive settings:
```bash
python run_dpsk_ocr_improved.py \
    --image_file document.jpg \
    --output_path ./output/conservative \
    --ngram_size 50 \
    --window_size 150
```

### Problem: Out of memory

Reduce image size:
```bash
python run_dpsk_ocr_improved.py \
    --image_file large_image.jpg \
    --output_path ./output/reduced \
    --base_size 512 \
    --image_size 512 \
    --no_crop_mode
```

## Performance Comparison

### Original Script (with hallucinations)
```bash
# Original - may produce hallucinations
cd /vercel/sandbox/DeepSeek-OCR-master/DeepSeek-OCR-hf
python run_dpsk_ocr.py
# Edit the file to set your image_file and output_path
```

### Improved Script (hallucination fix)
```bash
# Improved - with anti-hallucination mechanisms
python run_dpsk_ocr_improved.py \
    --image_file your_image.jpg \
    --output_path ./output
```

### vLLM (production recommended)
```bash
# vLLM - best performance and built-in anti-hallucination
cd ../DeepSeek-OCR-vllm
# Edit config.py to set INPUT_PATH and OUTPUT_PATH
python run_dpsk_ocr_image.py
```

## Parameter Reference

| Parameter | Default | Range | Use Case |
|-----------|---------|-------|----------|
| `--ngram_size` | 30 | 10-50 | Lower = more aggressive anti-repetition |
| `--window_size` | 90 | 40-150 | Lower = faster, less memory |
| `--temperature` | 0.0 | 0.0-1.0 | Always use 0.0 for OCR |
| `--base_size` | 1024 | 512-1280 | Higher = better quality, more memory |
| `--image_size` | 640 | 512-1280 | Crop size for dynamic mode |
| `--repetition_penalty` | 1.0 | 1.0-1.5 | Higher = penalize repetitions more |

## Getting Help

View all available options:
```bash
python run_dpsk_ocr_improved.py --help
```

For more details, see [HALLUCINATION_FIX.md](../../HALLUCINATION_FIX.md)
