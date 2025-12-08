# Dynamic Mode Selection for DeepSeek-OCR vLLM Deployment

## Overview

This document explains how to dynamically select OCR modes (Tiny, Small, Base, Large, Gundam) when using DeepSeek-OCR with vLLM deployment, addressing GitHub Issue #164.

## Problem

Previously, when deploying DeepSeek-OCR with vLLM, the mode parameters (`BASE_SIZE`, `IMAGE_SIZE`, `CROP_MODE`) were hardcoded in `config.py`. Users had to modify the config file and restart the service to change modes.

## Solution

The implementation now supports **dynamic mode selection** by passing optional parameters to the `tokenize_with_images()` method. You can override the config.py defaults on a per-request basis.

## Supported Modes

| Mode   | base_size | image_size | crop_mode | Vision Tokens | Use Case |
|--------|-----------|------------|-----------|---------------|----------|
| Tiny   | 512       | 512        | False     | 64            | Fast processing, simple documents |
| Small  | 640       | 640        | False     | 100           | Balanced speed/quality |
| Base   | 1024      | 1024       | False     | 256           | Standard quality |
| Large  | 1280      | 1280       | False     | 400           | High quality, detailed documents |
| Gundam | 1024      | 640        | True      | Dynamic       | Complex layouts, large documents |

## Usage Examples

### Example 1: Using Config Defaults

```python
from process.image_process import DeepseekOCRProcessor
from PIL import Image

image = Image.open("document.jpg").convert('RGB')

# Uses BASE_SIZE, IMAGE_SIZE, and CROP_MODE from config.py
image_features = DeepseekOCRProcessor().tokenize_with_images(
    images=[image],
    bos=True,
    eos=True,
    cropping=True  # Uses config.CROP_MODE if omitted
)
```

### Example 2: Small Mode (640×640, No Cropping)

```python
# Override to Small mode for faster processing
image_features = DeepseekOCRProcessor().tokenize_with_images(
    images=[image],
    bos=True,
    eos=True,
    cropping=False,
    base_size=640,
    image_size=640
)
```

### Example 3: Gundam Mode (Dynamic Resolution)

```python
# Use Gundam mode for complex documents with dynamic cropping
image_features = DeepseekOCRProcessor().tokenize_with_images(
    images=[image],
    bos=True,
    eos=True,
    cropping=True,
    base_size=1024,
    image_size=640
)
```

### Example 4: Large Mode (1280×1280, High Quality)

```python
# Use Large mode for detailed documents
image_features = DeepseekOCRProcessor().tokenize_with_images(
    images=[image],
    bos=True,
    eos=True,
    cropping=False,
    base_size=1280,
    image_size=1280
)
```

## Integration with vLLM Scripts

### Single Image Processing (run_dpsk_ocr_image.py)

```python
if '<image>' in PROMPT:
    # Option 1: Use config defaults
    image_features = DeepseekOCRProcessor().tokenize_with_images(
        images=[image], 
        bos=True, 
        eos=True, 
        cropping=CROP_MODE
    )
    
    # Option 2: Override to Small mode
    image_features = DeepseekOCRProcessor().tokenize_with_images(
        images=[image],
        bos=True,
        eos=True,
        cropping=False,
        base_size=640,
        image_size=640
    )
```

### PDF Batch Processing (run_dpsk_ocr_pdf.py)

```python
from functools import partial

# Option 1: Use config defaults for all pages
with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    batch_inputs = list(tqdm(
        executor.map(process_single_image, images),
        total=len(images),
        desc="Pre-processed images"
    ))

# Option 2: Use Small mode for all pages
process_with_small_mode = partial(
    process_single_image,
    base_size=640,
    image_size=640,
    cropping=False
)
with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    batch_inputs = list(tqdm(
        executor.map(process_with_small_mode, images),
        total=len(images),
        desc="Pre-processed images"
    ))
```

### Mixed Modes for Different Images

```python
# Process different images with different modes
batch_inputs = []

for image in images:
    # Decide mode based on image characteristics
    width, height = image.size
    
    if width <= 640 and height <= 640:
        # Use Small mode for small images
        features = DeepseekOCRProcessor().tokenize_with_images(
            images=[image],
            bos=True,
            eos=True,
            cropping=False,
            base_size=640,
            image_size=640
        )
    else:
        # Use Gundam mode for large images
        features = DeepseekOCRProcessor().tokenize_with_images(
            images=[image],
            bos=True,
            eos=True,
            cropping=True,
            base_size=1024,
            image_size=640
        )
    
    batch_inputs.append({
        "prompt": prompt,
        "multi_modal_data": {"image": features}
    })
```

## API Reference

### DeepseekOCRProcessor.tokenize_with_images()

```python
def tokenize_with_images(
    self,
    images: List[Image.Image],
    bos: bool = True,
    eos: bool = True,
    cropping: bool = True,
    base_size: int = None,      # NEW: Optional override
    image_size: int = None,     # NEW: Optional override
) -> List[Tuple]:
    """
    Tokenize images with optional mode parameters.
    
    Args:
        images: List of PIL images to process
        bos: Whether to add beginning of sequence token
        eos: Whether to add end of sequence token
        cropping: Whether to use dynamic cropping (Gundam mode)
        base_size: Override base_size (global view size). If None, uses config.BASE_SIZE
        image_size: Override image_size (local view size). If None, uses config.IMAGE_SIZE
    
    Returns:
        List containing [input_ids, pixel_values, images_crop, images_seq_mask, 
                        images_spatial_crop, num_image_tokens, image_shapes]
    """
```

## Performance Considerations

- **Tiny/Small modes**: Faster processing, lower GPU memory, suitable for simple documents
- **Base mode**: Balanced performance, recommended for most use cases
- **Large mode**: Higher quality but slower, more GPU memory required
- **Gundam mode**: Adaptive resolution, best for complex layouts and large documents

## Backward Compatibility

The implementation is **fully backward compatible**. Existing code that doesn't specify `base_size` or `image_size` will continue to work using the values from `config.py`.

## Testing

A test script is provided at `test_dynamic_modes.py` to verify all modes work correctly:

```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
python test_dynamic_modes.py
```

## Modified Files

1. `config.py` - Added documentation for dynamic mode usage
2. `process/image_process.py` - Added optional `base_size` and `image_size` parameters
3. `run_dpsk_ocr_image.py` - Added usage examples
4. `run_dpsk_ocr_pdf.py` - Updated `process_single_image()` to support mode parameters
5. `run_dpsk_ocr_eval_batch.py` - Updated `process_single_image()` to support mode parameters

## Conclusion

You can now dynamically select OCR modes when using vLLM deployment without modifying `config.py`. This provides flexibility to optimize processing based on document characteristics, performance requirements, or quality needs.
