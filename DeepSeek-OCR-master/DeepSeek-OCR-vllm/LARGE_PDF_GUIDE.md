# Quick Guide: Processing Large PDFs

## TL;DR - For Large PDFs (1000+ pages)

```bash
# 1. Edit config.py
INPUT_PATH = '/path/to/your/large.pdf'
OUTPUT_PATH = '/path/to/output'
BATCH_SIZE = 50  # Adjust based on your GPU memory

# 2. Run the batched script
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
python run_dpsk_ocr_pdf_batched.py
```

## Configuration Quick Reference

### GPU Memory Settings

| Your GPU Memory | Set BATCH_SIZE | Set MAX_CONCURRENCY |
|-----------------|----------------|---------------------|
| 8GB             | 20-25          | 25                  |
| 16GB            | 40-50          | 50                  |
| 24GB (A10)      | 50-75          | 75                  |
| 40GB+ (A100)    | 100-150        | 100                 |

### Edit `config.py`

```python
# Required settings
INPUT_PATH = '/path/to/your/document.pdf'
OUTPUT_PATH = '/path/to/output/directory'

# Memory optimization (adjust based on table above)
BATCH_SIZE = 50
MAX_CONCURRENCY = 75

# Optional: Enable memory monitoring
VERBOSE_MEMORY_STATS = True  # See memory usage in real-time
```

## Two Scripts Available

### 1. `run_dpsk_ocr_pdf_batched.py` ⭐ RECOMMENDED for large PDFs
- Processes in batches
- Constant memory usage
- Handles 2800+ pages
- Slightly slower but stable

### 2. `run_dpsk_ocr_pdf.py` - Original (now with memory cleanup)
- Good for <1000 pages
- Faster for small documents
- Enhanced with memory cleanup

## Monitoring Progress

### Watch GPU Memory
```bash
# In a separate terminal
watch -n 1 nvidia-smi
```

### Enable Verbose Output
In `config.py`:
```python
VERBOSE_MEMORY_STATS = True
```

## Troubleshooting

### Out of Memory?
1. Reduce `BATCH_SIZE` (try 25)
2. Reduce `MAX_CONCURRENCY` (try 25)
3. Set `MAX_CROPS = 4`

### Too Slow?
1. Increase `BATCH_SIZE` (if memory allows)
2. Increase `MAX_CONCURRENCY`
3. Disable verbose stats: `VERBOSE_MEMORY_STATS = False`

## Output Files

After processing, you'll get:
- `{filename}_det.mmd` - Full OCR output with detection info
- `{filename}.mmd` - Clean markdown output
- `{filename}_layouts.pdf` - PDF with bounding boxes
- `images/` - Extracted images from document

## Example: Processing 2800-page PDF

```python
# config.py
INPUT_PATH = '/data/large_document.pdf'
OUTPUT_PATH = '/data/output'
BATCH_SIZE = 50
MAX_CONCURRENCY = 75
ENABLE_MEMORY_CLEANUP = True
VERBOSE_MEMORY_STATS = True
```

```bash
python run_dpsk_ocr_pdf_batched.py
```

Expected time: ~35-40 minutes on g5.2xlarge (24GB GPU)

## Need Help?

See full documentation: `MEMORY_FIX_DOCUMENTATION.md`
