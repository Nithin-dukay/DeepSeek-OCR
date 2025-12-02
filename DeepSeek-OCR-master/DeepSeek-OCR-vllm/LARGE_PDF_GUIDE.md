# Large PDF Processing Guide

## Quick Start for 2800+ Page PDFs

### Problem Solved
✅ System no longer crashes when processing large PDFs (2800+ pages)  
✅ Automatic memory management prevents OOM errors  
✅ Resume capability if processing is interrupted  

---

## Configuration

Edit `config.py` and adjust these settings based on your GPU memory:

### For 24GB GPU (e.g., A10, RTX 3090, A5000):
```python
PDF_BATCH_SIZE = 50  # Process 50 pages at a time
GPU_MEMORY_THRESHOLD_GB = 20.0
```

### For 16GB GPU (e.g., RTX 4080, V100):
```python
PDF_BATCH_SIZE = 30  # Reduce batch size
GPU_MEMORY_THRESHOLD_GB = 14.0
```

### For 12GB GPU (e.g., RTX 3080, RTX 4070):
```python
PDF_BATCH_SIZE = 20  # Further reduce batch size
GPU_MEMORY_THRESHOLD_GB = 10.0
```

### For 8GB GPU (e.g., RTX 3070, RTX 4060):
```python
PDF_BATCH_SIZE = 10  # Minimum batch size
GPU_MEMORY_THRESHOLD_GB = 7.0
MAX_CONCURRENCY = 50  # Also reduce concurrency
```

---

## Usage

### 1. Set Input/Output Paths
In `config.py`:
```python
INPUT_PATH = '/path/to/your/large_document.pdf'
OUTPUT_PATH = '/path/to/output/directory'
```

### 2. Run Processing
```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
python run_dpsk_ocr_pdf.py
```

### 3. Monitor Progress
The script will show:
- Total page count
- Current chunk being processed
- Memory usage (if monitoring enabled)
- Checkpoint saves

Example output:
```
Total pages: 2800
Large PDF detected. Using chunked processing with batch size: 50

Processing pages 0 to 49 (50 pages)
Pre-processing pages 0-49: 100%|████████| 50/50 [00:15<00:00]
Running inference...
✓ Checkpoint saved: 50 pages processed

Processing pages 50 to 99 (50 pages)
...
```

---

## Troubleshooting

### Still Getting OOM (Out of Memory) Errors?

**Solution 1: Reduce Batch Size**
```python
PDF_BATCH_SIZE = 25  # or even 10 for very limited memory
```

**Solution 2: Reduce Concurrency**
```python
MAX_CONCURRENCY = 50  # or lower
```

**Solution 3: Reduce Image Workers**
```python
NUM_WORKERS = 32  # or lower
```

**Solution 4: Use Smaller Image Size**
```python
IMAGE_SIZE = 512  # instead of 640
BASE_SIZE = 640   # instead of 1024
```

### Processing Interrupted?

**No problem!** Just run the script again:
```bash
python run_dpsk_ocr_pdf.py
```

It will automatically resume from the last checkpoint:
```
✓ Checkpoint loaded: resuming from page 150
```

### Want to Start Fresh?

Delete the checkpoint:
```bash
rm -rf ./checkpoints/your_document_checkpoint.json
rm -rf ./checkpoints/your_document_checkpoint_images/
```

---

## Memory Monitoring

### Enable Detailed Memory Logging
In `config.py`:
```python
ENABLE_MEMORY_MONITORING = True
```

This will show memory usage after each stage:
```
============================================================
Memory Usage [After loading 50 images]
============================================================
GPU Memory:
  Allocated: 12.34 GB
  Reserved:  13.50 GB
  Free:      10.66 GB
  Total:     24.00 GB
CPU Memory:
  RSS:       8.45 GB
  Percent:   15.2%
============================================================
```

### Disable Memory Logging (Faster)
```python
ENABLE_MEMORY_MONITORING = False
```

---

## Performance Tips

### Faster Processing
- Increase `PDF_BATCH_SIZE` (if you have memory)
- Increase `MAX_CONCURRENCY`
- Increase `NUM_WORKERS`
- Disable memory monitoring: `ENABLE_MEMORY_MONITORING = False`

### More Stable (Less Memory)
- Decrease `PDF_BATCH_SIZE`
- Decrease `MAX_CONCURRENCY`
- Decrease `NUM_WORKERS`
- Enable memory monitoring: `ENABLE_MEMORY_MONITORING = True`

---

## Expected Processing Times

Based on A100-40G performance (~2500 tokens/s):

| Pages | Batch Size | Estimated Time |
|-------|------------|----------------|
| 400   | 50         | ~1-2 minutes   |
| 1000  | 50         | ~3-5 minutes   |
| 2800  | 50         | ~10-15 minutes |
| 5000  | 50         | ~20-30 minutes |

*Times vary based on GPU, document complexity, and settings*

---

## Advanced Configuration

### Checkpoint Settings
```python
CHECKPOINT_ENABLED = True  # Enable/disable checkpoints
CHECKPOINT_DIR = './checkpoints'  # Where to save checkpoints
```

### Memory Cleanup
```python
MEMORY_CLEANUP_FREQUENCY = 10  # Clear memory every N pages during preprocessing
```

---

## FAQ

**Q: How much disk space do checkpoints use?**  
A: Very little - typically 1-5MB per checkpoint. They're automatically deleted on successful completion.

**Q: Can I process multiple PDFs simultaneously?**  
A: Not recommended. Process one at a time to avoid memory issues.

**Q: What if my PDF has 10,000+ pages?**  
A: No problem! The chunked processing can handle unlimited pages. Just be patient - it will take longer.

**Q: Does this work with the image processing script?**  
A: The memory improvements apply to `run_dpsk_ocr_pdf.py`. For batch image processing, use `run_dpsk_ocr_eval_batch.py`.

**Q: Can I change settings while processing?**  
A: No, stop the script, change settings, and restart. It will resume from the checkpoint.

---

## Support

If you encounter issues:

1. Check memory usage: `nvidia-smi`
2. Enable memory monitoring in config
3. Try reducing `PDF_BATCH_SIZE`
4. Check the checkpoint directory for saved progress
5. Review error messages carefully

For persistent issues, provide:
- GPU model and memory
- PDF page count
- Your `config.py` settings
- Error messages (if any)

---

## Summary

✅ **Large PDFs now work reliably**  
✅ **Automatic memory management**  
✅ **Resume capability**  
✅ **Easy configuration**  
✅ **Backward compatible**  

Just set your paths in `config.py` and run `python run_dpsk_ocr_pdf.py`!
