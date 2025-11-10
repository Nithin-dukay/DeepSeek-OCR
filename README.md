<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->


<div align="center">
  <img src="assets/logo.svg" width="60%" alt="DeepSeek AI" />
</div>


<hr>
<div align="center">
  <a href="https://www.deepseek.com/" target="_blank">
    <img alt="Homepage" src="assets/badge.svg" />
  </a>
  <a href="https://huggingface.co/deepseek-ai/DeepSeek-OCR" target="_blank">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-DeepSeek%20AI-ffc107?color=ffc107&logoColor=white" />
  </a>

</div>

<div align="center">

  <a href="https://discord.gg/Tc7c45Zzu5" target="_blank">
    <img alt="Discord" src="https://img.shields.io/badge/Discord-DeepSeek%20AI-7289da?logo=discord&logoColor=white&color=7289da" />
  </a>
  <a href="https://twitter.com/deepseek_ai" target="_blank">
    <img alt="Twitter Follow" src="https://img.shields.io/badge/Twitter-deepseek_ai-white?logo=x&logoColor=white" />
  </a>

</div>



<p align="center">
  <a href="https://huggingface.co/deepseek-ai/DeepSeek-OCR"><b>📥 Model Download</b></a> |
  <a href="https://github.com/deepseek-ai/DeepSeek-OCR/blob/main/DeepSeek_OCR_paper.pdf"><b>📄 Paper Link</b></a> |
  <a href="https://arxiv.org/abs/2510.18234"><b>📄 Arxiv Paper Link</b></a> |
</p>

<h2>
<p align="center">
  <a href="">DeepSeek-OCR: Contexts Optical Compression</a>
</p>
</h2>

<p align="center">
<img src="assets/fig1.png" style="width: 1000px" align=center>
</p>
<p align="center">
<a href="">Explore the boundaries of visual-text compression.</a>       
</p>

## Release
- [2025/10/23]🚀🚀🚀 DeepSeek-OCR is now officially supported in upstream [vLLM](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-OCR.html#installing-vllm). Thanks to the [vLLM](https://github.com/vllm-project/vllm) team for their help.
- [2025/10/20]🚀🚀🚀 We release DeepSeek-OCR, a model to investigate the role of vision encoders from an LLM-centric viewpoint.

## ⚠️ IMPORTANT: Installation Order Matters!

**Before installing, please read this to avoid common errors:**

🚨 **Common Error**: `ImportError: cannot import name 'GenerationMixin' from 'transformers.generation'`

**Cause**: Installing packages in the wrong order or using incompatible versions.

**Solution**: Follow the correct installation order in [INSTALLATION.md](INSTALLATION.md)

### Quick Installation Check

Before you start, run this to validate your environment:
```bash
python check_environment.py
```

### Installation Methods

Choose ONE of these methods based on your needs:

1. **[Recommended] vLLM 0.8.5 (Local)** - Most stable, tested configuration
   ```bash
   bash setup_vllm_local.sh
   ```

2. **vLLM Nightly (Upstream)** - Latest features, officially supported by vLLM
   ```bash
   bash setup_vllm_upstream.sh
   ```

3. **Transformers Only** - Simpler but slower inference

📖 **For detailed instructions, troubleshooting, and platform-specific guides (Kaggle/Colab), see [INSTALLATION.md](INSTALLATION.md)**

## Contents
- [Install](#install)
- [vLLM Inference](#vllm-inference)
- [Transformers Inference](#transformers-inference)
- [Troubleshooting](#troubleshooting)
  




## Install

### ⚠️ Critical Installation Notes

**IMPORTANT**: Installation order matters! Installing packages in the wrong order will cause `ImportError: cannot import name 'GenerationMixin'`.

**For detailed installation instructions, see [INSTALLATION.md](INSTALLATION.md)**

### Quick Install (Automated)

**Option 1: vLLM 0.8.5 (Recommended for stability)**
```bash
git clone https://github.com/deepseek-ai/DeepSeek-OCR.git
cd DeepSeek-OCR
bash setup_vllm_local.sh
```

**Option 2: vLLM Nightly (Latest features)**
```bash
git clone https://github.com/deepseek-ai/DeepSeek-OCR.git
cd DeepSeek-OCR
bash setup_vllm_upstream.sh
```

### Manual Install (vLLM 0.8.5)

>Our tested environment is cuda11.8+torch2.6.0+transformers4.46.3

1. Clone this repository and navigate to the DeepSeek-OCR folder
```bash
git clone https://github.com/deepseek-ai/DeepSeek-OCR.git
cd DeepSeek-OCR
```

2. Create conda environment
```Shell
conda create -n deepseek-ocr python=3.12.9 -y
conda activate deepseek-ocr
```

3. **STEP 1: Install PyTorch FIRST** (Critical!)
```Shell
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
```

4. **STEP 2: Install vLLM** - Download the vllm-0.8.5 [whl](https://github.com/vllm-project/vllm/releases/tag/v0.8.5) 
```Shell
wget https://github.com/vllm-project/vllm/releases/download/v0.8.5/vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl
pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl
```

5. **STEP 3: Install requirements** (includes transformers==4.46.3)
```Shell
pip install -r requirements.txt
```

6. **STEP 4: Install flash-attn** (Optional but recommended)
```Shell
pip install flash-attn==2.7.3 --no-build-isolation
```

**Note:** You may see a warning like `vllm 0.8.5+cu118 requires transformers>=4.51.1`. This can be safely ignored - the project is designed to work with transformers 4.46.3 and vLLM 0.8.5 together.

### Verify Installation

After installation, verify your environment:
```bash
python check_environment.py
```

This will check for version compatibility and common issues.

## vLLM-Inference
- VLLM:
>**Note:** change the INPUT_PATH/OUTPUT_PATH and other settings in the DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py
```Shell
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
```
1. image: streaming output
```Shell
python run_dpsk_ocr_image.py
```
2. pdf: concurrency ~2500tokens/s(an A100-40G)
```Shell
python run_dpsk_ocr_pdf.py
```
3. batch eval for benchmarks
```Shell
python run_dpsk_ocr_eval_batch.py
```

**[2025/10/23] The version of upstream [vLLM](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-OCR.html#installing-vllm):**

```shell
uv venv
source .venv/bin/activate
# Until v0.11.1 release, you need to install vLLM from nightly build
uv pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
```

```python
from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
from PIL import Image

# Create model instance
llm = LLM(
    model="deepseek-ai/DeepSeek-OCR",
    enable_prefix_caching=False,
    mm_processor_cache_gb=0,
    logits_processors=[NGramPerReqLogitsProcessor]
)

# Prepare batched input with your image file
image_1 = Image.open("path/to/your/image_1.png").convert("RGB")
image_2 = Image.open("path/to/your/image_2.png").convert("RGB")
prompt = "<image>\nFree OCR."

model_input = [
    {
        "prompt": prompt,
        "multi_modal_data": {"image": image_1}
    },
    {
        "prompt": prompt,
        "multi_modal_data": {"image": image_2}
    }
]

sampling_param = SamplingParams(
            temperature=0.0,
            max_tokens=8192,
            # ngram logit processor args
            extra_args=dict(
                ngram_size=30,
                window_size=90,
                whitelist_token_ids={128821, 128822},  # whitelist: <td>, </td>
            ),
            skip_special_tokens=False,
        )
# Generate output
model_outputs = llm.generate(model_input, sampling_param)

# Print output
for output in model_outputs:
    print(output.outputs[0].text)
```
## Transformers-Inference
- Transformers
```python
from transformers import AutoModel, AutoTokenizer
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model_name = 'deepseek-ai/DeepSeek-OCR'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, _attn_implementation='flash_attention_2', trust_remote_code=True, use_safetensors=True)
model = model.eval().cuda().to(torch.bfloat16)

# prompt = "<image>\nFree OCR. "
prompt = "<image>\n<|grounding|>Convert the document to markdown. "
image_file = 'your_image.jpg'
output_path = 'your/output/dir'

res = model.infer(tokenizer, prompt=prompt, image_file=image_file, output_path = output_path, base_size = 1024, image_size = 640, crop_mode=True, save_results = True, test_compress = True)
```
or you can
```Shell
cd DeepSeek-OCR-master/DeepSeek-OCR-hf
python run_dpsk_ocr.py
```
## Support-Modes
The current open-source model supports the following modes:
- Native resolution:
  - Tiny: 512×512 （64 vision tokens）✅
  - Small: 640×640 （100 vision tokens）✅
  - Base: 1024×1024 （256 vision tokens）✅
  - Large: 1280×1280 （400 vision tokens）✅
- Dynamic resolution
  - Gundam: n×640×640 + 1×1024×1024 ✅

## Prompts examples
```python
# document: <image>\n<|grounding|>Convert the document to markdown.
# other image: <image>\n<|grounding|>OCR this image.
# without layouts: <image>\nFree OCR.
# figures in document: <image>\nParse the figure.
# general: <image>\nDescribe this image in detail.
# rec: <image>\nLocate <|ref|>xxxx<|/ref|> in the image.
# '先天下之忧而忧'
```


## Visualizations
<table>
<tr>
<td><img src="assets/show1.jpg" style="width: 500px"></td>
<td><img src="assets/show2.jpg" style="width: 500px"></td>
</tr>
<tr>
<td><img src="assets/show3.jpg" style="width: 500px"></td>
<td><img src="assets/show4.jpg" style="width: 500px"></td>
</tr>
</table>


## Troubleshooting

### Common Issues

#### 1. ImportError: cannot import name 'GenerationMixin'

**Error Message:**
```
ImportError: cannot import name 'GenerationMixin' from 'transformers.generation'
```

**Cause:** Version mismatch between transformers and vLLM, or incorrect installation order.

**Solutions:**

For vLLM 0.8.5:
```bash
pip uninstall transformers -y
pip install transformers==4.46.3
```

For vLLM nightly:
```bash
pip install --pre vllm --extra-index-url https://wheels.vllm.ai/nightly --force-reinstall
```

#### 2. CUDA Out of Memory

**Solutions:**
- Reduce `max_model_len` in your code (try 4096 instead of 8192)
- Use smaller image sizes (640x640 instead of 1024x1024)
- Reduce `MAX_CROPS` in `config.py` (try 4 or 6 instead of 9)

#### 3. Flash Attention Installation Fails

**Solution:** Flash attention is optional. Skip it and the model will use standard attention:
```bash
# Continue without flash-attn - the model will work fine
```

#### 4. Model Download is Slow

**Solutions:**
- Use HuggingFace mirror: `export HF_ENDPOINT=https://hf-mirror.com`
- Be patient - first download takes 5-10 minutes depending on network speed

### Platform-Specific Issues

#### Kaggle/Colab

For Kaggle and Google Colab users, see:
- [kaggle_notebook_example.py](kaggle_notebook_example.py) - Complete working example
- [INSTALLATION.md](INSTALLATION.md) - Platform-specific instructions

**Key points for Kaggle/Colab:**
1. Install PyTorch BEFORE vLLM
2. Do NOT install transformers separately when using vLLM nightly
3. Use the correct installation order (see kaggle_notebook_example.py)

### Getting Help

1. **Check your environment**: Run `python check_environment.py`
2. **Read the docs**: See [INSTALLATION.md](INSTALLATION.md) for detailed instructions
3. **Search issues**: Check [GitHub Issues](https://github.com/deepseek-ai/DeepSeek-OCR/issues)
4. **Create an issue**: If your problem isn't covered, create a new issue with:
   - Output of `check_environment.py`
   - Full error traceback
   - Your installation method
   - Platform (Kaggle/Colab/Local)

## Acknowledgement

We would like to thank [Vary](https://github.com/Ucas-HaoranWei/Vary/), [GOT-OCR2.0](https://github.com/Ucas-HaoranWei/GOT-OCR2.0/), [MinerU](https://github.com/opendatalab/MinerU), [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), [OneChart](https://github.com/LingyvKong/OneChart), [Slow Perception](https://github.com/Ucas-HaoranWei/Slow-Perception) for their valuable models and ideas.

We also appreciate the benchmarks: [Fox](https://github.com/ucaslcl/Fox), [OminiDocBench](https://github.com/opendatalab/OmniDocBench).

## Citation

```bibtex
@article{wei2025deepseek,
  title={DeepSeek-OCR: Contexts Optical Compression},
  author={Wei, Haoran and Sun, Yaofeng and Li, Yukun},
  journal={arXiv preprint arXiv:2510.18234},
  year={2025}
}
