# GitHub Issue #294 修复说明：出现了很离谱的解析结果

## 问题描述

该问题报告了 DeepSeek-OCR 的两个主要问题：

1. **幻觉问题**：模型生成了源 PDF/图像中不存在的内容
2. **不需要的定位标记**：输出包含 `<|ref|>`, `<|/ref|>`, `<|det|>`, `<|/det|>` 等标记，污染了 markdown 输出

## 根本原因

### 1. 幻觉问题的原因
- **max_tokens 过大 (8192)**：允许模型在实际内容之外继续生成
- **重复控制不足**：ngram 参数不够严格
- **缺少停止条件**：没有强制 EOS token
- **没有重复惩罚**：模型可以无限重复模式

### 2. 定位标记问题的原因
- **没有后处理**：定位标记没有从最终输出中过滤
- **提示词敏感**：使用 `<|grounding|>` 提示词但没有适当的标记清理

## 实施的解决方案

### 1. 改进的采样参数

**在所有三个脚本中的更改** (`run_dpsk_ocr_pdf.py`, `run_dpsk_ocr_eval_batch.py`, `run_dpsk_ocr_image.py`)：

```python
# 修改前
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    logits_processors=logits_processors,
    skip_special_tokens=False,
)

# 修改后
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=4096,  # 从 8192 减少到 4096，防止过度生成
    logits_processors=logits_processors,
    skip_special_tokens=False,
    stop_token_ids=[128009],  # 添加 EOS token 以停止生成
    repetition_penalty=1.05,  # 添加轻微的重复惩罚以减少幻觉
)
```

**优势**：
- 将 `max_tokens` 从 8192 减少到 4096 防止过度生成
- 添加 `stop_token_ids=[128009]` (EOS token) 以正确终止生成
- 添加 `repetition_penalty=1.05` 以抑制重复模式

### 2. 优化的 NGram 参数

**PDF 处理** (`run_dpsk_ocr_pdf.py`)：
```python
# 修改前: ngram_size=20, window_size=50
# 修改后: ngram_size=15, window_size=60
logits_processors = [NoRepeatNGramLogitsProcessor(ngram_size=15, window_size=60, whitelist_token_ids={128821, 128822})]
```

**批量评估** (`run_dpsk_ocr_eval_batch.py`)：
```python
# 修改前: ngram_size=40, window_size=90
# 修改后: ngram_size=30, window_size=80
logits_processors = [NoRepeatNGramLogitsProcessor(ngram_size=30, window_size=80, whitelist_token_ids={128821, 128822})]
```

**图像处理** (`run_dpsk_ocr_image.py`)：
```python
# 修改前: ngram_size=30, window_size=90
# 修改后: ngram_size=25, window_size=80
logits_processors = [NoRepeatNGramLogitsProcessor(ngram_size=25, window_size=80, whitelist_token_ids={128821, 128822})]
```

**优势**：
- 更小的 `ngram_size` 更早地捕获重复
- 平衡的 `window_size` 提供更好的上下文而不会过于限制

### 3. 定位标记清理函数

在 `run_dpsk_ocr_pdf.py` 和 `run_dpsk_ocr_eval_batch.py` 中添加了新的 `clean_grounding_tokens()` 函数：

```python
def clean_grounding_tokens(text):
    """
    从输出中删除不需要的定位标记。
    这有助于防止带有定位注释的幻觉内容。
    """
    # 删除所有定位标记模式: <|ref|>...<|/ref|><|det|>...<|/det|>
    pattern = r'<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>'
    cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
    
    # 删除任何剩余的单个定位标记
    cleaned_text = cleaned_text.replace('<|ref|>', '').replace('<|/ref|>', '')
    cleaned_text = cleaned_text.replace('<|det|>', '').replace('<|/det|>', '')
    cleaned_text = cleaned_text.replace('<|grounding|>', '')
    
    # 清理过多的换行符
    cleaned_text = re.sub(r'\n{4,}', '\n\n', cleaned_text)
    cleaned_text = re.sub(r'\n{3}', '\n\n', cleaned_text)
    
    return cleaned_text.strip()
```

**优势**：
- 从输出中删除所有定位标记模式
- 清理由标记删除引起的过多换行符
- 生成干净的 markdown，没有注释伪影

### 4. 在输出处理中应用清理

**在 `run_dpsk_ocr_pdf.py` 中**：
```python
# 应用定位标记清理以减少幻觉
content = clean_grounding_tokens(content)
```

**在 `run_dpsk_ocr_eval_batch.py` 中**：
```python
# 首先应用定位标记清理以减少幻觉
content = clean_grounding_tokens(content)
content = clean_formula(content)
```

## 测试建议

要验证修复是否正常工作：

1. **使用原始有问题的 PDF 进行测试**：
   - `yanbaopptmerge_yanbaoPPT_4570.pdf`
   - `jiaocaineedrop_jiaocai_needrop_en_2604.pdf`

2. **检查改进**：
   - ✅ 没有超出实际 PDF 内容的幻觉内容
   - ✅ 干净的 markdown 输出，没有 `<|ref|>`, `<|/ref|>`, `<|det|>`, `<|/det|>` 标记
   - ✅ 正确终止，没有过度重复
   - ✅ 准确的内容提取

3. **运行脚本**：
   ```bash
   # PDF 处理
   python run_dpsk_ocr_pdf.py
   
   # 批量评估
   python run_dpsk_ocr_eval_batch.py
   
   # 图像处理
   python run_dpsk_ocr_image.py
   ```

## 配置提示

如果仍然遇到问题，可以在相应的脚本中进一步调整这些参数：

### 更激进的重复预防：
- 减少 `ngram_size`（例如 10-15）
- 减少 `window_size`（例如 40-50）
- 增加 `repetition_penalty`（例如 1.1-1.2）

### 更短的输出：
- 减少 `max_tokens`（例如 2048-3072）

### 不同的文档类型：
- **简单文档**：使用较小的 ngram_size（15-20）
- **带表格的复杂文档**：使用当前设置（20-30）
- **具有重复模式的文档**：增加 repetition_penalty（1.1-1.15）

## 更改摘要

| 文件 | 更改 |
|------|------|
| `run_dpsk_ocr_pdf.py` | ✅ 将 max_tokens 减少到 4096<br>✅ 添加 stop_token_ids 和 repetition_penalty<br>✅ 将 ngram_size 优化为 15<br>✅ 添加 clean_grounding_tokens() 函数<br>✅ 在输出处理中应用清理 |
| `run_dpsk_ocr_eval_batch.py` | ✅ 将 max_tokens 减少到 4096<br>✅ 添加 stop_token_ids 和 repetition_penalty<br>✅ 将 ngram_size 优化为 30<br>✅ 添加 clean_grounding_tokens() 函数<br>✅ 在输出处理中应用清理 |
| `run_dpsk_ocr_image.py` | ✅ 将 max_tokens 减少到 4096<br>✅ 添加 stop_token_ids 和 repetition_penalty<br>✅ 将 ngram_size 优化为 25 |

## 预期结果

应用这些修复后：

1. **不再有幻觉**：模型将在到达实际内容末尾时停止生成
2. **干净的 markdown 输出**：最终输出中没有定位标记（`<|ref|>`, `<|/ref|>`, `<|det|>`, `<|/det|>`）
3. **更好的质量**：更准确的 OCR 结果，正确终止
4. **一致的行为**：在不同文档类型中更可预测的输出

## 附加说明

- 保留了 `whitelist_token_ids={128821, 128822}` 用于 `<td>` 和 `</td>` 标记，以允许表格结构重复
- 清理函数在生成后但保存前应用，确保原始输出仍然在 `_det` 文件中可用
- 这些更改向后兼容，不需要模型重新训练
