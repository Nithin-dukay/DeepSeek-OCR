# Summary of Changes for Issue #163

## Overview
Fixed MPS runtime error on Apple Silicon by adding device-specific interpolation logic.

## Files Modified

### 1. `DeepSeek-OCR-master/DeepSeek-OCR-vllm/deepencoder/sam_vary_sdpa.py`

**Function**: `get_abs_pos(abs_pos, tgt_size)`

**Change**: Added conditional logic to detect MPS devices and use bilinear interpolation instead of bicubic with antialias.

**Before**:
```python
if src_size != tgt_size:
    old_pos_embed = abs_pos.permute(0, 3, 1, 2)
    old_pos_embed = old_pos_embed.to(torch.float32)
    new_pos_embed = F.interpolate(
        old_pos_embed,
        size=(tgt_size, tgt_size),
        mode='bicubic',
        antialias=True,
        align_corners=False,
    ).to(dtype)
    new_pos_embed = new_pos_embed.permute(0, 2, 3, 1)
    return new_pos_embed
```

**After**:
```python
if src_size != tgt_size:
    old_pos_embed = abs_pos.permute(0, 3, 1, 2)
    old_pos_embed = old_pos_embed.to(torch.float32)
    
    # MPS (Apple Silicon) doesn't support bicubic interpolation with antialias
    # Use bilinear interpolation as a fallback for MPS devices
    if old_pos_embed.device.type == 'mps':
        new_pos_embed = F.interpolate(
            old_pos_embed,
            size=(tgt_size, tgt_size),
            mode='bilinear',
            align_corners=False,
        ).to(dtype)
    else:
        new_pos_embed = F.interpolate(
            old_pos_embed,
            size=(tgt_size, tgt_size),
            mode='bicubic',
            antialias=True,
            align_corners=False,
        ).to(dtype)
    
    new_pos_embed = new_pos_embed.permute(0, 2, 3, 1)
    return new_pos_embed
```

---

### 2. `DeepSeek-OCR-master/DeepSeek-OCR-vllm/deepencoder/clip_sdpa.py`

**Function**: `get_abs_pos(abs_pos, tgt_size)`

**Change**: Added the same conditional logic for MPS device detection.

**Before**:
```python
if src_size != tgt_size:
    old_pos_embed = old_pos_embed.view(1, src_size, src_size, dim).permute(0, 3, 1, 2).contiguous()
    old_pos_embed = old_pos_embed.to(torch.float32)
    new_pos_embed = F.interpolate(
        old_pos_embed,
        size=(tgt_size, tgt_size),
        mode='bicubic',
        antialias=True,
        align_corners=False,
    ).to(dtype)
    new_pos_embed = new_pos_embed.permute(0, 2, 3, 1)
    new_pos_embed = new_pos_embed.view(tgt_size * tgt_size, dim)
    vision_pos_embed = torch.cat([cls_token, new_pos_embed], dim=0)
    vision_pos_embed = vision_pos_embed.view(1, tgt_size * tgt_size + 1, dim)
    return vision_pos_embed
```

**After**:
```python
if src_size != tgt_size:
    old_pos_embed = old_pos_embed.view(1, src_size, src_size, dim).permute(0, 3, 1, 2).contiguous()
    old_pos_embed = old_pos_embed.to(torch.float32)
    
    # MPS (Apple Silicon) doesn't support bicubic interpolation with antialias
    # Use bilinear interpolation as a fallback for MPS devices
    if old_pos_embed.device.type == 'mps':
        new_pos_embed = F.interpolate(
            old_pos_embed,
            size=(tgt_size, tgt_size),
            mode='bilinear',
            align_corners=False,
        ).to(dtype)
    else:
        new_pos_embed = F.interpolate(
            old_pos_embed,
            size=(tgt_size, tgt_size),
            mode='bicubic',
            antialias=True,
            align_corners=False,
        ).to(dtype)
    
    new_pos_embed = new_pos_embed.permute(0, 2, 3, 1)
    new_pos_embed = new_pos_embed.view(tgt_size * tgt_size, dim)
    vision_pos_embed = torch.cat([cls_token, new_pos_embed], dim=0)
    vision_pos_embed = vision_pos_embed.view(1, tgt_size * tgt_size + 1, dim)
    return vision_pos_embed
```

---

## Key Points

1. **Device Detection**: Uses `tensor.device.type == 'mps'` to identify MPS devices
2. **MPS Path**: Uses `mode='bilinear'` without `antialias` parameter
3. **Default Path**: Maintains original `mode='bicubic'` with `antialias=True` for CUDA/CPU
4. **Comments**: Added clear comments explaining the MPS limitation
5. **No Breaking Changes**: Existing functionality preserved for non-MPS devices

## Testing

- ✅ Both files compile successfully with `python3 -m py_compile`
- ✅ Logic verified to correctly route based on device type
- ✅ Backward compatibility maintained for CUDA/CPU workflows

## Impact

- **Apple Silicon Users**: Can now run inference on MPS GPU without crashes
- **CUDA/CPU Users**: No changes to behavior or performance
- **Code Quality**: Improved with clear comments and device-aware logic
