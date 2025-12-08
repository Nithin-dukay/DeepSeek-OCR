# WeChat Group QR Code Maintenance Guide
# 微信群二维码维护指南

## English

### About WeChat Group QR Codes
WeChat group QR codes expire after **7 days** from creation. This is a WeChat platform limitation and requires regular updates.

### How to Update the QR Code

1. **Generate a New QR Code**
   - Open the WeChat group
   - Tap on the group settings (top right corner)
   - Select "Group QR Code"
   - Tap "..." and select "Save Image"

2. **Update the Repository**
   - Replace the file `assets/wechat_group.svg` (or `.png`/`.jpg`) with the new QR code image
   - Recommended format: PNG or JPG (300x300 pixels or larger)
   - File name should remain: `wechat_group.svg` or `wechat_group.png`

3. **Commit the Changes**
   ```bash
   git add assets/wechat_group.*
   git commit -m "chore: update WeChat group QR code"
   git push
   ```

4. **Close Related Issues**
   - Close any GitHub issues reporting expired QR codes
   - Reference the commit in the issue closure

### Automation Reminder
Consider setting a calendar reminder to update the QR code every **5-6 days** to prevent expiration.

---

## 中文

### 关于微信群二维码
微信群二维码在创建后 **7天** 过期。这是微信平台的限制，需要定期更新。

### 如何更新二维码

1. **生成新的二维码**
   - 打开微信群
   - 点击右上角群设置
   - 选择"群二维码"
   - 点击"..."并选择"保存图片"

2. **更新仓库**
   - 用新的二维码图片替换 `assets/wechat_group.svg`（或 `.png`/`.jpg`）
   - 推荐格式：PNG 或 JPG（300x300 像素或更大）
   - 文件名保持为：`wechat_group.svg` 或 `wechat_group.png`

3. **提交更改**
   ```bash
   git add assets/wechat_group.*
   git commit -m "chore: 更新微信群二维码"
   git push
   ```

4. **关闭相关 Issues**
   - 关闭任何报告二维码过期的 GitHub issues
   - 在关闭 issue 时引用提交记录

### 自动化提醒
建议设置日历提醒，每 **5-6天** 更新一次二维码，以防止过期。

---

## Quick Reference / 快速参考

| Task / 任务 | Action / 操作 |
|------------|--------------|
| QR Code Expiration / 二维码有效期 | 7 days / 7天 |
| Update Frequency / 更新频率 | Every 5-6 days / 每5-6天 |
| File Location / 文件位置 | `assets/wechat_group.*` |
| Recommended Size / 推荐尺寸 | 300x300px or larger / 300x300像素或更大 |

---

## Troubleshooting / 故障排除

### Issue: Users report QR code expired
**Solution:** Follow the update steps above immediately.

### 问题：用户反馈二维码已过期
**解决方案：** 立即按照上述步骤更新。

### Issue: QR code image not displaying in README
**Solution:** 
- Check file path is correct: `assets/wechat_group.*`
- Verify file format is supported (PNG, JPG, SVG)
- Clear browser cache and refresh

### 问题：README 中二维码图片不显示
**解决方案：**
- 检查文件路径是否正确：`assets/wechat_group.*`
- 验证文件格式是否支持（PNG、JPG、SVG）
- 清除浏览器缓存并刷新
