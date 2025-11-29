# Contributing to DeepSeek-OCR

Thank you for your interest in contributing to DeepSeek-OCR! We welcome contributions from the community.

## Ways to Contribute

### 1. 🐛 Report Bugs
- Use the [GitHub Issues](https://github.com/deepseek-ai/DeepSeek-OCR/issues) page
- Check if the issue already exists
- Provide detailed information:
  - OS and Python version
  - GPU model and VRAM
  - CUDA version
  - Steps to reproduce
  - Error messages and logs
  - Expected vs actual behavior

### 2. 💡 Suggest Features
- Open a feature request on GitHub Issues
- Describe the use case
- Explain why it would be useful
- Provide examples if possible

### 3. 📝 Improve Documentation
- Fix typos or unclear explanations
- Add examples or tutorials
- Improve code comments
- Translate documentation

### 4. 🔧 Submit Code
- Bug fixes
- Performance improvements
- New features
- Test coverage

## Development Setup

### 1. Fork and Clone
```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/DeepSeek-OCR.git
cd DeepSeek-OCR
```

### 2. Create a Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 3. Set Up Environment
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy
```

### 4. Make Changes
- Write clean, readable code
- Follow existing code style
- Add comments for complex logic
- Update documentation if needed

### 5. Test Your Changes
```bash
# Run existing tests
pytest tests/

# Test your specific changes
python your_test_script.py

# Check code style
black --check .
flake8 .
```

### 6. Commit and Push
```bash
# Stage your changes
git add .

# Commit with a clear message
git commit -m "feat: add support for new image format"
# or
git commit -m "fix: resolve CUDA memory leak in batch processing"

# Push to your fork
git push origin feature/your-feature-name
```

### 7. Create Pull Request
- Go to the original repository on GitHub
- Click "New Pull Request"
- Select your branch
- Fill in the PR template:
  - Description of changes
  - Related issues
  - Testing done
  - Screenshots (if applicable)

## Coding Guidelines

### Python Style
- Follow [PEP 8](https://pep8.org/)
- Use meaningful variable names
- Keep functions focused and small
- Add docstrings for public functions

```python
def process_image(image_path: str, mode: str = "base") -> dict:
    """
    Process an image using the specified mode.
    
    Args:
        image_path: Path to the input image
        mode: Processing mode (tiny/small/base/large/gundam)
        
    Returns:
        Dictionary containing OCR results
        
    Raises:
        FileNotFoundError: If image_path doesn't exist
        ValueError: If mode is invalid
    """
    # Implementation
    pass
```

### Commit Messages
Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

Examples:
```
feat: add support for TIFF images
fix: resolve memory leak in batch processing
docs: update installation instructions for Windows
refactor: simplify image preprocessing pipeline
test: add unit tests for OCR accuracy
```

### Code Review Process
1. Maintainers will review your PR
2. Address any feedback or requested changes
3. Once approved, your PR will be merged
4. Your contribution will be credited

## Testing Guidelines

### Unit Tests
```python
import pytest
from deepseek_ocr import process_image

def test_process_image_basic():
    result = process_image("test_image.jpg", mode="base")
    assert result is not None
    assert "text" in result
    
def test_process_image_invalid_mode():
    with pytest.raises(ValueError):
        process_image("test_image.jpg", mode="invalid")
```

### Integration Tests
- Test end-to-end workflows
- Test with real images
- Verify output quality

### Performance Tests
- Benchmark processing speed
- Monitor memory usage
- Test with various image sizes

## Documentation Guidelines

### README Updates
- Keep it concise and clear
- Use examples
- Update table of contents
- Test all code examples

### Code Comments
```python
# Good: Explains WHY
# Use dynamic cropping for images larger than 640x640 to preserve detail
if width > 640 or height > 640:
    crop_mode = True

# Bad: Explains WHAT (obvious from code)
# Set crop_mode to True
crop_mode = True
```

### API Documentation
- Document all public functions
- Include parameter types
- Provide usage examples
- Note any side effects

## Community Guidelines

### Be Respectful
- Be kind and courteous
- Respect different viewpoints
- Accept constructive criticism
- Focus on what's best for the project

### Be Collaborative
- Help others
- Share knowledge
- Give credit where due
- Celebrate contributions

### Be Professional
- Stay on topic
- Avoid spam or self-promotion
- Follow the code of conduct
- Maintain a positive environment

## Getting Help

### Questions?
- 💬 [Discord Community](https://discord.gg/Tc7c45Zzu5)
- 📧 GitHub Issues (for bug reports)
- 📚 [Documentation](README.md)

### Need Guidance?
- Check existing issues and PRs
- Ask in Discord
- Tag maintainers in your PR

## Recognition

Contributors will be:
- Listed in the project contributors
- Credited in release notes
- Acknowledged in the community

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to DeepSeek-OCR! 🚀

Every contribution, no matter how small, makes a difference.
