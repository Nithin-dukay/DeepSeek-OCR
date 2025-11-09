#!/bin/bash

# Apply Fix Script for GitHub Issue #110
# This script applies the fix to existing DeepSeek-OCR installation

set -e  # Exit on error

echo "========================================================================"
echo "  DeepSeek-OCR Issue #110 Fix Installer"
echo "========================================================================"
echo ""
echo "This script will:"
echo "  1. Check your environment"
echo "  2. Backup existing files"
echo "  3. Install the compatibility module"
echo "  4. Create fixed versions of run scripts"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Installation cancelled."
    exit 0
fi

# Check if we're in the right directory
if [ ! -d "DeepSeek-OCR-master/DeepSeek-OCR-vllm" ]; then
    echo "ERROR: DeepSeek-OCR-master/DeepSeek-OCR-vllm directory not found!"
    echo "Please run this script from the DeepSeek-OCR root directory."
    exit 1
fi

VLLM_DIR="DeepSeek-OCR-master/DeepSeek-OCR-vllm"

echo ""
echo "Step 1: Checking environment..."
echo "========================================================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Please install Python 3.8+."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✓ Python version: $PYTHON_VERSION"

# Check if vLLM is installed
if python3 -c "import vllm" 2>/dev/null; then
    VLLM_VERSION=$(python3 -c "import vllm; print(vllm.__version__)")
    echo "✓ vLLM version: $VLLM_VERSION"
else
    echo "⚠ vLLM not installed. You'll need to install it separately."
fi

echo ""
echo "Step 2: Creating backups..."
echo "========================================================================"

BACKUP_DIR="backups_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup existing run scripts
for script in run_dpsk_ocr_image.py run_dpsk_ocr_pdf.py run_dpsk_ocr_eval_batch.py; do
    if [ -f "$VLLM_DIR/$script" ]; then
        cp "$VLLM_DIR/$script" "$BACKUP_DIR/"
        echo "✓ Backed up $script"
    fi
done

# Backup config.py
if [ -f "$VLLM_DIR/config.py" ]; then
    cp "$VLLM_DIR/config.py" "$BACKUP_DIR/"
    echo "✓ Backed up config.py"
fi

echo "✓ Backups saved to: $BACKUP_DIR"

echo ""
echo "Step 3: Installing compatibility module..."
echo "========================================================================"

# Check if vllm_version_compat.py exists
if [ -f "vllm_version_compat.py" ]; then
    cp vllm_version_compat.py "$VLLM_DIR/"
    echo "✓ Installed vllm_version_compat.py"
elif [ -f "$VLLM_DIR/vllm_version_compat.py" ]; then
    echo "✓ vllm_version_compat.py already exists"
else
    echo "⚠ vllm_version_compat.py not found in current directory"
    echo "  Please ensure you have all solution files in the current directory"
fi

echo ""
echo "Step 4: Creating fixed run scripts..."
echo "========================================================================"

# Check if fixed scripts exist
if [ -f "run_dpsk_ocr_image_fixed.py" ]; then
    cp run_dpsk_ocr_image_fixed.py "$VLLM_DIR/"
    echo "✓ Installed run_dpsk_ocr_image_fixed.py"
elif [ -f "$VLLM_DIR/run_dpsk_ocr_image_fixed.py" ]; then
    echo "✓ run_dpsk_ocr_image_fixed.py already exists"
else
    echo "⚠ run_dpsk_ocr_image_fixed.py not found"
fi

echo ""
echo "Step 5: Installing documentation..."
echo "========================================================================"

# Copy documentation files
for doc in QUICK_FIX_GUIDE.md ISSUE_110_FIX.md TROUBLESHOOTING.md SOLUTION_SUMMARY.md SOLUTION_README.md; do
    if [ -f "$doc" ]; then
        cp "$doc" .
        echo "✓ Installed $doc"
    fi
done

# Copy test script
if [ -f "test_vllm_compatibility.py" ]; then
    cp test_vllm_compatibility.py .
    chmod +x test_vllm_compatibility.py
    echo "✓ Installed test_vllm_compatibility.py"
fi

echo ""
echo "========================================================================"
echo "  Installation Complete!"
echo "========================================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Test your environment:"
echo "   python3 test_vllm_compatibility.py"
echo ""
echo "2. Use the fixed scripts:"
echo "   cd $VLLM_DIR"
echo "   python run_dpsk_ocr_image_fixed.py"
echo ""
echo "3. Read the documentation:"
echo "   - QUICK_FIX_GUIDE.md for fast solutions"
echo "   - ISSUE_110_FIX.md for detailed information"
echo "   - TROUBLESHOOTING.md for common issues"
echo ""
echo "Your original files are backed up in: $BACKUP_DIR"
echo ""
echo "If you encounter any issues, run:"
echo "   python3 test_vllm_compatibility.py"
echo ""
echo "========================================================================"
