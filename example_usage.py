"""
Example usage of Enhanced DeepSeek-OCR
Demonstrates various features and use cases
"""

# Note: Import commented out for demonstration
# Uncomment when running with actual dependencies
# from enhanced_ocr_inference import DeepSeekOCRInference


def example_basic_usage():
    """Basic OCR extraction"""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    code = '''
# Initialize model
ocr = DeepSeekOCRInference(model_path="deepseek-ai/deepseek-ocr")

# Extract text from image
text = ocr.infer("path/to/image.jpg")

print(f"Extracted {len(text)} characters")
print(text)
'''
    print(code)


def example_with_anti_repetition():
    """OCR with enhanced anti-repetition settings"""
    print("\n" + "=" * 60)
    print("Example 2: Anti-Repetition Guardrails")
    print("=" * 60)
    
    code = '''
ocr = DeepSeekOCRInference()

# Use aggressive anti-repetition for problematic documents
text = ocr.infer(
    "historical_newspaper.jpg",
    repetition_penalty=1.5,      # Stronger penalty
    no_repeat_ngram_size=15,     # Larger n-gram window
    detect_repetition=True,       # Auto-detect loops
    retry_on_failure=True,        # Retry if repetition found
    max_retries=2,                # Up to 2 retries
)

print(f"Extracted {len(text)} characters with anti-repetition")
'''
    print(code)


def example_newspaper_columns():
    """Process multi-column newspaper layout"""
    print("\n" + "=" * 60)
    print("Example 3: Newspaper Column Splitting")
    print("=" * 60)
    
    code = '''
ocr = DeepSeekOCRInference()

# Split wide newspaper into columns
text = ocr.infer_with_column_split(
    "newspaper.jpg",
    num_columns=2,
    repetition_penalty=1.3,
    max_new_tokens=8192,  # Longer documents
)

print("Extracted text from 2-column layout")
print(f"Total length: {len(text)} characters")
'''
    print(code)


def example_batch_processing():
    """Process multiple images"""
    print("\n" + "=" * 60)
    print("Example 4: Batch Processing")
    print("=" * 60)
    
    code = '''
ocr = DeepSeekOCRInference()

image_paths = [
    "document1.jpg",
    "document2.jpg",
    "document3.jpg",
]

# Process all images
results = ocr.infer_batch(
    image_paths,
    repetition_penalty=1.2,
    detect_repetition=True,
)

for path, text in zip(image_paths, results):
    print(f"{path}: {len(text)} characters")
'''
    print(code)


def example_custom_prompts():
    """Use custom prompts for specific document types"""
    print("\n" + "=" * 60)
    print("Example 5: Custom Prompts")
    print("=" * 60)
    
    code = '''
ocr = DeepSeekOCRInference()

# For tables
table_text = ocr.infer(
    "table.jpg",
    prompt="Extract the table data, preserving rows and columns."
)

# For handwritten text
handwritten_text = ocr.infer(
    "handwritten.jpg",
    prompt="Carefully transcribe all handwritten text."
)

# For forms
form_text = ocr.infer(
    "form.jpg",
    prompt="Extract all text from this form, including field labels and values."
)
'''
    print(code)


def example_with_config():
    """Load settings from configuration file"""
    print("\n" + "=" * 60)
    print("Example 6: Using Configuration File")
    print("=" * 60)
    
    code = '''
import yaml

# Load config
with open("ocr_config.yaml") as f:
    config = yaml.safe_load(f)

ocr = DeepSeekOCRInference()

# Use newspaper preset
newspaper_config = config["newspaper"]
text = ocr.infer(
    "newspaper.jpg",
    max_new_tokens=newspaper_config["max_new_tokens"],
    repetition_penalty=newspaper_config["repetition_penalty"],
    no_repeat_ngram_size=newspaper_config["no_repeat_ngram_size"],
)
'''
    print(code)


def example_api_integration():
    """Example FastAPI integration"""
    print("\n" + "=" * 60)
    print("Example 7: FastAPI Integration")
    print("=" * 60)
    
    code = '''
from fastapi import FastAPI, File, UploadFile
from enhanced_ocr_inference import DeepSeekOCRInference
import tempfile

app = FastAPI()
ocr = DeepSeekOCRInference()

@app.post("/ocr")
async def extract_text(file: UploadFile = File(...)):
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    
    # Extract text with anti-repetition
    text = ocr.infer(
        tmp_path,
        detect_repetition=True,
        retry_on_failure=True,
    )
    
    return {"text": text, "length": len(text)}

# Run with: uvicorn example_usage:app --reload
'''
    
    print("FastAPI integration example:")
    print(code)


def example_gradio_interface():
    """Example Gradio interface"""
    print("\n" + "=" * 60)
    print("Example 8: Gradio Interface")
    print("=" * 60)
    
    code = '''
import gradio as gr
from enhanced_ocr_inference import DeepSeekOCRInference

ocr = DeepSeekOCRInference()

def process_image(image, rep_penalty, ngram_size):
    text = ocr.infer(
        image,
        repetition_penalty=rep_penalty,
        no_repeat_ngram_size=ngram_size,
        detect_repetition=True,
        retry_on_failure=True,
    )
    return text

demo = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="filepath", label="Upload Image"),
        gr.Slider(1.0, 2.0, value=1.2, label="Repetition Penalty"),
        gr.Slider(5, 20, value=10, step=1, label="N-gram Size"),
    ],
    outputs=gr.Textbox(label="Extracted Text", lines=20),
    title="Enhanced DeepSeek-OCR",
    description="Upload an image to extract text with anti-repetition guardrails",
)

demo.launch()
'''
    
    print("Gradio interface example:")
    print(code)


def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print(" Enhanced DeepSeek-OCR - Usage Examples")
    print("=" * 70)
    print("\nThese examples demonstrate the enhanced features:")
    print("  • Returns text (not None)")
    print("  • Anti-repetition guardrails")
    print("  • Chat template support")
    print("  • Automatic retry logic")
    print("  • Column splitting for newspapers")
    print("  • Batch processing")
    print("  • Custom prompts")
    print("  • Configuration presets")
    print("  • API integrations")
    print("\n" + "=" * 70)
    
    # Show all examples
    example_basic_usage()
    example_with_anti_repetition()
    example_newspaper_columns()
    example_batch_processing()
    example_custom_prompts()
    example_with_config()
    example_api_integration()
    example_gradio_interface()
    
    print("\n" + "=" * 70)
    print("To run these examples, ensure you have:")
    print("  1. Installed dependencies: pip install -r requirements.txt")
    print("  2. Downloaded the model: deepseek-ai/deepseek-ocr")
    print("  3. Prepared your images")
    print("\nFor more details, see ENHANCED_OCR_GUIDE.md")
    print("=" * 70)


if __name__ == "__main__":
    main()
