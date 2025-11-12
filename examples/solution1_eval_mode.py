"""
Solution 1: Using eval_mode=True to get the full output from model.infer()

This is the simplest solution - just set eval_mode=True to make the method
return the generated text instead of returning None.

Pros:
- Simple, works immediately
- No code changes needed
- Returns clean text output

Cons:
- No streaming (client waits for full generation)
- Not suitable for very long documents
"""

from transformers import AutoModel, AutoTokenizer
import torch
import os

# Configuration
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model_name = 'deepseek-ai/DeepSeek-OCR'

# Load model and tokenizer
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name, 
    _attn_implementation='flash_attention_2', 
    trust_remote_code=True, 
    use_safetensors=True
)
model = model.eval().cuda().to(torch.bfloat16)

# Setup
prompt = "<image>\n<|grounding|>Convert the document to markdown."
image_file = 'your_image.jpg'  # Replace with your image path
output_path = 'output'

print(f"\nProcessing image: {image_file}")
print(f"Prompt: {prompt}\n")

# Solution: Use eval_mode=True to get the output as a return value
result = model.infer(
    tokenizer, 
    prompt=prompt, 
    image_file=image_file, 
    output_path=output_path, 
    base_size=1024, 
    image_size=640, 
    crop_mode=True, 
    save_results=False,  # Don't save to file
    test_compress=False,
    eval_mode=True  # KEY: This makes the method return the output
)

# Now you can use the result
if result:
    print("=" * 80)
    print("GENERATED OUTPUT:")
    print("=" * 80)
    print(result)
    print("=" * 80)
    
    # Example: Send to client
    # In a web framework, you could do:
    # return jsonify({"text": result})
    # or
    # return Response(result, mimetype='text/plain')
    
    # Example: Save to custom location
    with open('output_custom.txt', 'w', encoding='utf-8') as f:
        f.write(result)
    print("\nOutput saved to: output_custom.txt")
    
    # Example: Process the result
    lines = result.split('\n')
    print(f"\nGenerated {len(lines)} lines of text")
    print(f"Total characters: {len(result)}")
else:
    print("No output generated (result is None)")
    print("Make sure eval_mode=True is set!")


# Example usage in a Flask API
"""
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/ocr', methods=['POST'])
def ocr_endpoint():
    image_path = request.json.get('image_path')
    prompt = request.json.get('prompt', '<image>\\nFree OCR.')
    
    result = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=image_path,
        output_path='temp',
        base_size=1024,
        image_size=640,
        crop_mode=True,
        eval_mode=True  # Returns the output
    )
    
    return jsonify({
        'success': True,
        'text': result,
        'length': len(result)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
"""


# Example usage in a FastAPI
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class OCRRequest(BaseModel):
    image_path: str
    prompt: str = '<image>\\nFree OCR.'

@app.post('/ocr')
async def ocr_endpoint(request: OCRRequest):
    try:
        result = model.infer(
            tokenizer,
            prompt=request.prompt,
            image_file=request.image_path,
            output_path='temp',
            base_size=1024,
            image_size=640,
            crop_mode=True,
            eval_mode=True  # Returns the output
        )
        
        return {
            'success': True,
            'text': result,
            'length': len(result)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
"""
