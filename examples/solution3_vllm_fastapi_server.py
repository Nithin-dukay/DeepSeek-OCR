"""
Solution 3: Production-Ready Streaming with vLLM and FastAPI

This solution uses the vLLM implementation (already included in the repository)
with FastAPI to provide a production-ready streaming API.

Pros:
- Native async streaming support
- High performance (2500+ tokens/s on A100)
- Production-ready
- Easy integration with web frameworks
- Supports Server-Sent Events (SSE) and WebSocket

Cons:
- Requires vLLM installation
- Different API than HuggingFace transformers

Based on: DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_image.py
"""

import asyncio
import os
import sys
import json
from typing import Optional
from pathlib import Path

import torch
if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"

os.environ['VLLM_USE_V1'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from fastapi import FastAPI, WebSocket, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import time

# Add the vLLM directory to path
vllm_dir = Path(__file__).parent.parent / "DeepSeek-OCR-master" / "DeepSeek-OCR-vllm"
sys.path.insert(0, str(vllm_dir))

from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.model_executor.models.registry import ModelRegistry
from deepseek_ocr import DeepseekOCRForCausalLM
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCRProcessor

# Register the model
ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)

# Configuration
MODEL_PATH = 'deepseek-ai/DeepSeek-OCR'
CROP_MODE = True
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="DeepSeek-OCR Streaming API",
    description="Production-ready OCR API with streaming support",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine instance
engine: Optional[AsyncLLMEngine] = None


# Request/Response models
class OCRRequest(BaseModel):
    image_path: str
    prompt: str = "<image>\n<|grounding|>Convert the document to markdown."
    temperature: float = 0.0
    max_tokens: int = 8192


class OCRResponse(BaseModel):
    success: bool
    text: str
    processing_time: float
    token_count: int


async def get_engine():
    """Get or create the vLLM engine."""
    global engine
    
    if engine is None:
        print("Initializing vLLM engine...")
        engine_args = AsyncEngineArgs(
            model=MODEL_PATH,
            hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
            block_size=256,
            max_model_len=8192,
            enforce_eager=False,
            trust_remote_code=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.75,
        )
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        print("Engine initialized successfully!")
    
    return engine


async def stream_generate(image_path: str, prompt: str, temperature: float = 0.0, 
                         max_tokens: int = 8192):
    """
    Generate OCR output with streaming.
    
    Yields:
        str: Chunks of generated text as they become available
    """
    engine = await get_engine()
    
    # Load and process image
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image: {str(e)}")
    
    # Process image with DeepSeekOCRProcessor
    processor = DeepseekOCRProcessor()
    processed_image = processor.tokenize_with_images(
        images=[image], 
        bos=True, 
        eos=True, 
        cropping=CROP_MODE
    )
    
    # Prepare request
    request = {
        "prompt": prompt,
        "multi_modal_data": {"image": processed_image}
    }
    
    # Setup sampling parameters with n-gram logits processor
    logits_processors = [
        NoRepeatNGramLogitsProcessor(
            ngram_size=30, 
            window_size=90, 
            whitelist_token_ids={128821, 128822}  # <td>, </td>
        )
    ]
    
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        logits_processors=logits_processors,
        skip_special_tokens=False,
    )
    
    # Generate with streaming
    request_id = f"request-{int(time.time() * 1000)}"
    printed_length = 0
    
    async for request_output in engine.generate(request, sampling_params, request_id):
        if request_output.outputs:
            full_text = request_output.outputs[0].text
            new_text = full_text[printed_length:]
            
            if new_text:
                yield new_text
                printed_length = len(full_text)


@app.on_event("startup")
async def startup_event():
    """Initialize the engine on startup."""
    await get_engine()


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "DeepSeek-OCR Streaming API",
        "version": "1.0.0",
        "endpoints": {
            "POST /ocr": "Full OCR (non-streaming)",
            "POST /ocr/stream": "Streaming OCR with SSE",
            "POST /ocr/upload": "Upload image and get OCR",
            "POST /ocr/upload/stream": "Upload image and stream OCR",
            "WebSocket /ws/ocr": "WebSocket streaming OCR"
        }
    }


@app.post("/ocr", response_model=OCRResponse)
async def ocr_full(request: OCRRequest):
    """
    Full OCR endpoint (non-streaming).
    Returns complete result after generation finishes.
    """
    start_time = time.time()
    
    try:
        # Collect all chunks
        full_text = ""
        token_count = 0
        
        async for chunk in stream_generate(
            request.image_path, 
            request.prompt, 
            request.temperature,
            request.max_tokens
        ):
            full_text += chunk
            token_count += 1
        
        processing_time = time.time() - start_time
        
        return OCRResponse(
            success=True,
            text=full_text,
            processing_time=processing_time,
            token_count=token_count
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ocr/stream")
async def ocr_stream(request: OCRRequest):
    """
    Streaming OCR endpoint using Server-Sent Events (SSE).
    Streams tokens as they are generated.
    """
    
    async def generate():
        try:
            # Send start event
            yield f"data: {json.dumps({'status': 'started'})}\n\n"
            
            # Stream tokens
            async for chunk in stream_generate(
                request.image_path, 
                request.prompt,
                request.temperature,
                request.max_tokens
            ):
                yield f"data: {json.dumps({'token': chunk})}\n\n"
            
            # Send completion event
            yield f"data: {json.dumps({'status': 'completed'})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.post("/ocr/upload")
async def ocr_upload(
    file: UploadFile = File(...),
    prompt: str = "<image>\n<|grounding|>Convert the document to markdown."
):
    """
    Upload an image and get OCR result (non-streaming).
    """
    # Save uploaded file
    file_path = UPLOAD_DIR / f"{int(time.time() * 1000)}_{file.filename}"
    
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process with OCR
        request = OCRRequest(image_path=str(file_path), prompt=prompt)
        result = await ocr_full(request)
        
        # Clean up
        file_path.unlink()
        
        return result
    
    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ocr/upload/stream")
async def ocr_upload_stream(
    file: UploadFile = File(...),
    prompt: str = "<image>\n<|grounding|>Convert the document to markdown."
):
    """
    Upload an image and stream OCR result using SSE.
    """
    # Save uploaded file
    file_path = UPLOAD_DIR / f"{int(time.time() * 1000)}_{file.filename}"
    
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    async def generate():
        try:
            # Send start event
            yield f"data: {json.dumps({'status': 'started'})}\n\n"
            
            # Stream tokens
            async for chunk in stream_generate(str(file_path), prompt):
                yield f"data: {json.dumps({'token': chunk})}\n\n"
            
            # Send completion event
            yield f"data: {json.dumps({'status': 'completed'})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            # Clean up file
            if file_path.exists():
                file_path.unlink()
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.websocket("/ws/ocr")
async def websocket_ocr(websocket: WebSocket):
    """
    WebSocket endpoint for streaming OCR.
    
    Client sends: {"image_path": "path/to/image.jpg", "prompt": "..."}
    Server streams: {"token": "..."} or {"status": "..."} or {"error": "..."}
    """
    await websocket.accept()
    
    try:
        # Receive request
        data = await websocket.receive_json()
        image_path = data.get('image_path')
        prompt = data.get('prompt', '<image>\n<|grounding|>Convert the document to markdown.')
        
        if not image_path:
            await websocket.send_json({'error': 'image_path is required'})
            return
        
        # Send status
        await websocket.send_json({'status': 'processing'})
        
        # Stream tokens
        async for chunk in stream_generate(image_path, prompt):
            await websocket.send_json({'token': chunk})
        
        # Send completion
        await websocket.send_json({'status': 'completed'})
        
    except Exception as e:
        await websocket.send_json({'error': str(e)})
    finally:
        await websocket.close()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "engine_initialized": engine is not None
    }


if __name__ == '__main__':
    import uvicorn
    
    print("=" * 80)
    print("DeepSeek-OCR Streaming API Server")
    print("=" * 80)
    print("\nStarting server on http://0.0.0.0:8000")
    print("\nAvailable endpoints:")
    print("  - POST   /ocr              - Full OCR (non-streaming)")
    print("  - POST   /ocr/stream       - Streaming OCR with SSE")
    print("  - POST   /ocr/upload       - Upload and OCR (non-streaming)")
    print("  - POST   /ocr/upload/stream - Upload and stream OCR")
    print("  - WS     /ws/ocr           - WebSocket streaming")
    print("  - GET    /health           - Health check")
    print("  - GET    /docs             - API documentation")
    print("\n" + "=" * 80)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
