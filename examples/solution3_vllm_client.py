"""
Client examples for consuming the streaming OCR API (Solution 3)

This file demonstrates various ways to consume the streaming API:
1. Python requests with SSE
2. JavaScript/Browser with EventSource
3. WebSocket client
4. File upload with streaming
"""

import requests
import json
import asyncio
import websockets
from pathlib import Path


# ============================================================================
# Example 1: Python Client with Server-Sent Events (SSE)
# ============================================================================

def example_sse_client(image_path: str, prompt: str = None):
    """
    Example: Consume streaming OCR using Server-Sent Events.
    """
    print("=" * 80)
    print("Example 1: SSE Streaming Client")
    print("=" * 80)
    
    url = "http://localhost:8000/ocr/stream"
    
    if prompt is None:
        prompt = "<image>\n<|grounding|>Convert the document to markdown."
    
    payload = {
        "image_path": image_path,
        "prompt": prompt
    }
    
    print(f"\nStreaming OCR for: {image_path}")
    print(f"Prompt: {prompt}\n")
    print("-" * 80)
    
    try:
        # Make streaming request
        response = requests.post(
            url,
            json=payload,
            stream=True,
            headers={"Accept": "text/event-stream"}
        )
        
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            return
        
        # Process SSE stream
        full_text = ""
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                
                # SSE format: "data: {json}"
                if line.startswith('data: '):
                    data_str = line[6:]  # Remove "data: " prefix
                    
                    try:
                        data = json.loads(data_str)
                        
                        if 'token' in data:
                            # Print token in real-time
                            print(data['token'], end='', flush=True)
                            full_text += data['token']
                        
                        elif 'status' in data:
                            if data['status'] == 'started':
                                print("[Stream started]")
                            elif data['status'] == 'completed':
                                print("\n\n[Stream completed]")
                        
                        elif 'error' in data:
                            print(f"\n\nError: {data['error']}")
                            return
                    
                    except json.JSONDecodeError:
                        pass
        
        print("-" * 80)
        print(f"\nTotal characters received: {len(full_text)}")
        
        return full_text
    
    except Exception as e:
        print(f"Error: {e}")
        return None


# ============================================================================
# Example 2: Non-Streaming Client (Full Response)
# ============================================================================

def example_full_client(image_path: str, prompt: str = None):
    """
    Example: Get full OCR result without streaming.
    """
    print("=" * 80)
    print("Example 2: Non-Streaming Client")
    print("=" * 80)
    
    url = "http://localhost:8000/ocr"
    
    if prompt is None:
        prompt = "<image>\n<|grounding|>Convert the document to markdown."
    
    payload = {
        "image_path": image_path,
        "prompt": prompt
    }
    
    print(f"\nProcessing: {image_path}")
    print("Waiting for complete result...\n")
    
    try:
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            
            print("-" * 80)
            print(result['text'])
            print("-" * 80)
            print(f"\nProcessing time: {result['processing_time']:.2f}s")
            print(f"Token count: {result['token_count']}")
            
            return result['text']
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
    
    except Exception as e:
        print(f"Error: {e}")
        return None


# ============================================================================
# Example 3: File Upload with Streaming
# ============================================================================

def example_upload_stream(file_path: str, prompt: str = None):
    """
    Example: Upload a file and stream OCR results.
    """
    print("=" * 80)
    print("Example 3: File Upload with Streaming")
    print("=" * 80)
    
    url = "http://localhost:8000/ocr/upload/stream"
    
    if prompt is None:
        prompt = "<image>\n<|grounding|>Convert the document to markdown."
    
    print(f"\nUploading and streaming: {file_path}\n")
    print("-" * 80)
    
    try:
        # Prepare file upload
        with open(file_path, 'rb') as f:
            files = {'file': (Path(file_path).name, f, 'image/jpeg')}
            data = {'prompt': prompt}
            
            # Make streaming request
            response = requests.post(
                url,
                files=files,
                data=data,
                stream=True,
                headers={"Accept": "text/event-stream"}
            )
        
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            return
        
        # Process SSE stream
        full_text = ""
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                
                if line.startswith('data: '):
                    data_str = line[6:]
                    
                    try:
                        data = json.loads(data_str)
                        
                        if 'token' in data:
                            print(data['token'], end='', flush=True)
                            full_text += data['token']
                        
                        elif 'status' in data:
                            if data['status'] == 'completed':
                                print("\n\n[Upload and streaming completed]")
                        
                        elif 'error' in data:
                            print(f"\n\nError: {data['error']}")
                            return
                    
                    except json.JSONDecodeError:
                        pass
        
        print("-" * 80)
        print(f"\nTotal characters received: {len(full_text)}")
        
        return full_text
    
    except Exception as e:
        print(f"Error: {e}")
        return None


# ============================================================================
# Example 4: WebSocket Client
# ============================================================================

async def example_websocket_client(image_path: str, prompt: str = None):
    """
    Example: Use WebSocket for streaming OCR.
    """
    print("=" * 80)
    print("Example 4: WebSocket Client")
    print("=" * 80)
    
    uri = "ws://localhost:8000/ws/ocr"
    
    if prompt is None:
        prompt = "<image>\n<|grounding|>Convert the document to markdown."
    
    print(f"\nConnecting to WebSocket: {uri}")
    print(f"Processing: {image_path}\n")
    print("-" * 80)
    
    try:
        async with websockets.connect(uri) as websocket:
            # Send request
            await websocket.send(json.dumps({
                "image_path": image_path,
                "prompt": prompt
            }))
            
            # Receive streaming response
            full_text = ""
            async for message in websocket:
                data = json.loads(message)
                
                if 'token' in data:
                    print(data['token'], end='', flush=True)
                    full_text += data['token']
                
                elif 'status' in data:
                    if data['status'] == 'processing':
                        print("[Processing started]")
                    elif data['status'] == 'completed':
                        print("\n\n[WebSocket stream completed]")
                        break
                
                elif 'error' in data:
                    print(f"\n\nError: {data['error']}")
                    break
            
            print("-" * 80)
            print(f"\nTotal characters received: {len(full_text)}")
            
            return full_text
    
    except Exception as e:
        print(f"Error: {e}")
        return None


# ============================================================================
# Example 5: JavaScript/Browser Client (Code Example)
# ============================================================================

def example_javascript_client():
    """
    Example: JavaScript code for browser-based streaming.
    """
    print("=" * 80)
    print("Example 5: JavaScript/Browser Client")
    print("=" * 80)
    
    js_code = '''
// Example 1: Using EventSource for SSE
function streamOCR(imagePath, prompt) {
    const url = 'http://localhost:8000/ocr/stream';
    
    // Make POST request to get streaming endpoint
    fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            image_path: imagePath,
            prompt: prompt || '<image>\\n<|grounding|>Convert the document to markdown.'
        })
    })
    .then(response => {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        function readStream() {
            reader.read().then(({ done, value }) => {
                if (done) {
                    console.log('Stream complete');
                    return;
                }
                
                const chunk = decoder.decode(value);
                const lines = chunk.split('\\n');
                
                lines.forEach(line => {
                    if (line.startsWith('data: ')) {
                        const data = JSON.parse(line.substring(6));
                        
                        if (data.token) {
                            // Display token in real-time
                            document.getElementById('output').textContent += data.token;
                        } else if (data.status === 'completed') {
                            console.log('OCR completed');
                        } else if (data.error) {
                            console.error('Error:', data.error);
                        }
                    }
                });
                
                readStream();
            });
        }
        
        readStream();
    })
    .catch(error => console.error('Error:', error));
}

// Example 2: File upload with streaming
function uploadAndStream(file, prompt) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('prompt', prompt || '<image>\\nFree OCR.');
    
    fetch('http://localhost:8000/ocr/upload/stream', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        function readStream() {
            reader.read().then(({ done, value }) => {
                if (done) return;
                
                const chunk = decoder.decode(value);
                const lines = chunk.split('\\n');
                
                lines.forEach(line => {
                    if (line.startsWith('data: ')) {
                        const data = JSON.parse(line.substring(6));
                        if (data.token) {
                            document.getElementById('output').textContent += data.token;
                        }
                    }
                });
                
                readStream();
            });
        }
        
        readStream();
    });
}

// Example 3: WebSocket streaming
function streamWithWebSocket(imagePath, prompt) {
    const ws = new WebSocket('ws://localhost:8000/ws/ocr');
    
    ws.onopen = () => {
        ws.send(JSON.stringify({
            image_path: imagePath,
            prompt: prompt || '<image>\\nFree OCR.'
        }));
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.token) {
            document.getElementById('output').textContent += data.token;
        } else if (data.status === 'completed') {
            console.log('OCR completed');
            ws.close();
        } else if (data.error) {
            console.error('Error:', data.error);
            ws.close();
        }
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
}

// HTML Example
/*
<!DOCTYPE html>
<html>
<head>
    <title>OCR Streaming Demo</title>
    <style>
        #output {
            white-space: pre-wrap;
            font-family: monospace;
            padding: 20px;
            border: 1px solid #ccc;
            min-height: 400px;
        }
    </style>
</head>
<body>
    <h1>DeepSeek-OCR Streaming Demo</h1>
    
    <div>
        <input type="file" id="fileInput" accept="image/*">
        <button onclick="handleUpload()">Upload and Stream</button>
    </div>
    
    <div id="output"></div>
    
    <script>
        function handleUpload() {
            const file = document.getElementById('fileInput').files[0];
            if (file) {
                document.getElementById('output').textContent = '';
                uploadAndStream(file, '<image>\\n<|grounding|>Convert the document to markdown.');
            }
        }
    </script>
</body>
</html>
*/
'''
    
    print(js_code)


# ============================================================================
# Main: Run Examples
# ============================================================================

def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("DeepSeek-OCR Streaming API Client Examples")
    print("=" * 80)
    print("\nMake sure the server is running:")
    print("  python examples/solution3_vllm_fastapi_server.py")
    print("\n" + "=" * 80)
    
    # Example image path (replace with your actual image)
    image_path = "your_image.jpg"
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code != 200:
            print("\n⚠️  Server is not responding. Please start the server first.")
            return
    except requests.exceptions.RequestException:
        print("\n⚠️  Cannot connect to server. Please start the server first:")
        print("     python examples/solution3_vllm_fastapi_server.py")
        return
    
    print("\n✅ Server is running!\n")
    
    # Uncomment the examples you want to run:
    
    # Example 1: SSE Streaming
    # example_sse_client(image_path)
    
    # Example 2: Full Response (Non-streaming)
    # example_full_client(image_path)
    
    # Example 3: File Upload with Streaming
    # example_upload_stream(image_path)
    
    # Example 4: WebSocket
    # asyncio.run(example_websocket_client(image_path))
    
    # Example 5: JavaScript code
    example_javascript_client()
    
    print("\n" + "=" * 80)
    print("To run the examples, uncomment the desired example in main()")
    print("and provide a valid image path.")
    print("=" * 80)


if __name__ == '__main__':
    main()
