# Dual Model LLM API (FastAPI + GPU)

A high-performance, concurrent FastAPI backend serving two Large Language Models simultaneously: a quantized GGUF model (via `llama-cpp-python`) and a full-precision Safetensors model (via Hugging Face `transformers`). 

Designed for NVIDIA GPUs, this API supports both standard JSON responses and real-time token streaming, with built-in concurrency controls to handle multiple simultaneous users safely without crashing the GPU.

## 📋 Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Local Installation](#local-installation)
- [Lightning AI Cloud Deployment](#lightning-ai-cloud-deployment)
- [Usage & API Testing](#usage--api-testing)
- [Architecture & Concurrency](#architecture--concurrency)

---

## ✨ Features
- **Dual Model Support**: Switch between fast GGUF (quantized) and high-quality Safetensors (full precision) models on the fly via a single API endpoint.
- **GPU Acceleration**: Full CUDA support for both `llama-cpp-python` and PyTorch.
- **Streaming & Non-Streaming**: Supports real-time token-by-token streaming (`text/plain`) and standard JSON responses.
- **Concurrent Request Handling**: Uses `asyncio.to_thread` and `threading.Lock` to accept multiple web requests simultaneously without crashing the GPU (Out-Of-Memory) or blocking the event loop.
- **Cloud-Ready**: Fully tested and deployed on Lightning AI (L40S GPU) with proxy timeout bypasses.

---

## 🛠️ Prerequisites
- **Hardware**: NVIDIA GPU with CUDA support (minimum 8GB VRAM, recommended 16GB+).
- **Software**: Python 3.10+, Docker (optional), Git.
- **Cloud**: Lightning AI account (for cloud deployment).

---

## 💻 Local Installation

### Option 1: Using Docker (Recommended)
1. Clone the repository:
   ```bash
   git clone https://github.com/SuriyaPrakash1928/llm-fastapi-gpu.git
   cd llm-fastapi-gpu
   ```
2. Place your models in the `./models` directory:
   - `./models/qwen2.5.gguf`
   - `./models/qwen2.5-safetensors/`
3. Build and run using Docker Compose:
   ```bash
   docker compose up --build
   ```

### Option 2: Using Python directly
1. Install dependencies (including pre-compiled CUDA wheels for `llama-cpp-python`):
   ```bash
   pip install -r requirements.txt --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122
   ```
2. Fix CUDA library paths (if you encounter `libcudart.so.12` or `libcublas.so.12` errors):
   ```bash
   export LD_LIBRARY_PATH=$(find /path/to/your/conda/envs/.../site-packages/nvidia -type d -name "lib" | tr '\n' ':')$LD_LIBRARY_PATH
   ```
3. Run the server:
   ```bash
   python app.py
   ```

---

## ⚡ Lightning AI Cloud Deployment (Step-by-Step)

If you want to deploy this on a cloud GPU (like the NVIDIA L40S) without managing local hardware, follow these steps:

### 1. Create the Studio
1. Go to [Lightning AI](https://lightning.ai) and log in.
2. Click **+ New Studio**.
3. Name it (e.g., `llm-fastapi-gpu-l40s`).
4. Under **Hardware/Machine Type**, select **L40S** (or A10G/T4).
5. Click **Start**.

### 2. Setup the Environment (Inside the Cloud Terminal)
Once the Studio loads, open the **Terminal** and run the following commands sequentially:

**Step A: Clone the Repository**
```bash
git clone https://github.com/SuriyaPrakash1928/llm-fastapi-gpu.git
cd llm-fastapi-gpu
```

**Step B: Install Dependencies**
```bash
pip install -r requirements.txt --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122
```

**Step C: Fix Missing CUDA Libraries (Crucial for Lightning AI)**
Lightning AI's default environment sometimes hides CUDA libraries from Python. Run this to fix `libcudart` and `libcublas` errors:
```bash
export LD_LIBRARY_PATH=$(find /system/conda/miniconda3/envs/cloudspace/lib/python3.12/site-packages/nvidia -type d -name "lib" | tr '\n' ':')$LD_LIBRARY_PATH

# Save it permanently so you don't have to run it every time you restart the Studio
echo 'export LD_LIBRARY_PATH=$(find /system/conda/miniconda3/envs/cloudspace/lib/python3.12/site-packages/nvidia -type d -name "lib" | tr "\n" ":")$LD_LIBRARY_PATH' >> ~/.bashrc
```

**Step D: Download the Models**
Create the models folder and download your weights (replace URLs with your actual HuggingFace model links):
```bash
mkdir -p models

# Download GGUF
wget -O models/qwen2.5.gguf "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/blob/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"

# Download Safetensors
pip install huggingface_hub
huggingface-cli download "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/model.safetensors"
```

### 3. Start the API Server
```bash
python app.py
```
*Wait for the logs to show `[SUCCESS] GGUF model loaded.` and `[SUCCESS] Safetensors model loaded.`*

### 4. Expose the Port
1. Look at the **Right-Hand Sidebar** in Lightning AI.
2. Click the **Ports** icon (🔌).
3. Find **Port 8000** and copy the public URL (e.g., `https://8000-xxxxx.cloudspaces.litng.ai`).

---

## 🚀 Usage & API Testing

You can test the API using Python, PowerShell, or cURL. 
**Note:** For cloud deployments (like Lightning AI), it is highly recommended to use `streaming: true` to prevent proxy timeouts when multiple users are queued.

### Python Test Script (Parallel Streaming)
Save this as `test_api.py` on your local machine:

```python
import requests
import concurrent.futures

API_URL = "https://YOUR_LIGHTNING_URL/v1/chat/completions"

def stream_request(user_id, prompt, quantization):
    payload = {
        "prompt": prompt,
        "quantization": quantization,
        "streaming": True, # Must be True for cloud deployments
        "max_tokens": 150,
        "temperature": 0.7
    }
    print(f"[User {user_id}] Live: ", end="", flush=True)
    with requests.post(API_URL, json=payload, stream=True) as r:
        for chunk in r.iter_content(chunk_size=1, decode_unicode=True):
            if chunk: print(chunk, end="", flush=True)
    print(f"\n[User {user_id}] Done!")

# Fire 3 parallel requests
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    executor.submit(stream_request, 1, "Write a poem about the ocean.", True)
    executor.submit(stream_request, 2, "What is the capital of France?", False)
    executor.submit(stream_request, 3, "Tell a short story about a cat.", True)
```

### PowerShell (Standard JSON)
```powershell
$url = "https://YOUR_LIGHTNING_URL/v1/chat/completions"
$body = @{
    prompt = "Write a short poem about AI"
    quantization = $true
    streaming = $false
    max_tokens = 150
    temperature = 0.7
} | ConvertTo-Json -Compress

Invoke-RestMethod -Uri $url -Method Post -Body $body -ContentType "application/json"
```

---

## 🏗️ Architecture & Concurrency

### How it handles multiple users:
Standard LLM inference blocks the CPU/GPU. If User 1 and User 2 send requests at the exact same millisecond, the server would normally freeze or crash with an Out-Of-Memory (OOM) error.

This API solves this using two mechanisms:
1. **`asyncio.to_thread` & `iterate_in_threadpool`**: Pushes the heavy synchronous GPU generation into background threads. This keeps the FastAPI main event loop free to instantly accept new HTTP connections (preventing 504 Gateway Timeouts).
2. **`threading.Lock()` (`gpu_lock`)**: Acts as a bouncer for the physical GPU. It forces the background threads to take turns. User 1 generates, then User 2, then User 3. This completely prevents CUDA OOM crashes while maintaining a highly responsive web server.

---
