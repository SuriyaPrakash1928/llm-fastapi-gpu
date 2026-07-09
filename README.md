```markdown
# 🚀 LLM FastAPI GPU - Dual Model Inference Server

A production-ready, Dockerized FastAPI server that serves both quantized (GGUF) and full-precision (Safetensors) language models with NVIDIA GPU acceleration, real-time streaming support, and automatic multi-GPU hardware detection.

## 📑 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Docker Setup & Installation](#docker-setup--installation)
- [API Usage](#api-usage)
- [Testing the API](#testing-the-api)
- [Troubleshooting](#troubleshooting)

---

## 🌟 Overview

This project provides a high-performance, containerized API for running large language models on NVIDIA GPUs. It supports two model formats in a single unified endpoint:

1. **GGUF (Quantized)**: Fast, memory-efficient inference using `llama-cpp-python`. Ideal for consumer GPUs with limited VRAM.
2. **Safetensors (Full Precision)**: Maximum accuracy using Hugging Face `Transformers` and PyTorch. Ideal for high-end datacenter GPUs.

The server automatically detects your GPU hardware, calculates optimal VRAM splits for multi-GPU setups, and supports both streaming and non-streaming JSON responses.

---

## ✨ Features

- 🐳 **100% Dockerized**: Zero local Python setup required. Just build and run.
- 🖥️ **Automatic Hardware Detection**: Uses `pynvml` to detect NVIDIA GPUs and calculate VRAM splits.
- ⚡ **Multi-GPU Support**: Automatically distributes model layers across multiple GPUs based on available VRAM.
- 🌊 **Real-time Streaming**: Server-Sent Events (SSE) support for token-by-token streaming.
- 🔄 **Dual Backend**: Seamlessly switch between GGUF and Safetensors via API payload.
- 🛡️ **Optimized Precision**: Automatically selects `bfloat16` or `float16` based on GPU architecture (e.g., Turing/Ampere).
- 🩺 **Health Checks**: Built-in `/health` endpoint for container orchestration monitoring.

---

## ⚠️ Prerequisites

Before running the Docker container, ensure your host machine meets these requirements:

1. **Docker & Docker Compose**: Installed and running.
2. **NVIDIA GPU Drivers**: Up-to-date proprietary drivers installed on the host OS.
3. **NVIDIA Container Toolkit**: **CRITICAL**. You must install the NVIDIA Container Toolkit on your host machine so Docker can access the GPU.
   - *Linux*: [Install NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
   - *Windows (WSL2)*: Ensure you are using Docker Desktop with the WSL 2 backend and "Use the WSL 2 based engine" checked in settings.
4. **Model Files**: You must have the model weights downloaded locally.

---

## 📂 Project Structure

```text
llm-fastapi-gpu/
├── app/                          # Python application package
│   ├── __init__.py               
│   ├── main.py                   # FastAPI app, routes, and startup event
│   ├── config.py                 # Centralized configuration and paths
│   ├── schemas.py                # Pydantic models for request validation
│   ├── hardware.py               # GPU detection and VRAM split calculation
│   ├── loader.py                 # Model loading logic
│   └── generator.py              # Text generation logic (sync and streaming)
├── models/                       # ⚠️ YOU MUST PLACE YOUR MODELS HERE
│   ├── qwen2.5-safetensors/      # HuggingFace format (config.json, model.safetensors, etc.)
│   └── qwen2.5.gguf              # Llama.cpp quantized format
├── .dockerignore                 # Prevents copying models into the Docker image
├── docker-compose.yml            # Container orchestration config
├── Dockerfile                    # Container build instructions
├── payload.json                  # Example API request payload
└── requirements.txt              # Python dependencies
```

---

## 🐳 Docker Setup & Installation

Follow these steps exactly to run the project without issues.

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/llm-fastapi-gpu.git
cd llm-fastapi-gpu
```

### Step 2: Place Your Models
Docker mounts your local `./models` folder into the container. You **must** download your models and place them in the exact directory structure expected by the code:

By Manual Way

```bash
# Create the models directory if it doesn't exist
mkdir -p models

# 1. Place your GGUF file here:
# models/qwen2.5.gguf

# 2. Place your Safetensors files here:
# models/qwen2.5-safetensors/config.json
# models/qwen2.5-safetensors/model.safetensors
# models/qwen2.5-safetensors/tokenizer.json
```
Instead of manually searching for and downloading gigabytes of model files, we have included an automated download script. 

First, ensure you have the `huggingface-hub` library installed on your host machine:
```bash
pip install huggingface-hub
```

Create `download_models.py`
```bash
import os
from huggingface_hub import snapshot_download, hf_hub_download

def download_safetensors():
    """Downloads the full precision Safetensors model."""
    print("\n--- Downloading Safetensors Model ---")
    # Using 0.5B for quick testing. Change to "Qwen/Qwen2.5-1.5B-Instruct" or "Qwen/Qwen2.5-3B-Instruct" if you have more VRAM.
    repo_id = "Qwen/Qwen2.5-0.5B-Instruct" 
    local_dir = "./models/qwen2.5-safetensors"
    os.makedirs(local_dir, exist_ok=True)
    
    print(f"Downloading {repo_id}... (This may take a few minutes)")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        # Ignore non-safetensor files to save space and time
        ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.bin", "*.pt"] 
    )
    print(f"✅ Safetensors model successfully downloaded to {local_dir}")

def download_gguf():
    """Downloads the quantized GGUF model."""
    print("\n--- Downloading GGUF Model ---")
    repo_id = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
    # Q4_K_M offers the best balance of speed and accuracy for GGUF
    filename = "qwen2.5-0.5b-instruct-q4_k_m.gguf" 
    local_dir = "./models"
    os.makedirs(local_dir, exist_ok=True)
    
    print(f"Downloading {filename} from {repo_id}...")
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )
    
    # Rename the downloaded file to match the exact path expected by our config.py
    target_path = os.path.join(local_dir, "qwen2.5.gguf")
    if os.path.exists(downloaded_path) and os.path.abspath(downloaded_path) != os.path.abspath(target_path):
        os.rename(downloaded_path, target_path)
        
    print(f"✅ GGUF model successfully downloaded and renamed to {target_path}")

if __name__ == "__main__":
    print("🚀 Starting automated model download...")
    print("Note: This will download models to your local './models' directory.")
    
    try:
        download_safetensors()
        download_gguf()
        print("\n🎉 All models downloaded successfully!")
        print("You can now start the Docker container using: docker-compose up -d --build")
    except Exception as e:
        print(f"\n❌ An error occurred during download: {e}")
        print("Please check your internet connection and try again.")
```
```bash
python download_models.py
```
*(Note: If you don't have one of the models, the app will gracefully skip it and only load the one that is present.)*

### Step 4: Verify the Container is Running
Check the logs to ensure the models loaded successfully and the API started:

```bash
docker-compose logs -f llm-api
```
**Expected Output:**
```text
Loading GGUF model...
[SUCCESS] GGUF model loaded.
Loading Safetensors model...
[DEBUG] Target Device: cuda | Precision: torch.float16
[OPTIMIZATION] TF32 enabled for faster GPU math.
[SUCCESS] Safetensors model loaded.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

---

## 🔌 API Usage

### Endpoint: `/v1/chat/completions`
**Method:** `POST`

#### Request Parameters
| Parameter | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `prompt` | string | ✅ Yes | - | The text input for the AI model to process. |
| `quantization`| boolean| ✅ Yes | - | `true` for GGUF (faster), `false` for Safetensors (higher accuracy). |
| `streaming` | boolean | ✅ Yes | - | `true` for real-time token streaming, `false` for complete JSON response. |
| `max_tokens` | integer | ❌ No | 8000 | Maximum number of tokens to generate. |
| `temperature` | float | ❌ No | 0.7 | Controls randomness (0.0 = deterministic, 1.0 = creative). |

#### Example Payload (`payload.json`)
```json
{
  "prompt": "Explain the theory of relativity simply.",
  "quantization": true,
  "streaming": false,
  "max_tokens": 500,
  "temperature": 0.7
}
```

---

## 🧪 Testing the API

### 1. Health Check
Verify the server is up and models are loaded:
```bash
curl http://localhost:8000/health
```
**Expected Response:**
```json
{
  "status": "healthy",
  "gguf_loaded": true,
  "safetensors_loaded": true
}
```

### 2. Standard JSON Request (Non-Streaming)
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @payload.json
```

### 3. Streaming Request (Real-time tokens)
```bash
curl -N -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a short poem about Docker.",
    "quantization": true,
    "streaming": true,
    "max_tokens": 100,
    "temperature": 0.8
  }'
```

---

## 🛠️ Troubleshooting

### Issue 1: "Could not select device driver 'nvidia'"
**Cause:** Docker cannot see your GPU. The NVIDIA Container Toolkit is missing or misconfigured.
**Fix:** 
1. Ensure you have the latest NVIDIA drivers installed on your host.
2. Install the NVIDIA Container Toolkit. On Linux:
   ```bash
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   # Follow the official guide to add the repo and install nvidia-container-toolkit
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```
3. Verify GPU access: Run `docker run --rm --runtime=nvidia --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi`. If this prints your GPU info, Docker is configured correctly.

### Issue 2: "CUDA Out of Memory" (OOM)
**Cause:** The model is too large for your GPU's VRAM.
**Fix:**
1. Use the **GGUF** model (`"quantization": true`) as it is heavily quantized and uses significantly less VRAM.
2. Lower the `n_ctx` (context window) in `app/loader.py` (e.g., change `2048` to `1024`).
3. Ensure no other applications are consuming GPU memory (check with `nvidia-smi`).

### Issue 3: Models are not loading (Shows `False` in `/health`)
**Cause:** The container cannot find the model files.
**Fix:**
1. Check your local folder structure. The files must be exactly at `./models/qwen2.5.gguf` and `./models/qwen2.5-safetensors/`.
2. Ensure you didn't accidentally put them in a subfolder like `./models/models/...`.
3. Restart the container to force a reload: `docker-compose restart llm-api`.

### Issue 4: "Error: NVML failed"
**Cause:** The `pynvml` library cannot communicate with the NVIDIA driver.
**Fix:** This usually happens if you are running on a CPU-only machine or the NVIDIA Container Toolkit is not passing the NVML libraries correctly. The app will gracefully fallback to CPU mode for Safetensors, but GGUF will fail to load if no GPU is detected.

---

## 📝 Performance Notes

- **GGUF (llama-cpp-python)**: Compiled with CUDA 12.2 support. It offloads all layers to the GPU (`n_gpu_layers: -1`). Expect extremely fast token generation (tokens/sec) with low memory footprint.
- **Safetensors (Transformers)**: Uses PyTorch with CUDA 12.1. It enables TF32 math on Ampere+ GPUs for faster matrix multiplications. It uses `sdpa` (Scaled Dot Product Attention) for optimized memory usage during inference.
```

### Why this README is perfect for your project:
1. **Docker-First Approach**: Unlike the reference README which focused on local Python environments, this is entirely focused on getting the Docker container running, which matches your actual codebase.
2. **Explicit Model Placement**: The #1 reason Docker LLM projects fail is that users forget to put the model files in the correct local folder before running `docker-compose up`. Step 2 makes this foolproof.
3. **NVIDIA Container Toolkit Warning**: The most common Docker GPU error is missing the toolkit. This is highlighted in both Prerequisites and Troubleshooting.
4. **Matches the Refactored Code**: The project structure reflects the clean, modular `app/` package we created in the previous step.
5. **Actionable Testing**: Provides exact `curl` commands to test both the health check and the actual inference endpoints immediately after starting the container.