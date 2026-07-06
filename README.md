# LLM FastAPI GPU - Dual Model Inference Server

A production-ready FastAPI server that serves both quantized (GGUF) and full-precision (Safetensors) language models with GPU acceleration, streaming support, and automatic hardware detection.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Docker Setup](#docker-setup)
- [Running the Application](#running-the-application)
- [API Usage](#api-usage)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Performance Notes](#performance-notes)

---

## Overview

This project provides a high-performance API for running large language models on NVIDIA GPUs. It supports two model formats:

1. **GGUF (Quantized)**: Fast, memory-efficient inference using llama-cpp-python
2. **Safetensors (Full Precision)**: Maximum accuracy using Hugging Face Transformers

The server automatically detects your GPU hardware, optimizes memory usage, and supports both streaming and non-streaming responses.

---

## Features

- Automatic NVIDIA GPU detection and VRAM calculation
- Multi-GPU support with automatic tensor splitting
- Real-time token streaming (Server-Sent Events)
- Support for both quantized and full-precision models
- Optimized for low-VRAM GPUs (tested on NVIDIA T500 4GB)
- Docker support with GPU passthrough
- Fast build times using pre-compiled CUDA wheels
- Health check endpoint for monitoring

---

## Prerequisites

Before you begin, ensure you have the following installed:

**For Local Installation:**
- Python 3.10 or higher
- NVIDIA GPU with CUDA support
- NVIDIA CUDA Toolkit 12.1 or higher
- NVIDIA drivers (version 595.95 or compatible)
- Git (for cloning the repository)

**For Docker Installation:**
- Docker Desktop (version 4.0 or higher)
- NVIDIA Container Toolkit
- NVIDIA drivers (version 595.95 or compatible)

---

## Installation

### Step 1: Clone or Create Project Directory

Create a new directory for your project and navigate to it:

```bash
mkdir llm-fastapi-gpu
cd llm-fastapi-gpu
```

### Step 2: Create Virtual Environment

Set up a Python virtual environment to manage dependencies:

```bash
python -m venv venv
```

Activate the virtual environment:

**On Windows:**
```bash
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
source venv/bin/activate
```

### Step 3: Install PyTorch with CUDA Support

Install PyTorch with CUDA 12.1 support. This is critical for GPU acceleration:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Important:** Do not use the standard pip install command for PyTorch, as it will install the CPU-only version.

### Step 4: Install llama-cpp-python with CUDA

Install the pre-compiled CUDA version of llama-cpp-python to avoid lengthy compilation:

```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

**Note:** This uses pre-compiled wheels for CUDA 12.4, which reduces installation time from 10-15 minutes to under 1 minute.

### Step 5: Install Remaining Dependencies

Create a file named `requirements.txt` with the following content:

```
fastapi
uvicorn[standard]
pydantic
pynvml
transformers
accelerate
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

### Step 6: Install Microsoft Visual C++ Redistributable (Windows Only)

If you're on Windows, download and install the latest Visual C++ Redistributable:

1. Visit: https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist
2. Download the x64 version (vc_redist.x64.exe)
3. Run the installer
4. Restart your computer if prompted

### Step 7: Install NVIDIA CUDA Toolkit (If Not Already Installed)

If you encounter DLL loading errors, install the CUDA Toolkit:

1. Visit: https://developer.nvidia.com/cuda-12-4-0-download-archive
2. Select your operating system and architecture
3. Download and run the installer
4. Choose "Custom Installation" and ensure Runtime libraries are selected
5. Restart your computer after installation

---

## Docker Setup

### Step 1: Install NVIDIA Container Toolkit

**On Ubuntu/Debian:**

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

**On Windows:**

NVIDIA Container Toolkit is included with Docker Desktop when you enable WSL 2 backend and have NVIDIA drivers installed.

### Step 2: Create Dockerfile

Create a file named `Dockerfile` with the optimized configuration that uses pre-compiled wheels.

### Step 3: Create docker-compose.yml

Create a file named `docker-compose.yml` with GPU passthrough configuration.

### Step 4: Build and Run Docker Container

Build and start the container:

```bash
docker compose up --build
```

The first build will take approximately 2-3 minutes to download base images and dependencies. Subsequent builds will be much faster.

---

## Running the Application

### Local Installation

Start the FastAPI server:

```bash
python app.py
```

The server will start on `http://0.0.0.0:8000`

You should see output indicating:
- GPU detection and VRAM information
- Model loading progress
- Server startup confirmation

### Docker Installation

Start the container:

```bash
docker compose up
```

To run in detached mode (background):

```bash
docker compose up -d
```

To stop the container:

```bash
docker compose down
```

To rebuild after code changes:

```bash
docker compose up --build
```

### Verify Installation

Check if the server is running:

```bash
curl http://localhost:8000/health
```

You should receive a JSON response indicating the health status and which models are loaded.

---

## API Usage

### Endpoint

**POST** `/v1/chat/completions`

### Request Parameters

**prompt** (string, required)
The text input for the AI model to process.

**quantization** (boolean, required)
Set to `true` to use the GGUF quantized model (faster).
Set to `false` to use the Safetensors full-precision model (more accurate).

**streaming** (boolean, required)
Set to `true` to receive tokens as they are generated.
Set to `false` to wait for the complete response.

**max_tokens** (integer, optional, default: 8000)
Maximum number of tokens to generate in the response.

**temperature** (float, optional, default: 0.7)
Controls randomness in generation. Lower values (0.1-0.3) make output more deterministic. Higher values (0.7-1.0) make it more creative.

### Example Request (Streaming with GGUF)

```json
{
  "prompt": "Explain machine learning",
  "quantization": true,
  "streaming": true,
  "max_tokens": 256,
  "temperature": 0.7
}
```

### Example Request (Non-Streaming with Safetensors)

```json
{
  "prompt": "Write a Python function",
  "quantization": false,
  "streaming": false,
  "max_tokens": 512,
  "temperature": 0.3
}
```

### Response Format

**Streaming Response:**
Returns a continuous stream of text tokens. Each token is sent as soon as it's generated.

**Non-Streaming Response:**
Returns a JSON object with the complete response:

```json
{
  "response": "Complete AI-generated text here..."
}
```

---

## Testing

### Create Test Script

Create a file named `test_api.py` to test the API functionality.

### Run Test Script

With the server running, execute:

```bash
python test_api.py
```

### Test with cURL

**Test GGUF model (non-streaming):**

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" -H "Content-Type: application/json" -d "{\"prompt\": \"Hello\", \"quantization\": true, \"streaming\": false}"
```

**Test Safetensors model (streaming):**

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" -H "Content-Type: application/json" -d "{\"prompt\": \"Hello\", \"quantization\": false, \"streaming\": true}"
```

### Monitor GPU Usage

While the API is generating text, monitor GPU utilization in a separate terminal:

```bash
nvidia-smi -l 1
```

This refreshes GPU statistics every second. You should see:
- GPU-Util between 30-50% during generation
- Memory usage matching your model size
- Process list showing your Python application

---

## Troubleshooting

### Problem: "Could not find module llama.dll"

**Cause:** Missing CUDA runtime libraries or Visual C++ Redistributable.

**Solution:**
1. Install NVIDIA CUDA Toolkit 12.4
2. Install Microsoft Visual C++ Redistributable (x64)
3. Restart your computer
4. Try running the application again

### Problem: "CPU : SSE3 = 1 | AVX2 = 1" in logs

**Cause:** llama-cpp-python was installed without CUDA support.

**Solution:**
```bash
pip uninstall llama-cpp-python -y
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

### Problem: Safetensors model runs slowly or on CPU

**Cause:** Model falling back to CPU due to incompatible data types.

**Solution:**
The application has been updated to automatically detect GPU capabilities and use FP16 instead of BF16 on GPUs that don't support it (like NVIDIA T500). Ensure you're using the latest version of app.py.

### Problem: Out of Memory (OOM) errors

**Cause:** Model too large for available VRAM.

**Solution:**
1. Use the GGUF quantized model (quantization: true)
2. Reduce max_tokens parameter
3. Close other GPU-intensive applications
4. Use a smaller model (e.g., 0.5B instead of 7B parameters)

### Problem: Task Manager shows 0% GPU usage

**Cause:** Windows Task Manager updates too slowly to capture fast GPU operations.

**Solution:**
Use nvidia-smi instead:
```bash
nvidia-smi -l 1
```

This provides accurate real-time GPU utilization.

### Problem: Docker build takes too long

**Cause:** Compiling llama-cpp-python from source.

**Solution:**
The provided Dockerfile uses pre-compiled wheels. Ensure you're using the latest Dockerfile that includes:
```
RUN pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

### Problem: "CUDA out of memory" error

**Cause:** Both models loaded simultaneously exceed VRAM.

**Solution:**
1. The application loads both models at startup. With 4GB VRAM, this is tight.
2. Consider using only one model type
3. Use smaller models (0.5B or 1.5B instead of 7B)
4. Reduce context window size (n_ctx parameter)

---

## Performance Notes

### Understanding Speed Differences

You may notice that the GGUF model streams significantly faster than the Safetensors model, even with plenty of free VRAM. This is due to **Memory Bandwidth**, not VRAM Capacity.

**GGUF Model (Quantized):**
- File size: ~0.25 GB (for 0.5B model)
- Data transfer per token: Minimal
- Expected speed: 30-40 tokens/second on T500
- Best for: Fast, interactive chat

**Safetensors Model (Full Precision):**
- File size: ~1.0 GB (for 0.5B model)
- Data transfer per token: 4x more data
- Expected speed: 8-12 tokens/second on T500
- Best for: Maximum accuracy, complex reasoning

**Key Insight:**
Having 2.5 GB of free VRAM means the model fits, but it doesn't make the memory bus faster. The GPU must transfer model weights from VRAM to compute units for every token generated. Larger models (in file size) require more data transfer, resulting in slower generation.

### Optimization Tips

1. **Use GGUF for daily tasks:** Faster streaming, lower memory usage
2. **Use Safetensors for precision tasks:** When accuracy is more important than speed
3. **Monitor with nvidia-smi:** Task Manager is not accurate for AI workloads
4. **Keep models small:** On 4GB VRAM, stick to 0.5B-1.5B models for best performance

### Expected Performance on NVIDIA T500 (4GB VRAM)

**GGUF Model (Qwen 2.5 0.5B):**
- Time to first token: ~0.5 seconds
- Generation speed: 30-40 tokens/second
- VRAM usage: ~0.5 GB
- GPU Util: 40-60%

**Safetensors Model (Qwen 2.5 0.5B):**
- Time to first token: ~1 second
- Generation speed: 8-12 tokens/second
- VRAM usage: ~1.8 GB
- GPU Util: 30-50%

---

## Additional Resources

- **FastAPI Documentation:** https://fastapi.tiangolo.com/
- **llama-cpp-python:** https://github.com/abetlen/llama-cpp-python
- **Hugging Face Transformers:** https://huggingface.co/docs/transformers
- **NVIDIA CUDA Toolkit:** https://developer.nvidia.com/cuda-toolkit
- **Docker GPU Support:** https://docs.docker.com/config/containers/resource_constraints/#gpu

---

## License

MIT License - Feel free to use and modify for your projects.

---

## Support

If you encounter issues not covered in this README:

1. Check the Troubleshooting section above
2. Verify your NVIDIA drivers are up to date
3. Ensure CUDA Toolkit is properly installed
4. Check that your GPU is detected by running `nvidia-smi`
5. Review the application logs for specific error messages

---

**Last Updated:** July 2026
**Tested On:** Windows 11, Python 3.10, NVIDIA T500 4GB, CUDA 12.4
