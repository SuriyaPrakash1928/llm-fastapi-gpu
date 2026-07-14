# Use the CUDA runtime image
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3 1

WORKDIR /app

# 1. Install PyTorch with CUDA 12.1 support (Official stable index for CUDA 12.x)
# Note: cu121 wheels are fully compatible with the CUDA 12.2 runtime base image.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# 3. Install requirements AND llama-cpp-python using the pre-compiled CUDA 12.2 wheels
RUN pip install --no-cache-dir -r requirements.txt \
    llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122

# 4. Copy ALL application code from the root directory (INCLUDING main.py)
COPY . .
COPY models/qwen2.5.gguf /app/models/qwen2.5.gguf

# 5. Ensure models directory exists
RUN mkdir -p /app/models

EXPOSE 8000

# Start the app (looks for main.py in the current WORKDIR: /app)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]