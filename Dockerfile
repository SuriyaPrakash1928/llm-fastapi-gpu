# Use the CUDA runtime image (much smaller than 'devel' since we aren't compiling C++ code)
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies (No need for cmake or build-essential anymore!)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3 1

WORKDIR /app

# 1. Install PyTorch with CUDA 12.1 support
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. Copy requirements
COPY requirements.txt .

# 3. Install requirements AND llama-cpp-python using the pre-compiled CUDA 12.2 wheels
# The --extra-index-url tells pip to grab the GPU-enabled llama-cpp-python wheel directly!
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122

# Copy application code
COPY app.py .
RUN mkdir -p /app/models

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]