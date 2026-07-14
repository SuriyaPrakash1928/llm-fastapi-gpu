import os

# Use the absolute Linux path because the code runs INSIDE the container
GGUF_MODEL_PATH = os.getenv("GGUF_MODEL_PATH", "/app/models/qwen2.5.gguf")
SAFETENSORS_MODEL_PATH = os.getenv("SAFETENSORS_MODEL_PATH", "/app/models/qwen2.5-safetensors")

# Generation defaults
DEFAULT_MAX_TOKENS = 8000
DEFAULT_TEMPERATURE = 0.7