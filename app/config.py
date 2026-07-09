import os

# Model paths (can be overridden via environment variables if needed)
GGUF_MODEL_PATH = os.getenv("GGUF_MODEL_PATH", "./models/qwen2.5.gguf")
SAFETENSORS_MODEL_PATH = os.getenv("SAFETENSORS_MODEL_PATH", "./models/qwen2.5-safetensors")

# Generation defaults
DEFAULT_MAX_TOKENS = 8000
DEFAULT_TEMPERATURE = 0.7