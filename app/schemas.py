from pydantic import BaseModel

class ChatRequest(BaseModel):
    prompt: str
    quantization: bool  # True for .gguf, False for .safetensors
    streaming: bool     # True for streaming response, False for standard JSON
    max_tokens: int = 8000
    temperature: float = 0.7