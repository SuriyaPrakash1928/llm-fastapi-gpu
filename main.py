from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
import uvicorn

from app.schemas import ChatRequest
from app.loader import ModelManager
from app.generator import (
    generate_gguf_stream, generate_gguf_sync,
    generate_safetensors_stream, generate_safetensors_sync
)

# --- FastAPI Setup ---
app = FastAPI(title="Dual Model LLM API")

@app.on_event("startup")
async def startup_event():
    # Initialize the ModelManager and attach it to the app state
    model_manager = ModelManager()
    model_manager.load_models()
    app.state.model_manager = model_manager

# --- API Endpoints ---
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest, req: Request):
    model_manager = req.app.state.model_manager
    
    if request.quantization:
        if not model_manager.llm_gguf: 
            raise HTTPException(status_code=500, detail="GGUF model not loaded.")
        if request.streaming: 
            return StreamingResponse(
                generate_gguf_stream(request.prompt, request.max_tokens, request.temperature, model_manager), 
                media_type="text/plain"
            )
        else: 
            return {"response": generate_gguf_sync(request.prompt, request.max_tokens, request.temperature, model_manager)}
    else:
        if not model_manager.llm_safetensors: 
            raise HTTPException(status_code=500, detail="Safetensors model not loaded.")
        if request.streaming: 
            return StreamingResponse(
                generate_safetensors_stream(request.prompt, request.max_tokens, request.temperature, model_manager), 
                media_type="text/plain"
            )
        else: 
            return {"response": generate_safetensors_sync(request.prompt, request.max_tokens, request.temperature, model_manager)}

@app.get("/health")
async def health_check(req: Request):
    model_manager = req.app.state.model_manager
    return {
        "status": "healthy", 
        "gguf_loaded": model_manager.llm_gguf is not None, 
        "safetensors_loaded": model_manager.llm_safetensors is not None
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)