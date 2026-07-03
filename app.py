import os
import time
import platform
import subprocess
import inspect
from threading import Thread

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

# Llama CPP for GGUF
from llama_cpp import Llama

# Transformers for Safetensors
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# 1. Import pynvml for NVIDIA hardware detection
try:
    import pynvml # type: ignore[import]
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

def detect_nvidia_gpus():
    nvidia_gpus = []
    if not PYNVML_AVAILABLE: return nvidia_gpus
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes): name = name.decode('utf-8')
            vram_gb = info.total / (1024**3)
            nvidia_gpus.append({"id": i, "name": name, "vram_gb": round(vram_gb, 2), "backend": "CUDA"})
        pynvml.nvmlShutdown()
    except Exception as e: print(f"[ERROR] NVML failed: {e}")
    return nvidia_gpus

def detect_intel_gpus():
    intel_gpus = []
    if platform.system() == "Windows":
        try:
            output = subprocess.check_output(['wmic', 'path', 'win32_videocontroller', 'get', 'name'], text=True, stderr=subprocess.DEVNULL)
            for line in output.strip().split('\n')[1:]:
                if "Intel" in line and line.strip(): intel_gpus.append({"name": line.strip(), "backend": "Vulkan/OpenCL"})
        except Exception: pass
    return intel_gpus

def analyze_and_calculate_split(nvidia_gpus, intel_gpus):
    if not nvidia_gpus: return None
    if len(nvidia_gpus) == 1: return None
    total_vram = sum(gpu["vram_gb"] for gpu in nvidia_gpus)
    return [gpu["vram_gb"] / total_vram for gpu in nvidia_gpus]

# --- FastAPI Setup ---
app = FastAPI(title="Dual Model LLM API")

class ChatRequest(BaseModel):
    prompt: str
    quantization: bool  # True for .gguf, False for .safetensors
    streaming: bool     # True for streaming response, False for standard JSON
    max_tokens: int = 8000
    temperature: float = 0.7

llm_gguf = None
llm_safetensors = None
tokenizer_safetensors = None

# Update these paths to your actual models
GGUF_MODEL_PATH = "./models/qwen2.5.gguf"
SAFETENSORS_MODEL_PATH = "./models/qwen2.5-safetensors" 

@app.on_event("startup")
async def startup_event():
    global llm_gguf, llm_safetensors, tokenizer_safetensors
    
    nvidia_gpus = detect_nvidia_gpus()
    intel_gpus = detect_intel_gpus()
    tensor_split_ratio = analyze_and_calculate_split(nvidia_gpus, intel_gpus)

    # 1. Load GGUF Model (Quantized)
    if os.path.exists(GGUF_MODEL_PATH):
        print("\nLoading GGUF model...")
        llama_kwargs = {
            "model_path": GGUF_MODEL_PATH,
            "n_gpu_layers": -1,  # Strictly run on GPU
            "n_ctx": 2048,       # Optimized context window
            "verbose": False      # Shows GPU offloading logs
        }
        if tensor_split_ratio is not None:
            llama_kwargs["tensor_split"] = tensor_split_ratio
            
        if 'split_mode' in inspect.signature(Llama).parameters:
            llama_kwargs["split_mode"] = 2 
            
        llm_gguf = Llama(**llama_kwargs)
        print("[SUCCESS] GGUF model loaded.")
    else:
        print(f"[WARNING] GGUF model not found at {GGUF_MODEL_PATH}.")

    # 2. Load Safetensors Model (Original/Full Precision)
    if os.path.exists(SAFETENSORS_MODEL_PATH):
        print("\nLoading Safetensors model...")
        try:
            tokenizer_safetensors = AutoTokenizer.from_pretrained(SAFETENSORS_MODEL_PATH, trust_remote_code=True)
            
            if torch.cuda.is_available():
                device = "cuda"
                # NVIDIA T500 (Turing) does NOT support BF16. 
                # We must fallback to FP16 on the GPU, NOT the CPU!
                if torch.cuda.is_bf16_supported():
                    torch_dtype = torch.bfloat16
                else:
                    torch_dtype = torch.float16
            else:
                device = "cpu"
                torch_dtype = torch.float32

            print(f"[DEBUG] Target Device: {device} | Precision: {torch_dtype}")
                
            llm_safetensors = AutoModelForCausalLM.from_pretrained(
                SAFETENSORS_MODEL_PATH,
                torch_dtype=torch_dtype,
                device_map=device,
                trust_remote_code=True,
                attn_implementation="sdpa"
            )
            
            if device == "cuda":
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print("[OPTIMIZATION] TF32 enabled for faster GPU math.")

            # Debug print to confirm GPU usage
            print(f"[DEBUG] Safetensors model is loaded on device: {next(llm_safetensors.parameters()).device}")
            print("[SUCCESS] Safetensors model loaded.")
        except Exception as e:
            print(f"[ERROR] Failed to load Safetensors model: {e}")
    else:
        print(f"[WARNING] Safetensors model not found at {SAFETENSORS_MODEL_PATH}.")

# --- Helper & Generation Functions ---
def get_safetensors_input(prompt: str):
    messages = [{"role": "user", "content": prompt}]
    if tokenizer_safetensors.chat_template:
        text = tokenizer_safetensors.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt
    return tokenizer_safetensors(text, return_tensors="pt").to(llm_safetensors.device)

def generate_gguf_stream(prompt: str, max_tokens: int, temperature: float):
    response = llm_gguf.create_chat_completion(messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens, temperature=temperature, stream=True)
    for chunk in response:
        delta = chunk['choices'][0]['delta']
        if 'content' in delta: yield delta['content']

def generate_gguf_sync(prompt: str, max_tokens: int, temperature: float) -> str:
    response = llm_gguf.create_chat_completion(messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens, temperature=temperature, stream=False)
    return response['choices'][0]['message']['content']

def generate_safetensors_stream(prompt: str, max_tokens: int, temperature: float):
    inputs = get_safetensors_input(prompt)
    streamer = TextIteratorStreamer(tokenizer_safetensors, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = dict(
        **inputs, 
        streamer=streamer,
        max_new_tokens=max_tokens,
        do_sample=temperature > 0, 
        pad_token_id=tokenizer_safetensors.eos_token_id, 
        eos_token_id=tokenizer_safetensors.eos_token_id
    )
    if temperature > 0: 
        generation_kwargs["temperature"] = temperature
        
    thread = Thread(target=llm_safetensors.generate, kwargs=generation_kwargs)
    thread.start()
    
    # OPTIMIZATION: Filter out empty strings/whitespace to prevent UI stutter
    for new_text in streamer: 
        if new_text:  # Only yield if there is actual text
            yield new_text
            
    thread.join()

def generate_safetensors_sync(prompt: str, max_tokens: int, temperature: float) -> str:
    inputs = get_safetensors_input(prompt)
    generation_kwargs = dict(**inputs, max_new_tokens=max_tokens, do_sample=temperature > 0, pad_token_id=tokenizer_safetensors.eos_token_id, eos_token_id=tokenizer_safetensors.eos_token_id)
    if temperature > 0: generation_kwargs["temperature"] = temperature
    outputs = llm_safetensors.generate(**generation_kwargs)
    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    return tokenizer_safetensors.decode(generated_tokens, skip_special_tokens=True)

# --- API Endpoints ---
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    if request.quantization:
        if not llm_gguf: raise HTTPException(status_code=500, detail="GGUF model not loaded.")
        if request.streaming: return StreamingResponse(generate_gguf_stream(request.prompt, request.max_tokens, request.temperature), media_type="text/plain")
        else: return {"response": generate_gguf_sync(request.prompt, request.max_tokens, request.temperature)}
    else:
        if not llm_safetensors: raise HTTPException(status_code=500, detail="Safetensors model not loaded.")
        if request.streaming: return StreamingResponse(generate_safetensors_stream(request.prompt, request.max_tokens, request.temperature), media_type="text/plain")
        else: return {"response": generate_safetensors_sync(request.prompt, request.max_tokens, request.temperature)}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "gguf_loaded": llm_gguf is not None, "safetensors_loaded": llm_safetensors is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)