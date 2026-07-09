import os
import inspect
from llama_cpp import Llama
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import GGUF_MODEL_PATH, SAFETENSORS_MODEL_PATH
from .hardware import detect_nvidia_gpus, detect_intel_gpus, analyze_and_calculate_split

class ModelManager:
    """Encapsulates the state and loading logic for both LLM models."""
    def __init__(self):
        self.llm_gguf = None
        self.llm_safetensors = None
        self.tokenizer_safetensors = None

    def load_models(self):
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
                "verbose": False     # Shows GPU offloading logs
            }
            if tensor_split_ratio is not None:
                llama_kwargs["tensor_split"] = tensor_split_ratio
                
            if 'split_mode' in inspect.signature(Llama).parameters:
                llama_kwargs["split_mode"] = 2 
                
            self.llm_gguf = Llama(**llama_kwargs)
            print("[SUCCESS] GGUF model loaded.")
        else:
            print(f"[WARNING] GGUF model not found at {GGUF_MODEL_PATH}.")

        # 2. Load Safetensors Model (Original/Full Precision)
        if os.path.exists(SAFETENSORS_MODEL_PATH):
            print("\nLoading Safetensors model...")
            try:
                self.tokenizer_safetensors = AutoTokenizer.from_pretrained(SAFETENSORS_MODEL_PATH, trust_remote_code=True)
                
                if torch.cuda.is_available():
                    device = "cuda"
                    # NVIDIA T500 (Turing) does NOT support BF16. 
                    if torch.cuda.is_bf16_supported():
                        dtype = torch.bfloat16
                    else:
                        dtype = torch.float16
                else:
                    device = "cpu"
                    dtype = torch.float32

                print(f"[DEBUG] Target Device: {device} | Precision: {dtype}")
                    
                self.llm_safetensors = AutoModelForCausalLM.from_pretrained(
                    SAFETENSORS_MODEL_PATH,
                    torch_dtype=dtype,
                    device_map=device,
                    trust_remote_code=True,
                    attn_implementation="sdpa"
                )
                
                if device == "cuda":
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    print("[OPTIMIZATION] TF32 enabled for faster GPU math.")

                print(f"[DEBUG] Safetensors model is loaded on device: {next(self.llm_safetensors.parameters()).device}")
                print("[SUCCESS] Safetensors model loaded.")
            except Exception as e:
                print(f"[ERROR] Failed to load Safetensors model: {e}")
        else:
            print(f"[WARNING] Safetensors model not found at {SAFETENSORS_MODEL_PATH}.")