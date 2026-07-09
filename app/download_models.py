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