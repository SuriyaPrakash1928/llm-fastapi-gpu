import platform
import subprocess

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
    except Exception as e: 
        print(f"[ERROR] NVML failed: {e}")
    return nvidia_gpus

def detect_intel_gpus():
    intel_gpus = []
    if platform.system() == "Windows":
        try:
            output = subprocess.check_output(['wmic', 'path', 'win32_videocontroller', 'get', 'name'], text=True, stderr=subprocess.DEVNULL)
            for line in output.strip().split('\n')[1:]:
                if "Intel" in line and line.strip(): 
                    intel_gpus.append({"name": line.strip(), "backend": "Vulkan/OpenCL"})
        except Exception: 
            pass
    return intel_gpus

def analyze_and_calculate_split(nvidia_gpus, intel_gpus):
    if not nvidia_gpus: return None
    if len(nvidia_gpus) == 1: return None
    total_vram = sum(gpu["vram_gb"] for gpu in nvidia_gpus)
    return [gpu["vram_gb"] / total_vram for gpu in nvidia_gpus]