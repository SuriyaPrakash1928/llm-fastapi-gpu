# llm-fastapi-gpu
Two models safetensors and gguf model runs based on the quantization input from the user.

**Dual-Model GPU-Accelerated LLM API**
**Project Overview**
  This project is a high-performance, asynchronous FastAPI server designed to dynamically route inference requests to either a quantized GGUF model or a full-precision Safetensors model. It is strictly optimized for NVIDIA GPU acceleration, featuring automatic hardware detection, multi-GPU tensor splitting, and real-time streaming capabilities.
**Key Features**
  Dual-Model Routing: Switch between the fast, quantized GGUF model and the high-precision Safetensors model via a simple configuration flag.
  Streaming and Non-Streaming: Supports both standard JSON responses and real-time, token-by-token streaming.
  Automatic GPU Detection: Automatically detects NVIDIA GPUs, calculates available VRAM, and configures tensor splitting for multi-GPU setups.
  Low-VRAM Optimized: Includes specific optimizations for low-VRAM GPUs, including TF32 math, Scaled Dot-Product Attention, and FP16 fallbacks.
  Docker Ready: Includes an optimized Docker configuration that uses pre-compiled CUDA wheels to significantly reduce build times.
Project Structure
  The project is organized into a main application file that handles the FastAPI server and model loading logic, Docker configuration files for containerization, a requirements file for Python dependencies, a testing script, and a dedicated models directory for storing the AI weights and tokenizers.
**Local Setup Instructions**
  To run the project locally, you must first create and activate a virtual environment.
  Next, install PyTorch specifically compiled for CUDA support to ensure the Safetensors model utilizes the GPU.
  Then, install the llama-cpp-python library using the pre-compiled CUDA wheels. This step is crucial as it avoids the lengthy C++ compilation process required for standard installations.
  After that, install the remaining standard Python dependencies listed in the requirements file.
  Finally, execute the main application file to start the FastAPI server.
**Docker Setup Instructions**
  For containerized deployment, ensure that Docker Desktop and the NVIDIA Container Toolkit are installed on your host machine.
  Use the provided Docker Compose configuration to build and run the container. This setup automatically handles GPU passthrough, allowing the containerized application to access your physical NVIDIA hardware.
**API Usage and Parameters**
  The API exposes a single endpoint for chat completions. It accepts a JSON payload containing the following parameters:
  Prompt: The text input provided by the user for the AI to respond to.
  Quantization: A boolean flag. Setting this to true routes the request to the fast GGUF model. Setting it to false routes it to the high-precision Safetensors model.
  Streaming: A boolean flag. Setting this to true enables real-time token streaming. Setting it to false waits for the complete response before returning it.
  Max Tokens: An integer defining the maximum length of the generated response.
  Temperature: A float value that controls the randomness and creativity of the AI's output.
**Performance and Hardware Insights**
  It is a common misconception that having free VRAM equates to faster generation speeds. Even if you have plenty of free VRAM, the full-precision Safetensors model will stream slower than the quantized GGUF model. This is due to Memory Bandwidth, not VRAM Capacity.
  The GGUF model compresses the model weights, resulting in a small file size. The GPU memory bus can transfer this small amount of data very quickly, resulting in a high token generation rate.
  Conversely, the Safetensors model uses full-precision weights, resulting in a file size that is several times larger. The GPU memory bus must transfer significantly more data for every single token generated.
  Having free VRAM simply means the model fits on the GPU. It does not make the memory bus faster. For maximum streaming speed on GPUs with limited memory bandwidth, such as laptop GPUs, it is highly recommended to use the quantized GGUF model.
**Troubleshooting and Technical Fixes**
  Issue: The system logs indicate CPU fallback instead of GPU usage.
  This occurs when the inference library is installed without CUDA support and silently defaults to the CPU. To fix this, uninstall the library and reinstall it using the pre-compiled CUDA wheels.
  Issue: The application fails to load due to missing module errors.
  This happens when Windows is missing the underlying NVIDIA CUDA runtime files required by the pre-compiled library. To fix this, install the official NVIDIA CUDA Toolkit and the latest Microsoft Visual C++ Redistributable, then restart your computer to update the system environment variables.
  Issue: The full-precision model stutters or runs extremely slowly.
  This occurs when the code attempts to use a specific data type that older or entry-level GPUs do not support, causing it to silently fall back to the CPU. The application has been updated with a specific hardware check that forces a compatible data type on GPUs that lack support for newer formats, ensuring the model stays strictly on the GPU.
  Issue: The Task Manager shows zero percent GPU usage while generating text.
  Windows Task Manager updates its graphs once per second. Small models generate tokens so quickly that the GPU finishes the calculations and idles before the Task Manager can register the activity. To accurately monitor the GPU, use the official NVIDIA command-line interface tool to view real-time utilization metrics.

