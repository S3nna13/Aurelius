# Hardware Profiles v2

## Profile Matrix

| Hardware | CPU Arch | RAM | GPU VRAM | Swift | Forge | Atlas | Primary Backend |
|---|---|---|---|---|---|---|---|
| Jetson Nano 2GB | arm64 | 2GB | Shared | Native Q3 | Split/Remote | Split/Remote | ONNX/TensorRT-Edge |
| Jetson Nano 4GB | arm64 | 4GB | Shared | Native Q4 | Extreme offload | Split/Remote | ONNX/TensorRT-Edge |
| Jetson Xavier NX | arm64 | 8GB | Shared | Native | Quantized/offload | Split/Remote | TensorRT-Edge |
| Jetson Orin Nano | arm64 | 8GB | Shared | Native | Quantized/split | Split/Remote | TensorRT-Edge |
| Jetson AGX Orin | arm64 | 16-32GB | Shared | Native | Native Q4 | Offload/split | TensorRT-LLM |
| Mac Silicon 8GB | arm64 | 8GB Unified | Shared | Native | Quantized | Split/Remote | MLX |
| Mac Silicon 16GB | arm64 | 16GB Unified | Shared | Native | Q4 local | Offload/split | MLX |
| Mac Silicon 32GB | arm64 | 32GB Unified | Shared | Native | Native Q4 | Q4 offload | MLX |
| Mac Silicon 64GB | arm64 | 64GB Unified | Shared | Native | Native | Q4 offload | MLX |
| Mac Ultra 128GB+ | arm64 | 128GB+ Unified | Shared | Native | Native | Native Q4 | MLX |
| RTX 8GB | x86_64 | 16GB+ | 8GB | Native | Q4/offload | Split/Remote | PyTorch/llama.cpp |
| RTX 12GB | x86_64 | 16GB+ | 12GB | Native | Q4 local | Split/Remote | PyTorch/llama.cpp |
| RTX 24GB | x86_64 | 16GB+ | 24GB | Native | Native Q4 | Q4 offload | vLLM/TensorRT-LLM |
| RTX 6000 Ada 48GB | x86_64 | 64GB+ | 48GB | Native | Native | Q4/native | TensorRT-LLM |
| RTX PRO 6000 BW 96GB | x86_64 | 128GB+ | 96GB | Native | Native | Native FP4 | TensorRT-LLM |
| Multi-GPU workstation | x86_64 | 128GB+ | 2x48GB+ | Native | Native | Native full | vLLM/TensorRT-LLM |

## Profile ID Convention

`{platform}_{memory_class}gb`

Examples: `mac_silicon_32gb`, `jetson_nano_4gb`, `rtx_24gb`, `linux_128gb`

## Platform Detection Rules

### Mac Silicon
- `platform.machine() == "arm64"` and `platform.system() == "Darwin"`
- Unified memory: `sysctl -n hw.memsize`
- MLX available: `import mlx`

### Jetson
- `/proc/device-tree/model` contains "Jetson" or "NVIDIA"
- Models: Nano, Xavier NX, Orin Nano, AGX Orin
- Unified memory always

### Desktop GPU
- `nvidia-smi --query-gpu=name,memory.total`
- CUDA version via `nvcc --version`
- TensorRT via `trtexec --version`
