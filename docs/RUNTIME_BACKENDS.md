# Runtime Backends Specification

## Supported Backends

| Backend | Type | Platform | Best For |
|---|---|---|---|
| pytorch_eager | Local CPU/GPU | All | Development, flexibility |
| torch_compile | Local GPU | CUDA/ROCm | Speed with PyTorch models |
| llama_cpp_gguf | Local CPU/GPU | All | GGUF quantized models, universal |
| mlx | Local GPU | Apple Silicon | Mac hardware acceleration |
| onnx_runtime | Local CPU | All | INT8 quantized, edge deployment |
| tensorrt_edge_llm | Local GPU | Jetson/Orin | Jetson edge inference |
| tensorrt_llm | Local GPU | NVIDIA GPUs | Production RTX/Blackwell serving |
| vllm | Local GPU | NVIDIA GPUs | High-throughput batch serving |
| remote_aurelius | Remote | Any | Thin clients, split execution |

## Selection Flow

1. Detect hardware → 2. Detect RAM/VRAM/unified memory → 3. Inspect requested model → 4. Inspect local artifacts → 5. Inspect remote endpoints → 6. Choose backend → 7. Choose quantization → 8. Choose context budget → 9. Choose capability mode → 10. Choose skill preload policy → 11. Produce capability report

## Backend Capabilities Matrix

| Backend | Quantization | CUDA | Metal | Context | Batching |
|---|---|---|---|---|---|
| pytorch_eager | bf16/fp16/fp8/q8/q4 | Partial | MPS | Large | Yes |
| torch_compile | bf16/fp16/fp8 | Full | No | Large | Yes |
| llama_cpp_gguf | q3/q4/q5/q8 | Partial | Partial | Medium | Minimal |
| mlx | bf16/q4 | No | Full | Medium | Medium |
| onnx_runtime | int8/fp16/fp32 | Partial | Partial | Small | Minimal |
| tensorrt_edge_llm | int8/fp16/q4 | Full | No | Medium | Medium |
| tensorrt_llm | fp4/fp8/fp16/bf16 | Full | No | Large | High |
| vllm | fp8/fp16/bf16 | Full | No | Large | High |
| remote_aurelius | Server-defined | N/A | N/A | Unlimited | Server-side |
