# Export Contracts — GGUF, MLX, ONNX, TensorRT-LLM

## Required Artifacts per Model

| Format | Swift | Forge | Atlas |
|---|---|---|---|
| safetensors | Required | Required | Required |
| GGUF Q3/Q4/Q5 | Required | Required | Optional/lite only |
| MLX 4-bit | Required | Required | Optional |
| ONNX INT8 | Required | Optional | No |
| TensorRT Edge | Optional | Required (Orin) | No |
| TensorRT-LLM FP8/FP4 | Optional | Required | Required |
| Remote container | Optional | Required | Required |

## Export Validation Checklist

Every exported artifact must pass:
1. Load — file loads without errors
2. Generate — produces text output
3. Tokenizer parity — same tokens as reference
4. Control token parity — all control tokens recognized
5. Memory protocol compatibility — memory ops work
6. Skill protocol compatibility — skill invocation works
7. Declared context test — handles declared context length
8. No silent fallback — produces valid output
9. Hardware smoke test — loads on target hardware class

## Export Paths

```
exports/
├── swift/
│   ├── safetensors/
│   ├── gguf-q3/
│   ├── gguf-q4/
│   ├── gguf-q5/
│   ├── onnx-int8/
│   └── mlx-4bit/
├── forge/
│   ├── safetensors/
│   ├── gguf-q4/
│   ├── gguf-q5/
│   ├── awq/
│   ├── gptq/
│   ├── mlx-4bit/
│   ├── tensorrt-edge/
│   └── tensorrt-llm-fp8/
└── atlas/
    ├── safetensors/
    ├── tensorrt-llm-fp8/
    ├── tensorrt-llm-fp4/
    ├── gguf-lite-q4/
    └── remote-container/
```

## CLI Export Commands

```
aurelius export gguf --model swift --quant q4 --output ./exports/
aurelius export mlx --model forge --output ./exports/
aurelius export onnx --model swift --quant int8 --output ./exports/
aurelius export tensorrt-llm --model atlas --fp8 --output ./exports/
aurelius export validate --all
```
