# RAM/VRAM-Aware Runtime — Memory Budget Policy

## Principle

OOM must never be the first fallback. The system monitors memory before, during, and after model loading, and degrades gracefully through a defined ladder.

## Memory Budget Manager

Tracks all memory consumers:

| Consumer | Swift | Forge | Atlas |
|---|---|---|---|
| Model weights | 0.5-1.5 GB | 3-8 GB | 14-32 GB |
| KV cache | 0.2-1 GB | 1-4 GB | 4-16 GB |
| Runtime memory | 0.5 GB | 0.5-1 GB | 1-2 GB |
| Vision/VLM buffers | 0 (none) | 0.5-1 GB | 1-4 GB |
| Prefix cache | disabled | 0.5-2 GB | 2-8 GB |
| Skill runtime | 0.1-0.5 GB | 0.2-1 GB | 0.5-2 GB |

Total system tracking:
- System RAM
- GPU VRAM (dedicated)
- Mac unified memory
- Jetson unified memory
- CPU shared memory
- OS reserved memory
- Safety margin (reserve)

## Budget Report

Every runtime profile produces a budget report:

```json
{
  "total_memory_gb": 32,
  "reserved_for_os_gb": 4,
  "available_for_aurelius_gb": 28,
  "budget": {
    "weights_gb": 7.2,
    "kv_cache_gb": 3.1,
    "runtime_memory_gb": 1.0,
    "vision_gb": 1.5,
    "prefix_cache_gb": 0.4,
    "skill_runtime_gb": 0.2,
    "safety_margin_gb": 2.0,
    "free_gb": 12.6
  },
  "pressure_level": "low",
  "recommended_model": "forge",
  "recommended_artifact": "forge-q4-mlx",
  "max_context": 32768
}
```

## Pressure Levels

| Level | Threshold | Action |
|---|---|---|
| low | < 50% used | Normal operation, all features |
| moderate | 50-70% used | Monitor closely, prepare for action |
| high | 70-85% used | Begin degradation ladder |
| critical | 85-95% used | Active degradation in progress |
| emergency | > 95% used | Last resort measures |

## Degradation Ladder (11 Steps)

When pressure exceeds the high threshold, the system degrades in strict order:

1. **Unload inactive skills** — Release memory from skills not actively in use
2. **Reduce batch size** — Drop parallel generation batches to 1
3. **Reduce context budget** — Shrink max context from 128K → 64K → 32K → 16K
4. **Quantize KV cache** — Apply KIVI-style 2-bit/4-bit KV quantization
5. **Shrink active prompt** — Compress or truncate non-essential context
6. **Evict prefix cache** — Drop cached prefix/suffix blocks
7. **Offload cold KV to CPU** — Move least-recently-accessed KV to unified/RAM
8. **Downshift/disable vision** — Turn off VLM encoder or reduce image resolution
9. **Switch artifact** — Fall back to smaller/quantized model artifact (Atlas → Forge → Swift)
10. **Switch to split/remote** — Offload inference to remote endpoint
11. **Ask user before cancel** — Present options: cancel, retry with smaller task, or wait for resources

## Memory Policies

| Policy | Profile Target |
|---|---|
| `conservative` | Jetson Nano/4GB, Mac 8GB, RTX 8GB — small context, aggressive quant, no VLM, no heavy neural memory |
| `balanced` | Default — Mac 16-32GB, RTX 12-24GB — moderate context, Q4/Q5, prefix cache, chunked prefill, CUA with verifier |
| `performance` | Mac 64GB+, RTX 4090/5090 — larger context, higher precision, larger batch, MTP, skills preloaded |
| `frontier` | RTX 6000 Ada/Blackwell, Mac Ultra — maximum context, full multimodal, Atlas mode, full CUA, skill composition |

## OOM Prevention

1. Budget check BEFORE model load — if insufficient, recommend appropriate artifact
2. Monitoring DURING generation — watch pressure levels every N steps
3. Preemptive degradation — start ladder before critical, not after
4. Recovery after completion — restore full capability when memory frees
5. Logging all pressure events — visible in `aurelius logs` and Mission Control

## Hardware-Specific Notes

### Mac Silicon (Unified Memory)
- Total unified memory shared between CPU and GPU
- No dedicated VRAM — all model weights + KV come from same pool
- OS aggressively compresses inactive memory
- Metal/MPS has different allocation patterns than CUDA

### Jetson (Unified Memory)
- Very constrained (2GB/4GB/8GB)
- Must use GGUF/ONNX exclusively
- Quantization mandatory (Q3/Q4/INT8)
- Context budget aggressively limited

### Dedicated GPU (RTX/Blackwell)
- Separate VRAM pool
- Can also spill to system RAM when needed
- TensorRT-LLM manages its own memory pools
- FP4/FP8 quantization available on Blackwell
