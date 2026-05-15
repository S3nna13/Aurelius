# Aurelius v2 Model Card — Three-Model Family

## Overview

Aurelius v2 defines exactly three public models, sharing one output protocol, one control token set, and one execution specification. Hardware variation is handled through artifacts, profiles, backends, and quantization — not additional model names.

## Model Hierarchy

| Model | Parameters | Purpose | Target Hardware |
|---|---|---|---|
| **Aurelius Swift** | ~0.6B dense | Edge, router, verifier, fallback | Jetson Nano+, Mac 8GB+, any GPU 2GB+ |
| **Aurelius Forge** | ~3B dense/hybrid | Default local agent, coding, CUA | Mac 16GB+, RTX 8-24GB+, Jetson Orin+ |
| **Aurelius Atlas** | ~32B total / ~8B active MoE | Frontier reasoning, orchestration | RTX 6000+, Blackwell, Mac Ultra, remote |

## Shared Architecture

All three models use:
- Decoder-only transformer base
- GQA (Grouped-Query Attention)
- RoPE / YaRN rotary embeddings
- SwiGLU activation
- RMSNorm
- Tied embeddings
- Shared protocol (Section 9 of master plan)

## Control Tokens

All models share the same control vocabulary:
`<|text|>`, `<|tool_call|>`, `<|computer_action|>`, `<|memory_op|>`, `<|skill_use|>`, `<|plan|>`, `<|critique|>`, `<|verify|>`, `<|ask_user|>`, `<|delegate|>`, `<|escalate|>`, `<|refuse_safe|>`, `<|final|>`, `<|screen|>`, `<|ax_tree|>`, `<|file_context|>`, `<|web_context|>`, `<|memory_context|>`, `<|skill_context|>`

## Swift — Edge / Small Model

### Target Spec
- ~0.6B dense parameters
- 8-12 layers depending on d_model/d_ff
- Single GPU or CPU inference target
- No MoE by default

### Architectural Decisions
- Standard decoder-only transformer
- GQA for KV efficiency
- AMC-Lite (Activation Memory Control, lightweight variant)
- DecisionHead for routing decisions
- ToolCallHead for structured tool invocation
- MemoryOpHead for memory operations
- SkillHead (lightweight retrieve/use only)
- VerifierHead (small, for CUA action validation)
- CUAActionHead in verifier/short-action mode only
- No MOE, no HOPE/CMS, no VLM stack, no full Lightning/DSA

### Artifacts
- safetensors (reference)
- GGUF Q3 / Q4 / Q5
- ONNX INT8
- MLX 4-bit
- TensorRT engine where feasible

### Primary Roles
- Answer simple prompts locally
- Route tasks to tools/skills/bigger models
- Verify CUA actions proposed by agents
- Summarize memory context
- Small tool calls
- Invoke built-in skills safely
- Run fully offline on constrained hardware
- Act as fallback when larger models unavailable

## Forge — Default Serious Agent

### Target Spec
- ~3B total parameters
- Optional hybrid MoE-lite in middle layers (shared expert)
- Shared-parameter MTP (Multi-Token Prediction)

### Architectural Decisions
- Decoder-only transformer
- MLA retrofit path (Multi-Latent Attention)
- AMC-2 in selected layers (medium-variant memory control)
- Optional middle-layer MoE-lite with shared expert
- DecisionHead, ToolCallHead, CUAActionHead, MemoryOpHead, SkillHead, CriticHead, VerifierHead, EscalationHead
- Compact OCR/VLM support for document/GUI perception
- Full PRAXIS alignment

### Artifacts
- safetensors (reference)
- GGUF Q4 / Q5
- MLX 4-bit
- AWQ / GPTQ quantized
- TensorRT-LLM profile
- TensorRT-Edge profile (Jetson Orin)
- FP8 package (optional)
- LoRA/S-LoRA adapter bundles

### Primary Roles
- Coding tasks (repo-scale)
- Local agent operation
- Tool use with safety gates
- Native skill library use
- CUA on capable hardware
- Multi-step task resolution
- Repo-scale context with compression/retrieval
- Escalate to Atlas for deep reasoning

## Atlas — Frontier Workstation Model

### Target Spec
- ~32B total / ~8B active MoE (preferred)
- Fallback under same public name: ~14B dense/hybrid

### Architectural Decisions
- Decoder-only base
- MLA (Multi-Latent Attention) mandatory
- DSA (Dynamic Sparse Attention) long-context path
- Lightning/persistent KV path (optional)
- Shared-parameter MTP
- Auxiliary-loss-free or bias-balanced MoE routing
- Shared expert + latent expert routing
- AMC-2
- Optional HOPE/CMS in deep layers
- Full action head stack
- Full SkillHead with composition/delegation
- Full PRAXIS v2
- Model-spec midtraining
- Agent RL
- Full multimodal GUI/document/webpage perception
- Long context: 128K → 1M depending backend

### Artifacts
- safetensors (reference)
- TensorRT-LLM FP8
- TensorRT-LLM FP4/NVFP4
- Remote serving container
- GGUF/MLX Atlas-lite (reduced, not full Atlas)

### Primary Roles
- Hard reasoning tasks
- Deep research with provenance
- Subagent orchestration
- Multiple native skill composition
- Large repo repair
- Full multimodal CUA
- Very long context handling
- Teaching Forge/Swift via distillation

## No-Silent-Fallback Rule

When any model name is requested, the runtime must report:
1. requested_model
2. actual_model (must match or clearly state it does not)
3. execution_mode (native_local, quantized_local, offloaded, split, remote, verifier_only)
4. backend
5. artifact
6. quantization
7. local_or_remote
8. fallback_reason (why not the requested form)
9. live_status (live, stub, remote, simulated)

If the user requests Atlas and gets Swift, this must be stated explicitly. Silent downgrade is prohibited.

## Distillation Chain

Atlas trains Forge teaches Swift:
- Atlas → Forge: reasoning, long-context, CUA imitation
- Forge → Swift: routing, verification, decisive action
- Shared: PRAXIS alignment, Safety gates, DAIES benchmarks
