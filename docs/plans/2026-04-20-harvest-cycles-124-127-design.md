# Aurelius Harvest Plan — Cycles 124–127
**Date:** 2026-04-20
**Sources:** Kimi K2.5 (2602.02276), GLM-5 (2602.15763), GPT-OSS-120B (2508.10925), Claude Code
**Surfaces:** agent, alignment, optimizers, model, training, longcontext, inference, chat, eval, data

---

## Decisions

1. **Cycle order:** Agent surface first (124), then model-architecture (125), then model+training (126), then vision/multimodal (127).
2. **Vision:** MoonViT-3D + Zero-Vision SFT included as Cycle 127 with multimodal data pipeline.
3. **Cycle size:** 6 modules per cycle (meta-prompt maximum).

---

## Gap Analysis — Already Implemented

| File | Status |
|------|--------|
| `src/model/mla.py` | ✓ do not re-implement |
| `src/inference/mxfp4_quant.py` | ✓ do not re-implement |
| `src/alignment/double_sided_is.py` | ✓ exists |
| `src/alignment/grpo*.py` | ✓ base GRPO exists; PARL/Toggle are distinct |
| `src/agent/react_loop.py`, `budget_bounded_loop.py` | ✓ single-agent; Swarm is distinct |

---

## Cycle 124 — Agent · Alignment · Optimizers
**Harvest basis:** Kimi K2.5 (2602.02276) + GLM-5 (2602.15763)
**Surfaces:** alignment ×3, agent ×2, optimizers ×1

| Module | File | Key Algorithm |
|--------|------|---------------|
| PARL | `src/alignment/parl.py` | r_PARL = λ₁·r_parallel + λ₂·r_finish + r_perf; λ annealed to 0 |
| Toggle | `src/alignment/toggle.py` | Phase 0: budget-limited; Phase 1: standard scaling; 25–30% token reduction |
| GRM | `src/alignment/grm.py` | LLM-as-judge multi-dim scoring; hybrid rule-based + generative |
| Agent Swarm | `src/agent/agent_swarm.py` | Orchestrator (trainable) + frozen subagents; critical-path = Σ(S_main + max S_sub) |
| MuonClip | `src/optimizers/muonclip.py` | Nesterov + per-head orthogonalization (Muon Split) + RL gradient clipping |
| Cross-Stage Distillation | `src/alignment/cross_stage_distillation.py` | L_CSD = L_RL + α·KL(π_θ ∥ π_teacher_k-1) |

---

## Cycle 125 — Model · Training · LongContext · Inference
**Harvest basis:** GLM-5 (2602.15763) + GPT-OSS-120B (2508.10925)
**Surfaces:** model ×2, training ×2, longcontext ×1, inference ×1

| Module | File | Key Algorithm |
|--------|------|---------------|
| DSA Attention | `src/model/dsa_attention.py` | Lightning Indexer learns top-k token selection; 2-stage: dense warm-up → sparse adapt |
| MTP Shared | `src/model/mtp_shared.py` | 3 MTP heads share projection weights; accept-rate 2.76 vs 2.55 baseline |
| Async RL Infra | `src/training/async_rl_infra.py` | Decoupled inference + training engines; Multi-Task Rollout Orchestrator; heartbeat fault tolerance |
| TITO Gateway | `src/training/tito_gateway.py` | Token-in-Token-out; eliminates re-tokenization mismatches across engine boundary |
| Hierarchical Context Mgr | `src/longcontext/hierarchical_context_mgr.py` | keep-recent-k → discard-all fallback; trigger at 80% max_len |
| Reasoning Level Controller | `src/inference/reasoning_level_controller.py` | System-prompt prefix → {low, medium, high} → (temperature, max_tokens, top_p) |

---

## Cycle 126 — Model · Chat · Eval · Training · Agent
**Harvest basis:** GLM-5 + GPT-OSS + Claude Code
**Surfaces:** model ×2, chat ×1, eval ×1, training ×1, agent ×1

| Module | File | Key Algorithm |
|--------|------|---------------|
| DP-aware MoE Routing | `src/model/dp_aware_moe_routing.py` | Consistent hashing: session_id → fixed DP rank; prevents cross-rank KV cache misses |
| MLA-256 | `src/model/mla_256.py` | head_dim 192→256, head_count ×0.67; Muon Split per-head orthogonalization |
| Harmony Template | `src/chat/harmony_template.py` | GPT-OSS Jinja2 chat template: scratchpad delimiters, tool-call format, system prompt position |
| Swarm Bench | `src/eval/swarm_bench.py` | Evaluates agent_swarm: critical-path steps, parallelism ratio, speedup vs single-agent |
| Slime Framework | `src/training/slime_framework.py` | Unified RL infra: rollout server, fault tolerance, task router → verifier → reward_fn |
| Plugin Hook Registry | `src/agent/plugin_hook.py` | HOOK_REGISTRY: pre/post tool_call, pre/post generation, on_error; callables registered at import |

---

## Cycle 127 — Vision / Multimodal
**Harvest basis:** Kimi K2.5 MoonViT-3D + Zero-Vision SFT (2602.02276)
**Surfaces:** model ×2, data ×2, alignment ×1, eval ×1

| Module | File | Key Algorithm |
|--------|------|---------------|
| MoonViT Patch Packer | `src/model/moonvit_patch_packer.py` | NaViT patch packing: 2D→1D sequence; variable resolution; spatiotemporal volume (4 frames × H/16×W/16) |
| Vision Projector | `src/model/vision_projector.py` | Linear projection from ViT hidden dim → LLM hidden dim; supports 4× temporal compression pooling |
| Vision Token Mixer | `src/data/vision_token_mixer.py` | Early-fusion: vision tokens mixed with text at constant ratio (10%) throughout training |
| Programmatic Image Tools | `src/data/programmatic_image_tools.py` | crop, detect_objects, pixel_distance, blob_count — proxy ops for Zero-Vision SFT |
| Zero-Vision SFT Trainer | `src/alignment/zero_vision_sft.py` | Text-only SFT activates visual reasoning via programmatic ops; outperforms text+vision SFT |
| Vision Grounding Eval | `src/eval/vision_grounding_eval.py` | F1 with soft IoU matching; normalized edit distance for OCR; absolute diff for counting |

---

## Hard Constraints (meta-prompt)

- Pure native PyTorch only — no transformers, einops, flash_attn, xformers, scipy, sklearn, etc.
- Tiny test config: n_layers=2, d_model=64, n_heads=4, n_kv_heads=2, head_dim=16, d_ff=128, vocab_size=256, max_seq_len=64
- New config keys default to feature OFF
- 10–16 unit tests per module (shapes, grads, determinism, edge cases, adversarial)
- Integration test: construct from AureliusConfig, exercise runtime path, assert regression guard
- Harvest to /tmp/harvest/<name>, rm -rf after porting
- Commit only on green full suite
- Push every 3 cycles or if >10 files touched
