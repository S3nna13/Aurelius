
## Cycle 132 — 2026-04-21
**Sources:** Hydra (speculative decoding), ChunkPrefill, GQA-Absorbed, ToolBench, PCGradV2, ARGS/PRM-Guided Search
**Modules:**
- `src/inference/hydra_speculative.py` — HydraSpeculative: batched multi-head draft + one-shot verify (16 tests)
- `src/inference/chunk_prefill_scheduler.py` — ChunkPrefillScheduler: decode-first interleaved prefill (20 tests)
- `src/model/gqa_absorbed.py` — GQAAbsorbedAttention: standard/absorbed numerically equivalent (25 tests)
- `src/eval/tool_bench.py` — ToolBench: tool-call selection/parameter/sequence accuracy (24 tests)
- `src/training/pcgrad_v2.py` — PCGradV2: cosine-adaptive conflicting gradient projection (19 tests)
- `src/inference/reward_guided_search.py` — RewardGuidedSearch: value-guided beam search with length penalty (15 tests)
**Tests added:** 119 | **Commits:** ee3a93f d998aae b637d4d c6d860f 89a62e8 40c36e4

## Cycle 133 — 2026-04-21
**Sources:** DAPO (2025), RLOO (Ahmadian et al. 2024), Wanda (Sun et al. 2023), Coreset selection, Dr. GRPO (2025), DoLa (Chuang et al. 2023)
**Modules:**
- `src/training/dapo_trainer.py` — DAPOTrainer: decoupled clip-high/low, entropy bonus, dynamic sampling filter (17 tests)
- `src/training/rloo_trainer.py` — RLOOTrainer: leave-one-out group mean baseline (18 tests)
- `src/model/wanda_pruner.py` — WandaPruner: |W|·‖X‖₂ activation-aware weight pruning, 2:4 semi-structured (15 tests)
- `src/data/coreset_selector.py` — CoresetSelector: k-center greedy and n-gram coverage data pruning (15 tests)
- `src/alignment/dr_grpo.py` — DrGRPOTrainer: bias-free GRPO, no std-norm, sequence-level loss (26 tests)
- `src/inference/dola_decoding.py` — DoLaDecoder: subtract/JSD layer contrast for factuality (17 tests)
**Tests added:** 108 | **Commits:** 2e0bbfd 5b59967 41fef9b 127cd78 9de7f53 c59d316

## Cycle 134 — 2026-04-21
**Sources:** ORPO (Hong et al. 2024), KTO (Ethayarajh et al. 2024), Min-P sampling (Nguyen 2024), Cross-Layer KV (MLKV/LCKV 2024), CoLT5 (Ainslie et al. 2023), IPO (Azar et al. 2024)
**Modules:**
- `src/alignment/orpo_trainer.py` — ORPOTrainer: odds-ratio preference opt without reference model (15 tests)
- `src/alignment/kto_trainer.py` — KTOTrainer: binary good/bad desirability labels, no pairs (15 tests)
- `src/inference/minp_sampler.py` — MinPSampler: dynamic min_p = α*p_max probability floor (15 tests)
- `src/longcontext/cross_layer_kv_sharing.py` — CrossLayerKVStack: odd layers reuse even-layer KV (15 tests)
- `src/model/colt5_conditional.py` — CoLT5FFN/Block: light/heavy conditional computation routing (15 tests)
- `src/training/ipo_trainer.py` — IPOTrainer: squared-loss regularized DPO avoiding overfit (17 tests)
**Tests added:** 92 | **Commits:** 1dbbc61 0fcd8ee 1d2b8e8 043fda4 6cc94ef 7a9bb0b

## Cycle 135 — 2026-04-21
**Sources:** Skeleton-of-Thought (Bao et al. 2023), Self-Rewarding LM (Yuan et al. 2024), LRU (Orvieto et al. 2023), Dolma/FineWeb quality filters, GAE token credit, CoT faithfulness eval
**Modules:**
- `src/inference/skeleton_of_thought.py` — SkeletonOfThoughtDecoder: skeleton → parallel point expansion (15 tests)
- `src/alignment/self_reward_trainer.py` — SelfRewardTrainer: LLM-as-own-judge DPO preference generation (15 tests)
- `src/model/linear_recurrent_unit.py` — LRULayer: complex-diagonal stable linear recurrence (16 tests)
- `src/data/quality_filter.py` — QualityFilter: entropy/n-gram/heuristic pretraining data filter (17 tests)
- `src/training/token_credit_assignment.py` — TokenCreditAssigner: uniform/discounted/GAE/end-decay (17 tests)
- `src/eval/reasoning_trace_eval.py` — ReasoningTraceEval: faithfulness/efficiency/redundancy CoT scoring (27 tests)
**Tests added:** 107 | **Commits:** 7836263 b0660a7 5b2c96b 3cfb35e 7c6142b b0660a7
