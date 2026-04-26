
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

## Cycle 136 — 2026-04-21
**Sources:** Reflexion (Shinn et al. 2023), FIM (Bavarian et al. 2022), Tree of Thought (Yao et al. 2023), Bradley-Terry RM (Ziegler 2019), CPO (Xu et al. 2024), Pass@k (Chen et al. 2021)
**Modules:**
- `src/agent/reflexion_agent.py` — ReflexionAgent: verbal RL through failure reflection (15 tests)
- `src/model/fim_lm.py` — FIMTransformer: SPM/PSM fill-in-the-middle reordering (17 tests)
- `src/inference/tree_of_thought.py` — TreeOfThoughtDecoder: BFS/DFS reasoning tree (21 tests)
- `src/training/offline_reward_modeling.py` — RewardModelTrainer: Bradley-Terry pairwise RM (15 tests)
- `src/alignment/cpo_trainer.py` — CPOTrainer: contrastive preference opt w/o reference model (16 tests)
- `src/eval/pass_at_k_eval.py` — PassAtKEvaluator: unbiased pass@k estimator (16 tests)
**Tests added:** 100 | **Commits:** c0ae8b2 24d240f 4b1ad38 9a92525 6754882 f6bc386

## Cycle 137 — 2026-04-21
**Sources:** VisionCrossAttn (Flamingo/LLaVA), AdaptiveKVEviction (H2O/SnapKV), MCTS-RL (AlphaZero), MultiAgentDebate (Du et al. 2023), NCE/InfoNCE/NT-Xent (van den Oord 2018, SimCLR), Speculative eval
**Modules:**
- `src/model/vision_cross_attention.py` — VisionCrossAttention: GQA cross-attn for vision-language (16 tests)
- `src/inference/adaptive_kv_eviction.py` — AdaptiveKVEvictionManager: attention-score dynamic eviction (20 tests)
- `src/training/mcts_rl_trainer.py` — MCTSRLTrainer: AlphaZero-style MCTS policy+value training (15 tests)
- `src/eval/multi_agent_debate_eval.py` — DebateEvaluator: drift/consensus/diversity scoring (17 tests)
- `src/training/nce_objectives.py` — NCEObjectives: NCE/InfoNCE/NT-Xent contrastive losses (16 tests)
- `src/eval/speculative_acceptance_eval.py` — SpeculativeAcceptanceEval: acceptance rate/speedup (17 tests)
**Tests added:** 101 | **Commits:** de5bcbb 5cdf282 c70001e cd40ecd ad84a43 d320040

## Cycle 213 — 2026-04-25
**Scope:** Skills Deepening — Wire real Aurelius skills into React frontend
**Modules:**
- `src/serving/aurelius_server.py` — Enhanced `/api/skills` with full fields; added `GET /api/skills/<id>` detail endpoint; added `POST /api/skills/execute` execution endpoint
- `frontend/src/pages/Skills.tsx` — Live skill catalog fetch, detail modal with instructions/scripts, variable input, execution UI with results
- `tests/serving/test_aurelius_server.py` — 5 new tests for skills fields, detail, fallback, execute missing body, execute with body
**Tests added:** 5 | **Total tests:** 850 passing (serving + agent skill tests)
**Commits:** 72c0864
**Security:** bandit 0 High findings | Foreign imports: clean

