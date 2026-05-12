# Dead code analysis — 513 files, 4.7 MB

This repository has accumulated research code across multiple cycles. Many files in
`src/model/` (211 of 254, 83%) and `src/training/` (302 of 325, 93%) are not
imported by their own `__init__.py` nor referenced anywhere else in the codebase.

## src/model/ (211 dead files, 1.9 MB)

These files are never imported by `src/model/__init__.py` nor referenced elsewhere:
act_routing, activation_functions, activation_patching, activation_search,
activations_norms, activations, adaptive_computation(_v2), adaptive_compute,
adaptive_span_attn, alibi_attention, aqlm_quant, attention_pruning,
attention_utils, attention_variants, attn_backend, audio_encoder,
bigbird_attention, bitnet(_quant), block_sparse_attention, byte_level_model,
checkpoint_migration, chunked_attention(_v2,_v3), chunked_local_attention,
colt5_conditional, compatibility, ... (210 total)

## src/training/ (302 dead files, 2.9 MB)

These files are never imported by `src/training/__init__.py` nor referenced elsewhere:
trainer (38KB), structured_pruning (34KB), model_soup (26KB),
continual_learning_v3 (21KB), loss_landscape (21KB), anthropic_training (21KB),
feature_distillation (20KB), adversarial_training_v2 (20KB), online_dpo (19KB),
federated_dp (19KB), process_reward_model (18KB), maml (17KB), ... (302 total)

## Recommendation

Before deletion, verify:
1. Check if tests/ references these files
2. Check if scripts/ or configs/ reference them
3. Archive to a branch before deleting
4. Delete in batches, verify no breakage after each

## What's actually active

src/model/ (43 files that ARE used):
attention.py, config.py, factory.py, family.py, ffn.py, fim_lm.py,
flash_mla.py, gqa_absorbed.py, head_registry.py, hlm.py, interface_contract.py,
interface_framework.py, lambda_attention.py, linear_recurrent_unit.py,
manifest.py, manifest_v2.py, mhc.py, mla_256.py, model_merging.py,
mod.py, moe.py, moonvit_patch_packer.py, mtp_shared.py,
multi_token_prediction.py, parallel_attention.py, release_track_router.py,
remode.py, rms_norm.py, striped_attention.py, transformer.py,
variant_adapter.py, vision_cross_attention.py, zamba_block.py,
architectures.py, matryoshka_embedding.py, dsa_attention.py,
csa_attention.py, hca_attention.py, dp_aware_moe_routing.py,
diffusion_llm.py, act.py, aqlm_quant.py, compatibility.py,
checkpoint_migration.py, chunked_local_attention.py, colt5_conditional.py

src/training/ (23 files that ARE used):
fsdp_lite.py, loss_variance_monitor.py, lr_range_test.py, token_dropout.py,
tool_call_supervision_loss.py, async_rl_infra.py, (and __init__.py exports)
