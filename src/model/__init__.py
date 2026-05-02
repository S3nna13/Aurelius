"""Aurelius 1.3B transformer model."""

import sys

from .aqlm_quant import AQLMCodebook, AQLMConfig, AQLMLinear
from .attention import GroupedQueryAttention, apply_rope, precompute_rope_frequencies
from .checkpoint_migration import (
    MIGRATION_REGISTRY,
    CheckpointMigrator,
    MigrationError,
    MigrationStep,
    get_migration_path,
    register_migration,
)
from .chunked_local_attention import ChunkedLocalAttention
from .colt5_conditional import CoLT5Block, CoLT5Config, CoLT5FFN
from .compatibility import (
    CompatibilityError,
    CompatibilityVerdict,
    SemverParts,
    assert_compatible,
    check_checkpoint_compatibility,
    check_manifest_compatibility,
    parse_semver,
)
from .config import AureliusConfig
from .csa_attention import CompressedSparseAttention
from .dp_aware_moe_routing import DPAwareMoERouter
from .dsa_attention import DSAAttention, DSAConfig, LightningIndexer
from .factory import (
    DEFAULT_BACKBONE_BUILDERS,
    FactoryError,
    build_backbone_from_manifest,
    build_from_variant_id,
    register_backbone_builder,
)
from .family import (
    AURELIUS_FAMILY,
    MODEL_FAMILY_REGISTRY,
    MODEL_VARIANT_REGISTRY,
    ModelFamily,
    ModelVariant,
    get_family,
    get_variant_by_id,
    register_family,
    register_variant,
)
from .ffn import SwiGLUFFN
from .fim_lm import FIMConfig, FIMDocument, FIMLossFilter, FIMTransformer
from .flash_mla import FlashMLAAttention, FlashMLAConfig
from .gqa_absorbed import GQAAbsorbedAttention, GQAAbsorbedConfig
from .hca_attention import HeavilyCompressedAttention
from .head_registry import (
    HEAD_REGISTRY,
    HeadFactoryError,
    HeadKind,
    HeadSpec,
    build_head,
    get_head,
    heads_by_kind,
    list_heads,
    register_head,
)
from .interface_contract import (
    InterfaceContractBundle,
    InterfaceContractError,
    InterfaceContractPaths,
    load_interface_contract_bundle,
    load_interface_contract_json,
    load_interface_contract_markdown,
    load_interface_contract_prompt_yaml_text,
    load_interface_contract_schema,
    resolve_interface_contract_paths,
    validate_interface_contract,
)
from .interface_framework import (
    ApprovalRequest,
    AureliusInterfaceFramework,
    BackgroundJob,
    Checkpoint,
    InterfaceFrameworkError,
    MessageEnvelope,
    ModePolicy,
    SkillBundle,
    TaskThread,
    TaskThreadSpec,
    Workstream,
)
from .lambda_attention import LambdaAttention
from .linear_recurrent_unit import LRUConfig, LRULayer
from .manifest import (
    AURELIUS_REFERENCE_MANIFEST,
    MODEL_MANIFEST_REGISTRY,
    FamilyManifest,
    ManifestValidationError,
    ReleaseTrack,
    dump_manifest,
    get_manifest,
    list_manifests,
    load_manifest,
    register_manifest,
)
from .manifest_v2 import (
    MANIFEST_SCHEMA_VERSION,
    compare_backend_contracts,
    is_v2_manifest,
    list_v2_manifests,
    upgrade_to_v2,
    v2_to_v1_dict,
)
from .mhc import ManifoldConstrainedHyperConnection, MHCLayer
from .mla_256 import MLA256Attention, MLA256Config
from .model_merging import (
    MERGING_REGISTRY,
    MergeError,
    MergeResult,
    MergeStrategy,
    ModelMerger,
    dare_merge,
    linear_merge,
    slerp_merge,
    ties_merge,
)
from .moonvit_patch_packer import MoonVitPatchPacker, MoonVitPatchPackerConfig
from .mtp_shared import SharedMTPHead
from .parallel_attention import ParallelAttentionBlock
from .release_track_router import (
    DEV_POLICY,
    INTERNAL_POLICY,
    POLICY_REGISTRY,
    PRODUCTION_POLICY,
    ReleaseTrackRouter,
    RouteDecision,
    RouterOverrideError,
    RouterPolicy,
)
from .rms_norm import RMSNorm
from .striped_attention import StripedAttention, StripedAttentionConfig
from .transformer import AureliusTransformer, TransformerBlock, count_parameters
from .variant_adapter import (
    VARIANT_ADAPTER_ATTACHMENTS,
    VARIANT_ADAPTER_REGISTRY,
    AdapterKind,
    AdapterValidationError,
    VariantAdapter,
    adapters_for_variant,
    attach_to_variant,
    get_adapter,
    list_adapters,
    register_adapter,
)
from .vision_cross_attention import VisionCrossAttention, VisionCrossAttnConfig
from .zamba_block import ZambaBlock, ZambaConfig, ZambaSharedAttention, ZambaSSMLayer

# ---------------------------------------------------------------------------
# Model component registry — additive, cycle-safe
# ---------------------------------------------------------------------------
MODEL_COMPONENT_REGISTRY: dict = {}
MODEL_COMPONENT_REGISTRY["dsa_attention"] = DSAAttention
MODEL_COMPONENT_REGISTRY["mtp_shared"] = SharedMTPHead
MODEL_COMPONENT_REGISTRY["dp_aware_moe_routing"] = DPAwareMoERouter
MODEL_COMPONENT_REGISTRY["mla_256"] = MLA256Attention
MODEL_COMPONENT_REGISTRY["moonvit_patch_packer"] = MoonVitPatchPacker
MODEL_COMPONENT_REGISTRY["flash_mla"] = FlashMLAAttention
MODEL_COMPONENT_REGISTRY["striped_attention"] = StripedAttention
MODEL_COMPONENT_REGISTRY["aqlm_linear"] = AQLMLinear
MODEL_COMPONENT_REGISTRY["zamba_block"] = ZambaBlock
MODEL_COMPONENT_REGISTRY["gqa_absorbed"] = GQAAbsorbedAttention
MODEL_COMPONENT_REGISTRY["colt5_ffn"] = CoLT5FFN
MODEL_COMPONENT_REGISTRY["colt5_block"] = CoLT5Block
MODEL_COMPONENT_REGISTRY["lru"] = LRULayer
MODEL_COMPONENT_REGISTRY["fim_transformer"] = FIMTransformer
MODEL_COMPONENT_REGISTRY["vision_cross_attention"] = VisionCrossAttention

from .matryoshka_embedding import MatryoshkaConfig, MatryoshkaEmbedding

MODEL_COMPONENT_REGISTRY["matryoshka_embedding"] = MatryoshkaEmbedding
MODEL_COMPONENT_REGISTRY["csa_attention"] = CompressedSparseAttention
MODEL_COMPONENT_REGISTRY["hca_attention"] = HeavilyCompressedAttention
MODEL_COMPONENT_REGISTRY["mhc"] = ManifoldConstrainedHyperConnection

_module = sys.modules[__name__]
sys.modules.setdefault("src.model", _module)
sys.modules.setdefault("model", _module)

from src.backends import (  # noqa: E402
    BACKEND_REGISTRY,
    ENGINE_ADAPTER_REGISTRY,
    select_backend_for_manifest,
)

__all__ = [
    "BACKEND_REGISTRY",
    "ENGINE_ADAPTER_REGISTRY",
    "select_backend_for_manifest",
    "VisionCrossAttention",
    "VisionCrossAttnConfig",
    "LRUConfig",
    "LRULayer",
    "FIMConfig",
    "FIMDocument",
    "FIMTransformer",
    "FIMLossFilter",
    "CoLT5Config",
    "CoLT5FFN",
    "CoLT5Block",
    "GQAAbsorbedAttention",
    "GQAAbsorbedConfig",
    "AQLMCodebook",
    "AQLMConfig",
    "AQLMLinear",
    "ZambaBlock",
    "ZambaConfig",
    "ZambaSSMLayer",
    "ZambaSharedAttention",
    "FlashMLAAttention",
    "FlashMLAConfig",
    "StripedAttention",
    "StripedAttentionConfig",
    "MoonVitPatchPacker",
    "MoonVitPatchPackerConfig",
    "DPAwareMoERouter",
    "MLA256Attention",
    "MLA256Config",
    "DSAAttention",
    "DSAConfig",
    "LightningIndexer",
    "MODEL_COMPONENT_REGISTRY",
    "MatryoshkaConfig",
    "MatryoshkaEmbedding",
    "AURELIUS_FAMILY",
    "AURELIUS_REFERENCE_MANIFEST",
    "AureliusConfig",
    "AureliusTransformer",
    "CheckpointMigrator",
    "ChunkedLocalAttention",
    "CompressedSparseAttention",
    "HeavilyCompressedAttention",
    "CompatibilityError",
    "CompatibilityVerdict",
    "DEFAULT_BACKBONE_BUILDERS",
    "FactoryError",
    "FamilyManifest",
    "GroupedQueryAttention",
    "HEAD_REGISTRY",
    "HeadFactoryError",
    "HeadKind",
    "HeadSpec",
    "LambdaAttention",
    "ManifoldConstrainedHyperConnection",
    "MHCLayer",
    "SharedMTPHead",
    "MERGING_REGISTRY",
    "MergeError",
    "MergeResult",
    "MergeStrategy",
    "ModelMerger",
    "dare_merge",
    "linear_merge",
    "slerp_merge",
    "ties_merge",
    "MIGRATION_REGISTRY",
    "MODEL_FAMILY_REGISTRY",
    "MODEL_MANIFEST_REGISTRY",
    "MODEL_VARIANT_REGISTRY",
    "ManifestValidationError",
    "MigrationError",
    "MigrationStep",
    "ModelFamily",
    "ModelVariant",
    "DEV_POLICY",
    "INTERNAL_POLICY",
    "POLICY_REGISTRY",
    "PRODUCTION_POLICY",
    "ParallelAttentionBlock",
    "RMSNorm",
    "ReleaseTrack",
    "ReleaseTrackRouter",
    "RouteDecision",
    "RouterOverrideError",
    "RouterPolicy",
    "SemverParts",
    "SwiGLUFFN",
    "TransformerBlock",
    "apply_rope",
    "assert_compatible",
    "build_backbone_from_manifest",
    "build_head",
    "build_from_variant_id",
    "check_checkpoint_compatibility",
    "check_manifest_compatibility",
    "count_parameters",
    "dump_manifest",
    "get_family",
    "get_head",
    "get_migration_path",
    "get_manifest",
    "get_variant_by_id",
    "heads_by_kind",
    "list_heads",
    "list_manifests",
    "load_manifest",
    "parse_semver",
    "precompute_rope_frequencies",
    "register_backbone_builder",
    "register_head",
    "register_migration",
    "register_family",
    "register_manifest",
    "register_variant",
    "VARIANT_ADAPTER_ATTACHMENTS",
    "VARIANT_ADAPTER_REGISTRY",
    "AdapterKind",
    "AdapterValidationError",
    "VariantAdapter",
    "adapters_for_variant",
    "attach_to_variant",
    "get_adapter",
    "list_adapters",
    "register_adapter",
    "MANIFEST_SCHEMA_VERSION",
    "is_v2_manifest",
    "upgrade_to_v2",
    "v2_to_v1_dict",
    "compare_backend_contracts",
    "list_v2_manifests",
    "InterfaceContractBundle",
    "InterfaceContractError",
    "InterfaceContractPaths",
    "resolve_interface_contract_paths",
    "load_interface_contract_bundle",
    "load_interface_contract_json",
    "load_interface_contract_markdown",
    "load_interface_contract_prompt_yaml_text",
    "load_interface_contract_schema",
    "validate_interface_contract",
    "ApprovalRequest",
    "AureliusInterfaceFramework",
    "BackgroundJob",
    "Checkpoint",
    "InterfaceFrameworkError",
    "MessageEnvelope",
    "ModePolicy",
    "SkillBundle",
    "TaskThread",
    "TaskThreadSpec",
    "Workstream",
    "SessionRecord",
    # additive model patterns
    "MoEConfig", "MoDConfig", "ReMoDEConfig", "HLMConfig",
    "SparseMoELayer", "BalancedMoEFFN", "SoftMoELayer", "ExpertChoiceLayer",
    "MoDRouter", "MoDLayer", "CapacityTracker",
    "ReMoDELayer", "ReMoDEBlock",
    "HLMRouter", "HLMProfile", "get_hlm_config",
    "DiffusionLLM", "DiffusionLLMConfig", "SemanticTokenizer", "DiscreteDiffusion", "DiffusionDecoder",
    "architectures",
    "mod",
]


from .remode import ReMoDEBlock, ReMoDELayer  # noqa: E402
from .hlm import HLMProfile, HLMRouter, get_hlm_config  # noqa: E402
from .diffusion_llm import DiffusionDecoder, DiffusionLLM, DiffusionLLMConfig, DiscreteDiffusion, SemanticTokenizer  # noqa: E402
from . import architectures  # noqa: E402
from .moe import (  # noqa: E402
    BalancedMoEFFN,
    ExpertChoiceLayer,
    SoftMoELayer,
)
from .config import HLMConfig, MoDConfig, MoEConfig, ReMoDEConfig  # noqa: E402
from . import mod  # noqa: E402

_LAZY_EXPORTS = {
    "SessionRecord": ("src.agent.session_manager", "SessionRecord"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    module = __import__(module_name, fromlist=[attr_name])
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
