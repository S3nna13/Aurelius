"""Aurelius 1.3B transformer model."""

from .dsa_attention import DSAAttention, DSAConfig, LightningIndexer
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
from .lambda_attention import LambdaAttention
from .mtp_shared import SharedMTPHead
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
from .dp_aware_moe_routing import DPAwareMoERouter
from .mla_256 import MLA256Attention, MLA256Config
from .moonvit_patch_packer import MoonVitPatchPacker, MoonVitPatchPackerConfig
from .rms_norm import RMSNorm
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

# ---------------------------------------------------------------------------
# Model component registry — additive, cycle-safe
# ---------------------------------------------------------------------------
MODEL_COMPONENT_REGISTRY: dict = {}
MODEL_COMPONENT_REGISTRY["dsa_attention"] = DSAAttention
MODEL_COMPONENT_REGISTRY["mtp_shared"] = SharedMTPHead
MODEL_COMPONENT_REGISTRY["dp_aware_moe_routing"] = DPAwareMoERouter
MODEL_COMPONENT_REGISTRY["mla_256"] = MLA256Attention
MODEL_COMPONENT_REGISTRY["moonvit_patch_packer"] = MoonVitPatchPacker

__all__ = [
    "MoonVitPatchPacker",
    "MoonVitPatchPackerConfig",
    "DPAwareMoERouter",
    "MLA256Attention",
    "MLA256Config",
    "DSAAttention",
    "DSAConfig",
    "LightningIndexer",
    "MODEL_COMPONENT_REGISTRY",
    "AURELIUS_FAMILY",
    "AURELIUS_REFERENCE_MANIFEST",
    "AureliusConfig",
    "AureliusTransformer",
    "CheckpointMigrator",
    "ChunkedLocalAttention",
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
]
