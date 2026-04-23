"""Aurelius multimodal surface — registry and routing layer.

This package wires the model-side vision/audio encoder and projector modules
(in ``src.model``) together via four named registries:

    VISION_ENCODER_REGISTRY     — maps name → vision encoder class
    AUDIO_ENCODER_REGISTRY      — maps name → audio encoder class
    MODALITY_PROJECTOR_REGISTRY — maps name → modality projector class
    MODALITY_TOKENIZER_REGISTRY — maps name → modality tokenizer class

Inspired by MoonshotAI/Kimi-K2 MoonViT (2602.02276), Llama 4 vision encoder,
Meta AI Apache-2.0, clean-room reimplementation.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Modality contract surface (stdlib-only, no torch dependency)
# ---------------------------------------------------------------------------

from src.multimodal.contract import (
    AUDIO_MODALITY_CONTRACT,
    DOCUMENT_MODALITY_CONTRACT,
    MODALITY_CONTRACT_REGISTRY,
    VISION_MODALITY_CONTRACT,
    ModalityContract,
    ModalityContractError,
    describe_modality_registry,
    dump_modality_contract,
    get_modality_contract,
    list_modality_contracts,
    load_modality_contract,
    register_modality_contract,
)

# ---------------------------------------------------------------------------
# Registry singletons and CRUD helpers
# ---------------------------------------------------------------------------

from src.multimodal.multimodal_registry import (
    AUDIO_ENCODER_REGISTRY,
    MODALITY_PROJECTOR_REGISTRY,
    MODALITY_TOKENIZER_REGISTRY,
    VISION_ENCODER_REGISTRY,
    MultimodalRegistryError,
    get_audio_encoder,
    get_modality_projector,
    get_modality_tokenizer,
    get_vision_encoder,
    list_audio_encoders,
    list_modality_projectors,
    list_modality_tokenizers,
    list_vision_encoders,
    register_audio_encoder,
    register_modality_projector,
    register_modality_tokenizer,
    register_vision_encoder,
)

# ---------------------------------------------------------------------------
# Modality tokenizer types
# ---------------------------------------------------------------------------

from src.multimodal.modality_tokenizer import (
    ModalityToken,
    ModalityTokenizer,
    ModalityTokenizerError,
    ModalityType,
    PassthroughModalityTokenizer,
)

# ---------------------------------------------------------------------------
# Register model-side vision encoders
# ---------------------------------------------------------------------------

from src.model.vision_encoder import (  # noqa: E402
    MultimodalEmbedding,
    PatchEmbedding,
    VisualProjection,
    VisionTransformer,
)

register_vision_encoder("VisionTransformer", VisionTransformer)
register_vision_encoder("PatchEmbedding", PatchEmbedding)
register_vision_encoder("VisualProjection", VisualProjection)
register_vision_encoder("MultimodalEmbedding", MultimodalEmbedding)

# ---------------------------------------------------------------------------
# Register model-side audio encoders
# ---------------------------------------------------------------------------

from src.model.audio_encoder import (  # noqa: E402
    MelSpectrogramExtractor,
    WhisperStyleEncoder,
)

register_audio_encoder("WhisperStyleEncoder", WhisperStyleEncoder)
register_audio_encoder("MelSpectrogramExtractor", MelSpectrogramExtractor)

# ---------------------------------------------------------------------------
# Register model-side modality projectors
# ---------------------------------------------------------------------------

from src.model.multimodal_projector import (  # noqa: E402
    ModalityProjector,
    MultiModalProjector,
)

register_modality_projector("ModalityProjector", ModalityProjector)
register_modality_projector("MultiModalProjector", MultiModalProjector)

from src.model.vision_projector import VisionProjector  # noqa: E402

register_modality_projector("VisionProjector", VisionProjector)

# ---------------------------------------------------------------------------
# Register built-in modality tokenizers
# ---------------------------------------------------------------------------

register_modality_tokenizer("PassthroughModalityTokenizer", PassthroughModalityTokenizer)

# ---------------------------------------------------------------------------
# Video encoder (temporal 3D patch, MoonViT-style)
# ---------------------------------------------------------------------------

from src.multimodal.video_encoder import (  # noqa: E402
    VIDEO_ENCODER_REGISTRY,
    Temporal3DPatchEmbed,
    TemporalPositionEncoding,
    VideoEncoder,
    VideoEncoderConfig,
)

register_vision_encoder("VideoEncoder", VideoEncoder)

# ---------------------------------------------------------------------------
# Cross-modal attention (Q-Former style)
# ---------------------------------------------------------------------------

from src.multimodal.cross_modal_attention import (  # noqa: E402
    CROSS_MODAL_REGISTRY,
    CrossModalAttentionLayer,
    CrossModalConfig,
    QFormer,
)

# ---------------------------------------------------------------------------
# Audio-speech fusion
# ---------------------------------------------------------------------------

from src.multimodal.audio_speech_fusion import (  # noqa: E402
    SPEECH_FUSION_REGISTRY,
    AudioTextAligner,
    SpeechFusionConfig,
    SpeechFusionEncoder,
)

# ---------------------------------------------------------------------------
# Document understanding (table/form/layout processor)
# ---------------------------------------------------------------------------

from src.multimodal.document_understanding import (  # noqa: E402
    DOCUMENT_EMBEDDER_REGISTRY,
    DOCUMENT_PARSER_REGISTRY,
    DocumentEmbedder,
    DocumentEmbedderConfig,
    DocumentPage,
    DocumentParser,
    DocumentRegion,
    DocumentRegionType,
    JSONLayoutParser,
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Contract surface
    "ModalityContract",
    "ModalityContractError",
    "MODALITY_CONTRACT_REGISTRY",
    "VISION_MODALITY_CONTRACT",
    "AUDIO_MODALITY_CONTRACT",
    "DOCUMENT_MODALITY_CONTRACT",
    "load_modality_contract",
    "dump_modality_contract",
    "register_modality_contract",
    "get_modality_contract",
    "list_modality_contracts",
    "describe_modality_registry",
    # Encoder/projector registries
    "VISION_ENCODER_REGISTRY",
    "AUDIO_ENCODER_REGISTRY",
    "MODALITY_PROJECTOR_REGISTRY",
    "MODALITY_TOKENIZER_REGISTRY",
    "MultimodalRegistryError",
    "register_vision_encoder",
    "get_vision_encoder",
    "list_vision_encoders",
    "register_audio_encoder",
    "get_audio_encoder",
    "list_audio_encoders",
    "register_modality_projector",
    "get_modality_projector",
    "list_modality_projectors",
    "register_modality_tokenizer",
    "get_modality_tokenizer",
    "list_modality_tokenizers",
    # Tokenizer types
    "ModalityType",
    "ModalityToken",
    "ModalityTokenizerError",
    "ModalityTokenizer",
    "PassthroughModalityTokenizer",
    # Video encoder
    "VideoEncoderConfig",
    "Temporal3DPatchEmbed",
    "TemporalPositionEncoding",
    "VideoEncoder",
    "VIDEO_ENCODER_REGISTRY",
    # Cross-modal attention (Q-Former)
    "CrossModalConfig",
    "CrossModalAttentionLayer",
    "QFormer",
    "CROSS_MODAL_REGISTRY",
    # Audio-speech fusion
    "SpeechFusionConfig",
    "AudioTextAligner",
    "SpeechFusionEncoder",
    "SPEECH_FUSION_REGISTRY",
    # Document understanding
    "DocumentRegionType",
    "DocumentRegion",
    "DocumentPage",
    "DocumentParser",
    "JSONLayoutParser",
    "DocumentEmbedderConfig",
    "DocumentEmbedder",
    "DOCUMENT_PARSER_REGISTRY",
    "DOCUMENT_EMBEDDER_REGISTRY",
]
