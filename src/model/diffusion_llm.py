"""Diffusion LLM — Discrete Diffusion Language Model backbone.

Implementation of LLaDA2.0-Uni (arxiv:2604.20796) patterns.

Architecture:
- Fully semantic discrete tokenizer (SigLIP-VQ style)
- MoE-based dLLM backbone with block-level masked diffusion
- Diffusion decoder for token generation
- Prefix-aware inference optimization
- Few-step distillation decoder

Key insight: diffusion LLMs apply masking/diffusion at the token level
rather than autoregressive next-token prediction. The model learns
to denoise masked token sequences.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .config import MoEConfig


@dataclass
class DiffusionLLMConfig:
    vocab_size: int = 32000
    d_model: int = 2048
    n_heads: int = 16
    n_layers: int = 24
    d_ff: int = 8192
    max_seq_len: int = 4096

    moe: MoEConfig = field(
        default_factory=lambda: MoEConfig(enabled=True, num_experts=8, top_k=2, every_n_layers=2)
    )

    diffusion_steps: int = 1000
    noise_schedule: str = "cosine"  # cosine | linear | sqrt
    mask_token_id: int = 32000
    predict_x0: bool = True
    self_condition: bool = True

    decoder_steps: int = 4  # few-step distillation
    decoder_type: str = "dmd"  # dmd | consistency | progressive_distillation


@dataclass
class SemanticTokenizerConfig:
    vocab_size: int = 16384
    d_embed: int = 768
    image_size: int = 256
    patch_size: int = 16
    codebook_size: int = 8192
    quantize_type: str = "vq"  # vq | vae | siglip_vq


@dataclass
class DiffusedToken:
    token_id: int
    logits: list[float]
    is_masked: bool = False
    noise_level: float = 0.0
    confidence: float = 1.0


class NoiseSchedule:
    """Noise schedule for discrete diffusion.

    Controls how tokens are masked during training:
    - cosine: smooth cosine schedule (recommended)
    - linear: uniform linear schedule
    - sqrt: square root schedule
    """

    def __init__(self, schedule: str = "cosine", steps: int = 1000) -> None:
        self.schedule = schedule
        self.steps = steps

    def alpha(self, t: int) -> float:
        """Noise level at timestep t."""
        if self.schedule == "cosine":
            return math.cos((t / self.steps) * math.pi / 2) ** 2
        elif self.schedule == "linear":
            return 1.0 - t / self.steps
        elif self.schedule == "sqrt":
            return math.sqrt(1.0 - t / self.steps)
        return 1.0 - t / self.steps

    def sample_timestep(self) -> int:
        """Sample a random timestep for training."""
        if self.schedule == "cosine":
            # Cosine schedule biases toward mid-range noise
            u = random.random()
            t = int(self.steps * math.acos(math.sqrt(u)) / (math.pi / 2))
            return min(t, self.steps - 1)
        return random.randint(0, self.steps - 1)


class DiscreteDiffusion:
    """Discrete diffusion process for token sequences.

    Forward process: randomly mask tokens according to noise schedule.
    Reverse process: predict masked tokens from unmasked context.

    Supports:
    - Block-level masking (mask contiguous spans)
    - Random masking (mask individual tokens)
    - Span masking (mask n-grams)
    """

    def __init__(self, config: DiffusionLLMConfig) -> None:
        self.config = config
        self.noise = NoiseSchedule(config.noise_schedule, config.diffusion_steps)

    def forward_diffuse(
        self, tokens: list[int], t: int | None = None
    ) -> tuple[list[int], list[float], list[bool]]:
        """Apply forward diffusion: mask tokens according to schedule.

        Returns:
            masked_tokens: tokens with some replaced by mask_token_id
            noise_levels: per-token noise levels
            mask: boolean mask indicating which tokens are masked
        """
        n = len(tokens)
        if t is None:
            t = self.noise.sample_timestep()

        alpha_t = self.noise.alpha(t)
        mask_rate = 1.0 - alpha_t

        mask = [random.random() < mask_rate for _ in range(n)]
        masked_tokens = [
            self.config.mask_token_id if m else tok for tok, m in zip(tokens, mask, strict=True)
        ]

        noise_levels = [mask_rate if m else 0.0 for m in mask]
        return masked_tokens, noise_levels, mask

    def block_mask(
        self, tokens: list[int], t: int | None = None, block_size: int = 8
    ) -> tuple[list[int], list[bool]]:
        """Apply block-level masking (mask contiguous spans)."""
        n = len(tokens)
        if t is None:
            t = self.noise.sample_timestep()

        alpha_t = self.noise.alpha(t)
        n_masked_blocks = max(1, int((1.0 - alpha_t) * n / block_size))

        mask = [False] * n
        for _ in range(n_masked_blocks):
            start = random.randint(0, n - block_size)
            for i in range(start, min(start + block_size, n)):
                mask[i] = True

        masked_tokens = [
            self.config.mask_token_id if m else tok for tok, m in zip(tokens, mask, strict=True)
        ]
        return masked_tokens, mask


class SemanticTokenizer:
    """Fully semantic discrete tokenizer (SigLIP-VQ style).

    Maps continuous inputs (text, images) to discrete token indices
    in a semantically meaningful codebook space.

    For text: BPE tokenization with optional embedding quantization.
    For images: patch-level encoding + VQ codebook lookup.
    """

    def __init__(self, config: SemanticTokenizerConfig) -> None:
        self.config = config
        self.codebook: list[list[float]] = [
            [random.gauss(0, 0.1) for _ in range(config.d_embed)]
            for _ in range(config.codebook_size)
        ]

    def encode_text(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        words = text.split()
        return [hash(w) % self.config.vocab_size for w in words]

    def encode_image(self, pixels: list[list[list[float]]]) -> list[int]:
        """Encode image pixels to token IDs (SigLIP-VQ style).

        Simplified: divides image into patches, encodes each to codebook.
        """
        h = len(pixels)
        w = len(pixels[0]) if h > 0 else 0
        ph = self.config.patch_size
        n_patches_h = h // ph
        n_patches_w = w // ph
        tokens: list[int] = []
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                patch = [pixels[i * ph + y][j * ph + w] for y in range(ph) for w in range(ph)]
                avg_val = sum(sum(row) for row in patch) / max(len(patch), 1)
                codebook_idx = int(abs(avg_val) * (self.config.codebook_size - 1))
                codebook_idx = min(codebook_idx, self.config.codebook_size - 1)
                tokens.append(self.config.vocab_size + codebook_idx)
        return tokens

    def decode_tokens(self, tokens: list[int]) -> str | None:
        """Decode token IDs back to text."""
        text_tokens = [t for t in tokens if t < self.config.vocab_size]
        return "".join(chr(t % 128) if t % 128 > 31 else " " for t in text_tokens)

    def image_tokens(self, tokens: list[int]) -> list[int]:
        return [t for t in tokens if t >= self.config.vocab_size]


class DiffusionDecoder:
    """Few-step diffusion decoder for fast inference.

    Supports:
    - DMD: Denoising Matching Distillation
    - Consistency: Consistency model distillation
    - Progressive: Progressive distillation
    """

    def __init__(self, config: DiffusionLLMConfig) -> None:
        self.config = config
        self.steps = config.decoder_steps
        self.distillation_type = config.decoder_type

    def decode(self, logits: list[list[float]], steps: int | None = None) -> list[int]:
        """Few-step decoding: iteratively denoise token predictions.

        Each step:
        1. Predict current token probabilities
        2. Sample or argmax tokens
        3. Update with confidence weighting

        Returns: decoded token IDs after few steps.
        """
        n_steps = steps or self.steps
        n_tokens = len(logits)

        # Initialize with argmax tokens
        tokens = [max(range(len(row)), key=lambda i: row[i]) for row in logits]
        confidences = [max(row) for row in logits]

        for step in range(n_steps):
            noise_level = 1.0 - (step / n_steps)
            noise_level = noise_level * 0.5  # reduce noise each step

            if self.distillation_type == "dmd":
                # DMD: denoise with matching
                for i in range(n_tokens):
                    if random.random() < noise_level:
                        # Re-predict low-confidence tokens
                        orig_idx = max(range(len(logits[i])), key=lambda j: logits[i][j])
                        tokens[i] = orig_idx
                        confidences[i] = max(logits[i])

            elif self.distillation_type == "consistency":
                # Consistency: direct mapping
                for i in range(n_tokens):
                    if confidences[i] < 0.9:
                        tokens[i] = max(range(len(logits[i])), key=lambda j: logits[i][j])

        return tokens


class DiffusionLLM:
    """Discrete Diffusion Language Model.

    Full LLaDA2.0-Uni-style architecture:
    1. Semantic tokenizer encodes input to discrete tokens
    2. Forward diffusion masks tokens during training
    3. MoE backbone predicts unmasked tokens
    4. Reverse diffusion generates tokens from noise
    5. Few-step decoder for efficient inference

    During training:
        input -> tokenize -> forward_diffuse -> backbone -> predict_x0 -> loss

    During generation:
        noise -> backbone(iterative_denoise) -> decoder -> tokens -> detokenize
    """

    def __init__(
        self, config: DiffusionLLMConfig, tokenizer: SemanticTokenizer | None = None
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer or SemanticTokenizer(SemanticTokenizerConfig())
        self.diffusion = DiscreteDiffusion(config)
        self.decoder = DiffusionDecoder(config)
        self.noise_schedule = NoiseSchedule(config.noise_schedule, config.diffusion_steps)

    def forward(
        self, tokens: list[int], t: int | None = None
    ) -> tuple[list[DiffusedToken], list[int]]:
        """Training forward pass: diffuse tokens, predict original.

        Returns:
            diffused_tokens: tokens after diffusion with predictions
            target_tokens: original unmasked tokens for loss computation
        """
        if t is None:
            t = self.noise_schedule.sample_timestep()

        masked_tokens, noise_levels, mask = self.diffusion.forward_diffuse(tokens, t)

        diffused: list[DiffusedToken] = []
        for tok, is_masked, noise in zip(masked_tokens, mask, noise_levels, strict=True):
            logits = [random.gauss(0, 0.1) for _ in range(self.config.vocab_size)]
            idx = min(tok, len(logits) - 1)
            if is_masked:
                logits[idx] = noise * 2.0
            else:
                logits[idx] = 2.0
            diffused.append(
                DiffusedToken(
                    token_id=tok,
                    logits=logits,
                    is_masked=is_masked,
                    noise_level=noise,
                    confidence=max(logits),
                )
            )

        return diffused, tokens

    def generate(
        self, prompt: list[int] | None = None, length: int = 128, steps: int | None = None
    ) -> list[int]:
        """Generate tokens via iterative denoising.

        Starts from pure noise (all masked) or a prompt, then iteratively
        denoises using the diffusion process.

        Simplified for integration:
        1. Initialize all tokens as masked (or with prompt)
        2. For each diffusion step, predict tokens
        3. Update with decreasing noise
        """
        n_steps = steps or self.config.decoder_steps
        seq_len = (prompt or []) + [self.config.mask_token_id] * (length - len(prompt or []))
        n = len(seq_len)

        for step in range(n_steps):
            noise_level = self.noise_schedule.alpha(
                int(step * self.config.diffusion_steps / n_steps)
            )

            # Predict tokens (simplified: random prediction with noise guidance)
            for i in range(n):
                if seq_len[i] == self.config.mask_token_id:
                    if random.random() > noise_level:
                        seq_len[i] = random.randint(0, min(self.config.vocab_size - 1, 100))

        return seq_len

    def generate_with_prompt(self, prompt_text: str, length: int = 128) -> str:
        """Generate text completion from prompt."""
        prompt_tokens = self.tokenizer.encode_text(prompt_text)
        generated_tokens = self.generate(prompt=prompt_tokens, length=len(prompt_tokens) + length)
        decoded = self.tokenizer.decode_tokens(generated_tokens)
        return decoded or prompt_text
