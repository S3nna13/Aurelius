"""Aurelius — Decoder-only transformer with GQA/CSA/HCA, SwiGLU, RoPE, and RMSNorm."""

from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt

from .attention import (
    GroupedQueryAttention,
    apply_rope,
    precompute_rope_frequencies,
    yarn_rope_frequencies,
)
from .config import AureliusConfig
from .ffn import SwiGLUFFN
from .moe import SparseMoELayer
from .rms_norm import RMSNorm


def _build_attention(config: AureliusConfig, layer_idx: int) -> nn.Module:
    if getattr(config, "use_diff_transformer", False):
        from .diff_transformer import DifferentialAttention

        return DifferentialAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            head_dim=config.head_dim,
            dropout=config.dropout,
        )

    if config.mla_enabled:
        from .mla_wrapper import MLACompatibleAttention

        return MLACompatibleAttention(config)

    if not config.hybrid_attention_enabled:
        return GroupedQueryAttention(config, layer_idx=layer_idx)

    from .csa_attention import CompressedSparseAttention
    from .hca_attention import HeavilyCompressedAttention

    if layer_idx < 2:
        return GroupedQueryAttention(config, layer_idx=layer_idx)

    layer_offset = layer_idx - 2
    if layer_offset % 2 == 0:
        return CompressedSparseAttention(config)
    else:
        return HeavilyCompressedAttention(config)


class TransformerBlock(nn.Module):
    """Single decoder layer: pre-norm attention + pre-norm FFN with residuals.

    Supports standard GQA, CSA, and HCA attention types.
    Supports standard residuals or mHC-enhanced residual connections.
    Supports dense FFN or SparseMoE FFN.
    """

    def __init__(self, config: AureliusConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attn = _build_attention(config, layer_idx)
        self.attn_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        use_moe = config.moe_enabled and (
            config.moe_every_n_layers > 0 and layer_idx % config.moe_every_n_layers == 0
        )
        if use_moe:
            self.ffn = SparseMoELayer(
                config.d_model,
                config.d_ff,
                config.moe_num_experts,
                config.moe_top_k,
            )
        else:
            self.ffn = SwiGLUFFN(config)

        if getattr(config, "teal_sparsity", 0.0) > 0:
            from src.inference.teal_sparsity import TEALSparsityWrapper

            self.ffn = TEALSparsityWrapper(self.ffn, sparsity=config.teal_sparsity)

        self.use_mhc = config.mhc_enabled
        if self.use_mhc:
            from .mhc import ManifoldConstrainedHyperConnection

            self.mhc_attn = ManifoldConstrainedHyperConnection(
                config.d_model,
                config.mhc_expansion_factor,
                config.mhc_sinkhorn_iterations,
            )
            self.mhc_ffn = ManifoldConstrainedHyperConnection(
                config.d_model,
                config.mhc_expansion_factor,
                config.mhc_sinkhorn_iterations,
            )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | dict | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | dict | None, torch.Tensor]:
        if self.use_mhc:
            attn_result = self.mhc_attn(self.attn_norm(x), self.attn, freqs_cis, mask, past_kv)
            if isinstance(attn_result, tuple):
                x, kv = attn_result
            else:
                x = attn_result
                kv = None
        else:
            attn_out, kv = self.attn(self.attn_norm(x), freqs_cis, mask, past_kv)
            x = x + attn_out

        if isinstance(self.ffn, SparseMoELayer):
            if self.use_mhc:
                ffn_result = self.mhc_ffn(self.ffn_norm(x), self.ffn)
                if isinstance(ffn_result, tuple):
                    x, aux_loss = ffn_result
                    return x, kv, aux_loss
                x = ffn_result
                aux_loss = x.new_zeros(())
            else:
                x, aux_loss = self.ffn(self.ffn_norm(x))
        else:
            if self.use_mhc:
                x = self.mhc_ffn(self.ffn_norm(x), self.ffn)
            else:
                x = x + self.ffn(self.ffn_norm(x))
            aux_loss = x.new_zeros(())

        return x, kv, aux_loss


class _NGPTBlockWrapper(nn.Module):
    """Wraps NGPTBlock to match the TransformerBlock interface."""

    def __init__(self, config: AureliusConfig, layer_idx: int = 0) -> None:
        super().__init__()
        from .ngpt import NGPTBlock, NGPTConfig

        ngpt_cfg = NGPTConfig(
            d_model=config.d_model,
            n_heads=config.n_heads,
            head_dim=config.head_dim,
            d_ff=config.d_ff,
            n_layers=config.n_layers,
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
        )
        self.block = NGPTBlock(ngpt_cfg)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | dict | None = None,
    ) -> tuple[torch.Tensor, None, torch.Tensor]:
        """Ignore freqs_cis and past_kv (nGPT does not use RoPE or KV cache)."""
        x = self.block(x, mask)
        return x, None, x.new_zeros(())


class AureliusTransformer(nn.Module):
    """Aurelius 1.3B decoder-only transformer.

    Architecture highlights:
        - 24 transformer blocks
        - Grouped-Query Attention (16 Q heads, 8 KV heads)
        - SwiGLU FFN (d_ff = 5632)
        - RoPE positional encoding (theta = 500,000)
        - RMSNorm pre-normalization
        - Tied input/output embeddings
        - No bias in any linear layer
    """

    def __init__(self, config: AureliusConfig | None = None) -> None:
        super().__init__()
        self.config = config or AureliusConfig()
        self.last_moe_aux_loss: torch.Tensor
        self.register_buffer("last_moe_aux_loss_buf", torch.tensor(0.0), persistent=False)

        # Token embedding
        if getattr(config, "use_ngpt", False):
            from .ngpt import NormalizedEmbedding

            self.embed = NormalizedEmbedding(self.config.vocab_size, self.config.d_model)
        else:
            self.embed = nn.Embedding(self.config.vocab_size, self.config.d_model)

        # Transformer blocks (with MoE support per layer)
        if getattr(config, "use_ngpt", False):
            self.layers = nn.ModuleList(
                [
                    _NGPTBlockWrapper(self.config, layer_idx=i)
                    for i in range(self.config.n_layers)
                ]
            )
        else:
            self.layers = nn.ModuleList(
                [
                    TransformerBlock(self.config, layer_idx=i)
                    for i in range(self.config.n_layers)
                ]
            )

        # Cross-layer KV sharing (optional)
        if getattr(self.config, "kv_sharing_factor", 1) > 1:
            from src.longcontext.cross_layer_kv_sharing import CrossLayerKVConfig, CrossLayerKVStack

            kv_cfg = CrossLayerKVConfig(
                n_layers=self.config.n_layers,
                d_model=self.config.d_model,
                n_heads=self.config.n_heads,
                n_kv_heads=self.config.n_kv_heads,
                head_dim=self.config.head_dim,
                share_every_n=self.config.kv_sharing_factor,
            )
            self._cross_layer_stack = CrossLayerKVStack(kv_cfg)
            self.layers = nn.ModuleList()

        # Final norm
        self.norm = RMSNorm(self.config.d_model, eps=self.config.rms_norm_eps)

        # Output head (tied with embedding)
        self.lm_head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)

        # Multi-Token Prediction (MTP)
        self.mtp: nn.Module | None = None
        if getattr(config, "mtp_heads", 0) > 0:
            from .mtp import MTPModule

            self.mtp = MTPModule(
                config.d_model,
                config.vocab_size,
                n_predict=config.mtp_heads,
                share_params=getattr(config, "mtp_share_params", True),
            )
        elif self.config.mtp_enabled:
            from .multi_token_prediction import MTPConfig, MultiTokenPredictionHead

            mtp_cfg = MTPConfig(
                depth=self.config.mtp_n_predict,
                lambda_mtp=self.config.mtp_loss_weight,
            )
            self.mtp = MultiTokenPredictionHead(self.config, mtp_cfg)
        if self.config.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        # TTT-Linear episodic memory layer (optional)
        if getattr(config, "use_ttt_episodic", False):
            from .ttt_layer import TTTConfig, TTTLinearLayer

            ttt_dim = getattr(config, "ttt_inner_dim", config.d_model)
            self.ttt_proj_in = nn.Linear(config.d_model, ttt_dim, bias=False)
            ttt_cfg = TTTConfig(
                d_model=ttt_dim,
                mini_batch_size=getattr(config, "ttt_mini_batch_size", 16),
                lr=config.ttt_inner_lr,
                use_ln=True,
            )
            self.ttt_layer = TTTLinearLayer(ttt_cfg)
            self.ttt_proj_out = nn.Linear(ttt_dim, config.d_model, bias=False)
        else:
            self.ttt_proj_in = None
            self.ttt_layer = None
            self.ttt_proj_out = None

        # Precompute RoPE frequencies (buffer — not a parameter, moves with .to())
        if self.config.rope_scaling_type == "yarn":
            freqs = yarn_rope_frequencies(
                self.config.head_dim,
                self.config.max_seq_len,
                self.config.rope_theta,
                scale=self.config.rope_scaling_factor,
                original_max_seq_len=self.config.rope_original_max_seq_len,
            )
        else:
            freqs = precompute_rope_frequencies(
                self.config.head_dim,
                self.config.max_seq_len,
                self.config.rope_theta,
            )

        self.register_buffer("freqs_cis", freqs, persistent=False)

        # MemoryManager for CPU-offloaded KV tiering
        if getattr(self.config, "cpu_kv_offload", False):
            from src.longcontext.memory_manager import MemoryManager

            self.kv_memory_manager = MemoryManager(
                cpu_kv_offload=True,
                cpu_kv_sink_size=self.config.cpu_kv_sink_size,
                cpu_kv_recent_size=self.config.cpu_kv_recent_size,
                cpu_kv_topk_blocks=self.config.cpu_kv_topk_blocks,
            )
            self._cpu_kv_offload_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        # QuestAttention page-level sparse attention
        if getattr(self.config, "quest_page_budget", 0) > 0:
            from src.inference.quest_attention import QuestAttention
            from src.longcontext.paged_kv_cache import PagedKVCache

            page_size = getattr(self.config, "page_size", 16)
            num_pages = (self.config.max_seq_len + page_size - 1) // page_size + self.config.quest_page_budget
            self.paged_kv_cache = PagedKVCache(
                n_heads=self.config.n_kv_heads,
                head_dim=self.config.head_dim,
                page_size=page_size,
                num_pages=num_pages,
            )
            self.quest_attention = QuestAttention(
                self.paged_kv_cache,
                page_budget=self.config.quest_page_budget,
            )

        # Initialize weights
        self._init_weights()

        if getattr(config, "use_liger_kernels", False):
            from src.model.liger_integration import apply_liger_kernels
            apply_liger_kernels(self)

    def _init_weights(self) -> None:
        """Small normal init for linear layers, scaled by depth for residual paths."""
        std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor] | dict | None] | None = None,
        return_hidden_states: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor, list[tuple[torch.Tensor, torch.Tensor] | dict | None]] | tuple[torch.Tensor | None, torch.Tensor, list[tuple[torch.Tensor, torch.Tensor] | dict | None], torch.Tensor]:
        """
        Args:
            input_ids: (batch, seq_len) — token indices.
            mask: Optional attention mask broadcastable to (B, H, S, S).
            labels: (batch, seq_len) — target token ids for computing cross-entropy loss.
            past_key_values: Per-layer KV cache from a previous forward pass.

        Returns:
            Tuple of (loss, logits, present_key_values):
                loss: Scalar cross-entropy loss if labels provided, else None.
                logits: (batch, seq_len, vocab_size).
                present_key_values: List of (k, v) tuples or KIVI compressed dicts, one per layer.
        """
        B, S = input_ids.shape
        assert S <= self.config.max_seq_len, (  # noqa: S101
            f"Sequence length {S} exceeds max_seq_len {self.config.max_seq_len}"
        )

        # Compute position offset from KV cache
        def _get_past_len(past_kv: tuple[torch.Tensor, torch.Tensor] | dict | None) -> int:
            if past_kv is None:
                return 0
            if isinstance(past_kv, dict):
                return int(past_kv.get("seq_len", 0))
            return past_kv[0].shape[1]

        past_len = (
            _get_past_len(past_key_values[0])
            if past_key_values is not None and len(past_key_values) > 0
            else 0
        )

        x = self.embed(input_ids)
        assert past_len + S <= self.config.max_seq_len, (  # noqa: S101
            f"past_len({past_len}) + S({S}) = {past_len + S} "
            f"exceeds max_seq_len {self.config.max_seq_len}"
        )
        freqs_cis = self.freqs_cis[past_len : past_len + S]

        if self.config.use_gradient_checkpointing and past_key_values is not None:
            raise ValueError(
                "Gradient checkpointing is incompatible with KV cache (past_key_values)"
            )

        present_key_values: list[tuple[torch.Tensor, torch.Tensor] | dict | None] = []
        moe_aux_loss = torch.tensor(0.0, device=x.device)

        if hasattr(self, "_cross_layer_stack"):
            x = self._cross_layer_stack(x, freqs_cis=freqs_cis)
        else:
            for i, layer in enumerate(self.layers):
                past_kv = past_key_values[i] if past_key_values is not None else None
                if self.config.use_gradient_checkpointing and self.training:

                    def make_ckpt_fn(item):
                        def fn(x, freqs_cis, mask):
                            out, kv, aux = item(x, freqs_cis, mask, None)  # noqa: E741
                            return out, kv[0], kv[1], aux

                        return fn

                    x, k, v, aux = ckpt(make_ckpt_fn(layer), x, freqs_cis, mask, use_reentrant=False)
                    kv = (k, v)
                else:
                    x, kv, aux = layer(x, freqs_cis, mask, past_kv)
                moe_aux_loss = moe_aux_loss + aux
                present_key_values.append(kv)

        # Optional TTT episodic memory processing
        if self.ttt_layer is not None:
            ttt_in = self.ttt_proj_in(x)
            ttt_out = self.ttt_layer(ttt_in)
            x = x + self.ttt_proj_out(ttt_out)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )
            loss = ce_loss + 0.01 * moe_aux_loss

            # Multi-Token Prediction auxiliary loss
            if self.mtp is not None:
                mtp_loss = self.mtp.compute_loss(x, labels)
                loss = loss + self.config.mtp_loss_weight * mtp_loss

        # Store aux loss for external monitoring (e.g., trainer logging)
        self.last_moe_aux_loss = moe_aux_loss.detach().to(x.device)

        if return_hidden_states:
            return loss, logits, present_key_values, x
        return loss, logits, present_key_values

    def _sample_logits(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
    ) -> torch.Tensor:
        """Sample a token from logits using temperature + top-p nucleus sampling."""
        if temperature != 1.0:
            logits = logits / temperature
        sorted_logits, sorted_indices = torch.sort(logits, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        sorted_mask = cumulative_probs <= (1.0 - top_p)
        sorted_mask[..., -1:] = False
        mask = sorted_mask.scatter(1, sorted_indices, sorted_mask)
        logits = logits.masked_fill(mask, float("-inf"))
        return torch.multinomial(logits.softmax(dim=-1), num_samples=1)

    def _reassemble_kv(
        self,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor] | None] | None,
    ) -> list[tuple[torch.Tensor, torch.Tensor] | None] | None:
        """Reassemble full KV caches from GPU sink/recent + CPU-offloaded middle."""
        if not hasattr(self, "kv_memory_manager") or not self.kv_memory_manager.cpu_kv_offload:
            return past_key_values
        if past_key_values is None:
            return None
        if not getattr(self, "_cpu_kv_offload_cache", {}):
            return past_key_values

        full: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        sink_size = self.kv_memory_manager.sink_size
        recent_size = self.kv_memory_manager.recent_size
        # Ensure any in-flight async H2D copies are complete before use.
        self.kv_memory_manager.synchronize_prefetch()
        for i, kv in enumerate(past_key_values):
            if kv is None:
                full.append(None)
                continue
            k, v = kv
            cpu_kv = self._cpu_kv_offload_cache.get(i)
            if cpu_kv is not None:
                ck, cv = cpu_kv
                device = k.device
                ck = ck.to(device, non_blocking=True)
                cv = cv.to(device, non_blocking=True)
                gpu_sink_k = k[:, :sink_size, ...]
                gpu_recent_k = k[:, -recent_size:, ...]
                gpu_sink_v = v[:, :sink_size, ...]
                gpu_recent_v = v[:, -recent_size:, ...]
                full_k = torch.cat([gpu_sink_k, ck, gpu_recent_k], dim=1)
                full_v = torch.cat([gpu_sink_v, cv, gpu_recent_v], dim=1)
                full.append((full_k, full_v))
            else:
                full.append(kv)
        return full

    def _compress_kv(
        self,
        present_key_values: list[tuple[torch.Tensor, torch.Tensor] | None] | None,
    ) -> list[tuple[torch.Tensor, torch.Tensor] | None] | None:
        """Offload middle KV blocks to CPU, keeping only sink + recent on GPU."""
        if not hasattr(self, "kv_memory_manager") or not self.kv_memory_manager.cpu_kv_offload:
            return present_key_values
        if present_key_values is None:
            return None
        if not hasattr(self, "_cpu_kv_offload_cache"):
            self._cpu_kv_offload_cache = {}

        compressed: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        sink_size = self.kv_memory_manager.sink_size
        recent_size = self.kv_memory_manager.recent_size
        budget = sink_size + recent_size

        for i, kv in enumerate(present_key_values):
            if kv is None:
                compressed.append(None)
                continue
            k, v = kv
            seq_len = k.shape[1]
            if seq_len > budget:
                sink_k = k[:, :sink_size, ...]
                sink_v = v[:, :sink_size, ...]
                recent_k = k[:, -recent_size:, ...]
                recent_v = v[:, -recent_size:, ...]
                middle_k = k[:, sink_size:-recent_size, ...]
                middle_v = v[:, sink_size:-recent_size, ...]
                self._cpu_kv_offload_cache[i] = (
                    middle_k.cpu().pin_memory() if middle_k.device.type == "cuda" else middle_k.cpu(),
                    middle_v.cpu().pin_memory() if middle_v.device.type == "cuda" else middle_v.cpu(),
                )
                compressed.append((
                    torch.cat([sink_k, recent_k], dim=1),
                    torch.cat([sink_v, recent_v], dim=1),
                ))
            else:
                compressed.append(kv)
                self._cpu_kv_offload_cache.pop(i, None)
        return compressed

    @torch.no_grad()
    def generate_paged(
        self,
        input_ids: torch.Tensor,
        paged_kv=None,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 0.9,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Generate with an optional server-level PagedKVCache for prefix reuse.

        The paged_kv argument is accepted for API compatibility with the serving
        layer but internal KV state is managed per-step by generate(). Prefix
        reuse from paged_kv is handled at the server level before this call.
        """
        return self.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 0.9,
        eos_token_id: int | None = None,
        reasoning_budget: int | None = None,
        think_token_id: int = 50400,
    ) -> torch.Tensor:
        """Autoregressive generation with top-p nucleus sampling and KV cache.

        Supports reasoning budget control (Nemotron 3): when reasoning_budget
        is set and the budget is reached, appends </think> and continues.

        Args:
            input_ids: (batch, prompt_len) — prompt token ids.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (1.0 = no scaling).
            top_p: Nucleus sampling threshold.
            eos_token_id: Stop generation when this token is produced.
            reasoning_budget: Max reasoning tokens before forced </think>.
            think_token_id: Token id for </think>.

        Returns:
            (batch, prompt_len + generated_len) token ids.
        """
        B, _ = input_ids.shape
        past_key_values = None
        cur_ids = input_ids
        think_tokens = 0
        think_closed = False
        finished = torch.zeros(B, dtype=torch.bool, device=input_ids.device)
        original_len = input_ids.shape[1]

        for step in range(max_new_tokens):
            if input_ids.shape[1] - original_len >= max_new_tokens:
                break

            forward_past = self._reassemble_kv(past_key_values) if past_key_values is not None else None

            if self.mtp is not None:
                out = self(cur_ids, past_key_values=forward_past, return_hidden_states=True)
                _, logits, present_key_values, hidden_states = out
            else:
                _, logits, present_key_values = self(cur_ids, past_key_values=forward_past)
                hidden_states = None

            past_key_values = self._compress_kv(present_key_values)
            next_logits = logits[:, -1, :]
            next_token = self._sample_logits(next_logits, temperature, top_p)

            # Speculative decoding with MTP drafts
            if (
                self.mtp is not None
                and hidden_states is not None
                and not finished.all()
                and input_ids.shape[1] - original_len < max_new_tokens - 1
            ):
                drafts = self.mtp.draft_tokens(hidden_states, temperature=temperature)
                if drafts.shape[1] > 0:
                    # Verification forward pass on full context + drafts
                    draft_ids = torch.cat([input_ids, drafts], dim=1)
                    _, draft_logits, draft_pkv = self(draft_ids, past_key_values=None)

                    accepted_tokens: list[torch.Tensor] = []
                    for i in range(drafts.shape[1]):
                        pos = input_ids.shape[1] + i - 1
                        if pos >= draft_logits.shape[1]:
                            break
                        draft_tok = drafts[:, i]
                        actual_logits = draft_logits[:, pos, :]
                        actual_tok = self._sample_logits(actual_logits, temperature, top_p).squeeze(-1)
                        accepted_tokens.append(actual_tok.unsqueeze(-1))
                        if not torch.equal(actual_tok, draft_tok):
                            break

                    if accepted_tokens:
                        accepted = torch.cat(accepted_tokens, dim=1)
                        input_ids = torch.cat([input_ids, accepted], dim=1)

                        # Only trust the KV cache if every draft was accepted
                        if len(accepted_tokens) == drafts.shape[1]:
                            past_key_values = self._compress_kv(draft_pkv)
                            cur_ids = accepted[:, -1:]
                        else:
                            past_key_values = None
                            cur_ids = input_ids

                        if eos_token_id is not None:
                            finished = finished | (accepted == eos_token_id).any(dim=1)
                        if finished.all():
                            break
                        continue

            if reasoning_budget is not None and not think_closed:
                think_tokens += 1
                if think_tokens >= reasoning_budget:
                    next_token = torch.full_like(next_token, think_token_id)
                    think_closed = True

            if eos_token_id is not None:
                finished = finished | (next_token.squeeze(-1) == eos_token_id)
                next_token[finished.unsqueeze(-1)] = eos_token_id

            cur_ids = next_token
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if eos_token_id is not None and finished.all():
                break

        return input_ids

    @torch.no_grad()
    def generate_stream(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 0.9,
        eos_token_id: int | None = None,
        reasoning_budget: int | None = None,
        think_token_id: int = 50400,
    ) -> Iterator[torch.Tensor]:
        """Autoregressive generation yielding one token at a time.

        Same algorithm as generate(), but yields each new token as it is
        produced instead of returning the full sequence at the end.

        Args:
            input_ids: (batch, prompt_len) — prompt token ids.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            eos_token_id: Stop generation when this token is produced.

        Yields:
            (batch, 1) token id tensor for each generated token.
        """

        B, _ = input_ids.shape
        past_key_values = None
        cur_ids = input_ids
        finished = torch.zeros(B, dtype=torch.bool, device=input_ids.device)

        for _ in range(max_new_tokens):
            forward_past = self._reassemble_kv(past_key_values) if past_key_values is not None else None
            _, logits, present_key_values = self(cur_ids, past_key_values=forward_past)
            past_key_values = self._compress_kv(present_key_values)
            next_logits = logits[:, -1, :]
            next_token = self._sample_logits(next_logits, temperature, top_p)

            if eos_token_id is not None:
                finished = finished | (next_token.squeeze(-1) == eos_token_id)

            yield next_token

            cur_ids = next_token

            if eos_token_id is not None and finished.all():
                return

    @torch.no_grad()
    def generate_with_quest(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 0.9,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive generation using Quest page-level sparse attention.

        Prompt KV is written into a PagedKVCache.  During each decode step only
        the top-``page_budget`` pages (selected by QuestAttention) are loaded
        and attended to.
        """
        if not hasattr(self, "quest_attention"):
            raise RuntimeError("QuestAttention is not enabled; set config.quest_page_budget > 0")
        if hasattr(self, "_cross_layer_stack"):
            raise RuntimeError("generate_with_quest is incompatible with cross-layer KV sharing")

        B, prompt_len = input_ids.shape
        if B != 1:
            raise ValueError("generate_with_quest currently supports batch_size=1 only")

        # --- Prompt phase -------------------------------------------------------
        _, logits, present_key_values = self(input_ids, past_key_values=None)

        # Allocate paged cache (over-allocate to be safe)
        request_id = "quest_gen"
        total_len = prompt_len + max_new_tokens
        self.paged_kv_cache.allocate(request_id, total_len)

        page_size = self.paged_kv_cache.page_size
        for layer_idx, (k_cache, v_cache) in enumerate(present_key_values):
            for pos in range(prompt_len):
                self.paged_kv_cache.write(request_id, pos, k_cache[0, pos], v_cache[0, pos])
                if (pos + 1) % page_size == 0:
                    logical_idx = pos // page_size
                    page_id = self.paged_kv_cache._tables[request_id].logical_pages[logical_idx]
                    self.quest_attention.update_page_stats(page_id, self.paged_kv_cache.K[page_id])

        # Sample first token from prompt logits
        cur_ids = self._sample_logits(logits[:, -1, :], temperature, top_p)

        output_ids: list[torch.Tensor] = []
        for step in range(max_new_tokens):
            output_ids.append(cur_ids)

            if eos_token_id is not None and (cur_ids == eos_token_id).all():
                break

            x = self.embed(cur_ids)  # (B, 1, d_model)
            past_len = prompt_len + step
            freqs_cis = self.freqs_cis[past_len : past_len + 1]

            for layer_idx, layer in enumerate(self.layers):
                if not hasattr(layer.attn, "q_proj"):
                    raise RuntimeError(
                        "generate_with_quest only supports attention layers with a q_proj attribute"
                    )

                norm_x = layer.attn_norm(x)
                q = layer.attn.q_proj(norm_x).view(B, 1, layer.attn.n_heads, layer.attn.head_dim)
                q = apply_rope(q, freqs_cis).squeeze(1)  # (B, n_heads, head_dim)
                # Reduce query heads to KV-head granularity for Quest scoring
                n_kv_heads = getattr(layer.attn, "n_kv_heads", layer.attn.n_heads)
                n_rep = layer.attn.n_heads // n_kv_heads
                if n_rep > 1:
                    query_vec = q[0].view(n_kv_heads, n_rep, layer.attn.head_dim).amax(dim=1)
                else:
                    query_vec = q[0]  # (n_heads, head_dim)

                selected_pages = self.quest_attention.select_pages(query_vec)
                k_sel, v_sel = self.quest_attention.gather_kv_for_pages(request_id, selected_pages)

                if k_sel.shape[0] > 0:
                    past_kv = (k_sel.unsqueeze(0).to(x.device), v_sel.unsqueeze(0).to(x.device))
                else:
                    past_kv = None

                x, kv, _ = layer(x, freqs_cis, mask=None, past_kv=past_kv)

                k_cache, v_cache = kv
                write_pos = past_len
                self.paged_kv_cache.write(request_id, write_pos, k_cache[0, -1, :], v_cache[0, -1, :])
                if (write_pos + 1) % page_size == 0:
                    logical_idx = write_pos // page_size
                    page_id = self.paged_kv_cache._tables[request_id].logical_pages[logical_idx]
                    self.quest_attention.update_page_stats(page_id, self.paged_kv_cache.K[page_id])

            x = self.norm(x)
            logits_step = self.lm_head(x)
            cur_ids = self._sample_logits(logits_step[:, -1, :], temperature, top_p)

        self.paged_kv_cache.deallocate(request_id)

        if output_ids:
            generated = torch.cat(output_ids, dim=1)
            return torch.cat([input_ids, generated], dim=1)
        return input_ids

    def count_parameters(self, *, count_embeddings: bool = True) -> dict[str, int]:
        """Count model parameters broken down by component.

        Args:
            count_embeddings: Whether to include embedding parameters.
                With tied embeddings, the embedding matrix is shared with lm_head,
                so we count it only once.

        Returns:
            Dictionary mapping component name to parameter count.
        """
        counts: dict[str, int] = {}

        # Embedding (counted once even when tied)
        if count_embeddings:
            counts["embedding"] = sum(p.numel() for p in self.embed.parameters())

        # Per-layer breakdown for first layer (all layers are identical).
        if len(self.layers) > 0:
            layer0 = self.layers[0]
            attn_params = sum(p.numel() for p in layer0.attn.parameters())
            ffn_params = sum(p.numel() for p in layer0.ffn.parameters())
            norm_params = sum(p.numel() for p in layer0.attn_norm.parameters()) + sum(
                p.numel() for p in layer0.ffn_norm.parameters()
            )
            per_layer = attn_params + ffn_params + norm_params

            counts["attention_per_layer"] = attn_params
            counts["ffn_per_layer"] = ffn_params
            counts["norm_per_layer"] = norm_params
            counts["per_layer_total"] = per_layer
            counts["all_layers"] = per_layer * self.config.n_layers
        elif hasattr(self, "_cross_layer_stack"):
            counts["attention_per_layer"] = 0
            counts["ffn_per_layer"] = 0
            counts["norm_per_layer"] = 0
            counts["per_layer_total"] = 0
            counts["all_layers"] = sum(p.numel() for p in self._cross_layer_stack.parameters())
        else:
            counts["attention_per_layer"] = 0
            counts["ffn_per_layer"] = 0
            counts["norm_per_layer"] = 0
            counts["per_layer_total"] = 0
            counts["all_layers"] = 0

        # Final norm
        counts["final_norm"] = sum(p.numel() for p in self.norm.parameters())

        # LM head (0 if tied, since we already counted embedding)
        if self.config.tie_embeddings:
            counts["lm_head (tied)"] = 0
        else:
            counts["lm_head"] = sum(p.numel() for p in self.lm_head.parameters())

        # Total unique parameters
        counts["total"] = sum(p.numel() for p in self.parameters())

        return counts


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in any PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    config = AureliusConfig()
    print("Aurelius Configuration:")
    print(f"  Layers:      {config.n_layers}")
    print(f"  Hidden dim:  {config.d_model}")
    print(f"  FFN dim:     {config.d_ff}")
    print(f"  Q heads:     {config.n_heads}")
    print(f"  KV heads:    {config.n_kv_heads}")
    print(f"  Head dim:    {config.head_dim}")
    print(f"  Vocab size:  {config.vocab_size:,}")
    print(f"  Max seq len: {config.max_seq_len:,}")
    print(f"  RoPE theta:  {config.rope_theta:,.0f}")
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = AureliusTransformer(config).to(device)

    # Parameter breakdown
    param_counts = model.count_parameters()
    print("\nParameter Breakdown:")
    for name, count in param_counts.items():
        print(f"  {name:.<30} {count:>15,}")

    total = count_parameters(model)
    print(f"\nTotal trainable parameters: {total:,} ({total / 1e9:.3f}B)")

    # Verify forward pass
    print("\nRunning forward pass verification...")
    batch_size = 2
    seq_len = 128
    tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    with torch.no_grad():
        _, logits, _ = model(tokens)

    print(f"  Input shape:  {tokens.shape}")
    print(f"  Output shape: {logits.shape}")
    assert logits.shape == (batch_size, seq_len, config.vocab_size), "Shape mismatch!"  # noqa: S101
    print("  Forward pass OK.")
