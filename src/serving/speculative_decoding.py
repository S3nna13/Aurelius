"""Speculative Decoding: draft-token verification for accelerated inference.

Implements the speculative decoding algorithm from Lev]ik et al. (2022)
"Fast Inference from Transformers via Speculative Decoding".

Architecture
------------
- DraftModel: A smaller/faster model that produces draft token sequences
- VerifierModel: The main model that verifies drafts in parallel
- SpeculativeScheduler: Coordinates the process, manages draft tree,
  handles acceptance/rejection and fallback generation

Workflow
--------
1. For each request, get draft tokens from draft model (k tokens)
2. Concatenate prompt + draft tokens and run verifier in parallel
3. Compute per-position acceptance probabilities (draft token distribution
   vs verifier distribution)
4. Sample verification outcomes; accept draft tokens up to first rejection
5. On rejection, sample the correct token from verifier and regenerate
   remaining draft tokens
6. Return accepted tokens + final remainder

Key components live here:
- SpeculativeDecodingConfig: parameters (draft_length, temperature, etc.)
- DraftModelAdapter: wraps any model to serve as a draft generator
- VerifierModelAdapter: wraps the main model, provides parallel scoring
- SpeculativeScheduler: state machine managing the decode loop
- SpeculativeStats: throughput/acceptance rate metrics

Performance
-----------
Throughput gain = 1 / (1 - acceptance_rate * (draft_length / compute_ratio))
Typical: 70-90% acceptance with 4-8 draft tokens → 2-3x speedup.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch
from torch import Tensor

from src._compat import StrEnum

_LOGGER = logging.getLogger("aurelius.serving.speculative")


# ---------------------------------------------------------------------------
# Configuration and types
# ---------------------------------------------------------------------------


class DraftStrategy(StrEnum):
    """How the draft model is selected."""

    FIXED_MODEL = "fixed_model"  # always use the same draft model
    DYNAMIC_ROUTING = "dynamic_routing"  # choose based on request tags
    MULTI_DRAFT = "multi_draft"  # run multiple draft models in parallel


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding."""

    draft_length: int = 4  # number of speculative tokens per step (k)
    temperature: float = 1.0  # sampling temperature (1.0 = greedy-like)
    top_k: int | None = None  # top-k filtering for draft model
    top_p: float | None = None  # nucleus filtering
    resample_every: int = 1  # resample draft model after N steps
    fallback_on_reject: bool = True  # if all drafts rejected, fall back to verifier
    max_concurrent_drafts: int = 1  # for multi-draft strategy
    log_acceptance_rates: bool = True  # track per-model acceptance


@dataclass
class ModelAdapter(Protocol):
    """Common interface for draft and verifier models."""

    def __call__(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """Return logits for next token prediction."""
        ...

    def generate_token(self, context: Tensor, **kwargs: Any) -> tuple[Tensor, Tensor]:
        """Generate a single token (next token + logits)."""
        ...

    def batch_generate(self, context_batch: Tensor, **kwargs: Any) -> tuple[Tensor, Tensor]:
        """Generate one token per batch item."""
        ...


@dataclass
class SpeculativeStats:
    """Runtime statistics for speculative decoding."""

    total_draft_tokens: int = 0
    accepted_tokens: int = 0
    rejected_tokens: int = 0
    fallback_calls: int = 0
    total_latency_ms: float = 0.0
    verifier_calls: int = 0

    @property
    def acceptance_rate(self) -> float:
        if self.total_draft_tokens == 0:
            return 0.0
        return self.accepted_tokens / self.total_draft_tokens

    @property
    def throughput_multiplier(self) -> float:
        """Estimated speedup factor vs non-speculative decoding."""
        return self.accepted_tokens / max(self.verifier_calls, 1)


# ---------------------------------------------------------------------------
# Draft Model Adapter
# ---------------------------------------------------------------------------


class DraftModelAdapter:
    """Wraps a small model to produce draft token sequences.

    The draft model should be significantly faster than the verifier
    (e.g., fewer layers, smaller hidden dim, quantized). This wrapper
    handles temperature, top-k/p, and manages draft generation state.
    """

    def __init__(
        self,
        model: ModelAdapter,
        config: SpeculativeConfig,
        device: str | torch.device = "cuda",
    ) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(device)
        self._draft_cache: Tensor | None = None  # cached draft tokens for reuse

    def generate_draft_sequence(
        self,
        context: Tensor,  # (batch, seq_len)
        temperature: float | None = None,
        top_k: int | None = None,
    ) -> Tensor:
        """Generate a draft sequence of length draft_length.

        Returns
        -------
        draft_ids: (batch, draft_length)
        """
        if temperature is None:
            temperature = self.config.temperature
        if top_k is None:
            top_k = self.config.top_k

        draft_tokens = []
        current = context

        for _ in range(self.config.draft_length):
            # Get next-token logits from draft model
            with torch.no_grad():
                logits = self.model(current)  # (B, seq_len, vocab)

            # Focus on last position
            next_logits = logits[:, -1, :] / temperature

            # Optional top-k
            if top_k is not None:
                top_k_logits, top_k_indices = torch.topk(next_logits, top_k, dim=-1)
                next_logits = torch.full_like(next_logits, float("-inf"))
                next_logits.scatter_(-1, top_k_indices, top_k_logits)

            # Sample next token (no repetition penalty for simplicity)
            probs = torch.nn.functional.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            draft_tokens.append(next_token)
            # Append to context for next iteration
            current = torch.cat([current, next_token], dim=1)

        return torch.cat(draft_tokens, dim=1)  # (B, draft_length)


# ---------------------------------------------------------------------------
# Verifier Model Adapter
# ---------------------------------------------------------------------------


class VerifierModelAdapter:
    """Wraps the main model for parallel verification of draft tokens.

    Given a batch of prompts + draft tokens, the verifier computes logits
    for the entire sequence in a single forward pass, then extracts the
    per-token distribution for verification.
    """

    def __init__(
        self,
        model: ModelAdapter,
        device: str | torch.device = "cuda",
    ) -> None:
        self.model = model
        self.device = torch.device(device)

    def verify_drafts(
        self,
        context: Tensor,  # (B, prompt_len)
        draft_ids: Tensor,  # (B, draft_length)
    ) -> tuple[Tensor, Tensor]:
        """Run verification forward pass on prompt+draft sequence.

        Returns
        -------
        draft_logits: (B, draft_length, vocab) — logits for each draft position
        verifier_logits: (batch, total_len, vocab) — full sequence logits
        """
        draft_len = draft_ids.shape[1]

        # Build verification inputs: prompt followed by drafts
        verifier_input = torch.cat([context, draft_ids], dim=1)  # (B, prompt_len + draft_len)

        # Run verifier in one pass
        with torch.no_grad():
            full_logits = self.model(verifier_input)  # (B, total_len, vocab)

        # Extract logits for the draft positions (just before the last token)
        # Draft positions are at indices: prompt_len .. prompt_len + draft_len - 1
        start = context.shape[1]
        end = start + draft_len
        draft_logits = full_logits[:, start:end, :]  # (B, draft_length, vocab)

        return draft_logits, full_logits


# ---------------------------------------------------------------------------
# Speculative Scheduler
# ---------------------------------------------------------------------------


class SpeculativeScheduler:
    """Coordinates draft generation, verification, and acceptance.

    This is the main orchestrator for speculative decoding. It tracks
    per-request state, handles the acceptance/rejection sampling, and
    produces the final sequence of accepted tokens.

    The scheduler maintains a mapping from request ID to request state
    (generation progress, accepted tokens, etc.) and can batch process
    multiple requests concurrently.
    """

    def __init__(
        self,
        draft_adapter: DraftModelAdapter,
        verifier_adapter: VerifierModelAdapter,
        config: SpeculativeConfig,
    ) -> None:
        self.draft = draft_adapter
        self.verifier = verifier_adapter
        self.config = config
        self.stats = SpeculativeStats()

        # Request state: request_id -> RequestState
        self._states: dict[str, RequestState] = {}

        # Tokenizer reference (must be set by caller)
        self.tokenizer: Any = None  # Any = PreTrainedTokenizer

    def register_request(
        self,
        request_id: str,
        prompt_ids: Tensor,  # (1, prompt_len)
        max_new_tokens: int = 100,
    ) -> None:
        """Register a new generation request."""
        self._states[request_id] = RequestState(
            request_id=request_id,
            prompt_ids=prompt_ids,
            generated_ids=[],
            draft_cache=None,
            position=0,
            max_new_tokens=max_new_tokens,
            finished=False,
        )

    def step_batch(self, request_ids: list[str]) -> dict[str, list[int]]:
        """Execute one speculative decoding step for a batch.

        Returns
        -------
        Mapping from request_id to newly generated token(s). Requests that
        finish will have the finished flag set in their state.
        """
        # 1. Build batch context: for each request, take prompt + accepted tokens
        batches = []
        for rid in request_ids:
            state = self._states[rid]
            if state.finished:
                continue
            # Current context = prompt + accepted + generated so far
            all_tokens = (
                torch.cat(
                    [
                        state.prompt_ids,
                        torch.tensor(
                            state.generated_ids,
                            dtype=torch.long,
                            device=self.draft.device,
                        ).unsqueeze(0),
                    ],
                    dim=1,
                )
                if state.generated_ids
                else state.prompt_ids
            )
            batches.append((rid, all_tokens))

        if not batches:
            return {}

        # Pad to same length? For now assume same tokenizer, so we can
        # just take the last context length. In practice need padding.
        # Simplified: batch size 1 common in chat; multi-batch later.
        rids, contexts = zip(*batches)
        context_batch = torch.cat(contexts, dim=0)  # (B, seq_len)

        # 2. Generate drafts for all requests in parallel
        with torch.no_grad():
            draft_ids = self.draft.generate_draft_sequence(context_batch)
        self.stats.total_draft_tokens += draft_ids.numel()

        # 3. Verify: run verifier on context+draft
        verifier_draft_logits, _ = self.verifier.verify_drafts(context_batch, draft_ids)

        # 4. Compute acceptance decisions per request
        results: dict[str, list[int]] = {}

        for i, rid in enumerate(rids):
            state = self._states[rid]
            context = contexts[i]  # (1, seq_len)
            draft_seq = draft_ids[i]  # (draft_length,)
            draft_logits = verifier_draft_logits[i]  # (draft_length, vocab)

            # Accept/reject sequence
            accepted_ids, rejected_pos, final_token = self._accept_reject_sequence(
                context, draft_seq, draft_logits, state
            )

            # Update state
            state.generated_ids.extend(accepted_ids)
            if final_token is not None:
                state.generated_ids.append(final_token)
            if len(state.generated_ids) >= state.max_new_tokens:
                state.finished = True

            results[rid] = accepted_ids + ([final_token] if final_token is not None else [])

        self.stats.verifier_calls += 1
        return results

    def _accept_reject_sequence(
        self,
        context: Tensor,
        draft_seq: Tensor,
        draft_logits: Tensor,
        state: RequestState,
    ) -> tuple[list[int], int | None, int | None]:
        """Decide which draft tokens to accept and what the next correct token is.

        Returns
        -------
        accepted_ids: list of accepted draft token IDs
        rejected_pos: position (0-index) where rejection occurred, or None if all accepted
        final_token: new token sampled from verifier at rejection point (or None if all accepted)
        """
        # We need to compare draft token distribution vs verifier distribution
        # at each draft position. The simplest approach (from original paper):
        # For each draft position i:
        #   - Get verifier distribution over vocab at position corresponding to
        #     context + all previously accepted draft tokens
        #   - Compute acceptance probability: p_verifier[draft_token] / p_draft[draft_token]
        #   - Sample uniform r; if r <= prob, accept; else reject and sample
        #     new token from verifier distribution.

        # For simplicity in this initial version: we approximate acceptance
        # by checking if verifier's argmax matches the draft token.
        # This gives ~ greedy-level quality and is simpler to implement.

        accepted = []
        rejected_pos = None
        final_token = None

        # Build context progressively: start with current context
        # We need to simulate autoregressive execution of drafts to get verifier positions
        # BUT we already have draft_logits from one batch run. We need to map
        # each draft position i to the correct verification step.
        # The verifier_draft_logits correspond to positions: context_len .. context_len+draft_len-1
        # So we can use them directly.

        for i in range(draft_seq.shape[0]):
            draft_token_id = draft_seq[i].item()
            # Get verifier distribution for this position
            verifier_probs = torch.softmax(draft_logits[i], dim=-1)
            draft_prob = verifier_probs[draft_token_id].item()

            # Simple acceptance: accept if draft_prob above threshold
            # This is the argmax-one-token-lag approximation
            if draft_prob > 0.5:  # can tune this threshold
                accepted.append(draft_token_id)
                self.stats.accepted_tokens += 1
            else:
                rejected_pos = i
                # Sample from verifier distribution at this position
                sample = torch.multinomial(verifier_probs, num_samples=1).item()
                final_token = sample
                self.stats.rejected_tokens += 1
                break

        if rejected_pos is None:
            # All drafts accepted; we still need to sample the NEXT token
            # beyond the draft sequence. This is from the verifier at the last position
            last_pos_logits = draft_logits[-1]
            verifier_probs = torch.softmax(last_pos_logits, dim=-1)
            final_token = torch.multinomial(verifier_probs, num_samples=1).item()
            # Note: final_token is the token after all accepted drafts

        return accepted, rejected_pos, final_token

    def get_stats(self) -> SpeculativeStats:
        """Return current runtime statistics."""
        return self.stats

    def reset_stats(self) -> None:
        """Clear statistics counters."""
        self.stats = SpeculativeStats()


# ---------------------------------------------------------------------------
# Request state tracking
# ---------------------------------------------------------------------------


@dataclass
class RequestState:
    """Per-request generation state for speculative decoding."""

    request_id: str
    prompt_ids: Tensor  # (1, prompt_len)
    generated_ids: list[int] = field(default_factory=list)
    draft_cache: Tensor | None = None  # cached draft tokens (optional)
    position: int = 0
    max_new_tokens: int = 100
    finished: bool = False


# ---------------------------------------------------------------------------
# Integration helper: wrap an existing inference endpoint
# ---------------------------------------------------------------------------


def wrap_inference_with_speculative(
    base_generate_fn: Any,
    draft_model: ModelAdapter,
    verifier_model: ModelAdapter,
    config: SpeculativeConfig | None = None,
    tokenizer: Any = None,
) -> SpeculativeScheduler:
    """Utility to add speculative decoding on top of an existing generate function.

    Parameters
    ----------
    base_generate_fn :
        Original synchronous generation function (e.g., Model.generate)
    draft_model, verifier_model :
        Model adapters conforming to ModelAdapter protocol
    config :
        Speculative decoding parameters
    tokenizer :
        Tokenizer for length tracking (optional)

    Returns
    -------
    Scheduler that can be used for batched speculative decoding.
    """
    if config is None:
        config = SpeculativeConfig()

    draft_adapter = DraftModelAdapter(draft_model, config)
    verifier_adapter = VerifierModelAdapter(verifier_model)

    scheduler = SpeculativeScheduler(draft_adapter, verifier_adapter, config)
    scheduler.tokenizer = tokenizer

    return scheduler
