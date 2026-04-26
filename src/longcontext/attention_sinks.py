"""StreamingLLM attention-sinks KV cache.

Implements the windowing strategy from Xiao et al. 2023,
"Efficient Streaming Language Models with Attention Sinks"
(arXiv:2309.17453).

The cache keeps:

* ``n_sinks`` permanent "sink" tokens (the first tokens of the stream),
  whose RoPE positions remain their original absolute positions
  ``0 .. n_sinks-1``.
* A rolling window of the most recent ``window_size`` tokens. When the
  total cached length would exceed ``n_sinks + window_size``, the oldest
  non-sink entries are evicted from the middle of the buffer.

Position-id policy (shifted, per StreamingLLM paper):

    Returned position ids for the window slots are
    ``n_sinks, n_sinks+1, ..., n_sinks + window_actual - 1`` -- i.e. the
    window tokens are re-indexed contiguously after the sinks regardless
    of their true source position in the stream. This matches the
    paper's finding that preserving *relative* positions inside the
    window matters more than the absolute source position; using the
    original absolute positions ("un-shifted") causes attention logit
    blow-ups as the stream grows because RoPE was never trained on
    extrapolated positions.

    Tradeoff: shifted positions mean the model cannot distinguish
    "token N of the stream" from "token N+K of the stream" once a token
    has rolled into the window; this is acceptable for streaming decode
    where only local structure in the window matters. If a downstream
    caller needs true absolute positions they should track them
    externally.

This module is opt-in; importing it has no effect on the default
model forward path. It mutates no existing module.
"""

from __future__ import annotations

import torch
from torch import Tensor


class AttentionSinkCache:
    """Rolling KV cache with permanent attention sinks.

    Shapes:
        Keys / values passed to :meth:`append` and returned by it have
        shape ``[B, H_kv, T, D]`` where ``T`` is the number of tokens in
        that call for ``append`` and the current cache length for the
        return value.

    Parameters
    ----------
    n_sinks:
        Number of permanent sink tokens kept at the head of the cache.
        Must be ``>= 0``.
    window_size:
        Maximum number of rolling-window tokens kept after the sinks.
        Must be ``>= 0``.
    head_dim:
        Per-head hidden size ``D``. Must be ``> 0``.
    n_kv_heads:
        Number of key/value heads ``H_kv``. Must be ``> 0``.
    """

    def __init__(
        self,
        n_sinks: int = 4,
        window_size: int = 512,
        head_dim: int = 64,
        n_kv_heads: int = 8,
    ) -> None:
        if not isinstance(n_sinks, int) or n_sinks < 0:
            raise ValueError(f"n_sinks must be a non-negative int, got {n_sinks!r}")
        if not isinstance(window_size, int) or window_size < 0:
            raise ValueError(f"window_size must be a non-negative int, got {window_size!r}")
        if n_sinks + window_size == 0:
            raise ValueError("n_sinks + window_size must be > 0 (cache would have zero capacity)")
        if not isinstance(head_dim, int) or head_dim <= 0:
            raise ValueError(f"head_dim must be a positive int, got {head_dim!r}")
        if not isinstance(n_kv_heads, int) or n_kv_heads <= 0:
            raise ValueError(f"n_kv_heads must be a positive int, got {n_kv_heads!r}")

        self.n_sinks = n_sinks
        self.window_size = window_size
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads

        # Lazily initialized on first append (once we know B, dtype, device).
        self._sink_k: Tensor | None = None
        self._sink_v: Tensor | None = None
        self._win_k: Tensor | None = None
        self._win_v: Tensor | None = None
        # True count of tokens currently occupying sinks / window slots.
        self._n_sink_filled: int = 0
        self._n_win_filled: int = 0
        # Total tokens ever appended (for debugging / callers).
        self._total_seen: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Drop all cached state. Safe to call at any time."""
        self._sink_k = None
        self._sink_v = None
        self._win_k = None
        self._win_v = None
        self._n_sink_filled = 0
        self._n_win_filled = 0
        self._total_seen = 0

    def num_cached_tokens(self) -> int:
        """Return the number of tokens currently in the cache."""
        return self._n_sink_filled + self._n_win_filled

    @property
    def budget(self) -> int:
        """Maximum cache size (``n_sinks + window_size``)."""
        return self.n_sinks + self.window_size

    def append(
        self,
        new_k: Tensor,
        new_v: Tensor,
        current_seq_pos: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Append new KV entries to the cache and return the full view.

        Parameters
        ----------
        new_k, new_v:
            Tensors of shape ``[B, H_kv, T_new, D]`` with matching
            dtype/device.
        current_seq_pos:
            Position (in the untruncated stream) of ``new_k[:, :, 0]``.
            Used purely for bookkeeping; positions returned to the caller
            are derived from cache slot indices per the shifted-RoPE
            policy. If ``None``, assumed equal to the running total.

        Returns
        -------
        cached_k, cached_v:
            Tensors of shape ``[B, H_kv, n_sinks_actual + window_actual, D]``.
        cached_positions:
            1-D :class:`torch.long` tensor of length
            ``n_sinks_actual + window_actual`` giving the RoPE position
            id for each cache slot. Sink slots carry their true absolute
            positions ``0..n_sinks_actual-1``; window slots carry shifted
            positions ``n_sinks..n_sinks + window_actual - 1``.
        """
        self._validate_input(new_k, "new_k")
        self._validate_input(new_v, "new_v")
        if new_k.shape != new_v.shape:
            raise ValueError(
                f"new_k and new_v must share shape; got {tuple(new_k.shape)} "
                f"vs {tuple(new_v.shape)}"
            )
        if new_k.dtype != new_v.dtype:
            raise ValueError(
                f"new_k and new_v must share dtype; got {new_k.dtype} vs {new_v.dtype}"
            )
        if new_k.device != new_v.device:
            raise ValueError(
                f"new_k and new_v must share device; got {new_k.device} vs {new_v.device}"
            )

        B, H, T_new, D = new_k.shape
        if H != self.n_kv_heads:
            raise ValueError(
                f"new_k has {H} kv heads but cache was configured for {self.n_kv_heads}"
            )
        if D != self.head_dim:
            raise ValueError(f"new_k has head_dim={D} but cache was configured for {self.head_dim}")

        self._ensure_buffers(B, new_k.dtype, new_k.device)
        assert self._sink_k is not None  # for type-checkers  # noqa: S101
        assert self._sink_v is not None  # noqa: S101
        assert self._win_k is not None  # noqa: S101
        assert self._win_v is not None  # noqa: S101

        # Sanity-check batch size against any previously cached state.
        if self._sink_k.shape[0] != B:
            raise ValueError(
                f"batch size {B} does not match cached batch size "
                f"{self._sink_k.shape[0]}; call reset() before changing B"
            )

        if current_seq_pos is not None and current_seq_pos != self._total_seen:
            # Allow but note: caller's idea of position disagrees with
            # ours. We trust our own counter for bookkeeping since the
            # position output is derived from slot indices anyway.
            pass
        self._total_seen += T_new

        # Ingest new tokens one at a time. T_new is typically small
        # (1 for decode, a prompt-length for prefill); this is O(T_new)
        # not O(cache) because buffers are preallocated and we use slice
        # assignments.
        k_iter = new_k
        v_iter = new_v

        # Step 1: fill sinks if any slots remain.
        if self.n_sinks > 0 and self._n_sink_filled < self.n_sinks:
            take = min(self.n_sinks - self._n_sink_filled, T_new)
            dst = slice(self._n_sink_filled, self._n_sink_filled + take)
            self._sink_k[:, :, dst, :] = k_iter[:, :, :take, :]
            self._sink_v[:, :, dst, :] = v_iter[:, :, :take, :]
            self._n_sink_filled += take
            k_iter = k_iter[:, :, take:, :]
            v_iter = v_iter[:, :, take:, :]

        remaining = k_iter.shape[2]
        if remaining == 0:
            return self._materialize_view()

        # Step 2: push remainder into the rolling window. If window_size
        # is 0 we simply drop them -- only sinks are kept.
        if self.window_size == 0:
            return self._materialize_view()

        # If the incoming chunk alone exceeds the window, keep only its
        # tail: older positions would be evicted anyway.
        if remaining >= self.window_size:
            tail_start = remaining - self.window_size
            self._win_k[:, :, :, :] = k_iter[:, :, tail_start:, :]
            self._win_v[:, :, :, :] = v_iter[:, :, tail_start:, :]
            self._n_win_filled = self.window_size
            return self._materialize_view()

        # Otherwise, append-with-eviction. Evict from the middle (i.e.
        # the oldest window slots, which live right after the sinks in
        # logical order). Concretely, shift the live window contents
        # left by the overflow amount and write the new tokens at the
        # tail.
        free = self.window_size - self._n_win_filled
        if remaining <= free:
            dst = slice(self._n_win_filled, self._n_win_filled + remaining)
            self._win_k[:, :, dst, :] = k_iter
            self._win_v[:, :, dst, :] = v_iter
            self._n_win_filled += remaining
        else:
            overflow = remaining - free
            # Shift existing live content left by `overflow`.
            keep_src = slice(overflow, self._n_win_filled)
            keep_dst = slice(0, self._n_win_filled - overflow)
            # Clone to avoid aliasing on overlapping copy.
            self._win_k[:, :, keep_dst, :] = self._win_k[:, :, keep_src, :].clone()
            self._win_v[:, :, keep_dst, :] = self._win_v[:, :, keep_src, :].clone()
            # Write all `remaining` new tokens at the tail.
            new_dst = slice(self.window_size - remaining, self.window_size)
            self._win_k[:, :, new_dst, :] = k_iter
            self._win_v[:, :, new_dst, :] = v_iter
            self._n_win_filled = self.window_size

        return self._materialize_view()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _validate_input(self, t: Tensor, name: str) -> None:
        if not isinstance(t, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(t).__name__}")
        if t.dim() != 4:
            raise ValueError(f"{name} must be 4-D [B, H_kv, T, D], got shape {tuple(t.shape)}")

    def _ensure_buffers(self, B: int, dtype: torch.dtype, device: torch.device) -> None:
        if self._sink_k is not None:
            return
        H, D = self.n_kv_heads, self.head_dim
        # Allocate zero-initialized buffers. We never read unfilled
        # slots (views are sliced by filled counts).
        self._sink_k = torch.zeros((B, H, self.n_sinks, D), dtype=dtype, device=device)
        self._sink_v = torch.zeros((B, H, self.n_sinks, D), dtype=dtype, device=device)
        self._win_k = torch.zeros((B, H, self.window_size, D), dtype=dtype, device=device)
        self._win_v = torch.zeros((B, H, self.window_size, D), dtype=dtype, device=device)

    def _materialize_view(self) -> tuple[Tensor, Tensor, Tensor]:
        assert self._sink_k is not None  # noqa: S101
        assert self._sink_v is not None  # noqa: S101
        assert self._win_k is not None  # noqa: S101
        assert self._win_v is not None  # noqa: S101

        ns, nw = self._n_sink_filled, self._n_win_filled
        if self.window_size == 0:
            # Window buffer exists but is empty; nw is always 0.
            sink_k = self._sink_k[:, :, :ns, :]
            sink_v = self._sink_v[:, :, :ns, :]
            cached_k = sink_k
            cached_v = sink_v
        elif ns == self.n_sinks and nw == self.window_size:
            # Fast path: full cache, no slicing needed.
            cached_k = torch.cat([self._sink_k, self._win_k], dim=2)
            cached_v = torch.cat([self._sink_v, self._win_v], dim=2)
        else:
            sink_k = self._sink_k[:, :, :ns, :]
            sink_v = self._sink_v[:, :, :ns, :]
            # Window is always packed to the *left* while filling, then
            # to the full buffer once full. During fill, live slots are
            # [0:nw).
            win_k = self._win_k[:, :, :nw, :]
            win_v = self._win_v[:, :, :nw, :]
            if ns == 0:
                cached_k, cached_v = win_k, win_v
            elif nw == 0:
                cached_k, cached_v = sink_k, sink_v
            else:
                cached_k = torch.cat([sink_k, win_k], dim=2)
                cached_v = torch.cat([sink_v, win_v], dim=2)

        positions = self._build_positions(ns, nw, cached_k.device)
        return cached_k, cached_v, positions

    def _build_positions(self, ns: int, nw: int, device: torch.device) -> Tensor:
        # Sinks: 0 .. ns-1 (true absolute).
        # Window: ns .. ns+nw-1 (shifted -- see module docstring).
        sink_pos = torch.arange(ns, dtype=torch.long, device=device)
        win_pos = torch.arange(self.n_sinks, self.n_sinks + nw, dtype=torch.long, device=device)
        if ns == 0:
            return win_pos
        if nw == 0:
            return sink_pos
        return torch.cat([sink_pos, win_pos], dim=0)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"AttentionSinkCache(n_sinks={self.n_sinks}, "
            f"window_size={self.window_size}, head_dim={self.head_dim}, "
            f"n_kv_heads={self.n_kv_heads}, "
            f"cached={self.num_cached_tokens()}/{self.budget})"
        )


__all__ = ["AttentionSinkCache"]
