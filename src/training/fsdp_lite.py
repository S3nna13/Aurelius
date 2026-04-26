"""FSDP-Lite: single-process parameter-sharding wrapper (educational).

This module simulates Fully Sharded Data Parallel (FSDP) parameter sharding in
a single process. It splits each parameter tensor into ``world_size`` logical
shards and stores only the slice belonging to ``rank``. During the forward
pass, the full parameters are reconstituted ("all-gathered") via an in-process
stub, the wrapped module runs, and the gathered parameters are then dropped
back to shards to simulate memory savings.

Key properties
--------------
* ``world_size == 1`` is a no-op pass-through: ``shard_tensor`` returns the
  original tensor unchanged and ``FSDPLite`` simply forwards to the wrapped
  module.
* Tensors whose flattened length is not divisible by ``world_size`` are
  **zero-padded** on the final shard (documented, not an error). ``gather``
  trims the padding when the original length is known.
* No use of ``torch.distributed`` — pure ``torch`` only. The ``allgather_stub``
  is a plain list concatenation. Real distributed code can swap it later
  without changing the public surface.
* ``FSDPLite.parameters()`` returns only the local-rank shard tensors, which is
  what an optimizer needs to see in a real FSDP setup.

This is an educational scaffold. It intentionally keeps the full unsharded
parameter data resident inside the wrapped module so that the forward pass
still works correctly; the "drop" after forward merely re-writes the local
shard from the (possibly updated) parameter data, reflecting the shape of the
real API rather than providing true memory savings.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

__all__ = [
    "ShardSpec",
    "FSDPLite",
    "shard_tensor",
    "gather_tensor",
]


@dataclass
class ShardSpec:
    """Specification of the simulated distributed group.

    Parameters
    ----------
    world_size:
        Total number of simulated ranks. Must be >= 1.
    rank:
        The rank this wrapper represents. Must satisfy ``0 <= rank < world_size``.
    flatten:
        If True (default), parameters are flattened before sharding. If False,
        sharding is performed along dim 0 of the original shape (requires
        dim 0 to be >= world_size).
    """

    world_size: int
    rank: int
    flatten: bool = True

    def __post_init__(self) -> None:
        if self.world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {self.world_size}")
        if self.rank < 0 or self.rank >= self.world_size:
            raise ValueError(
                f"rank must satisfy 0 <= rank < world_size; got rank={self.rank}, "
                f"world_size={self.world_size}"
            )


def _shard_length(numel: int, world_size: int) -> int:
    """Per-shard length after zero-padding so numel is divisible by world_size."""
    return (numel + world_size - 1) // world_size


def shard_tensor(t: torch.Tensor, spec: ShardSpec) -> torch.Tensor:
    """Return the shard of ``t`` that belongs to ``spec.rank``.

    The tensor is flattened (when ``spec.flatten``) then padded with zeros at
    the tail so its length is divisible by ``world_size``. The result is the
    contiguous slice ``[rank * L : (rank + 1) * L]`` where ``L`` is the
    per-shard length.

    If ``world_size == 1`` the original tensor is returned unchanged (no copy,
    no flatten, no pad) — this is a documented no-op fast path.
    """
    if spec.world_size == 1:
        return t

    if spec.flatten:
        flat = t.reshape(-1)
    else:
        if t.dim() == 0:
            raise ValueError("Cannot shard a 0-dim tensor with flatten=False")
        if t.size(0) < spec.world_size:
            raise ValueError(
                f"flatten=False requires size(0) >= world_size, got {t.size(0)} < {spec.world_size}"
            )
        flat = t

    n = flat.size(0)
    per = _shard_length(n, spec.world_size)
    pad = per * spec.world_size - n
    if pad > 0:
        pad_shape = list(flat.shape)
        pad_shape[0] = pad
        padding = torch.zeros(pad_shape, dtype=flat.dtype, device=flat.device)
        flat = torch.cat([flat, padding], dim=0)

    start = spec.rank * per
    end = start + per
    return flat[start:end].detach().clone()


def gather_tensor(
    shards: list[torch.Tensor],
    original_numel: int | None = None,
    original_shape: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Simulated all-gather: concatenate shards and restore the original shape.

    Parameters
    ----------
    shards:
        One tensor per rank, in rank order. Every shard must have the same
        leading-dim length (they were produced by :func:`shard_tensor`).
    original_numel:
        If provided, the concatenated buffer is truncated to this many elements
        (this removes the zero padding introduced by ``shard_tensor``). If
        omitted, no trimming is performed.
    original_shape:
        If provided, the result is reshaped to this shape after trimming.
    """
    if len(shards) == 0:
        raise ValueError("gather_tensor requires at least one shard")

    if len(shards) == 1 and original_numel is None and original_shape is None:
        return shards[0]

    flat = torch.cat([s.reshape(-1) for s in shards], dim=0)
    if original_numel is not None:
        flat = flat[:original_numel]
    if original_shape is not None:
        flat = flat.reshape(*original_shape)
    return flat


class FSDPLite(nn.Module):
    """Wrap an ``nn.Module`` so its parameters are logically sharded.

    During ``forward``:
      1. the local shard for each parameter is combined with the (simulated)
         shards from other ranks via an in-process all-gather stub,
      2. the inner module runs with its full-precision parameters,
      3. the local shard buffer is re-extracted from the (possibly updated)
         parameter data — this models the "reshard" step that frees gathered
         memory in real FSDP.

    Only the local-rank shards are exposed via :meth:`parameters`, so an
    optimizer constructed from ``fsdp.parameters()`` will only update this
    rank's slice — mirroring real FSDP's behavior.

    When ``spec.world_size == 1`` this is a transparent pass-through.
    """

    def __init__(self, module: nn.Module, spec: ShardSpec) -> None:
        super().__init__()
        self.spec = spec
        self.inner = module

        # For each inner parameter we record:
        #   - original shape
        #   - original numel
        #   - simulated "other rank" shards (list of full-length shards, stored
        #     as plain tensors — this is what the all-gather stub returns on
        #     behalf of peer ranks).
        #   - the local shard stored as an nn.Parameter so it shows up in
        #     self.parameters() (and so gradients flow to it).
        self._param_shapes: dict[str, tuple[int, ...]] = {}
        self._param_numels: dict[str, int] = {}
        self._peer_shards: dict[str, list[torch.Tensor]] = {}
        self._local_shards = nn.ParameterDict()
        self._param_names: list[str] = []

        for name, p in list(module.named_parameters()):
            safe = name.replace(".", "__")
            self._param_names.append(name)
            self._param_shapes[safe] = tuple(p.shape)
            self._param_numels[safe] = p.numel()

            # Build all shards from the current full parameter (simulates an
            # initial distribution where every rank already has its own slice).
            all_shards: list[torch.Tensor] = []
            if spec.world_size == 1:
                all_shards = [p.data.reshape(-1).detach().clone()]
            else:
                for r in range(spec.world_size):
                    rspec = ShardSpec(world_size=spec.world_size, rank=r, flatten=spec.flatten)
                    all_shards.append(shard_tensor(p.data, rspec))

            local = all_shards[spec.rank].detach().clone()
            local_param = nn.Parameter(local, requires_grad=p.requires_grad)
            self._local_shards[safe] = local_param

            peer = [s.detach().clone() for i, s in enumerate(all_shards) if i != spec.rank]
            self._peer_shards[safe] = peer

    # ------------------------------------------------------------------ utils
    def _safe(self, name: str) -> str:
        return name.replace(".", "__")

    def _allgather_stub(self, safe: str) -> torch.Tensor:
        """Simulated all-gather: reassemble the full tensor for parameter ``safe``."""
        local = self._local_shards[safe]
        peers = self._peer_shards[safe]

        if self.spec.world_size == 1:
            return local.reshape(self._param_shapes[safe])

        ordered: list[torch.Tensor] = []
        peer_iter = iter(peers)
        for r in range(self.spec.world_size):
            if r == self.spec.rank:
                ordered.append(local)
            else:
                ordered.append(next(peer_iter))

        return gather_tensor(
            ordered,
            original_numel=self._param_numels[safe],
            original_shape=self._param_shapes[safe],
        )

    def gather_full_param(self, name: str) -> torch.Tensor:
        """Return the reconstructed full tensor for parameter ``name``."""
        safe = self._safe(name)
        if safe not in self._param_shapes:
            raise KeyError(f"unknown parameter: {name!r}")
        return self._allgather_stub(safe)

    # ------------------------------------------------------------- fwd / api
    def _write_full_params_into_inner(self) -> dict[str, torch.Tensor]:
        """Replace inner params with gathered tensors; return backups for restore."""
        backups: dict[str, torch.Tensor] = {}
        for name in self._param_names:
            safe = self._safe(name)
            full = self._allgather_stub(safe)
            mod, attr = self._resolve(name)
            backups[name] = getattr(mod, attr)
            # Replace the Parameter slot with a plain tensor that still carries
            # gradient back to the local shard (through the gather op).
            delattr(mod, attr)
            setattr(mod, attr, full)
        return backups

    def _restore_inner(self, backups: dict[str, torch.Tensor]) -> None:
        for name, original in backups.items():
            mod, attr = self._resolve(name)
            if hasattr(mod, attr):
                try:
                    delattr(mod, attr)
                except AttributeError:
                    pass
            setattr(mod, attr, original)

    def _resolve(self, dotted: str) -> tuple[nn.Module, str]:
        parts = dotted.split(".")
        mod: nn.Module = self.inner
        for p in parts[:-1]:
            mod = getattr(mod, p)
        return mod, parts[-1]

    def forward(self, *args, **kwargs):  # type: ignore[override]
        if self.spec.world_size == 1:
            return self.inner(*args, **kwargs)

        backups = self._write_full_params_into_inner()
        try:
            out = self.inner(*args, **kwargs)
        finally:
            self._restore_inner(backups)
        # Re-extract the local shard from the (possibly updated) inner param data.
        # In real FSDP this is the "reshard" step that frees the gathered buffers.
        # Here the local shard is the Parameter itself, so nothing to do — the
        # grads have already flowed into self._local_shards[safe].
        return out

    # parameters() inherited from nn.Module already returns only the
    # registered nn.Parameter objects, which are exactly the local shards
    # (plus nothing from inner, since inner's parameters are held as plain
    # tensors during forward and via gathered reassembly — but inner keeps its
    # own Parameter objects too). Override to explicitly expose only the local
    # shards so an optimizer sees the sharded view.

    def parameters(self, recurse: bool = True):  # type: ignore[override]
        return iter(self._local_shards.values())

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ):  # type: ignore[override]
        for name in self._param_names:
            safe = self._safe(name)
            yield (f"{prefix}{name}" if prefix else name), self._local_shards[safe]
