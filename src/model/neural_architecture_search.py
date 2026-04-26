"""Differentiable Architecture Search (DARTS) for neural architecture search.

Implements bilevel optimization over a DAG-structured search space where each
edge is a MixedOperation — a weighted sum of candidate ops whose weights are
the architecture parameters optimized on a separate validation set.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DARTSConfig:
    """Hyperparameters for the DARTS search space and bilevel training."""

    n_cells: int = 4
    n_nodes: int = 4
    d_model: int = 32
    vocab_size: int = 64
    arch_lr: float = 3e-4
    model_lr: float = 1e-3
    temperature: float = 1.0


# ---------------------------------------------------------------------------
# Candidate operations
# ---------------------------------------------------------------------------


class _ZeroOp(nn.Module):
    """Output is all zeros — allows an edge to be 'dropped'."""

    def forward(self, x: Tensor) -> Tensor:
        return torch.zeros_like(x)


class _IdentityOp(nn.Module):
    """Pass-through: output equals input."""

    def forward(self, x: Tensor) -> Tensor:
        return x


class _LinearOp(nn.Module):
    """Full-rank linear projection."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class _Conv1dK3Op(nn.Module):
    """Kernel-3 depthwise-style 1-D convolution over the sequence dimension.

    Expects input of shape (B, T, D).  Uses causal (left) padding so the
    output length always matches the input length.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=2, groups=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, D) -> (B, D, T) -> conv -> (B, D, T+2) -> trim -> (B, T, D)
        bt = x.transpose(1, 2)  # (B, D, T)
        out = self.conv(bt)  # (B, D, T+2)  due to padding=2
        out = out[:, :, : bt.size(2)]  # causal trim back to (B, D, T)
        return out.transpose(1, 2)  # (B, T, D)


def _build_ops(d_model: int) -> dict[str, nn.Module]:
    """Return the fixed set of candidate operations for every edge."""
    return {
        "linear": _LinearOp(d_model),
        "conv1d_k3": _Conv1dK3Op(d_model),
        "zero": _ZeroOp(),
        "identity": _IdentityOp(),
    }


# ---------------------------------------------------------------------------
# MixedOperation
# ---------------------------------------------------------------------------


class MixedOperation(nn.Module):
    """A single edge in the DARTS DAG — a learned mixture of candidate ops.

    During search the output is a (Gumbel-)softmax-weighted sum of all op
    outputs, keeping every path differentiable.  After search the single
    best-weighted op is extracted via :meth:`get_best_op`.
    """

    def __init__(self, d_model: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature
        ops_dict = _build_ops(d_model)
        # Preserve insertion order (Python 3.7+) so index ↔ name mapping is stable.
        self._op_names: list[str] = list(ops_dict.keys())
        self.operations: dict[str, nn.Module] = {}
        # Register each sub-module properly so parameters are tracked.
        for name, op in ops_dict.items():
            self.operations[name] = op
            self.add_module(name, op)
        n = len(self._op_names)
        # Uniform initialisation — no prior preference.
        self.arch_weights = nn.Parameter(torch.zeros(n))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_probs(self) -> Tensor:
        """Softmax probabilities over operations (shape: n_ops)."""
        return F.softmax(self.arch_weights / self.temperature, dim=0)

    def get_best_op(self) -> str:
        """Operation name with the highest architecture weight."""
        idx = int(self.arch_weights.argmax().item())
        return self._op_names[idx]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """Soft-weighted sum of all op outputs.

        In training mode Gumbel noise is added for exploration; in eval mode
        plain softmax is used for reproducibility.
        """
        if self.training:
            weights = F.gumbel_softmax(
                self.arch_weights,
                tau=self.temperature,
                hard=False,
            )
        else:
            weights = self.get_probs()

        out: Tensor | None = None
        for i, name in enumerate(self._op_names):
            op_out = self.operations[name](x) * weights[i]
            out = op_out if out is None else out + op_out
        assert out is not None  # noqa: S101
        return out


# ---------------------------------------------------------------------------
# Cell
# ---------------------------------------------------------------------------


class Cell(nn.Module):
    """A single DARTS cell: a small DAG with *n_nodes* intermediate nodes.

    Input node indices 0…(k-1) are the cell's *inputs* list.
    Intermediate nodes 0…(n_nodes-1) each receive contributions from all
    preceding nodes (inputs + earlier intermediates).

    The ``edges`` ModuleList is stored in row-major order:
        for dst in range(n_nodes):
            for src in range(n_inputs + dst):
                edges[edge_idx] = MixedOperation(src -> dst)

    Cell output: concatenation of all intermediate node tensors, projected
    back to d_model.
    """

    def __init__(
        self,
        n_nodes: int,
        d_model: int,
        n_inputs: int = 1,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.d_model = d_model
        self.n_inputs = n_inputs

        # Build one MixedOperation per (src, dst) pair.
        edge_list: list[MixedOperation] = []
        self._edge_map: list[tuple[int, int]] = []  # (src, dst) index pairs
        for dst in range(n_nodes):
            n_predecessors = n_inputs + dst
            for src in range(n_predecessors):
                edge_list.append(MixedOperation(d_model, temperature=temperature))
                self._edge_map.append((src, dst))

        self.edges = nn.ModuleList(edge_list)
        self.n_ops: int = len(self.edges)

        # Project concatenated intermediate outputs back to d_model.
        self.proj = nn.Linear(n_nodes * d_model, d_model, bias=False)

    def forward(self, inputs: list[Tensor]) -> Tensor:
        """Compute cell output.

        Args:
            inputs: list of tensors, each (B, T, d_model).  Length must equal
                    ``self.n_inputs``.

        Returns:
            (B, T, d_model) tensor.
        """
        assert len(inputs) == self.n_inputs, (  # noqa: S101
            f"Cell expects {self.n_inputs} input(s), got {len(inputs)}"
        )
        # All nodes: inputs first, then intermediate outputs.
        nodes: list[Tensor] = list(inputs)

        edge_idx = 0
        for dst in range(self.n_nodes):
            n_predecessors = self.n_inputs + dst
            # Sum contributions from all predecessors.
            h: Tensor | None = None
            for src in range(n_predecessors):
                contrib = self.edges[edge_idx](nodes[src])
                h = contrib if h is None else h + contrib
                edge_idx += 1
            assert h is not None  # noqa: S101
            nodes.append(h)  # intermediate node output

        # Concatenate intermediate node outputs along feature dim.
        intermediate = nodes[self.n_inputs :]  # n_nodes tensors
        cat = torch.cat(intermediate, dim=-1)  # (B, T, n_nodes*d_model)
        return self.proj(cat)  # (B, T, d_model)


# ---------------------------------------------------------------------------
# DARTSSearchSpace
# ---------------------------------------------------------------------------


class DARTSSearchSpace(nn.Module):
    """Full DARTS search network: embedding → cells → language-model head.

    Each cell is a :class:`Cell` with learnable mixed operations on edges.
    Architecture parameters (``arch_weights``) are kept separate from model
    weights so bilevel optimisation can update them independently.
    """

    def __init__(self, config: DARTSConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model

        # Token embedding (stem).
        self.stem = nn.Embedding(config.vocab_size, d)

        # Stack of searchable cells.
        self.cells = nn.ModuleList(
            [
                Cell(config.n_nodes, d, n_inputs=1, temperature=config.temperature)
                for _ in range(config.n_cells)
            ]
        )

        # LM head.
        self.head = nn.Linear(d, config.vocab_size, bias=False)

    # ------------------------------------------------------------------

    def forward(self, x: torch.LongTensor) -> Tensor:  # type: ignore[override]
        """Forward pass.

        Args:
            x: integer token ids, shape (B, T).

        Returns:
            logits, shape (B, T, vocab_size).
        """
        h = self.stem(x)  # (B, T, d_model)
        for cell in self.cells:
            h = cell([h])  # each cell takes one input
        return self.head(h)  # (B, T, vocab_size)

    # ------------------------------------------------------------------

    def arch_parameters(self) -> list[nn.Parameter]:
        """All architecture weight parameters (one per edge per cell)."""
        params: list[nn.Parameter] = []
        for cell in self.cells:
            for edge in cell.edges:
                params.append(edge.arch_weights)
        return params

    def model_parameters(self) -> list[nn.Parameter]:
        """All non-architecture parameters (embeddings, projections, op weights)."""
        arch_ids = {id(p) for p in self.arch_parameters()}
        return [p for p in self.parameters() if id(p) not in arch_ids]


# ---------------------------------------------------------------------------
# DARTSTrainer
# ---------------------------------------------------------------------------


class DARTSTrainer:
    """Bilevel DARTS trainer.

    The outer loop updates **architecture weights** on the validation set;
    the inner loop updates **model weights** on the training set.

    Usage::

        trainer = DARTSTrainer(model, train_data_fn, val_data_fn, arch_lr, model_lr)
        for step in range(n_steps):
            train_batch = train_data_fn()
            val_batch   = val_data_fn()
            train_loss, val_loss = trainer.bilevel_step(train_batch, val_batch)
    """

    def __init__(
        self,
        model: DARTSSearchSpace,
        train_data_fn,
        val_data_fn,
        arch_lr: float = 3e-4,
        model_lr: float = 1e-3,
    ) -> None:
        self.model = model
        self.train_data_fn = train_data_fn
        self.val_data_fn = val_data_fn

        self.model_optimizer = torch.optim.Adam(model.model_parameters(), lr=model_lr)
        self.arch_optimizer = torch.optim.Adam(model.arch_parameters(), lr=arch_lr)

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def bilevel_step(
        self,
        train_batch: tuple[Tensor, Tensor],
        val_batch: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        """One bilevel optimisation step.

        Step 1 — update model weights using *train_batch*.
        Step 2 — update architecture weights using *val_batch*.

        Args:
            train_batch: (input_ids, target_ids) for model weight update.
            val_batch:   (input_ids, target_ids) for arch weight update.

        Returns:
            (train_loss, val_loss) as scalar detached Tensors.
        """
        train_x, train_y = train_batch
        val_x, val_y = val_batch

        # --- Step 1: update model weights ---
        self.model.train()
        self.model_optimizer.zero_grad()
        train_logits = self.model(train_x)
        # Flatten for cross-entropy: (B*T, vocab)
        train_loss = F.cross_entropy(
            train_logits.reshape(-1, train_logits.size(-1)),
            train_y.reshape(-1),
        )
        train_loss.backward()
        self.model_optimizer.step()

        # --- Step 2: update architecture weights ---
        self.arch_optimizer.zero_grad()
        val_logits = self.model(val_x)
        val_loss = F.cross_entropy(
            val_logits.reshape(-1, val_logits.size(-1)),
            val_y.reshape(-1),
        )
        val_loss.backward()
        self.arch_optimizer.step()

        return train_loss.detach(), val_loss.detach()

    # ------------------------------------------------------------------

    def derive_architecture(self) -> dict[str, str]:
        """Return the best operation for every edge across all cells.

        Returns:
            Dict mapping ``"cell{c}_edge{e}"`` to the operation name chosen
            by argmax of that edge's architecture weights.
        """
        result: dict[str, str] = {}
        for c, cell in enumerate(self.model.cells):
            for e, edge in enumerate(cell.edges):
                key = f"cell{c}_edge{e}"
                result[key] = edge.get_best_op()
        return result


# ---------------------------------------------------------------------------
# DiscretizedCell
# ---------------------------------------------------------------------------


class DiscretizedCell(nn.Module):
    """A fixed (non-mixed) cell derived from a searched :class:`Cell`.

    Each edge uses only its highest-weighted operation, making inference
    cheaper and the architecture human-readable.
    """

    def __init__(self, source_cell: Cell, derived_ops: dict[str, str]) -> None:
        """
        Args:
            source_cell:  the :class:`Cell` that was searched.
            derived_ops:  mapping ``"edge{e}"`` -> op name, as returned by
                          :meth:`DARTSTrainer.derive_architecture` for this cell
                          (keys without the ``"cell{c}_"`` prefix).
        """
        super().__init__()
        self.n_nodes = source_cell.n_nodes
        self.n_inputs = source_cell.n_inputs
        self.d_model = source_cell.d_model
        self._edge_map = source_cell._edge_map

        # Build fixed ops: one nn.Module per edge.
        fixed_ops: list[nn.Module] = []
        for e, mixed_edge in enumerate(source_cell.edges):
            op_name = derived_ops.get(f"edge{e}", mixed_edge.get_best_op())
            # Retrieve the selected op's state dict and clone it.
            op_module = mixed_edge.operations[op_name]
            import copy

            fixed_ops.append(copy.deepcopy(op_module))

        self.fixed_edges = nn.ModuleList(fixed_ops)

        # Reuse the projection layer.
        import copy

        self.proj = copy.deepcopy(source_cell.proj)

    def forward(self, inputs: list[Tensor]) -> Tensor:
        """Identical DAG traversal as :class:`Cell` but using fixed ops."""
        assert len(inputs) == self.n_inputs  # noqa: S101
        nodes: list[Tensor] = list(inputs)

        edge_idx = 0
        for dst in range(self.n_nodes):
            n_predecessors = self.n_inputs + dst
            h: Tensor | None = None
            for src in range(n_predecessors):
                contrib = self.fixed_edges[edge_idx](nodes[src])
                h = contrib if h is None else h + contrib
                edge_idx += 1
            assert h is not None  # noqa: S101
            nodes.append(h)

        intermediate = nodes[self.n_inputs :]
        cat = torch.cat(intermediate, dim=-1)
        return self.proj(cat)
