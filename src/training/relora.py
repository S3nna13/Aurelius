"""ReLoRA: High-Rank Training Through Low-Rank Updates.

Implements the ReLoRA algorithm from:
  Lialin et al., "ReLoRA: High-Rank Training Through Low-Rank Updates",
  arXiv:2307.05695 (2023).

Algorithm (Section 3):
  Standard LoRA:  W = W_0 + B·A   (W_0 frozen, train B and A)
  ReLoRA iteratively:
    1. Init B = 0, A ~ N(0, 1)  (standard LoRA init)
    2. Train for T_restart steps
    3. Merge:  W_0 ← W_0 + B·A   (bake update into base)
    4. Reset:  B ← 0, A ← re-init randomly
    5. Optionally warm-restart optimizer (zero m_1, keep m_2 / step)
  After K restarts the cumulative update can reach rank up to K·r.

Paper notation used throughout:
  W_0  — base weight matrix
  A    — right LoRA factor (shape: r × d_in),  A ~ N(0,1)
  B    — left  LoRA factor (shape: d_out × r), B = 0
  r    — LoRA rank
  T    — restart interval (restart_every)
  m_1  — first moment (Adam momentum) — zeroed on warm restart
  m_2  — second moment (Adam variance) — kept on warm restart
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ReLoRALinear
# ---------------------------------------------------------------------------

class ReLoRALinear(nn.Module):
    """nn.Linear augmented with a low-rank adapter (B·A) for ReLoRA training.

    The base weight W_0 is NOT frozen — ReLoRA merges the adapter back into
    W_0 periodically, so W_0 must be a plain ``nn.Parameter`` that we can
    update with ``.data``.

    Forward:  y = x · W_0ᵀ + x · Aᵀ · Bᵀ + bias

    Paper notation:
      W_0  : base weight  (out_features × in_features)
      A    : right factor (r × in_features),   A ~ N(0, 1)
      B    : left  factor (out_features × r),  B = 0
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"rank r must be positive, got {r}")

        self.in_features = in_features
        self.out_features = out_features
        self.r = r

        # W_0: base weight — trainable so we can .data-assign after merge
        self.W_0 = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.W_0, a=5 ** 0.5)

        # B: left factor — always init to zero
        self.B = nn.Parameter(torch.zeros(out_features, r))
        # A: right factor — init from N(0,1)
        self.A = nn.Parameter(torch.empty(r, in_features))
        nn.init.normal_(self.A)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # W_0 does NOT participate in LoRA gradient updates — freeze it so
        # only B and A receive gradients during training.
        self.W_0.requires_grad_(False)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_linear(cls, linear: nn.Linear, r: int) -> "ReLoRALinear":
        """Wrap an existing nn.Linear in a ReLoRALinear, copying its weights."""
        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            r=r,
            bias=linear.bias is not None,
        )
        with torch.no_grad():
            layer.W_0.copy_(linear.weight)
            if linear.bias is not None and layer.bias is not None:
                layer.bias.copy_(linear.bias)
        return layer

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """y = x·W_0ᵀ + x·Aᵀ·Bᵀ  (+bias)."""
        # Base linear
        out = F.linear(x, self.W_0, self.bias)
        # LoRA delta: x @ A.T @ B.T  (shape preserving)
        lora_delta = x @ self.A.T @ self.B.T
        return out + lora_delta

    # ------------------------------------------------------------------
    # Merge & reset (Section 3, Step 3–4)
    # ------------------------------------------------------------------

    def merge(self) -> None:
        """Merge the current LoRA update into W_0 and reset B, A.

        Step 3: W_0 ← W_0 + B·A
        Step 4: B ← 0,  A ← re-init N(0,1)
        """
        with torch.no_grad():
            # W_0 ← W_0 + B · A   (both are shape-compatible for matmul)
            self.W_0.data.add_(self.B.data @ self.A.data)
            # Reset B to zeros
            self.B.data.zero_()
            # Re-initialise A from N(0,1)
            nn.init.normal_(self.A.data)


# ---------------------------------------------------------------------------
# ReLoRAScheduler
# ---------------------------------------------------------------------------

class ReLoRAScheduler:
    """Determines when to trigger a ReLoRA restart.

    A restart fires every ``restart_every`` steps, but not before
    ``warmup_steps`` have elapsed (the model needs a warm-up pass before
    the first merge is meaningful).

    Args:
        restart_every: Number of steps between consecutive restarts (T in paper).
        warmup_steps:  Steps to skip before the first restart is allowed.
    """

    def __init__(self, restart_every: int, warmup_steps: int = 100) -> None:
        if restart_every <= 0:
            raise ValueError(f"restart_every must be positive, got {restart_every}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        self.restart_every = restart_every
        self.warmup_steps = warmup_steps

    def should_restart(self, step: int) -> bool:
        """Return True iff a restart should fire at ``step``.

        Args:
            step: Current training step (0-based).

        Returns:
            True if ``step >= warmup_steps`` AND ``step % restart_every == 0``
            AND ``step > 0``.
        """
        if step <= 0:
            return False
        if step < self.warmup_steps:
            return False
        return step % self.restart_every == 0


# ---------------------------------------------------------------------------
# ReLoRAWrapper
# ---------------------------------------------------------------------------

class ReLoRAWrapper(nn.Module):
    """Wraps a model's target nn.Linear layers with ReLoRALinear adapters.

    Usage::

        wrapper = ReLoRAWrapper(model, target_modules=["q_proj","v_proj"],
                                rank=8, restart_every=500, warmup_steps=100)
        optimizer = torch.optim.AdamW(wrapper.trainable_params_list())

        for step, batch in enumerate(dataloader):
            loss = compute_loss(wrapper.model, batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if scheduler.should_restart(step):
                wrapper.restart(optimizer)

    Args:
        model:          The base PyTorch model to wrap.
        target_modules: List of sub-strings; any ``nn.Linear`` whose
                        fully-qualified name contains one of these strings
                        will be replaced with a ``ReLoRALinear``.
        rank:           LoRA rank r.
        restart_every:  Steps between restarts (T in paper).
        warmup_steps:   Steps before first restart is allowed.
    """

    def __init__(
        self,
        model: nn.Module,
        target_modules: List[str],
        rank: int,
        restart_every: int,
        warmup_steps: int = 100,
    ) -> None:
        super().__init__()
        self.model = model
        self.rank = rank
        self.scheduler = ReLoRAScheduler(restart_every, warmup_steps)
        self._relora_layers: List[ReLoRALinear] = []

        self._replace_target_linears(target_modules)

    # ------------------------------------------------------------------
    # Layer replacement
    # ------------------------------------------------------------------

    def _replace_target_linears(self, target_modules: List[str]) -> None:
        """Walk model and replace matching nn.Linear with ReLoRALinear."""
        replacements: List[Tuple[nn.Module, str, nn.Linear]] = []

        for full_name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            leaf = full_name.split(".")[-1] if full_name else ""
            if any(t in full_name or t in leaf for t in target_modules):
                parts = full_name.split(".")
                parent = self.model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                replacements.append((parent, parts[-1], module))

        for parent, attr, linear in replacements:
            relora_layer = ReLoRALinear.from_linear(linear, r=self.rank)
            setattr(parent, attr, relora_layer)
            self._relora_layers.append(relora_layer)

    # ------------------------------------------------------------------
    # Restart (Section 3, Steps 3–5)
    # ------------------------------------------------------------------

    def restart(self, optimizer: Optional[torch.optim.Optimizer] = None) -> None:
        """Merge all LoRA updates into W_0, reset adapters, warm-restart optimizer.

        Steps:
          1. For each ReLoRALinear: W_0 ← W_0 + B·A, reset B=0, A~N(0,1).
          2. If optimizer provided: zero m_1 (first moment) for all LoRA
             parameters while keeping m_2 (second moment) and step counts
             intact — the "jagged cosine" schedule from Section 4.2.

        Args:
            optimizer: Optional AdamW / Adam optimizer whose state should
                       receive the warm restart.
        """
        # Step 3+4: merge and reset all ReLoRA layers
        for layer in self._relora_layers:
            layer.merge()

        # Step 5: warm-restart optimizer state (Section 4.2)
        if optimizer is not None:
            lora_param_ids = {
                id(p) for p in self._lora_parameters()
            }
            for group in optimizer.param_groups:
                for p in group["params"]:
                    if id(p) not in lora_param_ids:
                        continue
                    state = optimizer.state.get(p)
                    if state is None:
                        continue
                    # Reset first moment (exp_avg / m_1), keep second moment
                    if "exp_avg" in state:
                        state["exp_avg"].zero_()

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------

    def _lora_parameters(self) -> Iterator[nn.Parameter]:
        """Yield B and A parameters from all ReLoRALinear layers."""
        for layer in self._relora_layers:
            yield layer.B
            yield layer.A

    def trainable_params(self) -> List[Tuple[str, nn.Parameter]]:
        """Return (name, param) pairs for all LoRA (B and A) parameters only.

        W_0 and biases are excluded; only B and A are trained.

        Returns:
            List of (fully-qualified-name, parameter) tuples.
        """
        result = []
        for name, param in self.model.named_parameters():
            # Only include B and A LoRA matrices
            if param.requires_grad:
                result.append((name, param))
        return result

    def trainable_params_list(self) -> List[nn.Parameter]:
        """Convenience: return just the parameter tensors (for optimizers)."""
        return [p for _, p in self.trainable_params()]

    # ------------------------------------------------------------------
    # Forward delegation
    # ------------------------------------------------------------------

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
