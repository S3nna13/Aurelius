"""ZeRO Stage-1 style optimizer state sharding in pure PyTorch.

This module simulates the optimizer-state partitioning described in ZeRO
(Rajbhandari et al., 2019). The implementation focuses on Stage 1: partition
the Adam optimizer states across ``N_d`` data-parallel ranks while leaving the
model parameters and gradients logically replicated.

The update uses the paper-style variables directly:

* ``N_d``: data-parallel degree
* ``theta_t``: flattened parameter vector at step ``t``
* ``g_t``: flattened gradient vector at step ``t``
* ``m_t``: first-moment optimizer state
* ``v_t``: second-moment optimizer state
* ``P_r``: contiguous parameter partition owned by rank ``r``

The code is single-process and educational by design: each logical rank owns a
slice of ``m_t`` and ``v_t``, computes its local update on ``P_r``, and the
updated ``theta_t`` is gathered back into the original parameter tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn


@dataclass(frozen=True)
class OptimizerStateShardingConfig:
    """Configuration for ZeRO Stage-1 style optimizer state sharding."""

    N_d: int = 2
    alpha: float = 1e-3
    beta_1: float = 0.9
    beta_2: float = 0.999
    lambda_: float = 0.0
    epsilon: float = 1e-8
    bias_correction: bool = True
    state_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        if self.N_d < 1:
            raise ValueError("N_d must be >= 1")
        if self.alpha <= 0.0:
            raise ValueError("alpha must be > 0")
        if not 0.0 <= self.beta_1 < 1.0:
            raise ValueError("beta_1 must be in [0, 1)")
        if not 0.0 <= self.beta_2 < 1.0:
            raise ValueError("beta_2 must be in [0, 1)")
        if self.lambda_ < 0.0:
            raise ValueError("lambda_ must be >= 0")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be > 0")
        if not torch.empty((), dtype=self.state_dtype).is_floating_point():
            raise TypeError("state_dtype must be a floating-point torch dtype")


@dataclass(frozen=True)
class FlatParameterInfo:
    """Metadata needed to map a flat vector back to parameter tensors."""

    name: str
    start: int
    end: int
    shape: torch.Size
    dtype: torch.dtype
    device: torch.device


@dataclass(frozen=True)
class ParameterPartition:
    """Contiguous parameter partition ``P_r`` owned by logical rank ``r``."""

    rank: int
    start: int
    end: int

    @property
    def numel(self) -> int:
        return self.end - self.start


class OptimizerStateSharding:
    """AdamW-style optimizer with ZeRO Stage-1 optimizer-state sharding.

    The optimizer stores ``m_t`` and ``v_t`` only for the local contiguous
    partition ``P_r`` assigned to each logical rank ``r``. Parameters are
    flattened into ``theta_t`` for the update, then written back to the original
    tensors with their original shapes and dtypes preserved.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        config: OptimizerStateShardingConfig | None = None,
    ) -> None:
        self.config = config or OptimizerStateShardingConfig()
        self._params = [param for param in params if param.requires_grad]
        if not self._params:
            raise ValueError("OptimizerStateSharding requires at least one trainable parameter")
        if any(param.is_sparse for param in self._params):
            raise TypeError("Sparse parameters are not supported")

        self._flat_parameter_info = self._build_flat_parameter_info()
        self.P_r = self._build_parameter_partitions()
        self.t = 0
        self.m_t = [
            torch.zeros(
                partition.numel, dtype=self.config.state_dtype, device=self._params[0].device
            )
            for partition in self.P_r
        ]
        self.v_t = [
            torch.zeros(
                partition.numel, dtype=self.config.state_dtype, device=self._params[0].device
            )
            for partition in self.P_r
        ]

    @property
    def flat_parameter_info(self) -> tuple[FlatParameterInfo, ...]:
        return self._flat_parameter_info

    def _build_flat_parameter_info(self) -> tuple[FlatParameterInfo, ...]:
        info: list[FlatParameterInfo] = []
        start = 0
        for index, param in enumerate(self._params):
            end = start + param.numel()
            info.append(
                FlatParameterInfo(
                    name=f"param_{index}",
                    start=start,
                    end=end,
                    shape=param.shape,
                    dtype=param.dtype,
                    device=param.device,
                )
            )
            start = end
        return tuple(info)

    def _build_parameter_partitions(self) -> tuple[ParameterPartition, ...]:
        total_numel = self.numel()
        base = total_numel // self.config.N_d
        remainder = total_numel % self.config.N_d
        partitions: list[ParameterPartition] = []
        start = 0
        for rank in range(self.config.N_d):
            shard_width = base + int(rank < remainder)
            end = start + shard_width
            partitions.append(ParameterPartition(rank=rank, start=start, end=end))
            start = end
        return tuple(partitions)

    def numel(self) -> int:
        return sum(param.numel() for param in self._params)

    def flatten_parameters(self) -> torch.Tensor:
        theta_t = [
            param.detach().reshape(-1).to(dtype=self.config.state_dtype) for param in self._params
        ]
        return torch.cat(theta_t, dim=0)

    def flatten_gradients(self) -> torch.Tensor:
        g_t: list[torch.Tensor] = []
        for param in self._params:
            if param.grad is None:
                raise RuntimeError("All trainable parameters must have gradients before step()")
            if param.grad.is_sparse:
                raise TypeError("Sparse gradients are not supported")
            g_t.append(param.grad.detach().reshape(-1).to(dtype=self.config.state_dtype))
        return torch.cat(g_t, dim=0)

    def partition_vector(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if x.ndim != 1:
            raise ValueError("partition_vector expects a 1D tensor")
        if x.numel() != self.numel():
            raise ValueError("partition_vector received a tensor with incompatible numel")
        return tuple(x[partition.start : partition.end] for partition in self.P_r)

    def gather_vector(self, x_r: Iterable[torch.Tensor]) -> torch.Tensor:
        x_r_tuple = tuple(x_r)
        if len(x_r_tuple) != self.config.N_d:
            raise ValueError("gather_vector requires one shard per rank")
        expected = [partition.numel for partition in self.P_r]
        actual = [shard.numel() for shard in x_r_tuple]
        if actual != expected:
            raise ValueError(
                f"Shard sizes do not match partitions: expected {expected}, got {actual}"
            )
        return torch.cat(x_r_tuple, dim=0)

    def optimizer_state(self, rank: int) -> dict[str, torch.Tensor | int]:
        partition = self.P_r[rank]
        return {
            "rank": rank,
            "start": partition.start,
            "end": partition.end,
            "m_t": self.m_t[rank],
            "v_t": self.v_t[rank],
            "t": self.t,
        }

    @torch.no_grad()
    def step(self) -> torch.Tensor:
        theta_t = self.flatten_parameters()
        g_t = self.flatten_gradients()

        theta_r = self.partition_vector(theta_t)
        g_r = self.partition_vector(g_t)

        self.t += 1
        theta_t_next_r: list[torch.Tensor] = []
        beta_1_power = self.config.beta_1**self.t
        beta_2_power = self.config.beta_2**self.t

        for rank, (theta_part, grad_part) in enumerate(zip(theta_r, g_r)):
            m_part = self.m_t[rank]
            v_part = self.v_t[rank]

            m_part.mul_(self.config.beta_1).add_(grad_part, alpha=1.0 - self.config.beta_1)
            v_part.mul_(self.config.beta_2).addcmul_(
                grad_part, grad_part, value=1.0 - self.config.beta_2
            )

            if self.config.bias_correction:
                m_hat_t = m_part / (1.0 - beta_1_power)
                v_hat_t = v_part / (1.0 - beta_2_power)
            else:
                m_hat_t = m_part
                v_hat_t = v_part

            update_t = m_hat_t / (v_hat_t.sqrt() + self.config.epsilon)
            if self.config.lambda_ != 0.0:
                update_t = update_t + self.config.lambda_ * theta_part

            theta_t_next_r.append(theta_part - self.config.alpha * update_t)

        theta_t_next = self.gather_vector(theta_t_next_r)
        self._write_back(theta_t_next)
        return theta_t_next

    @torch.no_grad()
    def zero_grad(self, set_to_none: bool = False) -> None:
        for param in self._params:
            if param.grad is None:
                continue
            if set_to_none:
                param.grad = None
            else:
                param.grad.zero_()

    @torch.no_grad()
    def _write_back(self, theta_t: torch.Tensor) -> None:
        for param, info in zip(self._params, self._flat_parameter_info):
            param.copy_(
                theta_t[info.start : info.end]
                .view(info.shape)
                .to(dtype=info.dtype, device=info.device)
            )
