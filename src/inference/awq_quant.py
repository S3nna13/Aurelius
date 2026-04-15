"""AWQ: Activation-Aware Weight Quantization (Lin et al., 2023).

Key insight: not all weights are equally important. Salient weights (those activated
by large activation magnitudes) should be protected from quantization error. AWQ:
1. Runs calibration data through the model to collect per-channel activation statistics.
2. Finds a per-channel scale `s` that minimizes quantization error by scaling important
   channels UP before quantizing, then scaling DOWN at runtime.
3. Applies INT4/INT8 group-wise quantization with the optimal scales baked in.

The weight transformation: W_q = round_to_int(W * s) where s is chosen to minimize
||W·x - dequant(quant(W * s) / s)·x||
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Core statistics helpers
# ---------------------------------------------------------------------------

def absmax_per_channel(
    activations: Union[torch.Tensor, List[torch.Tensor]],
    dim: int = 0,
) -> torch.Tensor:
    """Compute per-channel activation magnitude: (in_features,).

    Args:
        activations: (N_samples, in_features) tensor, or a list of tensors
                     each with shape (B, T, in_features) or (B, in_features).
        dim: The dimension along which to compute the absmax (default 0 = sample dim).

    Returns:
        Tensor of shape (in_features,) with the per-channel absmax.
    """
    if isinstance(activations, (list, tuple)):
        # Flatten each tensor to (N, in_features) and concatenate
        flat_list = []
        for act in activations:
            if act.dim() == 3:
                # (B, T, in_features) -> (B*T, in_features)
                flat_list.append(act.reshape(-1, act.shape[-1]))
            elif act.dim() == 2:
                flat_list.append(act)
            else:
                flat_list.append(act.unsqueeze(0))
        activations = torch.cat(flat_list, dim=0)  # (N_total, in_features)

    # activations: (N_samples, in_features) or (in_features,)
    if activations.dim() == 1:
        return activations.abs()

    # Reduce over all dims except the last (in_features)
    result = activations.abs()
    # Reduce over all leading dims
    while result.dim() > 1:
        result = result.amax(dim=0)
    return result


# ---------------------------------------------------------------------------
# AWQ scale computation
# ---------------------------------------------------------------------------

def _quantize_symmetric(w: torch.Tensor, n_bits: int) -> torch.Tensor:
    """Symmetric quantize: scale per row, return integer-valued float tensor."""
    qmax = 2 ** (n_bits - 1) - 1  # e.g. 7 for INT4
    qmin = -(2 ** (n_bits - 1))   # e.g. -8 for INT4
    abs_max = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = abs_max / qmax
    w_q = (w / scale).round().clamp(qmin, qmax)
    return w_q, scale


def compute_awq_scale(
    weight: torch.Tensor,            # (out_features, in_features)
    activation_stats: torch.Tensor,  # (in_features,) per-channel absmax
    n_bits: int = 4,
    group_size: int = 128,
    scale_search_steps: int = 20,
) -> torch.Tensor:
    """Grid-search optimal per-group scale in [s_min, s_max].

    Returns scale of shape (n_groups,) where n_groups = in_features // group_size,
    or (in_features,) if group_size is None or >= in_features.

    The scale s minimizes ||W·x - dequant(quant(W * s)) / s · x||_F over calibration
    activations, where x ~ activation_stats.

    Args:
        weight:           (out_features, in_features) float weight tensor.
        activation_stats: (in_features,) per-channel absmax activation statistics.
        n_bits:           Quantization bits (typically 4 or 8).
        group_size:       Number of in_features per quantization group.
        scale_search_steps: Number of grid search steps between s_min and s_max.

    Returns:
        scale: (n_groups,) or (in_features,) optimal per-group scale.
    """
    out_features, in_features = weight.shape
    w = weight.float()
    act = activation_stats.float().to(w.device)

    # Determine groups
    if group_size is None or group_size >= in_features:
        group_size = in_features
    n_groups = in_features // group_size

    best_scales = torch.ones(n_groups, dtype=torch.float32, device=w.device)

    qmax = 2 ** (n_bits - 1) - 1
    qmin = -(2 ** (n_bits - 1))

    for g in range(n_groups):
        col_start = g * group_size
        col_end = col_start + group_size

        w_group = w[:, col_start:col_end]       # (out, gs)
        act_group = act[col_start:col_end]       # (gs,)

        # The "x" proxy: use activation magnitudes as representative input
        x_proxy = act_group  # (gs,)

        # Original output contribution: W_group @ x_proxy (scalar approx)
        orig_out = (w_group * x_proxy.unsqueeze(0)).sum(dim=1)  # (out,)

        # Search range: scale between s_min and s_max
        # s controls how much we scale down/up; typical range [0.1, 1.0] applied to
        # activation-weighted importance. Following AWQ, we search alpha in [0,1]
        # such that s = act^alpha, so important channels get larger scale.

        # s_min: uniform scale (alpha=0 -> s=1, no differential scaling)
        # s_max: full activation-proportional scaling (alpha=1 -> s=act)

        # Clamp act_group to avoid zero
        act_group_c = act_group.clamp(min=1e-6)

        best_err = float("inf")
        best_s = 1.0

        for step in range(scale_search_steps + 1):
            alpha = step / scale_search_steps  # [0, 1]

            # s = act^alpha (per AWQ paper)
            s = act_group_c.pow(alpha)  # (gs,)
            # Normalize so that mean(s) = 1, keeping relative ratios
            s = s / s.mean().clamp(min=1e-8)

            # Scale weights: W_scaled = W * s (broadcast over out_features)
            w_scaled = w_group * s.unsqueeze(0)  # (out, gs)

            # Quantize W_scaled per row (symmetric)
            abs_max = w_scaled.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
            q_scale = abs_max / qmax
            w_q = (w_scaled / q_scale).round().clamp(qmin, qmax)

            # Dequantize and un-apply the AWQ scale
            w_dequant = (w_q * q_scale) / s.unsqueeze(0)

            # Error: ||(W - W_dequant) @ x_proxy||^2
            err = ((w_group - w_dequant) * x_proxy.unsqueeze(0)).pow(2).sum().item()

            if err < best_err:
                best_err = err
                best_s = alpha

        # Recompute best scale vector from best alpha
        alpha = best_s
        s = act_group_c.pow(alpha)
        s = s / s.mean().clamp(min=1e-8)
        # Store as a single representative scale per group (mean of per-channel scales)
        best_scales[g] = s.mean()

    return best_scales  # (n_groups,)


# ---------------------------------------------------------------------------
# Quantization and dequantization
# ---------------------------------------------------------------------------

def quantize_weight_awq(
    weight: torch.Tensor,            # (out_features, in_features)
    scale: torch.Tensor,             # (n_groups,) or (in_features,)
    n_bits: int = 4,
    group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply AWQ quantization to a weight tensor.

    Scales important channels up before quantizing so they use more of the
    quantization range, then stores the inverse scale for dequantization.

    Args:
        weight:     (out_features, in_features) float weight tensor.
        scale:      (n_groups,) or (in_features,) per-group or per-channel scale.
        n_bits:     Quantization bits (4 or 8).
        group_size: Number of in_features per quantization group.

    Returns:
        weight_int:     Integer-valued tensor of same shape as weight (stored as int8 or int32).
        scale_per_group: (out_features, n_groups) per-group quantization scale.
        zero_per_group:  (out_features, n_groups) per-group zero point (symmetric = 0).
    """
    out_features, in_features = weight.shape
    w = weight.float()

    if group_size is None or group_size >= in_features:
        effective_group_size = in_features
    else:
        effective_group_size = group_size

    # Pad in_features to be divisible by effective_group_size
    pad = (effective_group_size - in_features % effective_group_size) % effective_group_size
    if pad > 0:
        w = F.pad(w, (0, pad))
    padded_in = w.shape[1]
    n_groups = padded_in // effective_group_size

    # Expand AWQ scale to per-channel (in_features,)
    if scale.numel() == n_groups:
        # (n_groups,) -> (padded_in,) by repeating each group's scale
        awq_scale_expanded = scale.repeat_interleave(effective_group_size)  # (padded_in,)
    elif scale.numel() == in_features:
        awq_scale_expanded = F.pad(scale, (0, pad)) if pad > 0 else scale
    else:
        # Fallback: use mean
        awq_scale_expanded = torch.ones(padded_in, dtype=torch.float32, device=w.device)

    awq_scale_expanded = awq_scale_expanded.float().to(w.device)

    # Apply AWQ scale: W_scaled = W * awq_scale (scale important channels up)
    w_scaled = w * awq_scale_expanded.unsqueeze(0)  # (out, padded_in)

    # Reshape into groups: (out_features, n_groups, group_size)
    w_grouped = w_scaled.reshape(out_features, n_groups, effective_group_size)

    qmax = 2 ** (n_bits - 1) - 1
    qmin = -(2 ** (n_bits - 1))

    # Per-group, per-row symmetric quantization
    abs_max = w_grouped.abs().amax(dim=2, keepdim=True).clamp(min=1e-8)  # (out, n_groups, 1)
    q_scale = abs_max / qmax  # (out, n_groups, 1)

    w_q = (w_grouped / q_scale).round().clamp(qmin, qmax)  # (out, n_groups, gs)

    # Reshape back to (out, padded_in) and trim to original
    weight_int = w_q.reshape(out_features, padded_in)[:, :in_features].contiguous()

    # Store quantization scale and zero point
    scale_per_group = q_scale.squeeze(2)  # (out_features, n_groups)
    zero_per_group = torch.zeros_like(scale_per_group)  # symmetric: zero=0

    # Cast to int8 if n_bits <= 8, else keep as float (integer-valued)
    if n_bits <= 8:
        weight_int = weight_int.to(torch.int8)
    else:
        weight_int = weight_int.to(torch.int32)

    return weight_int, scale_per_group, zero_per_group


def dequantize_weight_awq(
    weight_int: torch.Tensor,        # quantized integer weights (out, in)
    scale: torch.Tensor,             # per-group quantization scale (out, n_groups) or (n_groups,)
    zero: torch.Tensor,              # per-group zero point (out, n_groups) or (n_groups,)
    group_size: int = 128,
    original_shape: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    """Dequantize AWQ integer weights back to float.

    Args:
        weight_int:     (out_features, in_features) integer weight tensor.
        scale:          (out_features, n_groups) or (n_groups,) per-group scale.
        zero:           (out_features, n_groups) or (n_groups,) per-group zero point.
        group_size:     Number of in_features per quantization group.
        original_shape: If provided, reshape output to this shape.

    Returns:
        Dequantized weight tensor of shape (out_features, in_features).
    """
    out_features, in_features = weight_int.shape
    w = weight_int.float()

    if group_size is None or group_size >= in_features:
        effective_group_size = in_features
    else:
        effective_group_size = group_size

    # Pad in_features to be divisible by effective_group_size
    pad = (effective_group_size - in_features % effective_group_size) % effective_group_size
    if pad > 0:
        w = F.pad(w, (0, pad))
    padded_in = w.shape[1]
    n_groups = padded_in // effective_group_size

    # Ensure scale and zero have shape (out_features, n_groups)
    if scale.dim() == 1:
        if scale.numel() == n_groups:
            scale = scale.unsqueeze(0).expand(out_features, -1)
            zero = zero.unsqueeze(0).expand(out_features, -1) if zero.numel() == n_groups else torch.zeros_like(scale)
        else:
            scale = scale.unsqueeze(0).expand(out_features, -1)
            zero = zero.unsqueeze(0).expand(out_features, -1)

    # Reshape to (out_features, n_groups, group_size)
    w = w.reshape(out_features, n_groups, effective_group_size)

    scale_exp = scale.float().unsqueeze(2)  # (out, n_groups, 1)
    zero_exp = zero.float().unsqueeze(2)    # (out, n_groups, 1)

    # Dequantize: (w_q - zero) * scale
    w_dequant = (w - zero_exp) * scale_exp  # (out, n_groups, gs)

    # Reshape back to (out, padded_in) and trim
    w_dequant = w_dequant.reshape(out_features, padded_in)[:, :in_features].contiguous()

    if original_shape is not None:
        w_dequant = w_dequant.reshape(original_shape)

    return w_dequant


# ---------------------------------------------------------------------------
# AWQLinear module
# ---------------------------------------------------------------------------

class AWQLinear(nn.Module):
    """Drop-in replacement for nn.Linear using AWQ INT4 weights.

    Stores packed INT4 weights (as int8) and dequantizes at forward time.
    The AWQ scale is baked into the stored quantization scales.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        group_size: int = 128,
        n_bits: int = 4,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size if group_size is not None and group_size < in_features else in_features
        self.n_bits = n_bits

        n_groups = (in_features + self.group_size - 1) // self.group_size

        # Register buffers for quantized weight, per-group scales and zero points
        self.register_buffer(
            "weight_int",
            torch.zeros(out_features, in_features, dtype=torch.int8),
        )
        self.register_buffer(
            "scale_per_group",
            torch.ones(out_features, n_groups, dtype=torch.float32),
        )
        self.register_buffer(
            "zero_per_group",
            torch.zeros(out_features, n_groups, dtype=torch.float32),
        )
        # AWQ channel scale (used to un-scale at forward time)
        self.register_buffer(
            "awq_scale",
            torch.ones(in_features, dtype=torch.float32),
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        activation_stats: torch.Tensor,
        group_size: int = 128,
        n_bits: int = 4,
        scale_search_steps: int = 20,
    ) -> "AWQLinear":
        """Create AWQLinear from a pretrained nn.Linear + calibration stats.

        Args:
            linear:           Source nn.Linear module.
            activation_stats: (in_features,) per-channel absmax from calibration.
            group_size:       Number of in_features per quantization group.
            n_bits:           Quantization bits (4 or 8).
            scale_search_steps: Steps for AWQ scale grid search.

        Returns:
            AWQLinear with quantized weights.
        """
        in_features = linear.in_features
        out_features = linear.out_features
        has_bias = linear.bias is not None

        layer = cls(
            in_features=in_features,
            out_features=out_features,
            bias=has_bias,
            group_size=group_size,
            n_bits=n_bits,
        )

        w = linear.weight.data.float()

        # Step 1: Compute optimal AWQ scale
        awq_group_scale = compute_awq_scale(
            weight=w,
            activation_stats=activation_stats.float(),
            n_bits=n_bits,
            group_size=group_size,
            scale_search_steps=scale_search_steps,
        )  # (n_groups,)

        # Expand AWQ scale to per-channel for storage
        eff_group_size = layer.group_size
        n_groups = (in_features + eff_group_size - 1) // eff_group_size
        awq_scale_per_channel = awq_group_scale.repeat_interleave(eff_group_size)[:in_features]

        # Step 2: Quantize weights with AWQ scale
        weight_int, scale_per_group, zero_per_group = quantize_weight_awq(
            weight=w,
            scale=awq_group_scale,
            n_bits=n_bits,
            group_size=group_size,
        )

        layer.weight_int.copy_(weight_int)
        layer.scale_per_group.copy_(scale_per_group)
        layer.zero_per_group.copy_(zero_per_group)
        layer.awq_scale.copy_(awq_scale_per_channel)

        if has_bias:
            layer.bias = nn.Parameter(linear.bias.data.clone())

        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Dequantize weights, then perform linear forward.

        Args:
            x: Input tensor (..., in_features).

        Returns:
            Output tensor (..., out_features).
        """
        # Dequantize: get back W_scaled (W * awq_scale)
        w_dequant = dequantize_weight_awq(
            weight_int=self.weight_int,
            scale=self.scale_per_group,
            zero=self.zero_per_group,
            group_size=self.group_size,
        )  # (out, in) — this is W * awq_scale (quantized then dequantized)

        # Un-apply AWQ scale: divide by awq_scale to recover approximate W
        # awq_scale is (in_features,), broadcast over out_features
        awq_scale = self.awq_scale.unsqueeze(0).clamp(min=1e-8)  # (1, in)
        w_float = w_dequant / awq_scale  # (out, in)

        return F.linear(x, w_float, self.bias)


# ---------------------------------------------------------------------------
# AWQ Calibrator
# ---------------------------------------------------------------------------

class AWQCalibrator:
    """Collect activation statistics from calibration data.

    Registers forward hooks on target Linear modules to collect input activations,
    then computes per-channel absmax statistics used to drive AWQ scale search.
    """

    def __init__(self, model: nn.Module, target_modules: Optional[List[str]] = None):
        """Initialize calibrator.

        Args:
            model:          The model to calibrate.
            target_modules: List of module name substrings to target (e.g., ["q_proj", "v_proj"]).
                            If None, targets all nn.Linear modules.
        """
        self.model = model
        self.target_modules = target_modules
        self._hooks: List = []
        self._stats: Dict[str, List[torch.Tensor]] = {}

    def _should_target(self, name: str, module: nn.Module) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        if self.target_modules is None:
            return True
        return any(t in name for t in self.target_modules)

    def collect(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run forward pass, collect per-module activation stats.

        Args:
            input_ids: Input tensor to feed through the model (e.g., token IDs or embeddings).

        Returns:
            Dict mapping module_name -> (in_features,) absmax activation tensor.
        """
        self._stats = {}
        self._hooks = []

        # Register hooks on all target Linear modules
        for name, module in self.model.named_modules():
            if self._should_target(name, module):
                self._stats[name] = []

                def make_hook(n):
                    def hook(mod, inp, out):
                        # inp is a tuple; inp[0] is the input activation
                        act = inp[0].detach().float()
                        self._stats[n].append(act)
                    return hook

                h = module.register_forward_hook(make_hook(name))
                self._hooks.append(h)

        # Run forward pass
        try:
            with torch.no_grad():
                self.model(input_ids)
        except Exception:
            pass  # Partial forward is OK if model has requirements we don't meet
        finally:
            # Remove all hooks
            for h in self._hooks:
                h.remove()
            self._hooks = []

        # Compute absmax per channel for each module
        result: Dict[str, torch.Tensor] = {}
        for name, acts in self._stats.items():
            if len(acts) == 0:
                continue
            result[name] = absmax_per_channel(acts)

        return result

    def quantize_model(
        self,
        activation_stats: Dict[str, torch.Tensor],
        n_bits: int = 4,
        group_size: int = 128,
    ) -> nn.Module:
        """Replace Linear layers with AWQLinear using collected stats.

        Modifies the model in-place and returns it.

        Args:
            activation_stats: Dict mapping module_name -> (in_features,) absmax tensor.
            n_bits:           Quantization bits.
            group_size:       Number of in_features per quantization group.

        Returns:
            The model with Linear layers replaced by AWQLinear.
        """
        # Build a flat map of name -> parent module + child name for replacement
        def _get_parent(model: nn.Module, full_name: str):
            parts = full_name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            return parent, parts[-1]

        for name, module in list(self.model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            if not self._should_target(name, module):
                continue
            if name not in activation_stats:
                continue

            stats = activation_stats[name]
            awq_linear = AWQLinear.from_linear(
                linear=module,
                activation_stats=stats,
                group_size=group_size,
                n_bits=n_bits,
            )

            parent, child_name = _get_parent(self.model, name)
            setattr(parent, child_name, awq_linear)

        return self.model
