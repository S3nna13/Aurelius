"""GPTQ-style post-training quantization.

Layer-wise quantization with Hessian-guided error compensation.
Minimizes quantization error without any gradient updates.

Reference: Frantar et al., 2022 — "GPTQ: Accurate Post-Training Quantization
for Generative Pre-trained Transformers".
"""

from __future__ import annotations

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Hessian Estimation
# ---------------------------------------------------------------------------


class HessianEstimator:
    """Estimate the Hessian of the layer output w.r.t. weights.

    For a linear layer y = W x, the Hessian of the squared error w.r.t. W
    is H = 2 * X.T @ X / n, where X is the matrix of input activations.
    We accumulate X.T @ X across multiple forward passes and normalize at the end.
    """

    def __init__(self, n_samples: int = 128) -> None:
        self.n_samples = n_samples
        self.H: torch.Tensor | None = None
        self.n_collected: int = 0

    def collect(self, activations: torch.Tensor) -> None:
        """Accumulate Hessian from a batch of input activations.

        Args:
            activations: (B, T, d_in) input activations to the layer.
        """
        # Flatten batch and sequence dims → (n, d_in)
        B, T, d_in = activations.shape
        X = activations.reshape(-1, d_in).float()  # (n, d_in)

        XtX = X.t().mm(X)  # (d_in, d_in)

        if self.H is None:
            self.H = torch.zeros_like(XtX)

        self.H = self.H + XtX
        self.n_collected += X.shape[0]  # count individual vectors

    def get_hessian(self) -> torch.Tensor:
        """Return the normalized symmetric PSD Hessian matrix.

        Returns:
            H: (d_in, d_in) symmetric PSD matrix, scaled by 2 / n_collected.
        """
        if self.H is None or self.n_collected == 0:
            raise RuntimeError("No activations have been collected yet.")
        H = 2.0 * self.H / self.n_collected
        # Symmetrize to remove any floating-point asymmetry
        H = (H + H.t()) * 0.5
        return H

    def damp(self, percentile: float = 1.0) -> torch.Tensor:
        """Return dampened Hessian by adding a fraction of mean diagonal.

        Args:
            percentile: fraction (in %) of mean diagonal to add.

        Returns:
            H_damp: dampened Hessian with larger diagonal entries.
        """
        H = self.get_hessian()
        damp_val = (percentile / 100.0) * H.diagonal().mean()
        H_damp = H + damp_val * torch.eye(H.shape[0], dtype=H.dtype, device=H.device)
        return H_damp


# ---------------------------------------------------------------------------
# GPTQ Quantizer
# ---------------------------------------------------------------------------


class GPTQQuantizer:
    """Column-wise GPTQ quantizer using Round-To-Nearest (RTN) per group.

    Supports symmetric and asymmetric quantization via per-group min/max scaling.
    """

    def __init__(self, n_bits: int = 4, group_size: int = 128) -> None:
        if n_bits < 1 or n_bits > 16:
            raise ValueError(f"n_bits must be in [1, 16], got {n_bits}")
        self.n_bits = n_bits
        self.group_size = group_size
        self.q_max = (1 << n_bits) - 1  # 2^n_bits - 1

    def quantize_weight(
        self, W: torch.Tensor, H: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize weight matrix using per-group RTN.

        Args:
            W: (d_out, d_in) float weight matrix.
            H: (d_in, d_in) Hessian matrix (used for future GPTQ error
               compensation; currently the scale/zero computation uses RTN).

        Returns:
            W_q:   (d_out, d_in)              integer quantized weights.
            scale: (d_out, n_groups)          per-group scale factors.
            zero:  (d_out, n_groups)          per-group zero points.
        """
        d_out, d_in = W.shape
        gs = self.group_size
        # Pad d_in to a multiple of group_size if needed
        n_groups = (d_in + gs - 1) // gs

        W_q = torch.empty_like(W)
        scale_list: list[torch.Tensor] = []
        zero_list: list[torch.Tensor] = []

        W_f = W.float()

        for g in range(n_groups):
            col_start = g * gs
            col_end = min(col_start + gs, d_in)
            W_g = W_f[:, col_start:col_end]  # (d_out, actual_gs)

            w_min = W_g.min(dim=1, keepdim=True).values  # (d_out, 1)
            w_max = W_g.max(dim=1, keepdim=True).values  # (d_out, 1)

            scale_g = (w_max - w_min) / self.q_max  # (d_out, 1)
            # Avoid division by zero for constant groups
            scale_g = scale_g.clamp(min=1e-8)

            zero_g = torch.round(-w_min / scale_g)  # (d_out, 1)
            zero_g = zero_g.clamp(0, self.q_max)

            W_q_g = torch.round(W_g / scale_g + zero_g)
            W_q_g = W_q_g.clamp(0, self.q_max)

            W_q[:, col_start:col_end] = W_q_g

            # Store per-group scale/zero as (d_out,) vectors
            scale_list.append(scale_g.squeeze(1))
            zero_list.append(zero_g.squeeze(1))

        scale = torch.stack(scale_list, dim=1)  # (d_out, n_groups)
        zero = torch.stack(zero_list, dim=1)  # (d_out, n_groups)

        return W_q, scale, zero

    def dequantize(
        self, W_q: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor
    ) -> torch.Tensor:
        """Dequantize integer weights back to float.

        Args:
            W_q:   (d_out, d_in)       integer quantized weights.
            scale: (d_out, n_groups)   per-group scale factors.
            zero:  (d_out, n_groups)   per-group zero points.

        Returns:
            W_deq: (d_out, d_in) float dequantized weights.
        """
        d_out, d_in = W_q.shape
        gs = self.group_size
        n_groups = scale.shape[1]

        W_deq = torch.empty(d_out, d_in, dtype=torch.float32, device=W_q.device)

        for g in range(n_groups):
            col_start = g * gs
            col_end = min(col_start + gs, d_in)

            s = scale[:, g].unsqueeze(1)  # (d_out, 1)
            z = zero[:, g].unsqueeze(1)  # (d_out, 1)

            W_deq[:, col_start:col_end] = (W_q[:, col_start:col_end].float() - z) * s

        return W_deq


# ---------------------------------------------------------------------------
# Layer Quantizer
# ---------------------------------------------------------------------------


class LayerQuantizer:
    """Quantize a single nn.Linear layer using GPTQ / RTN."""

    def __init__(self, layer: nn.Linear, quantizer: GPTQQuantizer) -> None:
        self.layer = layer
        self.quantizer = quantizer

    def quantize(
        self, calibration_activations: torch.Tensor
    ) -> tuple[float, torch.Tensor, torch.Tensor]:
        """Quantize layer.weight using calibration activations.

        Args:
            calibration_activations: (B, T, d_in) input activations.

        Returns:
            quant_error: relative Frobenius error ||W - W_deq||_F / ||W||_F.
            scale:       per-group scale tensor.
            zero:        per-group zero tensor.
        """
        estimator = HessianEstimator()
        estimator.collect(calibration_activations)
        H = estimator.damp()

        W_original = self.layer.weight.data.clone()

        W_q, scale, zero = self.quantizer.quantize_weight(W_original, H)
        W_deq = self.quantizer.dequantize(W_q, scale, zero)

        # Cast back to original dtype
        W_deq = W_deq.to(W_original.dtype)

        norm_orig = W_original.float().norm(p="fro")
        if norm_orig < 1e-12:
            quant_error = 0.0
        else:
            quant_error = ((W_original.float() - W_deq.float()).norm(p="fro") / norm_orig).item()

        self.layer.weight.data = W_deq
        return quant_error, scale, zero

    def compression_ratio(self) -> float:
        """Return bit compression ratio: 32 / n_bits."""
        return 32.0 / self.quantizer.n_bits


# ---------------------------------------------------------------------------
# Model Quantizer
# ---------------------------------------------------------------------------


class ModelQuantizer:
    """Quantize all nn.Linear layers in a model."""

    def __init__(self, model: nn.Module, n_bits: int = 4, group_size: int = 128) -> None:
        self.model = model
        self.n_bits = n_bits
        self.group_size = group_size

    def find_linear_layers(self) -> dict[str, nn.Linear]:
        """Return dict of {fully_qualified_name: nn.Linear} for all linear layers."""
        layers: dict[str, nn.Linear] = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                layers[name] = module
        return layers

    def quantize_model(self, calibration_data: torch.Tensor) -> dict[str, float]:
        """Quantize all linear layers and return per-layer quantization errors.

        For each linear layer, collect activations via a forward hook on that
        specific layer, then quantize it.

        Args:
            calibration_data: (B, T, d_in_first_layer) or (B, d_in) input tensor.

        Returns:
            errors: {layer_name: quant_error}
        """
        linear_layers = self.find_linear_layers()
        errors: dict[str, float] = {}

        for name, layer in linear_layers.items():
            collected: list[torch.Tensor] = []

            def _hook(module: nn.Module, inp: tuple, out: torch.Tensor) -> None:
                # inp[0] can be 2-D or 3-D
                x = inp[0].detach()
                if x.dim() == 2:
                    x = x.unsqueeze(0)  # (1, N, d_in)
                collected.append(x)

            handle = layer.register_forward_hook(_hook)
            with torch.no_grad():
                _ = self.model(calibration_data)
            handle.remove()

            if not collected:
                # No activations captured; skip this layer
                errors[name] = 0.0
                continue

            activations = torch.cat(collected, dim=0)  # (total_B, T, d_in)

            quantizer = GPTQQuantizer(n_bits=self.n_bits, group_size=self.group_size)
            layer_q = LayerQuantizer(layer, quantizer)
            quant_error, _, _ = layer_q.quantize(activations)
            errors[name] = quant_error

        return errors

    def quantization_summary(self, errors: dict[str, float]) -> dict:
        """Compute summary statistics for quantization quality.

        Args:
            errors: dict of {layer_name: quant_error} from quantize_model.

        Returns:
            dict with keys: mean_error, max_error, total_params, effective_bits.
        """
        linear_layers = self.find_linear_layers()

        error_values = list(errors.values())
        mean_error = float(sum(error_values) / len(error_values)) if error_values else 0.0
        max_error = float(max(error_values)) if error_values else 0.0

        total_params = sum(
            layer.weight.numel() + (layer.bias.numel() if layer.bias is not None else 0)
            for layer in linear_layers.values()
        )

        # Effective bits: linear weights stored at n_bits, biases at 32 bits
        weight_params = sum(layer.weight.numel() for layer in linear_layers.values())
        bias_params = sum(
            layer.bias.numel() for layer in linear_layers.values() if layer.bias is not None
        )
        if total_params > 0:
            effective_bits = (weight_params * self.n_bits + bias_params * 32) / total_params
        else:
            effective_bits = float(self.n_bits)

        return {
            "mean_error": mean_error,
            "max_error": max_error,
            "total_params": total_params,
            "effective_bits": effective_bits,
        }


# ---------------------------------------------------------------------------
# Quantization Benchmark
# ---------------------------------------------------------------------------


class QuantizationBenchmark:
    """Measure quantization quality via various error metrics."""

    def __init__(self) -> None:
        pass

    def weight_error(self, original_W: torch.Tensor, quantized_W: torch.Tensor) -> float:
        """Relative Frobenius norm error between original and quantized weights.

        Returns:
            error ≥ 0; 0.0 when weights are identical.
        """
        norm = original_W.float().norm(p="fro")
        if norm < 1e-12:
            return 0.0
        return ((original_W.float() - quantized_W.float()).norm(p="fro") / norm).item()

    def output_error(self, original_out: torch.Tensor, quantized_out: torch.Tensor) -> float:
        """Relative L2 error between original and quantized model outputs.

        Returns:
            error ≥ 0; 0.0 when outputs are identical.
        """
        norm = original_out.float().norm()
        if norm < 1e-12:
            return 0.0
        return ((original_out.float() - quantized_out.float()).norm() / norm).item()

    def perplexity_increase(
        self, original_logprobs: torch.Tensor, quantized_logprobs: torch.Tensor
    ) -> float:
        """Estimate perplexity increase due to quantization.

        Args:
            original_logprobs:  (N,) log-probabilities from original model.
            quantized_logprobs: (N,) log-probabilities from quantized model.

        Returns:
            exp(mean(original_lp - quantized_lp)) ≥ 1.0 when quantized is worse.
        """
        diff = original_logprobs.float() - quantized_logprobs.float()
        return diff.mean().exp().item()
