#!/usr/bin/env python3
"""Classify attention heads into retrieval vs. streaming for DuoAttention.

This calibration script produces a JSON configuration that maps each layer's
heads to either "retrieval" (full KV) or "streaming" (sink + recent window).

Because the current ``GroupedQueryAttention`` uses fused SDPA kernels that do
not expose raw attention scores, the default mode falls back to a layer-wise
heuristic (deeper layers → more streaming heads).  When ``--entropy-mode`` is
passed, the script temporarily enables manual attention computation inside
``GroupedQueryAttention`` to capture probability matrices, computes per-head
entropy, and classifies heads accordingly.

Example::

    python scripts/classify_attention_heads.py \
        --output configs/duo_attention_heads.json \
        --n-samples 128 --seq-len 512
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# Ensure local src/ takes precedence over any installed package.
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import torch


def compute_entropy(attn_scores: torch.Tensor) -> float:
    """Compute average attention entropy over a score matrix.

    Args:
        attn_scores: Tensor of shape ``(..., seq_len)`` or similar.

    Returns:
        Mean entropy in nats.
    """
    probs = torch.softmax(attn_scores, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean().item()
    return entropy


def _default_classification(
    n_layers: int,
    n_heads: int,
    retrieval_ratio: float = 0.3,
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """Generate a default head split based on layer depth.

    Earlier layers get more retrieval heads; deeper layers get more streaming.
    This follows the empirical observation that lower layers tend to attend
    broadly (retrieval) while upper layers specialize (streaming).
    """
    retrieval_heads: dict[int, list[int]] = {}
    streaming_heads: dict[int, list[int]] = {}

    for layer_idx in range(n_layers):
        # Linearly interpolate: first layer gets ~retrieval_ratio, last layer ~0.1
        ratio = retrieval_ratio * (1.0 - layer_idx / max(n_layers - 1, 1))
        ratio = max(ratio, 0.1)
        n_retrieval = max(1, math.floor(n_heads * ratio))
        retrieval_heads[layer_idx] = list(range(n_retrieval))
        streaming_heads[layer_idx] = list(range(n_retrieval, n_heads))

    return retrieval_heads, streaming_heads


@contextmanager
def _score_capture_context():
    """Context manager that enables attention-score capture in GQA layers."""
    from src.model.attention import GroupedQueryAttention

    GroupedQueryAttention.enable_score_capture()
    try:
        yield
    finally:
        GroupedQueryAttention.disable_score_capture()


def _build_calibration_config(base_config) -> Any:
    """Return a smaller config for fast offline calibration.

    Keeps *n_layers* and *n_heads* identical to the base config so that the
    produced JSON is compatible, but reduces *d_model*, *head_dim*, and *d_ff*
    to make forward passes cheap on CPU.
    """
    from src.model.config import AureliusConfig

    fast_head_dim = max(32, base_config.head_dim // 4)
    fast_d_model = base_config.n_heads * fast_head_dim
    fast_d_ff = max(1024, base_config.d_ff // 4)

    return AureliusConfig(
        d_model=fast_d_model,
        n_layers=base_config.n_layers,
        n_heads=base_config.n_heads,
        n_kv_heads=base_config.n_kv_heads,
        head_dim=fast_head_dim,
        d_ff=fast_d_ff,
        vocab_size=base_config.vocab_size,
        max_seq_len=base_config.max_seq_len,
        rope_theta=base_config.rope_theta,
        rms_norm_eps=base_config.rms_norm_eps,
        dropout=0.0,
        tie_embeddings=base_config.tie_embeddings,
    )


def _entropy_based_classification(
    model: torch.nn.Module,
    n_layers: int,
    n_heads: int,
    calibration_ids: torch.Tensor,
    top_k_ratio: float = 0.3,
    batch_size: int = 16,
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """Calibrate head roles by measuring attention-score entropy.

    Temporarily enables manual attention computation inside
    ``GroupedQueryAttention`` so that probability matrices can be captured.
    Entropy is averaged over all calibration samples for each head in each
    layer.  High-entropy heads (broad attention) are classified as *retrieval*;
    low-entropy heads (focused attention) are classified as *streaming*.

    Args:
        model: The transformer model (already in eval mode).
        n_layers: Number of transformer layers.
        n_heads: Number of attention heads per layer.
        calibration_ids: Integer tensor of shape ``(n_samples, seq_len)``.
        top_k_ratio: Fraction of heads per layer to mark as retrieval.
        batch_size: How many samples to process in a single forward pass.

    Returns:
        Tuple of (retrieval_heads, streaming_heads) dictionaries.
    """
    from src.model.attention import GroupedQueryAttention

    device = next(model.parameters()).device
    calibration_ids = calibration_ids.to(device)
    n_samples = calibration_ids.shape[0]

    # Accumulate entropy per layer and head across all samples.
    entropy_sums: list[list[float]] = [[0.0 for _ in range(n_heads)] for _ in range(n_layers)]
    sample_counts: list[list[int]] = [[0 for _ in range(n_heads)] for _ in range(n_layers)]

    # Map module id -> layer index for clean attribution.
    layer_index_by_module: dict[int, int] = {}
    if hasattr(model, "_cross_layer_stack"):
        # CrossLayerKVStack does not use standard per-layer GQA; skip entropy mode.
        raise RuntimeError(
            "CrossLayerKVStack is active; entropy-based calibration is not supported."
        )

    for layer_idx, layer in enumerate(model.layers):
        if hasattr(layer, "attn") and isinstance(layer.attn, GroupedQueryAttention):
            layer_index_by_module[id(layer.attn)] = layer_idx

    with torch.no_grad(), _score_capture_context():
        for start in range(0, n_samples, batch_size):
            batch = calibration_ids[start : start + batch_size]
            _ = model(batch)

            captured = GroupedQueryAttention.get_captured_scores()
            for module_id, probs in captured.items():
                layer_idx = layer_index_by_module.get(module_id)
                if layer_idx is None:
                    continue
                # probs shape: (batch, n_heads, q_len, k_len)
                if probs.shape[1] != n_heads:
                    continue
                # Compute entropy per head, averaged over query positions.
                # entropy shape: (batch, n_heads)
                head_entropies = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean(dim=-1)
                for b in range(head_entropies.shape[0]):
                    for h in range(n_heads):
                        entropy_sums[layer_idx][h] += head_entropies[b, h].item()
                        sample_counts[layer_idx][h] += 1

            # Clear captured scores between batches to keep memory bounded.
            GroupedQueryAttention._captured_scores.clear()

    # Average entropies and classify.
    retrieval_heads: dict[int, list[int]] = {}
    streaming_heads: dict[int, list[int]] = {}

    for layer_idx in range(n_layers):
        avg_entropies = []
        for h in range(n_heads):
            count = sample_counts[layer_idx][h]
            avg = entropy_sums[layer_idx][h] / max(count, 1)
            avg_entropies.append((h, avg))

        # Sort descending by entropy — high entropy = broad = retrieval.
        avg_entropies.sort(key=lambda x: -x[1])
        n_retrieval = max(1, int(n_heads * top_k_ratio))
        retrieval_heads[layer_idx] = [h for h, _ in avg_entropies[:n_retrieval]]
        streaming_heads[layer_idx] = [h for h, _ in avg_entropies[n_retrieval:]]

    return retrieval_heads, streaming_heads


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate DuoAttention head classification.")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to a checkpoint to load (optional).",
    )
    parser.add_argument(
        "--calibration-data",
        default=None,
        help="Path to calibration texts, one per line (optional).",
    )
    parser.add_argument(
        "--output",
        default="configs/duo_attention_heads.json",
        help="Where to write the classification JSON.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=128,
        help="Number of calibration samples.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
        help="Sequence length for synthetic calibration data.",
    )
    parser.add_argument(
        "--sink-size",
        type=int,
        default=4,
        help="Number of sink tokens for streaming heads.",
    )
    parser.add_argument(
        "--recent-size",
        type=int,
        default=512,
        help="Number of recent tokens for streaming heads.",
    )
    parser.add_argument(
        "--entropy-mode",
        action="store_true",
        help="Use entropy-based calibration (requires attention score hooks).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        default=True,
        help="Use a reduced-dimension model for fast CPU calibration (default: True).",
    )
    parser.add_argument(
        "--no-fast",
        action="store_false",
        dest="fast",
        help="Disable fast calibration and use the full model architecture.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for entropy-based calibration forward passes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    from src.model.config import AureliusConfig
    from src.model.transformer import AureliusTransformer

    base_config = AureliusConfig()

    # When a checkpoint is provided we must honour its architecture; otherwise
    # we can use a stripped-down config for speedy CPU calibration.
    if args.model_path is not None:
        config = base_config
        use_fast = False
    else:
        config = _build_calibration_config(base_config) if args.fast else base_config
        use_fast = args.fast

    model = AureliusTransformer(config)
    if args.model_path is not None:
        state = torch.load(args.model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
    model.eval()

    n_layers = base_config.n_layers
    n_heads = base_config.n_heads

    # ------------------------------------------------------------------
    # Calibration data
    # ------------------------------------------------------------------
    if args.calibration_data and os.path.isfile(args.calibration_data):
        with open(args.calibration_data, encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        # Tokenizer is not available by default; fall back to random IDs
        print(f"Loaded {len(lines)} calibration lines; using synthetic token IDs.")

    # Synthetic token IDs for forward-pass probing
    calibration_ids = torch.randint(0, config.vocab_size, (args.n_samples, args.seq_len))

    # ------------------------------------------------------------------
    # Classify heads
    # ------------------------------------------------------------------
    method = "heuristic"
    if args.entropy_mode:
        try:
            retrieval_heads, streaming_heads = _entropy_based_classification(
                model,
                n_layers,
                n_heads,
                calibration_ids,
                batch_size=args.batch_size,
            )
            method = "entropy"
            if use_fast:
                method += " (fast-calibration)"
        except Exception as exc:
            print(f"WARNING: Entropy-based calibration failed: {exc}")
            print("Falling back to default heuristic classification.")
            retrieval_heads, streaming_heads = _default_classification(n_layers, n_heads)
    else:
        retrieval_heads, streaming_heads = _default_classification(n_layers, n_heads)

    result: dict[str, Any] = {
        "retrieval_heads": retrieval_heads,
        "streaming_heads": streaming_heads,
        "sink_size": args.sink_size,
        "recent_size": args.recent_size,
        "metadata": {
            "model": "AureliusTransformer",
            "n_layers": n_layers,
            "n_heads": n_heads,
            "calibration_samples": args.n_samples,
            "calibration_seq_len": args.seq_len,
            "method": method,
        },
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved DuoAttention head classification to {args.output}")

    # Print summary
    total_retrieval = sum(len(v) for v in retrieval_heads.values())
    total_streaming = sum(len(v) for v in streaming_heads.values())
    print(f"  Total retrieval heads: {total_retrieval} / {n_layers * n_heads}")
    print(f"  Total streaming heads: {total_streaming} / {n_layers * n_heads}")


if __name__ == "__main__":
    main()
