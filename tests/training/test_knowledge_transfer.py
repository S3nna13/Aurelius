"""Tests for KnowledgeTransfer weight transfer utilities."""

import torch
import torch.nn as nn

from src.training.knowledge_transfer import (
    KnowledgeTransfer,
    TransferConfig,
    TransferResult,
)


def _linear_pair(in_dim=4, out_dim=4):
    torch.manual_seed(0)
    src = nn.Linear(in_dim, out_dim)
    torch.manual_seed(99)
    tgt = nn.Linear(in_dim, out_dim)
    return src, tgt


def _seq(in_dim=4, hidden=4, out_dim=4):
    torch.manual_seed(0)
    src = nn.Sequential(nn.Linear(in_dim, hidden), nn.Linear(hidden, out_dim))
    torch.manual_seed(99)
    tgt = nn.Sequential(nn.Linear(in_dim, hidden), nn.Linear(hidden, out_dim))
    return src, tgt


class TestTransferConfig:
    def test_defaults(self):
        cfg = TransferConfig()
        assert cfg.layers_to_transfer is None
        assert cfg.transfer_mode == "copy"
        assert cfg.interpolation_alpha == 0.5

    def test_custom(self):
        cfg = TransferConfig(
            layers_to_transfer=[0, 1], transfer_mode="interpolate", interpolation_alpha=0.3
        )
        assert cfg.layers_to_transfer == [0, 1]
        assert cfg.interpolation_alpha == 0.3


class TestTransferResult:
    def test_fields(self):
        r = TransferResult(n_params_transferred=100, n_layers_transferred=2, mode="copy")
        assert r.n_params_transferred == 100
        assert r.n_layers_transferred == 2
        assert r.mode == "copy"


class TestCompatibleKeys:
    def test_matching_keys(self):
        src, tgt = _linear_pair()
        kt = KnowledgeTransfer()
        keys = kt.compatible_keys(src, tgt)
        assert "weight" in keys
        assert "bias" in keys

    def test_incompatible_shape_excluded(self):
        torch.manual_seed(0)
        src = nn.Linear(4, 4)
        torch.manual_seed(1)
        tgt = nn.Linear(4, 8)
        kt = KnowledgeTransfer()
        keys = kt.compatible_keys(src, tgt)
        assert len(keys) == 0

    def test_partial_mismatch(self):
        torch.manual_seed(0)
        src = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 8))
        torch.manual_seed(1)
        tgt = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        kt = KnowledgeTransfer()
        keys = kt.compatible_keys(src, tgt)
        assert "0.weight" in keys
        assert "1.weight" not in keys


class TestCopyTransfer:
    def test_weights_equal_after_copy(self):
        src, tgt = _linear_pair()
        kt = KnowledgeTransfer(TransferConfig(transfer_mode="copy"))
        kt.transfer(src, tgt)
        assert torch.allclose(src.weight, tgt.weight)
        assert torch.allclose(src.bias, tgt.bias)

    def test_result_mode(self):
        src, tgt = _linear_pair()
        kt = KnowledgeTransfer(TransferConfig(transfer_mode="copy"))
        result = kt.transfer(src, tgt)
        assert result.mode == "copy"

    def test_result_n_params_positive(self):
        src, tgt = _linear_pair()
        kt = KnowledgeTransfer()
        result = kt.transfer(src, tgt)
        assert result.n_params_transferred > 0

    def test_source_unchanged(self):
        src, tgt = _linear_pair()
        orig_weight = src.weight.clone()
        kt = KnowledgeTransfer()
        kt.transfer(src, tgt)
        assert torch.allclose(src.weight, orig_weight)


class TestInterpolateTransfer:
    def test_weights_interpolated(self):
        src, tgt = _linear_pair()
        alpha = 0.5
        cfg = TransferConfig(transfer_mode="interpolate", interpolation_alpha=alpha)
        tgt_orig_weight = tgt.weight.clone()
        kt = KnowledgeTransfer(cfg)
        kt.transfer(src, tgt)
        expected = alpha * src.weight + (1.0 - alpha) * tgt_orig_weight
        assert torch.allclose(tgt.weight, expected, atol=1e-6)

    def test_alpha_zero_preserves_target(self):
        src, tgt = _linear_pair()
        tgt_orig = tgt.weight.clone()
        cfg = TransferConfig(transfer_mode="interpolate", interpolation_alpha=0.0)
        kt = KnowledgeTransfer(cfg)
        kt.transfer(src, tgt)
        assert torch.allclose(tgt.weight, tgt_orig, atol=1e-6)

    def test_alpha_one_equals_copy(self):
        src, tgt = _linear_pair()
        cfg = TransferConfig(transfer_mode="interpolate", interpolation_alpha=1.0)
        kt = KnowledgeTransfer(cfg)
        kt.transfer(src, tgt)
        assert torch.allclose(tgt.weight, src.weight, atol=1e-6)


class TestExpandTransfer:
    def test_same_shape_copies(self):
        src, tgt = _linear_pair()
        cfg = TransferConfig(transfer_mode="expand")
        kt = KnowledgeTransfer(cfg)
        result = kt.transfer(src, tgt)
        assert torch.allclose(src.weight, tgt.weight)
        assert result.n_params_transferred > 0


class TestPartialTransfer:
    def test_pattern_weight_only(self):
        src, tgt = _linear_pair()
        tgt_bias_orig = tgt.bias.clone()
        kt = KnowledgeTransfer()
        result = kt.partial_transfer(src, tgt, "weight")
        assert torch.allclose(tgt.weight, src.weight)
        assert torch.allclose(tgt.bias, tgt_bias_orig)
        assert result.n_params_transferred == src.weight.numel()

    def test_pattern_bias_only(self):
        src, tgt = _linear_pair()
        tgt_weight_orig = tgt.weight.clone()
        kt = KnowledgeTransfer()
        kt.partial_transfer(src, tgt, "bias")
        assert torch.allclose(tgt.bias, src.bias)
        assert torch.allclose(tgt.weight, tgt_weight_orig)

    def test_no_match_transfers_nothing(self):
        src, tgt = _linear_pair()
        tgt_weight_orig = tgt.weight.clone()
        kt = KnowledgeTransfer()
        result = kt.partial_transfer(src, tgt, "nonexistent_key_xyz")
        assert result.n_params_transferred == 0
        assert torch.allclose(tgt.weight, tgt_weight_orig)

    def test_result_is_transfer_result(self):
        src, tgt = _linear_pair()
        kt = KnowledgeTransfer()
        result = kt.partial_transfer(src, tgt, "weight")
        assert isinstance(result, TransferResult)
