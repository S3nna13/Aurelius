"""Tests for EAGLE-2 dynamic draft-tree decoding."""

from __future__ import annotations

import torch

from src.inference.eagle2_decoding import (
    EAGLE2Config,
    EAGLE2Decoder,
    build_dynamic_draft_tree,
    dynamic_branch_width,
    eagle2_loss_reference,
)


def _make_config(**kwargs) -> EAGLE2Config:
    defaults = dict(
        d_model=64,
        vocab_size=256,
        D=3,
        K=3,
        d_hidden=128,
        beta_threshold=0.25,
        max_nodes=32,
    )
    defaults.update(kwargs)
    return EAGLE2Config(**defaults)


def _make_decoder(seed: int = 0, **kwargs) -> EAGLE2Decoder:
    torch.manual_seed(seed)
    return EAGLE2Decoder(_make_config(**kwargs))


def _make_inputs(batch: int = 2, seq_len: int = 5, seed: int = 0):
    torch.manual_seed(seed)
    H = torch.randn(batch, seq_len, 64)
    input_ids = torch.randint(0, 256, (batch, seq_len))
    labels = torch.randint(0, 256, (batch, seq_len))
    attention_mask = torch.ones(batch, seq_len, dtype=torch.bool)
    return H, input_ids, labels, attention_mask


def test_forward_shapes_and_dtypes_on_tiny_config():
    decoder = _make_decoder()
    H, input_ids, _, attention_mask = _make_inputs()

    outputs = decoder(H=H, input_ids=input_ids, attention_mask=attention_mask)

    assert outputs["q"].shape == (2, 5, 3, 256)
    assert outputs["beta"].shape == (2, 5, 3)
    assert outputs["k"].shape == (2, 5, 3)
    assert outputs["q"].dtype == H.dtype
    assert outputs["beta"].dtype == H.dtype
    assert outputs["k"].dtype == torch.long


def test_loss_backward_produces_finite_grads_on_all_params():
    decoder = _make_decoder()
    H, input_ids, labels, attention_mask = _make_inputs()
    H.requires_grad_(True)

    loss = decoder(
        H=H,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )["loss"]
    assert loss is not None
    loss.backward()

    for name, param in decoder.named_parameters():
        assert param.grad is not None, f"Missing grad for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite grad for {name}"


def test_forward_is_deterministic_under_manual_seed():
    H, input_ids, labels, attention_mask = _make_inputs(seed=7)

    decoder_a = _make_decoder(seed=123)
    decoder_b = _make_decoder(seed=123)

    out_a = decoder_a(H=H, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    out_b = decoder_b(H=H, input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    assert torch.allclose(out_a["q"], out_b["q"])
    assert torch.allclose(out_a["beta"], out_b["beta"])
    assert torch.equal(out_a["k"], out_b["k"])
    assert torch.allclose(out_a["loss"], out_b["loss"])


def test_batch_one_seq_len_one_edge_case():
    decoder = _make_decoder()
    H, input_ids, labels, attention_mask = _make_inputs(batch=1, seq_len=1, seed=3)

    outputs = decoder(H=H, input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    assert outputs["q"].shape == (1, 1, 3, 256)
    assert outputs["beta"].shape == (1, 1, 3)
    assert torch.isfinite(outputs["loss"])


def test_padded_positions_are_zeroed_and_pruned():
    decoder = _make_decoder()
    H, input_ids, _, attention_mask = _make_inputs(batch=2, seq_len=4, seed=4)
    attention_mask[1, 2:] = False

    outputs = decoder(H=H, input_ids=input_ids, attention_mask=attention_mask)

    assert torch.count_nonzero(outputs["q"][1, 2:]) == 0
    assert torch.count_nonzero(outputs["beta"][1, 2:]) == 0
    assert torch.count_nonzero(outputs["k"][1, 2:]) == 0


def test_numerical_stability_on_extreme_hidden_states():
    decoder = _make_decoder()
    H, input_ids, labels, attention_mask = _make_inputs(seed=5)
    H = H * 1e4

    outputs = decoder(H=H, input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    assert torch.isfinite(outputs["q"]).all()
    assert torch.isfinite(outputs["beta"]).all()
    assert torch.isfinite(outputs["loss"])


def test_reference_loss_matches_module_loss():
    decoder = _make_decoder(seed=17)
    H, input_ids, labels, attention_mask = _make_inputs(seed=9)

    outputs = decoder(H=H, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    reference = eagle2_loss_reference(
        q=outputs["q"],
        beta=outputs["beta"],
        labels=labels,
        attention_mask=attention_mask,
    )

    assert torch.allclose(outputs["loss"], reference, atol=1e-5)


def test_dynamic_branch_width_stays_in_bounds_and_dtype():
    beta = torch.tensor([[-100.0, 0.0, 100.0]])
    k = dynamic_branch_width(beta, K=4, threshold=0.25)

    assert k.dtype == torch.long
    assert torch.equal(k, torch.tensor([[0, 2, 4]]))


def test_build_dynamic_draft_tree_depth_and_parent_consistency():
    config = _make_config(D=2, K=2, max_nodes=8)
    q = torch.tensor(
        [
            [
                [4.0, 3.0, 1.0, -1.0],
                [2.0, 0.5, -2.0, -3.0],
            ]
        ]
    )
    beta = torch.tensor([[10.0, 10.0]])

    tree = build_dynamic_draft_tree(q=q, beta=beta, config=config)

    active = tree.node_mask[0]
    assert active.sum().item() == 6
    assert torch.equal(tree.depth[0, :6], torch.tensor([1, 1, 2, 2, 2, 2]))
    assert torch.equal(tree.parent_index[0, :2], torch.tensor([-1, -1]))
    assert torch.equal(tree.parent_index[0, 2:6], torch.tensor([0, 0, 1, 1]))


def test_propose_tree_uses_last_valid_prefix_position():
    decoder = _make_decoder(seed=21)
    H, input_ids, _, attention_mask = _make_inputs(batch=2, seq_len=5, seed=8)
    attention_mask[1, 3:] = False

    tree = decoder.propose_tree(H=H, input_ids=input_ids, attention_mask=attention_mask)

    assert tree.token_ids.shape == (2, decoder.config.max_nodes)
    assert tree.node_mask[0].any()
    assert tree.node_mask[1].any()


def test_propose_tree_is_deterministic():
    decoder = _make_decoder(seed=33)
    H, input_ids, _, attention_mask = _make_inputs(seed=11)

    tree_a = decoder.propose_tree(H=H, input_ids=input_ids, attention_mask=attention_mask)
    tree_b = decoder.propose_tree(H=H, input_ids=input_ids, attention_mask=attention_mask)

    assert torch.equal(tree_a.token_ids, tree_b.token_ids)
    assert torch.equal(tree_a.parent_index, tree_b.parent_index)
    assert torch.equal(tree_a.depth, tree_b.depth)
    assert torch.equal(tree_a.node_mask, tree_b.node_mask)
    assert torch.allclose(tree_a.log_prob, tree_b.log_prob)


def test_all_masked_prefix_returns_empty_tree():
    decoder = _make_decoder(seed=55)
    H, input_ids, _, attention_mask = _make_inputs(batch=1, seq_len=4, seed=6)
    attention_mask.zero_()

    tree = decoder.propose_tree(H=H, input_ids=input_ids, attention_mask=attention_mask)

    assert not tree.node_mask.any()
    assert torch.equal(tree.parent_index, torch.full_like(tree.parent_index, -1))
