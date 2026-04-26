"""Tests for split learning coordinator."""

from __future__ import annotations

from src.federation.split_learning import (
    SPLIT_LEARNING_REGISTRY,
    GradientPacket,
    SmashData,
    SplitConfig,
    SplitLearningCoordinator,
)


def _coord() -> SplitLearningCoordinator:
    return SplitLearningCoordinator(SplitConfig())


def test_split_config_defaults() -> None:
    cfg = SplitConfig()
    assert cfg.cut_layer == 6
    assert cfg.client_lr == 1e-3
    assert cfg.server_lr == 1e-3
    assert cfg.num_clients == 3


def test_split_config_override() -> None:
    cfg = SplitConfig(cut_layer=4, num_clients=8)
    assert cfg.cut_layer == 4
    assert cfg.num_clients == 8


def test_register_client_adds_once() -> None:
    c = _coord()
    c.register_client("c1")
    c.register_client("c1")
    assert c.client_ids() == ["c1"]


def test_register_multiple_clients() -> None:
    c = _coord()
    c.register_client("a")
    c.register_client("b")
    c.register_client("c")
    assert len(c.client_ids()) == 3


def test_client_ids_returns_copy() -> None:
    c = _coord()
    c.register_client("a")
    ids = c.client_ids()
    ids.append("x")
    assert c.client_ids() == ["a"]


def test_smash_data_defaults_batch_id() -> None:
    s = SmashData(client_id="c1", layer_index=6, activations=[0.1], labels=[1])
    assert isinstance(s.batch_id, str)
    assert len(s.batch_id) == 8


def test_smash_data_unique_batch_ids() -> None:
    a = SmashData(client_id="c1", layer_index=6, activations=[0.1], labels=[0])
    b = SmashData(client_id="c1", layer_index=6, activations=[0.1], labels=[0])
    assert a.batch_id != b.batch_id


def test_smash_data_explicit_batch_id() -> None:
    s = SmashData(
        client_id="c1",
        layer_index=6,
        activations=[0.1],
        labels=[0],
        batch_id="abc12345",
    )
    assert s.batch_id == "abc12345"


def test_receive_smash_stores_data() -> None:
    c = _coord()
    s = SmashData(client_id="c1", layer_index=6, activations=[0.2], labels=[1])
    c.receive_smash(s)
    assert c.pending_batches() == 1


def test_receive_smash_multiple() -> None:
    c = _coord()
    for i in range(5):
        c.receive_smash(
            SmashData(
                client_id="c1",
                layer_index=6,
                activations=[float(i)],
                labels=[i],
                batch_id=f"b{i:07d}",
            )
        )
    assert c.pending_batches() == 5


def test_pending_batches_zero_initially() -> None:
    assert _coord().pending_batches() == 0


def test_compute_server_gradient_identity() -> None:
    c = _coord()
    s = SmashData(client_id="c1", layer_index=6, activations=[1.0, 2.0, 3.0], labels=[0])
    packet = c.compute_server_gradient(s, lambda x: list(x))
    assert packet.gradients == [0.0, 0.0, 0.0]


def test_compute_server_gradient_diff() -> None:
    c = _coord()
    s = SmashData(client_id="c1", layer_index=6, activations=[1.0, 2.0, 3.0], labels=[0])
    packet = c.compute_server_gradient(s, lambda x: [v + 1.0 for v in x])
    assert packet.gradients == [1.0, 1.0, 1.0]


def test_compute_server_gradient_shorter_output() -> None:
    c = _coord()
    s = SmashData(client_id="c1", layer_index=6, activations=[1.0, 2.0, 3.0], labels=[0])
    packet = c.compute_server_gradient(s, lambda x: [x[0] * 2.0])
    assert packet.gradients == [1.0]


def test_compute_server_gradient_preserves_client_id() -> None:
    c = _coord()
    s = SmashData(client_id="alice", layer_index=6, activations=[0.0], labels=[1])
    packet = c.compute_server_gradient(s, lambda x: [0.5])
    assert packet.client_id == "alice"


def test_compute_server_gradient_preserves_batch_id() -> None:
    c = _coord()
    s = SmashData(
        client_id="c1",
        layer_index=6,
        activations=[0.0],
        labels=[1],
        batch_id="deadbeef",
    )
    packet = c.compute_server_gradient(s, lambda x: [0.5])
    assert packet.batch_id == "deadbeef"


def test_send_gradient_returns_packet() -> None:
    c = _coord()
    p = GradientPacket(client_id="c1", batch_id="b1", gradients=[0.1, 0.2])
    assert c.send_gradient(p) is p


def test_send_gradient_preserves_content() -> None:
    c = _coord()
    p = GradientPacket(client_id="c1", batch_id="b1", gradients=[0.1, 0.2])
    out = c.send_gradient(p)
    assert out.gradients == [0.1, 0.2]


def test_smash_data_is_frozen() -> None:
    s = SmashData(client_id="c1", layer_index=6, activations=[0.0], labels=[0])
    try:
        s.client_id = "other"  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("SmashData should be frozen")


def test_gradient_packet_is_frozen() -> None:
    p = GradientPacket(client_id="c1", batch_id="b1", gradients=[0.1])
    try:
        p.client_id = "other"  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("GradientPacket should be frozen")


def test_registry_has_default() -> None:
    assert "default" in SPLIT_LEARNING_REGISTRY
    assert SPLIT_LEARNING_REGISTRY["default"] is SplitLearningCoordinator


def test_coordinator_stores_config() -> None:
    cfg = SplitConfig(cut_layer=3)
    c = SplitLearningCoordinator(cfg)
    assert c.config.cut_layer == 3


def test_multiple_clients_independent_smash() -> None:
    c = _coord()
    c.receive_smash(
        SmashData(
            client_id="a",
            layer_index=6,
            activations=[0.1],
            labels=[0],
            batch_id="b0000001",
        )
    )
    c.receive_smash(
        SmashData(
            client_id="b",
            layer_index=6,
            activations=[0.1],
            labels=[0],
            batch_id="b0000001",
        )
    )
    assert c.pending_batches() == 2


def test_same_batch_id_overwrites_same_client() -> None:
    c = _coord()
    c.receive_smash(
        SmashData(
            client_id="a",
            layer_index=6,
            activations=[0.1],
            labels=[0],
            batch_id="dup00000",
        )
    )
    c.receive_smash(
        SmashData(
            client_id="a",
            layer_index=6,
            activations=[0.2],
            labels=[1],
            batch_id="dup00000",
        )
    )
    assert c.pending_batches() == 1


def test_compute_gradient_empty_output() -> None:
    c = _coord()
    s = SmashData(client_id="c1", layer_index=6, activations=[1.0], labels=[0])
    packet = c.compute_server_gradient(s, lambda x: [])
    assert packet.gradients == []


def test_client_ids_empty_initial() -> None:
    assert _coord().client_ids() == []
