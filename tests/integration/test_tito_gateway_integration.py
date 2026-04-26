"""Integration tests for the TITO Gateway registry wiring (GLM-5 §4.1).

Verifies that:
1. "tito" is present in TRAINING_REGISTRY after importing src.training.
2. A TITOGateway can be constructed via the registry entry and processes a batch.
3. Pre-existing registry keys are not disturbed by the additive wiring.
"""

from src.training import TRAINING_REGISTRY
from src.training.tito_gateway import TITOConfig

# ---------------------------------------------------------------------------
# 1. Registry key existence
# ---------------------------------------------------------------------------


def test_tito_key_in_training_registry():
    """'tito' must be registered in TRAINING_REGISTRY."""
    assert "tito" in TRAINING_REGISTRY, (
        "TRAINING_REGISTRY is missing 'tito'. "
        "Ensure src/training/__init__.py contains "
        'TRAINING_REGISTRY["tito"] = TITOGateway.'
    )


# ---------------------------------------------------------------------------
# 2. Construct via registry and exercise wrap_batch
# ---------------------------------------------------------------------------


def test_registry_construct_and_wrap_batch():
    """Gateway constructed from the registry class should canonicalize a batch."""
    GatewayClass = TRAINING_REGISTRY["tito"]

    config = TITOConfig(
        vocab_size=5000,
        pad_id=0,
        unk_id=1,
        id_remap={4096: 10},  # remap an inference-side special token
    )
    gw = GatewayClass(config)

    batch = [
        [0, 1, 42],
        [4096, 999, 0],  # 4096 → 10 via remap
        [],
    ]
    result = gw.wrap_batch(batch)

    assert result[0] == [0, 1, 42]
    assert result[1] == [10, 999, 0]  # remap applied
    assert result[2] == []


# ---------------------------------------------------------------------------
# 3. Pre-existing registry key is undisturbed
# ---------------------------------------------------------------------------


def test_existing_registry_key_still_present():
    """Adding 'tito' must not remove pre-existing registry entries."""
    # The tool_call_supervision key was wired before TITO; it must still exist.
    from src.training import AUXILIARY_LOSS_REGISTRY

    assert "tool_call_supervision" in AUXILIARY_LOSS_REGISTRY, (
        "AUXILIARY_LOSS_REGISTRY lost its 'tool_call_supervision' entry — "
        "the __init__.py edit was not purely additive."
    )
