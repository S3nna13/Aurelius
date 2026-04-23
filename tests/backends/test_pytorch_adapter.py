"""Tests for the reference PyTorch backend adapter.

This is one of the explicit-exemption test files allowed to import
``torch`` directly per the backend-surface integration contract.
"""

from __future__ import annotations

import pytest
import torch

from src.backends import BACKEND_REGISTRY
from src.backends.base import (
    BackendAdapter,
    BackendAdapterError,
    BackendContract,
)
from src.backends.pytorch_adapter import (
    PyTorchAdapter,
    PyTorchTensorNS,
    register,
)


# ---------------------------------------------------------------------------
# BackendContract validation
# ---------------------------------------------------------------------------


def test_contract_happy_path() -> None:
    c = BackendContract(
        backend_name="pytorch",
        engine_contract="1.0.0",
        adapter_contract="1.0.0",
        supports_training=True,
        supports_inference=True,
        capability_tags=("autograd",),
    )
    assert c.backend_name == "pytorch"
    assert c.capability_tags == ("autograd",)


def test_contract_rejects_uppercase_backend_name() -> None:
    with pytest.raises(BackendAdapterError, match="backend_name"):
        BackendContract(
            backend_name="PyTorch",
            engine_contract="1.0.0",
            adapter_contract="1.0.0",
            supports_training=True,
            supports_inference=True,
            capability_tags=(),
        )


def test_contract_rejects_space_in_backend_name() -> None:
    with pytest.raises(BackendAdapterError, match="backend_name"):
        BackendContract(
            backend_name="py torch",
            engine_contract="1.0.0",
            adapter_contract="1.0.0",
            supports_training=True,
            supports_inference=True,
            capability_tags=(),
        )


def test_contract_rejects_empty_backend_name() -> None:
    with pytest.raises(BackendAdapterError, match="backend_name"):
        BackendContract(
            backend_name="",
            engine_contract="1.0.0",
            adapter_contract="1.0.0",
            supports_training=True,
            supports_inference=True,
            capability_tags=(),
        )


def test_contract_rejects_bad_engine_semver() -> None:
    with pytest.raises(BackendAdapterError, match="engine_contract"):
        BackendContract(
            backend_name="pytorch",
            engine_contract="1.0",
            adapter_contract="1.0.0",
            supports_training=True,
            supports_inference=True,
            capability_tags=(),
        )


def test_contract_rejects_bad_adapter_semver() -> None:
    with pytest.raises(BackendAdapterError, match="adapter_contract"):
        BackendContract(
            backend_name="pytorch",
            engine_contract="1.0.0",
            adapter_contract="v1.0.0",
            supports_training=True,
            supports_inference=True,
            capability_tags=(),
        )


def test_contract_rejects_non_tuple_capability_tags() -> None:
    with pytest.raises(BackendAdapterError, match="capability_tags"):
        BackendContract(
            backend_name="pytorch",
            engine_contract="1.0.0",
            adapter_contract="1.0.0",
            supports_training=True,
            supports_inference=True,
            capability_tags=["autograd"],  # type: ignore[arg-type]
        )


def test_contract_rejects_non_str_capability_tag() -> None:
    with pytest.raises(BackendAdapterError, match="capability_tags"):
        BackendContract(
            backend_name="pytorch",
            engine_contract="1.0.0",
            adapter_contract="1.0.0",
            supports_training=True,
            supports_inference=True,
            capability_tags=("autograd", 7),  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# PyTorchAdapter
# ---------------------------------------------------------------------------


def test_pytorch_adapter_is_backend_adapter_instance() -> None:
    adapter = PyTorchAdapter()
    assert isinstance(adapter, BackendAdapter)


def test_pytorch_adapter_contract_identity() -> None:
    adapter = PyTorchAdapter()
    assert adapter.contract.backend_name == "pytorch"
    assert adapter.contract.engine_contract == "1.0.0"
    assert adapter.contract.adapter_contract == "1.0.0"
    assert adapter.contract.supports_training is True
    assert adapter.contract.supports_inference is True
    assert "cuda" in adapter.contract.capability_tags


def test_tensor_ns_zeros_shape_and_dtype() -> None:
    ns = PyTorchAdapter().tensor_ns()
    t = ns.zeros((2, 3))
    assert isinstance(t, torch.Tensor)
    assert tuple(t.shape) == (2, 3)
    assert t.dtype == torch.float32
    assert torch.all(t == 0)


def test_tensor_ns_ones_one_dim() -> None:
    ns = PyTorchAdapter().tensor_ns()
    t = ns.ones((4,))
    assert isinstance(t, torch.Tensor)
    assert tuple(t.shape) == (4,)
    assert torch.all(t == 1)


def test_tensor_ns_as_array_passthrough_and_coerce() -> None:
    ns = PyTorchAdapter().tensor_ns()
    existing = torch.tensor([1.0, 2.0])
    assert ns.as_array(existing) is existing
    coerced = ns.as_array([1, 2, 3])
    assert isinstance(coerced, torch.Tensor)
    assert tuple(coerced.shape) == (3,)


def test_tensor_ns_zeros_with_dtype_string() -> None:
    ns = PyTorchAdapter().tensor_ns()
    t = ns.zeros((2,), dtype="int64")
    assert t.dtype == torch.int64


def test_tensor_ns_rejects_bad_dtype_string() -> None:
    ns = PyTorchAdapter().tensor_ns()
    with pytest.raises(BackendAdapterError, match="unknown torch dtype"):
        ns.zeros((2,), dtype="not_a_dtype")


def test_dtype_of_for_tensor_returns_str() -> None:
    adapter = PyTorchAdapter()
    t = torch.zeros((1,), dtype=torch.float32)
    assert adapter.dtype_of(t) == "torch.float32"


def test_dtype_of_rejects_non_tensor() -> None:
    adapter = PyTorchAdapter()
    with pytest.raises(BackendAdapterError, match="dtype_of"):
        adapter.dtype_of([1, 2, 3])


def test_device_of_cpu_tensor() -> None:
    adapter = PyTorchAdapter()
    t = torch.zeros((1,))
    assert adapter.device_of(t).startswith("cpu")


def test_device_of_rejects_non_tensor() -> None:
    adapter = PyTorchAdapter()
    with pytest.raises(BackendAdapterError, match="device_of"):
        adapter.device_of("not a tensor")


def test_is_available_is_true() -> None:
    assert PyTorchAdapter().is_available() is True


def test_runtime_info_contains_required_keys() -> None:
    info = PyTorchAdapter().runtime_info()
    assert info["backend_name"] == "pytorch"
    assert info["engine_contract"] == "1.0.0"
    assert info["adapter_contract"] == "1.0.0"
    assert info["available"] is True
    assert "torch_version" in info
    assert info["torch_version"] == torch.__version__


def test_tensor_ns_is_pytorch_tensor_ns() -> None:
    adapter = PyTorchAdapter()
    assert isinstance(adapter.tensor_ns(), PyTorchTensorNS)


# ---------------------------------------------------------------------------
# register() idempotence + registry membership
# ---------------------------------------------------------------------------


def test_register_is_idempotent_and_present_in_registry() -> None:
    # src.backends import should have already placed "pytorch" in the
    # registry; calling register() again must not raise.
    assert "pytorch" in BACKEND_REGISTRY
    before = BACKEND_REGISTRY["pytorch"]
    register()
    register()
    assert BACKEND_REGISTRY["pytorch"] is before


def test_registered_adapter_matches_contract_metadata() -> None:
    adapter = BACKEND_REGISTRY["pytorch"]
    assert isinstance(adapter, PyTorchAdapter)
    assert adapter.contract.backend_name == "pytorch"
