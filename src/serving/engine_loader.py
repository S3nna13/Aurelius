"""Engine loader — selects and instantiates the inference backend.

Provides :func:`build_engine`, which returns a ``generate_fn`` with the
signature ``Callable[[ChatRequest], str]`` and a backend label string.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .api_server import ChatRequest
else:
    ChatRequest = "ChatRequest"

from .agentic_runtime import build_agentic_request_generate_fn
from .chat_session import load_model_for_chat
from .vllm_engine import VLLMEngine


def make_mock_generate_fn() -> Callable[[ChatRequest], str]:
    def _generate(request: ChatRequest) -> str:
        last_user_message = ""
        for msg in reversed(request.messages):
            if msg.get("role") == "user":
                last_user_message = msg.get("content", "")
                break
        return f"Mock response to: {last_user_message}"

    return _generate


def build_engine(
    backend: str,
    model_path: str,
    *,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.90,
    dtype: str = "auto",
    quantization: str | None = None,
    max_num_seqs: int = 256,
    speculative_decoding: bool = False,
    n_spec_tokens: int = 5,
    model_revision: str | None = None,
) -> tuple[Callable[[ChatRequest], str], str, object | None]:
    """Construct and return a generate function and backend label.

    Args:
        backend: One of ``"vllm"``, ``"agentic"``, ``"mock"``, or ``""`` (auto-select).
        model_path: Path or repo ID for the model. May be empty for mock.
        tensor_parallel_size: GPU count for vLLM tensor parallelism.
        gpu_memory_utilization: Fraction of GPU memory for vLLM KV cache.
        dtype: Weight dtype for vLLM (``"auto"``, ``"float16"``, ...).
        quantization: Quantization method or None.
        max_num_seqs: Max concurrent sequences for vLLM.
        speculative_decoding: Enable speculative decoding in vLLM.
        n_spec_tokens: Number of speculative tokens.

    Returns:
        A tuple of ``(generate_fn, backend_label, engine_obj)`` where
        ``engine_obj`` is the underlying engine instance (or ``None`` for mock).
    """
    if backend == "vllm":
        engine = VLLMEngine(
            model_path=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            quantization=quantization,
            max_num_seqs=max_num_seqs,
            speculative_decoding=speculative_decoding,
            n_spec_tokens=n_spec_tokens,
        )
        hf_revision = model_revision or os.environ.get("AURELIUS_HF_REVISION")
        model_is_local = Path(model_path).expanduser().exists()

        def generate_from_engine(request: ChatRequest) -> str:
            try:
                from transformers import AutoTokenizer
            except ImportError:
                raise ImportError(
                    "transformers is required for vLLM backend; install with: "
                    "pip install transformers"
                )

            if not model_is_local and not hf_revision:
                raise ValueError(
                    "Hugging Face model downloads require AURELIUS_HF_REVISION to pin a revision"
                )

            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                revision=hf_revision,
            )
            prompt_tokens = []
            if request.system:
                system_text = request.system
                sys_msg = {"role": "system", "content": system_text}
                prompt_tokens = tokenizer.apply_chat_template(
                    [sys_msg] + request.messages,
                    tokenize=True,
                    add_generation_prompt=True,
                )
            else:
                prompt_tokens = tokenizer.apply_chat_template(
                    request.messages,
                    tokenize=True,
                    add_generation_prompt=True,
                )

            text = engine.generate(
                input_ids=prompt_tokens,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
            )
            return text

        return generate_from_engine, "vllm", engine

    if backend == "onnx":
        raise NotImplementedError("ONNX backend not yet implemented")

    if backend == "agentic":
        if not model_path:
            raise ValueError("agentic backend requires a model_path")
        model, tokenizer = load_model_for_chat(model_path)
        return build_agentic_request_generate_fn(model, tokenizer), "agentic", None

    if backend == "mock" or (backend == "" and model_path == ""):
        return make_mock_generate_fn(), "mock", None

    return make_mock_generate_fn(), "mock", None
