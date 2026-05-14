"""vLLM engine adapter with paged attention support.

Provides a VLLMEngine class that wraps the vLLM LLM interface for
efficient paged KV cache inference. Lazy-imports vllm so the module
can be imported even when vllm is not installed.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class VLLMEngine:
    """vLLM-backed inference engine with paged attention.

    This class wraps the vLLM ``LLM`` interface to expose synchronous
    and streaming generation. It is constructed via :func:`build_engine`
    in ``engine_loader.py`` and is not intended to be instantiated
    directly by application code.

    Args:
        model_path: Path to the model directory or HuggingFace repo ID.
        tensor_parallel_size: Number of GPUs to split across. Defaults to 1.
        gpu_memory_utilization: Fraction of GPU memory to use for the KV
            cache. Defaults to 0.90.
        dtype: Data type for model weights (``"auto"``, ``"float16"``,
            ``"bfloat16"``). Defaults to ``"auto"``.
        quantization: Quantization method (``"fp8"``, ``"int8"``, ``"awq"``,
            ``"gptq"``) or None. Defaults to None.
        max_num_seqs: Maximum number of sequences to process concurrently.
            Defaults to 256.
        speculative_decoding: Whether to enable speculative decoding.
            Defaults to False.
        n_spec_tokens: Number of speculative tokens for speculative decoding.
            Defaults to 5.
    """

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        dtype: str = "auto",
        quantization: str | None = None,
        max_num_seqs: int = 256,
        speculative_decoding: bool = False,
        n_spec_tokens: int = 5,
    ):
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.dtype = dtype
        self.quantization = quantization
        self.max_num_seqs = max_num_seqs
        self.speculative_decoding = speculative_decoding
        self.n_spec_tokens = n_spec_tokens
        self._llm = None

    def _get_llm(self):
        """Lazily initialise the vLLM LLM instance."""
        if self._llm is not None:
            return self._llm

        try:
            import vllm
        except ImportError as exc:
            raise ImportError("vLLM is not installed; install with: pip install vllm") from exc

        kwargs: dict = {
            "model": self.model_path,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "dtype": self.dtype,
            "max_num_seqs": self.max_num_seqs,
        }

        if self.quantization and self.quantization != "none":
            kwargs["quantization"] = self.quantization

        if self.speculative_decoding:
            try:
                kwargs["speculative_model"] = None
                kwargs["n_spec_tokens"] = self.n_spec_tokens
            except TypeError:
                pass

        self._llm = vllm.LLM(**kwargs)
        return self._llm

    def generate(
        self,
        input_ids: list[int],
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        stop: list[str] | None = None,
    ) -> str:
        """Run synchronous generation on the given token IDs.

        Args:
            input_ids: List of token IDs forming the input prompt.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature (0.0 = deterministic).
            stop: List of stop strings. Generation stops when any are encountered.

        Returns:
            The generated text (stop strings excluded).
        """
        llm = self._get_llm()

        sampling_params = {
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }
        if stop:
            sampling_params["stop"] = stop

        outputs = llm.generate(prompt_token_ids=[input_ids], sampling_params=sampling_params)
        output = outputs[0]
        return output.outputs[0].text

    def generate_streaming(
        self,
        input_ids: list[int],
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        stop: list[str] | None = None,
    ) -> Generator[str, None, None]:
        """Run streaming generation on the given token IDs.

        Args:
            input_ids: List of token IDs forming the input prompt.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature (0.0 = deterministic).
            stop: List of stop strings. Generation stops when any are encountered.

        Yields:
            Text chunks as they are generated.
        """
        llm = self._get_llm()

        sampling_params = {
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }
        if stop:
            sampling_params["stop"] = stop

        result = llm.generate(prompt_token_ids=[input_ids], sampling_params=sampling_params)
        output = result[0]
        stream_output = output.outputs[0]
        events = getattr(stream_output, "events", None)
        if not events:
            text = getattr(stream_output, "text", "")
            if text:
                yield text
            return
        for event in events:
            if event.finish_reason is not None:
                break
            yield event.text

    def is_available(self) -> bool:
        """Return True if vllm is importable, False otherwise."""
        try:
            import vllm  # noqa: F401

            return True
        except ImportError:
            return False

    def engine_config(self) -> dict:
        """Return the engine configuration as a dict."""
        return {
            "model_path": self.model_path,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "dtype": self.dtype,
            "quantization": self.quantization,
            "max_num_seqs": self.max_num_seqs,
            "speculative_decoding": self.speculative_decoding,
            "n_spec_tokens": self.n_spec_tokens,
            "vllm_available": self.is_available(),
        }
