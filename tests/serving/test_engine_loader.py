from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch


class TestBuildEngineMock(unittest.TestCase):
    def test_build_engine_mock(self):
        from src.serving.engine_loader import build_engine

        gen_fn, label, engine_obj = build_engine("mock", "")
        self.assertEqual(label, "mock")
        mock_request = MagicMock()
        mock_request.messages = [{"role": "user", "content": "hello"}]
        result = gen_fn(mock_request)
        self.assertEqual(result, "Mock response to: hello")


class TestBuildEngineEmptyBackend(unittest.TestCase):
    def test_build_engine_empty_backend_uses_mock(self):
        from src.serving.engine_loader import build_engine

        gen_fn, label, engine_obj = build_engine("", "")
        self.assertEqual(label, "mock")


class TestBuildEngineONNX(unittest.TestCase):
    def test_build_engine_onnx_raises(self):
        from src.serving.engine_loader import build_engine

        with self.assertRaises(NotImplementedError):
            build_engine("onnx", "/model")


class TestBuildEngineAgentic(unittest.TestCase):
    @patch(
        "src.serving.engine_loader.build_agentic_request_generate_fn",
        return_value=lambda request: "agentic response",
    )
    @patch(
        "src.serving.engine_loader.load_model_for_chat",
        return_value=(MagicMock(name="model"), MagicMock(name="tokenizer")),
    )
    def test_build_engine_agentic_returns_callable_and_label(
        self,
        mock_load_model_for_chat,
        mock_build_agentic_request_generate_fn,
    ):
        from src.serving.engine_loader import build_engine

        gen_fn, label, engine_obj = build_engine("agentic", "/model/path")
        self.assertEqual(label, "agentic")
        mock_load_model_for_chat.assert_called_once_with("/model/path")
        mock_build_agentic_request_generate_fn.assert_called_once()
        self.assertTrue(callable(gen_fn))
        self.assertEqual(gen_fn(MagicMock()), "agentic response")


class TestBuildEngineVLLM(unittest.TestCase):
    @patch("src.serving.engine_loader.VLLMEngine", return_value=MagicMock())
    def test_build_engine_vllm_returns_callable_and_label(self, mock_vllm_engine_cls):
        from src.serving.engine_loader import build_engine

        gen_fn, label, engine_obj = build_engine("vllm", "/model/path")
        self.assertEqual(label, "vllm")
        mock_vllm_engine_cls.assert_called_once()
        self.assertTrue(callable(gen_fn))

    @patch("src.serving.engine_loader.VLLMEngine", return_value=MagicMock())
    def test_build_engine_vllm_passes_speculative_decoding(self, mock_vllm_engine_cls):
        from src.serving.engine_loader import build_engine

        gen_fn, label, engine_obj = build_engine(
            "vllm",
            "/model/path",
            speculative_decoding=True,
            n_spec_tokens=3,
        )
        mock_vllm_engine_cls.assert_called_once_with(
            model_path="/model/path",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,
            dtype="auto",
            quantization=None,
            max_num_seqs=256,
            speculative_decoding=True,
            n_spec_tokens=3,
        )


class TestBuildEngineQuantization(unittest.TestCase):
    @patch("src.serving.engine_loader.VLLMEngine", return_value=MagicMock())
    def test_build_engine_quantization_fp8(self, mock_vllm_engine_cls):
        from src.serving.engine_loader import build_engine

        build_engine("vllm", "/model", quantization="fp8")
        mock_vllm_engine_cls.assert_called_once()
        call_kwargs = mock_vllm_engine_cls.call_args.kwargs
        self.assertEqual(call_kwargs["quantization"], "fp8")


if __name__ == "__main__":
    unittest.main()
