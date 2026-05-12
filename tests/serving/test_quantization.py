from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch


class TestVLLMEngineQuantization(unittest.TestCase):
    def test_vllm_engine_quantization_default(self):
        from src.serving.vllm_engine import VLLMEngine

        engine = VLLMEngine(model_path="/model")
        self.assertIsNone(engine.quantization)
        config = engine.engine_config()
        self.assertIsNone(config["quantization"])

    def test_vllm_engine_quantization_fp8(self):
        from src.serving.vllm_engine import VLLMEngine

        engine = VLLMEngine(model_path="/model", quantization="fp8")
        self.assertEqual(engine.quantization, "fp8")
        config = engine.engine_config()
        self.assertEqual(config["quantization"], "fp8")

    def test_vllm_engine_quantization_int8(self):
        from src.serving.vllm_engine import VLLMEngine

        engine = VLLMEngine(model_path="/model", quantization="int8")
        self.assertEqual(engine.quantization, "int8")

    def test_vllm_engine_quantization_awq(self):
        from src.serving.vllm_engine import VLLMEngine

        engine = VLLMEngine(model_path="/model", quantization="awq")
        self.assertEqual(engine.quantization, "awq")


class TestBuildEngineQuantization(unittest.TestCase):
    @patch("src.serving.engine_loader.VLLMEngine", return_value=MagicMock())
    def test_build_engine_quantization_fp8(self, mock_vllm_engine_cls):
        from src.serving.engine_loader import build_engine

        build_engine("vllm", "/model", quantization="fp8")
        call_kwargs = mock_vllm_engine_cls.call_args.kwargs
        self.assertEqual(call_kwargs["quantization"], "fp8")


if __name__ == "__main__":
    unittest.main()
