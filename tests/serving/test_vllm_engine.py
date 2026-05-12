from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


class TestVLLMEngineInitDefaults(unittest.TestCase):
    def test_vllm_engine_init_defaults(self):
        from src.serving.vllm_engine import VLLMEngine

        engine = VLLMEngine(model_path="/model")
        self.assertEqual(engine.model_path, "/model")
        self.assertEqual(engine.tensor_parallel_size, 1)
        self.assertEqual(engine.gpu_memory_utilization, 0.90)
        self.assertEqual(engine.dtype, "auto")
        self.assertIsNone(engine.quantization)
        self.assertEqual(engine.max_num_seqs, 256)
        self.assertFalse(engine.speculative_decoding)
        self.assertEqual(engine.n_spec_tokens, 5)
        self.assertIsNone(engine._llm)


class TestVLLMEngineInitCustom(unittest.TestCase):
    def test_vllm_engine_init_custom(self):
        from src.serving.vllm_engine import VLLMEngine

        engine = VLLMEngine(
            model_path="/model",
            tensor_parallel_size=2,
            gpu_memory_utilization=0.8,
            speculative_decoding=True,
            n_spec_tokens=3,
        )
        self.assertEqual(engine.tensor_parallel_size, 2)
        self.assertEqual(engine.gpu_memory_utilization, 0.8)
        self.assertTrue(engine.speculative_decoding)
        self.assertEqual(engine.n_spec_tokens, 3)


class TestVLLMEngineIsAvailable(unittest.TestCase):
    def test_vllm_engine_is_available_false_when_not_installed(self):
        from src.serving.vllm_engine import VLLMEngine

        engine = VLLMEngine(model_path="/model")
        with patch.dict(sys.modules, {"vllm": None}):
            with patch("builtins.__import__", side_effect=ImportError("no vllm")):
                self.assertFalse(engine.is_available())

    def test_vllm_engine_is_available_true_when_installed(self):
        from src.serving.vllm_engine import VLLMEngine

        engine = VLLMEngine(model_path="/model")
        mock_vllm = MagicMock()
        with patch.dict(sys.modules, {"vllm": mock_vllm}):
            self.assertTrue(engine.is_available())


class TestVLLMEngineConfig(unittest.TestCase):
    def test_vllm_engine_engine_config_returns_dict(self):
        from src.serving.vllm_engine import VLLMEngine

        engine = VLLMEngine(
            model_path="/model",
            tensor_parallel_size=2,
            gpu_memory_utilization=0.8,
            speculative_decoding=True,
            n_spec_tokens=3,
        )
        config = engine.engine_config()
        self.assertIsInstance(config, dict)
        self.assertEqual(config["model_path"], "/model")
        self.assertEqual(config["tensor_parallel_size"], 2)
        self.assertEqual(config["gpu_memory_utilization"], 0.8)
        self.assertEqual(config["speculative_decoding"], True)
        self.assertIn("vllm_available", config)


class TestVLLMEngineGenerate(unittest.TestCase):
    def test_vllm_engine_generate_raises_import_error_when_not_available(self):
        from src.serving.vllm_engine import VLLMEngine

        engine = VLLMEngine(model_path="/model")
        original_import = __builtins__["__import__"]

        def failing_import(name, *args, **kwargs):
            if name == "vllm":
                raise ImportError("no vllm")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=failing_import):
            engine._llm = None
            with self.assertRaises(ImportError):
                engine.generate(input_ids=[1, 2, 3])

    def test_vllm_engine_lazy_load_only_loads_on_generate(self):
        from src.serving.vllm_engine import VLLMEngine

        engine = VLLMEngine(model_path="/model")
        self.assertIsNone(engine._llm)

        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="hello world")]

        mock_llm_instance = MagicMock()
        mock_llm_instance.generate.return_value = [mock_output]

        mock_vllm_module = MagicMock()
        mock_vllm_module.LLM.return_value = mock_llm_instance

        with patch.dict(sys.modules, {"vllm": mock_vllm_module}):
            result = engine.generate(input_ids=[1, 2, 3])

        self.assertIsNotNone(engine._llm)
        self.assertEqual(result, "hello world")

    def test_vllm_engine_generate_streaming(self):
        from src.serving.vllm_engine import VLLMEngine

        engine = VLLMEngine(model_path="/model")
        self.assertIsNone(engine._llm)

        mock_event1 = MagicMock()
        mock_event1.finish_reason = None
        mock_event1.text = "hello"

        mock_event2 = MagicMock()
        mock_event2.finish_reason = "stop"
        mock_event2.text = " world"

        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(events=[mock_event1, mock_event2])]

        mock_llm_instance = MagicMock()
        mock_llm_instance.generate.return_value = [mock_output]

        mock_vllm_module = MagicMock()
        mock_vllm_module.LLM.return_value = mock_llm_instance

        with patch.dict(sys.modules, {"vllm": mock_vllm_module}):
            chunks = list(engine.generate_streaming(input_ids=[1, 2, 3]))

        self.assertEqual(chunks, ["hello"])

    def test_vllm_engine_generate_streaming_falls_back_to_text(self):
        from src.serving.vllm_engine import VLLMEngine

        engine = VLLMEngine(model_path="/model")

        mock_output = MagicMock()
        mock_output.outputs = [SimpleNamespace(text="hello world", events=None)]

        mock_llm_instance = MagicMock()
        mock_llm_instance.generate.return_value = [mock_output]

        mock_vllm_module = MagicMock()
        mock_vllm_module.LLM.return_value = mock_llm_instance

        with patch.dict(sys.modules, {"vllm": mock_vllm_module}):
            chunks = list(engine.generate_streaming(input_ids=[1, 2, 3]))

        self.assertEqual(chunks, ["hello world"])


if __name__ == "__main__":
    unittest.main()
