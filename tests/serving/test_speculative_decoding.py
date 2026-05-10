from __future__ import annotations

import unittest


class TestSpeculativeDecoderInit(unittest.TestCase):
    def test_speculative_decoder_init(self):
        from src.serving.vllm_engine import VLLMEngine

        engine = VLLMEngine(
            model_path="/model",
            speculative_decoding=True,
            n_spec_tokens=5,
        )
        self.assertTrue(engine.speculative_decoding)
        self.assertEqual(engine.n_spec_tokens, 5)


class TestSpeculativeDecoderConfigPropagation(unittest.TestCase):
    def test_speculative_decoder_config_propagation(self):
        from src.serving.vllm_engine import VLLMEngine

        engine = VLLMEngine(
            model_path="/model",
            speculative_decoding=True,
            n_spec_tokens=7,
        )
        config = engine.engine_config()
        self.assertTrue(config["speculative_decoding"])
        self.assertEqual(config["n_spec_tokens"], 7)

    def test_vllm_engine_speculative_config_not_set_when_disabled(self):
        from src.serving.vllm_engine import VLLMEngine

        engine = VLLMEngine(
            model_path="/model",
            speculative_decoding=False,
        )
        config = engine.engine_config()
        self.assertFalse(config["speculative_decoding"])


if __name__ == "__main__":
    unittest.main()
