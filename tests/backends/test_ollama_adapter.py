from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from src.backends.ollama_adapter import OLLAMA_REGISTRY, OllamaAdapter, OllamaConfig


def _make_response(data: dict, status: int = 200):
    resp = MagicMock()
    resp.status = status
    resp.read.return_value = json.dumps(data).encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestOllamaConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = OllamaConfig()
        assert cfg.host == "http://localhost:11434"
        assert cfg.model == "llama3"
        assert cfg.timeout_s == 60.0
        assert cfg.stream is False


class TestOllamaAdapterGenerate(unittest.TestCase):
    def test_generate_returns_response_field(self):
        adapter = OllamaAdapter()
        mock_resp = _make_response({"response": "hello world"})
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = adapter.generate("say hi")
        assert result == "hello world"

    def test_generate_sends_correct_body(self):
        adapter = OllamaAdapter(OllamaConfig(model="mistral"))
        mock_resp = _make_response({"response": "ok"})
        captured = []

        def fake_urlopen(req, timeout=None):
            captured.append(json.loads(req.data))
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            adapter.generate("test", max_tokens=128, temperature=0.5)

        body = captured[0]
        assert body["model"] == "mistral"
        assert body["prompt"] == "test"
        assert body["options"]["num_predict"] == 128
        assert body["options"]["temperature"] == 0.5
        assert body["stream"] is False

    def test_generate_raises_runtime_error_on_network_failure(self):
        adapter = OllamaAdapter()
        with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            with self.assertRaises(RuntimeError) as ctx:
                adapter.generate("hello")
        assert "Ollama unavailable" in str(ctx.exception)


class TestOllamaAdapterListModels(unittest.TestCase):
    def test_list_models_parses_tags(self):
        adapter = OllamaAdapter()
        data = {"models": [{"name": "llama3"}, {"name": "mistral"}]}
        mock_resp = _make_response(data)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            models = adapter.list_models()
        assert models == ["llama3", "mistral"]

    def test_list_models_empty(self):
        adapter = OllamaAdapter()
        mock_resp = _make_response({"models": []})
        with patch("urllib.request.urlopen", return_value=mock_resp):
            models = adapter.list_models()
        assert models == []

    def test_list_models_raises_on_error(self):
        adapter = OllamaAdapter()
        with patch("urllib.request.urlopen", side_effect=OSError("network down")):
            with self.assertRaises(RuntimeError):
                adapter.list_models()


class TestOllamaAdapterIsAvailable(unittest.TestCase):
    def test_is_available_true_on_200(self):
        adapter = OllamaAdapter()
        mock_resp = _make_response({"models": []}, status=200)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            assert adapter.is_available() is True

    def test_is_available_false_on_network_error(self):
        adapter = OllamaAdapter()
        with patch("urllib.request.urlopen", side_effect=OSError("refused")):
            assert adapter.is_available() is False

    def test_is_available_false_on_exception(self):
        adapter = OllamaAdapter()
        with patch("urllib.request.urlopen", side_effect=Exception("boom")):
            assert adapter.is_available() is False


class TestOllamaRegistry(unittest.TestCase):
    def test_registry_has_ollama_key(self):
        assert "ollama" in OLLAMA_REGISTRY

    def test_registry_value_is_ollama_adapter_class(self):
        assert OLLAMA_REGISTRY["ollama"] is OllamaAdapter


if __name__ == "__main__":
    unittest.main()
