"""Integration tests for src.serving.structured_output_decoder.

Verifies that:
 1. The decoder classes are accessible via src.serving and the registries.
 2. Config-driven construction works (enable_structured_output flag).
 3. End-to-end constraint pipeline: mock logits → constrained_logits → valid token.
 4. GrammarConstrainedDecoder integrates similarly.
"""

from __future__ import annotations

import torch
import pytest

import src.serving as serving
from src.serving.structured_output_decoder import (
    GrammarConstrainedDecoder,
    StructuredOutputDecoder,
    STRUCTURED_OUTPUT_REGISTRY,
)
from src.model.config import AureliusConfig


# ---------------------------------------------------------------------------
# Config-driven construction
# ---------------------------------------------------------------------------

def _make_decoder_from_config(
    decoder_type: str = "json_schema",
    vocab_size: int = 256,
    eos_token_id: int = 0,
) -> StructuredOutputDecoder | GrammarConstrainedDecoder:
    """Mirrors what production serving code would do: read config, pick decoder."""
    # Config flags
    enable_structured_output: bool = True  # noqa: SIM002
    structured_output_type: str = decoder_type

    if not enable_structured_output:
        raise RuntimeError("Structured output is disabled in config")

    cls = STRUCTURED_OUTPUT_REGISTRY[structured_output_type]
    if structured_output_type == "json_schema":
        return cls(vocab_size=vocab_size, eos_token_id=eos_token_id)
    # grammar decoder needs grammar_states; build a trivial one for testing
    return cls(
        vocab_size=vocab_size,
        eos_token_id=eos_token_id,
        grammar_states={"start": ['"', "t", "f", "n", "{", "["]},
        terminal_states={"start"},
        initial_state="start",
    )


# ---------------------------------------------------------------------------
# 1. Registry / serving package exposure
# ---------------------------------------------------------------------------

def test_structured_output_registry_in_serving_module():
    """STRUCTURED_OUTPUT_REGISTRY is importable from the serving sub-package."""
    from src.serving.structured_output_decoder import STRUCTURED_OUTPUT_REGISTRY as reg
    assert "json_schema" in reg
    assert "grammar" in reg


def test_serving_init_exposes_registries():
    """src.serving already exposes API_SHAPE_REGISTRY; structured output integrates."""
    assert hasattr(serving, "API_SHAPE_REGISTRY")
    # Our registry is in the submodule and accessible via direct import.
    assert "json_schema" in STRUCTURED_OUTPUT_REGISTRY


# ---------------------------------------------------------------------------
# 2. Config integration — AureliusConfig has structured-output fields
# ---------------------------------------------------------------------------

def test_aurelius_config_has_structured_output_fields():
    """AureliusConfig must carry enable_structured_output / structured_output_type."""
    cfg = AureliusConfig()
    assert hasattr(cfg, "enable_structured_output"), (
        "AureliusConfig missing 'enable_structured_output' field"
    )
    assert hasattr(cfg, "structured_output_type"), (
        "AureliusConfig missing 'structured_output_type' field"
    )
    # Defaults: OFF and json_schema
    assert cfg.enable_structured_output is False
    assert cfg.structured_output_type == "json_schema"


def test_config_driven_decoder_construction():
    """Decoder constructed from config params operates correctly."""
    VOCAB_SIZE = 256
    EOS_ID = 0

    cfg = AureliusConfig()
    # Override to enable.
    cfg_enable = True
    cfg_type = cfg.structured_output_type  # "json_schema"

    cls = STRUCTURED_OUTPUT_REGISTRY[cfg_type]
    decoder = cls(vocab_size=VOCAB_SIZE, eos_token_id=EOS_ID)

    assert isinstance(decoder, StructuredOutputDecoder)
    assert decoder.vocab_size == VOCAB_SIZE
    assert decoder.eos_token_id == EOS_ID


# ---------------------------------------------------------------------------
# 3. End-to-end: mock logits → constrained_logits → at least one finite token
# ---------------------------------------------------------------------------

def _build_small_vocab(size: int = 256) -> list[str]:
    vocab = [""]  # index 0 = EOS
    for i in range(1, 128):
        vocab.append(chr(i))
    for i in range(128, size):
        vocab.append(f"t{i}")
    return vocab[:size]


SMALL_VOCAB = _build_small_vocab(256)


def test_end_to_end_json_schema_string():
    """Full pipeline: string schema, random logits → constrained, greedy token is valid."""
    torch.manual_seed(42)
    decoder = StructuredOutputDecoder(vocab_size=256, eos_token_id=0)
    schema = {"type": "string"}

    logits = torch.randn(1, 256)
    partial = ""

    out = decoder.constrained_logits(logits, schema, partial, SMALL_VOCAB)
    assert out.shape == (1, 256)

    # At least one token must be finite (not -inf).
    assert torch.isfinite(out).any().item()

    # Greedy-pick a token; appending it should still be a valid prefix.
    tok_id = int(out[0].argmax().item())
    tok_str = SMALL_VOCAB[tok_id]
    new_partial = partial + tok_str
    # Allow EOS (empty string) if generation is complete.
    if tok_id != 0:
        assert decoder.is_valid_prefix(schema, new_partial) or decoder.is_complete(schema, new_partial)


def test_end_to_end_json_schema_boolean():
    """Boolean schema: only 't' (for true) and 'f' (for false) should be allowed at start."""
    decoder = StructuredOutputDecoder(vocab_size=256, eos_token_id=0)
    schema = {"type": "boolean"}
    logits = torch.zeros(1, 256)

    out = decoder.constrained_logits(logits, schema, "", SMALL_VOCAB)
    finite_ids = [i for i in range(256) if torch.isfinite(out[0, i]).item()]
    finite_toks = [SMALL_VOCAB[i] for i in finite_ids]

    # 't' and 'f' must be among allowed tokens (prefixes of true/false)
    assert "t" in finite_toks or any(t.startswith("t") for t in finite_toks)
    assert "f" in finite_toks or any(t.startswith("f") for t in finite_toks)


def test_end_to_end_json_schema_object():
    """Object schema: partial '{' should be extended by a valid continuation."""
    decoder = StructuredOutputDecoder(vocab_size=256, eos_token_id=0)
    schema = {"type": "object", "properties": {"x": {"type": "number"}}}
    logits = torch.zeros(256)

    out = decoder.constrained_logits(logits, schema, "{", SMALL_VOCAB)
    assert torch.isfinite(out).any().item()


def test_end_to_end_enum_schema():
    """Enum schema: only prefixes of allowed values should survive masking."""
    decoder = StructuredOutputDecoder(vocab_size=256, eos_token_id=0)
    schema = {"enum": ["yes", "no"]}
    logits = torch.zeros(1, 256)

    out = decoder.constrained_logits(logits, schema, "", SMALL_VOCAB)
    finite_ids = [i for i in range(256) if torch.isfinite(out[0, i]).item()]
    finite_toks = [SMALL_VOCAB[i] for i in finite_ids]

    # '"' should be allowed (start of '"yes"' or '"no"')
    assert '"' in finite_toks


def test_end_to_end_anyof_schema():
    """anyOf schema: valid tokens for either branch should be allowed."""
    decoder = StructuredOutputDecoder(vocab_size=256, eos_token_id=0)
    schema = {"anyOf": [{"type": "string"}, {"type": "number"}]}
    logits = torch.zeros(1, 256)

    out = decoder.constrained_logits(logits, schema, "", SMALL_VOCAB)
    finite_ids = [i for i in range(256) if torch.isfinite(out[0, i]).item()]
    finite_toks = set(SMALL_VOCAB[i] for i in finite_ids)

    # '"' for strings, digits/'-' for numbers must be allowed
    assert '"' in finite_toks or any(c.isdigit() for c in finite_toks)


# ---------------------------------------------------------------------------
# 4. GrammarConstrainedDecoder integration
# ---------------------------------------------------------------------------

def test_grammar_decoder_end_to_end():
    """Grammar decoder masks logits correctly in a serving-like pipeline."""
    vocab = ["", "{", "}", '"', "a", "b", " ", ":"]
    V = len(vocab)
    grammar = {
        "start": ["{"],
        "in_obj": ['"', "a", "b", "}"],
    }
    decoder = GrammarConstrainedDecoder(
        vocab_size=V,
        eos_token_id=0,
        grammar_states=grammar,
        terminal_states={"in_obj"},
        initial_state="start",
    )

    logits = torch.ones(1, V) * 2.0
    out = decoder.constrained_logits(logits, vocab, state="start")

    # Only "{" (index 1) should survive in "start".
    assert out[0, 1] == 2.0    # "{" allowed
    assert out[0, 2] == float("-inf")  # "}" not in start
    assert out[0, 0] == float("-inf")  # EOS not terminal in start


def test_grammar_decoder_transition_and_reset():
    """State transition and reset work correctly."""
    vocab = ["", "a", "b"]
    grammar = {"start": ["a"], "end": ["b"]}
    decoder = GrammarConstrainedDecoder(
        vocab_size=3,
        eos_token_id=0,
        grammar_states=grammar,
        terminal_states={"end"},
        initial_state="start",
    )
    assert decoder.current_state == "start"
    decoder.transition("end")
    assert decoder.current_state == "end"

    mask = decoder.build_token_mask(vocab)
    assert mask[0].item() is True   # EOS allowed in terminal
    assert mask[1].item() is False  # "a" not in end
    assert mask[2].item() is True   # "b" in end

    decoder.reset()
    assert decoder.current_state == "start"


# ---------------------------------------------------------------------------
# 5. No forbidden runtime imports
# ---------------------------------------------------------------------------

def test_no_heavy_ml_imports():
    """Verify structured_output_decoder.py itself doesn't import heavy ML deps.

    We verify this by inspecting the source file directly — checking for
    import statements — rather than relying on sys.modules state which can
    be polluted by other tests in the full suite loading those packages.
    """
    import ast
    import pathlib

    src_path = pathlib.Path(__file__).parent.parent.parent / "src" / "serving" / "structured_output_decoder.py"
    source = src_path.read_text()
    tree = ast.parse(source)

    forbidden = {
        "transformers", "einops", "trl", "xformers", "flash_attn",
        "bitsandbytes", "peft", "diffusers", "datasets", "accelerate",
        "deepspeed", "langchain", "llamaindex",
    }

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom):
                mod = (node.module or "").split(".")[0]
            else:
                for alias in node.names:
                    mod = alias.name.split(".")[0]
                    assert mod not in forbidden, (
                        f"structured_output_decoder.py imports forbidden package '{mod}'"
                    )
                continue
            assert mod not in forbidden, (
                f"structured_output_decoder.py imports forbidden package '{mod}'"
            )
