"""Tests for src/data/tokenization_pipeline.py"""

from src.data.tokenization_pipeline import (
    TokenizationConfig,
    TokenizationPipeline,
    TokenizedOutput,
)

# ---------------------------------------------------------------------------
# TokenizationConfig defaults
# ---------------------------------------------------------------------------


def test_config_default_max_length():
    cfg = TokenizationConfig()
    assert cfg.max_length == 512


def test_config_default_pad_token_id():
    cfg = TokenizationConfig()
    assert cfg.pad_token_id == 0


def test_config_default_bos_token_id():
    cfg = TokenizationConfig()
    assert cfg.bos_token_id == 1


def test_config_default_eos_token_id():
    cfg = TokenizationConfig()
    assert cfg.eos_token_id == 2


def test_config_default_unk_token_id():
    cfg = TokenizationConfig()
    assert cfg.unk_token_id == 3


def test_config_default_truncation():
    cfg = TokenizationConfig()
    assert cfg.truncation is True


def test_config_default_padding():
    cfg = TokenizationConfig()
    assert cfg.padding is True


def test_config_custom():
    cfg = TokenizationConfig(max_length=64, pad_token_id=99)
    assert cfg.max_length == 64
    assert cfg.pad_token_id == 99


# ---------------------------------------------------------------------------
# TokenizedOutput fields
# ---------------------------------------------------------------------------


def test_tokenized_output_fields():
    out = TokenizedOutput(input_ids=[1, 5, 2], attention_mask=[1, 1, 1], length=3, truncated=False)
    assert out.input_ids == [1, 5, 2]
    assert out.attention_mask == [1, 1, 1]
    assert out.length == 3
    assert out.truncated is False


def test_tokenized_output_truncated_flag():
    out = TokenizedOutput(input_ids=[1, 2], attention_mask=[1, 1], length=2, truncated=True)
    assert out.truncated is True


# ---------------------------------------------------------------------------
# TokenizationPipeline.normalize
# ---------------------------------------------------------------------------


def test_normalize_lowercase():
    pipeline = TokenizationPipeline()
    assert pipeline.normalize("Hello World") == "hello world"


def test_normalize_strips_whitespace():
    pipeline = TokenizationPipeline()
    assert pipeline.normalize("  hello  ") == "hello"


def test_normalize_collapses_whitespace():
    pipeline = TokenizationPipeline()
    assert pipeline.normalize("foo  bar\t\nbaz") == "foo bar baz"


def test_normalize_empty():
    pipeline = TokenizationPipeline()
    assert pipeline.normalize("") == ""


def test_normalize_already_normalized():
    pipeline = TokenizationPipeline()
    assert pipeline.normalize("hello world") == "hello world"


# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------


def test_tokenize_starts_with_bos():
    pipeline = TokenizationPipeline()
    ids = pipeline.tokenize("hello")
    assert ids[0] == pipeline.config.bos_token_id


def test_tokenize_ends_with_eos():
    pipeline = TokenizationPipeline()
    ids = pipeline.tokenize("hello")
    assert ids[-1] == pipeline.config.eos_token_id


def test_tokenize_length_equals_normalized_plus_two():
    pipeline = TokenizationPipeline()
    text = "hi"
    ids = pipeline.tokenize(text)
    normalized = pipeline.normalize(text)
    assert len(ids) == len(normalized) + 2


def test_tokenize_empty_string():
    pipeline = TokenizationPipeline()
    ids = pipeline.tokenize("")
    assert ids[0] == pipeline.config.bos_token_id
    assert ids[-1] == pipeline.config.eos_token_id
    assert len(ids) == 2


def test_char_to_id_in_vocab():
    pipeline = TokenizationPipeline(vocab_size=32000)
    for ch in "abcABC123":
        assert 0 <= pipeline.char_to_id(ch) < 32000


# ---------------------------------------------------------------------------
# encode: truncation
# ---------------------------------------------------------------------------


def test_encode_length_equals_max_length():
    pipeline = TokenizationPipeline()
    long_text = "a" * 600
    out = pipeline.encode(long_text)
    assert out.length == pipeline.config.max_length


def test_encode_truncated_flag_true():
    pipeline = TokenizationPipeline()
    long_text = "a" * 600
    out = pipeline.encode(long_text)
    assert out.truncated is True


def test_encode_truncated_last_token_is_eos():
    pipeline = TokenizationPipeline()
    long_text = "a" * 600
    out = pipeline.encode(long_text)
    assert out.input_ids[-1] == pipeline.config.eos_token_id


def test_encode_not_truncated_for_short_text():
    pipeline = TokenizationPipeline()
    out = pipeline.encode("hi")
    assert out.truncated is False


# ---------------------------------------------------------------------------
# encode: padding
# ---------------------------------------------------------------------------


def test_encode_padded_to_max_length():
    pipeline = TokenizationPipeline()
    out = pipeline.encode("hi")
    assert out.length == pipeline.config.max_length


def test_encode_padding_uses_pad_token_id():
    pipeline = TokenizationPipeline()
    out = pipeline.encode("hi")
    # There should be padding tokens at the end
    assert out.input_ids[-1] == pipeline.config.pad_token_id


def test_encode_attention_mask_real_tokens_are_one():
    pipeline = TokenizationPipeline()
    out = pipeline.encode("hi")
    # BOS token (id=1) -> attention_mask should be 1 since id != pad_token_id(0)
    assert out.attention_mask[0] == 1


def test_encode_attention_mask_pad_tokens_are_zero():
    pipeline = TokenizationPipeline()
    out = pipeline.encode("hi")
    # Last tokens are padding (id=0), mask should be 0
    assert out.attention_mask[-1] == 0


def test_encode_attention_mask_length_matches_input_ids():
    pipeline = TokenizationPipeline()
    out = pipeline.encode("hello world")
    assert len(out.attention_mask) == len(out.input_ids)


# ---------------------------------------------------------------------------
# encode: no truncation / no padding
# ---------------------------------------------------------------------------


def test_encode_no_truncation_no_padding_preserves_length():
    cfg = TokenizationConfig(max_length=512, truncation=False, padding=False)
    pipeline = TokenizationPipeline(config=cfg)
    text = "hi"
    out = pipeline.encode(text)
    expected_len = len(pipeline.normalize(text)) + 2  # BOS + chars + EOS
    assert out.length == expected_len


def test_encode_no_truncation_long_text():
    cfg = TokenizationConfig(max_length=5, truncation=False, padding=False)
    pipeline = TokenizationPipeline(config=cfg)
    out = pipeline.encode("hello world")
    assert out.truncated is False
    assert out.length > 5


# ---------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------


def test_decode_returns_string():
    pipeline = TokenizationPipeline()
    result = pipeline.decode([1, 65, 66, 2])
    assert isinstance(result, str)


def test_decode_filters_bos():
    pipeline = TokenizationPipeline()
    result = pipeline.decode([1, 65, 66])
    assert len(result) == 2


def test_decode_filters_eos():
    pipeline = TokenizationPipeline()
    result = pipeline.decode([65, 66, 2])
    assert len(result) == 2


def test_decode_filters_pad():
    pipeline = TokenizationPipeline()
    result = pipeline.decode([65, 0, 66])
    assert len(result) == 2


def test_decode_empty_after_filtering():
    pipeline = TokenizationPipeline()
    result = pipeline.decode([0, 1, 2])
    assert result == ""


# ---------------------------------------------------------------------------
# batch_encode
# ---------------------------------------------------------------------------


def test_batch_encode_returns_list():
    pipeline = TokenizationPipeline()
    results = pipeline.batch_encode(["hello", "world", "foo"])
    assert isinstance(results, list)


def test_batch_encode_length_matches_input():
    pipeline = TokenizationPipeline()
    texts = ["hello", "world", "foo", "bar"]
    results = pipeline.batch_encode(texts)
    assert len(results) == len(texts)


def test_batch_encode_each_is_tokenized_output():
    pipeline = TokenizationPipeline()
    results = pipeline.batch_encode(["hello", "world"])
    for r in results:
        assert isinstance(r, TokenizedOutput)


def test_batch_encode_empty_list():
    pipeline = TokenizationPipeline()
    results = pipeline.batch_encode([])
    assert results == []


def test_batch_encode_all_same_length_with_padding():
    pipeline = TokenizationPipeline()
    results = pipeline.batch_encode(["hi", "hello world", "a longer sentence here"])
    lengths = [r.length for r in results]
    assert len(set(lengths)) == 1  # all padded to max_length
