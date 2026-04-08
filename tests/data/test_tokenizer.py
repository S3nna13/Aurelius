import pytest
from src.data.simple_tokenizer import (
    TokenizerConfig, CharTokenizer, SimpleTokenizer, VocabBuilder, _tokenize_words
)

def test_tokenize_words():
    tokens = _tokenize_words("Hello, World! How are you?")
    assert "hello" in tokens
    assert "world" in tokens

def test_char_tokenizer_encode_decode():
    tok = CharTokenizer()
    text = "Hello world"
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    assert decoded == text

def test_char_tokenizer_bos_eos():
    tok = CharTokenizer()
    ids = tok.encode("hi", add_bos=True, add_eos=True)
    assert ids[0] == tok.token_to_id["<bos>"]
    assert ids[-1] == tok.token_to_id["<eos>"]
    assert len(ids) == 4  # bos + h + i + eos

def test_char_tokenizer_unk():
    tok = CharTokenizer()
    ids = tok.encode("\x00")  # null char not in vocab
    assert ids[0] == tok.unk_id

def test_char_tokenizer_vocab_size():
    tok = CharTokenizer()
    assert tok.vocab_size > 90  # 4 specials + 95 printable + 2 whitespace

def test_vocab_builder_builds_vocab():
    builder = VocabBuilder(max_vocab_size=100, min_freq=1)
    builder.update("the cat sat on the mat the cat")
    vocab = builder.build()
    assert "the" in vocab
    assert "cat" in vocab
    assert "<unk>" in vocab

def test_vocab_builder_respects_min_freq():
    builder = VocabBuilder(min_freq=3)
    builder.update("the the the cat cat cat dog")
    vocab = builder.build()
    assert "the" in vocab
    assert "cat" in vocab
    assert "dog" not in vocab  # only 1 occurrence

def test_simple_tokenizer_encode():
    vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3, "hello": 4, "world": 5}
    tok = SimpleTokenizer(vocab)
    ids = tok.encode("hello world")
    assert ids == [4, 5]

def test_simple_tokenizer_unk():
    vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    tok = SimpleTokenizer(vocab)
    ids = tok.encode("unknown word")
    assert all(i == 1 for i in ids)  # all map to UNK

def test_vocab_builder_tokenizer_roundtrip():
    builder = VocabBuilder(min_freq=1)
    corpus = "the quick brown fox jumps over the lazy dog"
    builder.update(corpus)
    tok = builder.build_tokenizer()
    ids = tok.encode(corpus)
    assert len(ids) > 0
    assert all(isinstance(i, int) for i in ids)
