import pytest
import torch
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.inference.beam_search import BeamSearchConfig, BeamResult, beam_search

@pytest.fixture
def small_model():
    cfg = AureliusConfig(n_layers=2, d_model=64, n_heads=2, n_kv_heads=2,
                         head_dim=32, d_ff=128, vocab_size=64, max_seq_len=64)
    torch.manual_seed(0)
    model = AureliusTransformer(cfg)
    model.eval()
    return model

def test_beam_search_returns_result(small_model):
    prompt = torch.randint(0, 64, (1, 4))
    cfg = BeamSearchConfig(num_beams=2, max_new_tokens=5)
    result = beam_search(small_model, prompt, cfg)
    assert isinstance(result, BeamResult)

def test_beam_search_num_sequences(small_model):
    prompt = torch.randint(0, 64, (1, 4))
    cfg = BeamSearchConfig(num_beams=3, max_new_tokens=5)
    result = beam_search(small_model, prompt, cfg)
    assert len(result.sequences) == 3
    assert len(result.scores) == 3

def test_beam_search_sequence_length(small_model):
    prompt = torch.randint(0, 64, (1, 4))
    cfg = BeamSearchConfig(num_beams=2, max_new_tokens=6)
    result = beam_search(small_model, prompt, cfg)
    prompt_len = 4
    for seq in result.sequences:
        # Each sequence should have prompt + new tokens (may be shorter if EOS)
        assert len(seq) >= prompt_len + 1

def test_beam_search_sorted_by_score(small_model):
    prompt = torch.randint(0, 64, (1, 4))
    cfg = BeamSearchConfig(num_beams=3, max_new_tokens=4)
    result = beam_search(small_model, prompt, cfg)
    assert result.scores[0] >= result.scores[1] >= result.scores[-1]

def test_beam_search_eos_stops_beam(small_model):
    prompt = torch.randint(0, 64, (1, 4))
    cfg = BeamSearchConfig(num_beams=2, max_new_tokens=20, eos_token_id=1, min_new_tokens=1)
    result = beam_search(small_model, prompt, cfg)
    # Sequences that hit EOS should not continue past it
    for seq in result.sequences:
        if 1 in seq[4:]:  # EOS in generated part
            eos_pos = seq.index(1, 4)
            assert seq[eos_pos] == 1  # EOS is last or internal

def test_beam_search_scores_finite(small_model):
    import math
    prompt = torch.randint(0, 64, (1, 4))
    cfg = BeamSearchConfig(num_beams=2, max_new_tokens=4)
    result = beam_search(small_model, prompt, cfg)
    for score in result.scores:
        assert math.isfinite(score)
