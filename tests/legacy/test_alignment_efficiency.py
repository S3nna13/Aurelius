import torch
import sys
sys.path.insert(0, '/Users/christienantonio/aurelius')
from alignment_impl import *
from efficiency_impl import *

B, D, T = 2, 64, 8

def test_dpo_loss_shapes():
    loss_fn = DPOLoss()
    c, r, rc, rr = [torch.randn(B) for _ in range(4)]
    loss, acc = loss_fn(c, r, rc, rr)
    assert loss.shape == ()
    assert acc.shape == ()

def test_constitutional_classifier_shapes():
    cc = ConstitutionalClassifier(d_model=D, n_principles=8)
    x = torch.randn(3, D)
    scores, types = cc(x)
    assert scores.shape == (3, 8)
    viol, sugg = cc.critique(x)
    assert viol.shape == (3, 8)
    assert sugg.shape == (3, D)

def test_evidence_retriever_shapes():
    er = EvidenceRetriever(d_model=D)
    q = torch.randn(B, D)
    keys = torch.randn(10, D)
    scores = er(q, keys)
    assert scores.shape == (B, 10)

def test_claim_verifier_shapes():
    cv = ClaimVerifier(d_model=D)
    claim = torch.randn(B, D)
    evidence = torch.randn(B, 5, D)
    scores = cv(claim, evidence)
    assert scores.shape == (B, 5)

def test_process_reward_model_shapes():
    prm = ProcessRewardModel(d_model=D)
    steps = torch.randn(B, T, D)
    rewards = prm(steps)
    assert rewards.shape == (B, T)
    outcome = prm.compute_outcome_reward(rewards)
    assert outcome.shape == (B,)

def test_rarr_forward():
    rarr = RARRRetriever(d_model=D)
    for i in range(3):
        rarr.add_evidence(torch.randn(D), torch.randn(D), f"src{i}")
    claim = torch.randn(B, D)
    ev, conf = rarr(claim)
    assert len(ev) <= 5
    assert conf.shape == ()

def test_tiled_flash_attention_shapes():
    attn = TiledFlashAttention(d_model=256, n_heads=8, block_size=32)
    x = torch.randn(2, 64, 256)
    y = attn(x)
    assert y.shape == (2, 64, 256)

def test_paged_kv_manager():
    mgr = PagedKVManager(n_layers=2, n_heads=4, head_dim=64, block_size=16, max_blocks=8, device='cpu')
    blocks = mgr.alloc(2)
    assert len(blocks) == 2
    k, v = torch.randn(4, 64), torch.randn(4, 64)
    mgr.write(0, blocks[0], 0, k, v)
    rk, rv = mgr.read(0, blocks[0], 1)
    assert rk.shape == (1, 4, 64)
    assert rv.shape == (1, 4, 64)
    mgr.free(blocks)

def test_streaming_cache_shapes():
    sc = StreamingCache(d_model=D, window_size=8, n_sink=2)
    x = torch.randn(2, 4, D)
    out, cache = sc(x)
    assert out.shape == (2, 4, D)
    assert cache.shape[1] <= 10

def test_distributed_training_manager_shapes():
    dtm = DistributedTrainingManager(d_model=D, d_ff=256, n_heads=4, n_pp_stages=2)
    x = torch.randn(2, T, D)
    tp_out = dtm.tensor_parallel_forward(x)
    assert tp_out.shape == (2, T, D)
    mbs = [torch.randn(1, T, D), torch.randn(2, T, D)]
    pp_out = dtm.pipeline_parallel_forward(mbs)
    assert len(pp_out) == 2
    assert pp_out[0].shape == (1, T, D)
    assert pp_out[1].shape == (2, T, D)
