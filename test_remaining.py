import torch
import sys
sys.path.insert(0, '/Users/christienantonio/aurelius')
from rust_bridge import _PyFallbackPageTable, _py_estimate_memory, _py_save_checkpoint
from train_3b import count_parameters
from generate_report import make_style, PRIMARY, SECONDARY, DARK_BG
from agent_train import ImitationDataset
from kv_cache_quant import KVCacheQuantizer
from fused_kernels import FlashAttention3Wrapper
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import Color
import tempfile, os

def test_py_fallback_page_table():
    pt = _PyFallbackPageTable(capacity=3)
    assert pt.register_page(1, 0.8, 1024, True) == "ok"
    assert pt.register_page(2, 0.5, 2048, False) == "ok"
    assert pt.register_page(3, 0.9, 512, True) == "ok"
    assert pt.register_page(4, 0.1, 128, False).startswith("full:")
    assert pt.access(1) == "gpu"
    assert pt.access(99) == "absent"
    assert pt.promote_to_gpu(2) == "promoted"
    stats = pt.stats()
    assert "pages=3" in stats

def test_py_estimate_memory():
    result = _py_estimate_memory(768, 3072, 12, 512, 4, 2)
    assert isinstance(result, str)
    assert "MB" in result

def test_py_save_checkpoint():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.pt")
        tensors = {"w": torch.randn(3, 3), "b": torch.zeros(3)}
        _py_save_checkpoint(path, tensors)
        loaded = torch.load(path)
        assert "w" in loaded and "b" in loaded
        assert loaded["w"].shape == (3, 3)

def test_count_parameters():
    model = torch.nn.Linear(10, 5, bias=False)
    result = count_parameters(model)
    assert result["total"] == 50
    assert len(result["components"]) > 0

def test_make_style():
    style = make_style("test", fontSize=14)
    assert isinstance(style, ParagraphStyle)
    assert style.name == "test"
    assert style.fontSize == 14

def test_color_constants():
    assert isinstance(PRIMARY, Color)
    assert isinstance(SECONDARY, Color)
    assert isinstance(DARK_BG, Color)

def test_imitation_dataset():
    demos = [{'input_ids': torch.zeros(128).long(), 'labels': torch.zeros(128).long()}]
    ds = ImitationDataset(demos)
    batch = ds.sample_batch(1)
    assert batch['input_ids'].shape == (1, 128), f"Got {batch['input_ids'].shape}"
    assert batch['labels'].shape == (1, 128)
    assert batch['tool_labels'] is None

def test_kv_cache_quantizer_roundtrip():
    q = KVCacheQuantizer(bits=8, block_size=64)
    x = torch.randn(2, 4, 512, 128)
    qx, scale = q.quantize(x)
    rx = q.dequantize(qx, scale)
    assert qx.shape == x.shape
    assert rx.shape == x.shape
    assert torch.allclose(rx, x, atol=0.5)

def test_flash_attention_wrapper_forward():
    m = FlashAttention3Wrapper(d_model=128, n_heads=4, causal=False)
    x = torch.randn(1, 8, 128)
    cos = torch.ones(1, 1, 32)
    sin = torch.zeros(1, 1, 32)
    out = m(x, cos, sin)
    assert out.shape == (1, 8, 128)
