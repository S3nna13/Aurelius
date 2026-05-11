"""Tests for new AureliusConfig factory methods."""

from src.model.config import AureliusConfig


def test_flash_284b_config():
    config = AureliusConfig.aurelius_flash_284b()
    assert config.d_model == 4096
    assert config.n_layers == 43
    assert config.max_seq_len == 1_000_000
    assert config.hybrid_attention_enabled is True
    assert config.mhc_enabled is True
    assert config.moe_num_experts == 256
    assert config.moe_top_k == 6
    assert config.moe_every_n_layers == 1
    assert config.attention_compression_rate_csa == 4
    assert config.attention_compression_rate_hca == 128
    assert config.attention_top_k == 512
    assert config.mhc_expansion_factor == 4


def test_2_7b_config():
    config = AureliusConfig.aurelius_2_7b()
    assert config.d_model == 2560
    assert config.n_layers == 32
    assert config.n_heads == 20
    assert config.n_kv_heads == 5
    assert config.max_seq_len == 8192
    assert config.tie_embeddings is True
    assert config.use_gradient_checkpointing is True
    assert config.d_model == config.n_heads * config.head_dim
    assert config.n_heads % config.n_kv_heads == 0


def test_3b_config():
    config = AureliusConfig.aurelius_3b()
    assert config.d_model == 3072
    assert config.n_layers == 28
    assert config.n_heads == 24
    assert config.n_kv_heads == 6
    assert config.max_seq_len == 4096
    assert config.tie_embeddings is True
    assert config.use_gradient_checkpointing is True
    assert config.rope_scaling_type == "yarn"
    assert config.rope_scaling_factor == 4.0
    assert config.rope_original_max_seq_len == 4096
    assert config.d_model == config.n_heads * config.head_dim
    assert config.n_heads % config.n_kv_heads == 0


def test_pro_1_6t_config():
    config = AureliusConfig.aurelius_pro_1_6t()
    assert config.d_model == 7168
    assert config.n_layers == 61
    assert config.max_seq_len == 1_000_000
    assert config.hybrid_attention_enabled is True
    assert config.mhc_enabled is True
    assert config.moe_num_experts == 384
    assert config.moe_top_k == 6
    assert config.attention_top_k == 1024
    assert config.attention_output_projection_groups == 16
    assert config.attention_query_compression_dim == 1536


def test_flash_config_validation():
    config = AureliusConfig.aurelius_flash_284b()
    assert config.d_model == config.n_heads * config.head_dim
    assert config.n_heads % config.n_kv_heads == 0


def test_pro_config_validation():
    config = AureliusConfig.aurelius_pro_1_6t()
    assert config.d_model == config.n_heads * config.head_dim
    assert config.n_heads % config.n_kv_heads == 0


def test_hybrid_attention_flags_propagation():
    flash = AureliusConfig.aurelius_flash_284b()
    pro = AureliusConfig.aurelius_pro_1_6t()
    assert flash.hybrid_attention_enabled is True
    assert flash.mhc_enabled is True
    assert pro.hybrid_attention_enabled is True
    assert pro.mhc_enabled is True


def test_default_config_no_hybrid():
    config = AureliusConfig()
    assert config.hybrid_attention_enabled is False
    assert config.mhc_enabled is False
