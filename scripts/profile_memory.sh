#!/usr/bin/env bash
# =============================================================================
# Aurelius memory profiler — measure peak memory at each model size
#
# Usage:
#   bash scripts/profile_memory.sh               # Profile all sizes
#   bash scripts/profile_memory.sh 2.7b           # Profile specific size
#
# Profiles memory for model construction only (no training data needed).
# Tests: 1.4B (baseline), 2.7B, 3.0B, and MoE 5B.
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

SIZE="${1:-all}"

profile_size() {
    local name="$1"
    local classmethod="$2"

    echo "============================================"
    echo "  Profiling: ${name}"
    echo "============================================"

    python3 -c "
import torch
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

config = AureliusConfig.${classmethod}()
n_params = sum(v.numel() for v in config.__dict__.values() if isinstance(v, int))

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f'  Device: {device}')
print(f'  Config: d_model={config.d_model}, n_layers={config.n_layers}, '
      f'n_heads={config.n_heads}, n_kv_heads={config.n_kv_heads}, d_ff={config.d_ff}')

model = AureliusTransformer(config).to(device)

# Count parameters
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'  Total params: {total:,} ({total/1e9:.2f}B)')
print(f'  Trainable:    {trainable:,} ({trainable/1e9:.2f}B)')

# Memory estimate (bf16)
bytes_per_param = 2  # bf16
model_memory_gb = (total * bytes_per_param) / 1e9
optim_adamw_gb = (total * 4 * 2) / 1e9     # 2 fp32 states
optim_muon_gb = (total * 4) / 1e9           # 1 fp32 state (momentum only)
activations_gb = model_memory_gb * 0.5      # estimate with grad_ckpt

print(f'  Memory estimates (bf16, bs=1):')
print(f'    Model weights:    {model_memory_gb:.1f} GB')
print(f'    AdamW states:     {optim_adamw_gb:.1f} GB')
print(f'    Muon states:      {optim_muon_gb:.1f} GB')
print(f'    Activations (est): {activations_gb:.1f} GB')
print(f'    Total (Muon):     {model_memory_gb + optim_muon_gb + activations_gb:.1f} GB')

# Peak memory on MPS
if device.type == 'mps':
    torch.mps.empty_cache()
    print(f'  MPS allocated: {torch.mps.current_allocated_memory() / 1e9:.2f} GB')
elif device.type == 'cuda':
    torch.cuda.empty_cache()
    print(f'  CUDA allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB')

del model
print()
"
}

case "${SIZE}" in
    all)
        profile_size "1.4B (baseline)" "aurelius_1_3b"
        profile_size "2.7B" "aurelius_2_7b"
        profile_size "3.0B" "aurelius_3b"
        profile_size "MoE 5-6B" "aurelius_moe_5b"
        ;;
    1.4b|1.4B)
        profile_size "1.4B (baseline)" "default"
        ;;
    2.7b|2.7B)
        profile_size "2.7B" "aurelius_2_7b"
        ;;
    3b|3B|3.0b|3.0B)
        profile_size "3.0B" "aurelius_3b"
        ;;
    moe|MoE)
        profile_size "MoE 5-6B" "aurelius_moe_5b"
        ;;
    *)
        echo "Unknown size: ${SIZE}"
        echo "Usage: $0 [all|1.4b|2.7b|3b|moe]"
        exit 1
        ;;
esac
