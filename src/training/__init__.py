from src.training.fsdp_lite import FSDPLite, ShardSpec, gather_tensor, shard_tensor
from src.training.loss_variance_monitor import LossStats, LossVarianceMonitor
from src.training.lr_range_test import LRRangeTest, LRRangeTestResult
from src.training.token_dropout import TokenDropout

__all__ = [
    "FSDPLite",
    "ShardSpec",
    "shard_tensor",
    "gather_tensor",
    "TokenDropout",
    "LossStats",
    "LossVarianceMonitor",
    "LRRangeTest",
    "LRRangeTestResult",
]

from src.training.tool_call_supervision_loss import ToolCallSupervisionLoss  # noqa: E402

AUXILIARY_LOSS_REGISTRY: dict[str, type] = {}
AUXILIARY_LOSS_REGISTRY.setdefault("tool_call_supervision", ToolCallSupervisionLoss)

__all__ += ["ToolCallSupervisionLoss", "AUXILIARY_LOSS_REGISTRY"]
