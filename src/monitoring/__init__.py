"""Aurelius monitoring surface: metrics, alerting, health checks."""
__all__ = [
    "MetricType", "MetricSample", "MetricsCollector", "METRICS_COLLECTOR",
    "AlertSeverity", "AlertRule", "AlertManager", "ALERT_MANAGER",
    "HealthStatus", "HealthCheck", "HealthChecker", "HEALTH_CHECKER",
    "AgentHealthMonitor", "AGENT_HEALTH_MONITOR_REGISTRY",
    "MONITORING_REGISTRY",
]
from .metrics_collector import MetricType, MetricSample, MetricsCollector, METRICS_COLLECTOR
from .alert_manager import AlertSeverity, AlertRule, AlertManager, ALERT_MANAGER
from .health_checker import HealthStatus, HealthCheck, HealthChecker, HEALTH_CHECKER
from .agent_health_monitor import AgentHealthMonitor, AGENT_HEALTH_MONITOR_REGISTRY

MONITORING_REGISTRY: dict[str, object] = {
    "metrics": METRICS_COLLECTOR,
    "alerts": ALERT_MANAGER,
    "health": HEALTH_CHECKER,
}

# --- Message metrics (cycle-197) ---------------------------------------------
from .message_metrics import (  # noqa: F401
    InstrumentedMessageBus,
    MessageMetrics,
    MessageMetricsConfig,
    MESSAGE_METRICS_REGISTRY,
    DEFAULT_MESSAGE_METRICS,
)
MONITORING_REGISTRY["message_metrics"] = DEFAULT_MESSAGE_METRICS

# --- Agent health monitor --------------------------------------------------
from .agent_health_monitor import DEFAULT_AGENT_HEALTH_MONITOR  # noqa: F401
MONITORING_REGISTRY["agent_health"] = DEFAULT_AGENT_HEALTH_MONITOR
