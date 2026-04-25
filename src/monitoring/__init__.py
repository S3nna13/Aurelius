"""Aurelius monitoring surface: metrics, alerting, health checks."""
__all__ = [
    "MetricType", "MetricSample", "MetricsCollector", "METRICS_COLLECTOR",
    "AlertSeverity", "AlertRule", "AlertManager", "ALERT_MANAGER",
    "HealthStatus", "HealthCheck", "HealthChecker", "HEALTH_CHECKER",
    "MONITORING_REGISTRY",
]
from .metrics_collector import MetricType, MetricSample, MetricsCollector, METRICS_COLLECTOR
from .alert_manager import AlertSeverity, AlertRule, AlertManager, ALERT_MANAGER
from .health_checker import HealthStatus, HealthCheck, HealthChecker, HEALTH_CHECKER

MONITORING_REGISTRY: dict[str, object] = {
    "metrics": METRICS_COLLECTOR,
    "alerts": ALERT_MANAGER,
    "health": HEALTH_CHECKER,
}

# --- metric exporter (additive) ------------------------------------
from .metric_exporter import ExportFormat, MetricPoint, MetricExporter, METRIC_EXPORTER_REGISTRY  # noqa: F401
MONITORING_REGISTRY.update({"exporter": MetricExporter})
__all__ += [
    "ExportFormat",
    "MetricPoint",
    "MetricExporter",
    "METRIC_EXPORTER_REGISTRY",
]
