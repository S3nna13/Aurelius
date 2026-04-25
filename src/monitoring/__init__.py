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
