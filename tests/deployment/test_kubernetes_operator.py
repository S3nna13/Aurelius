from __future__ import annotations

import json

import pytest

from src.deployment.kubernetes_operator import (
    K8S_OPERATOR_REGISTRY,
    KubernetesOperator,
    ModelDeploymentSpec,
)


@pytest.fixture()
def spec() -> ModelDeploymentSpec:
    return ModelDeploymentSpec(
        name="aurelius", replicas=3, gpu_count=2, memory_mb=8192, cpu_cores=2.0, port=9090
    )


@pytest.fixture()
def op() -> KubernetesOperator:
    return KubernetesOperator(namespace="ml")


def test_generate_deployment_api_version(op, spec):
    d = op.generate_deployment(spec)
    assert d["apiVersion"] == "apps/v1"
    assert d["kind"] == "Deployment"


def test_generate_deployment_replicas(op, spec):
    d = op.generate_deployment(spec)
    assert d["spec"]["replicas"] == 3


def test_generate_deployment_resources(op, spec):
    d = op.generate_deployment(spec)
    container = d["spec"]["template"]["spec"]["containers"][0]
    assert container["resources"]["requests"]["memory"] == "8192Mi"
    assert container["resources"]["requests"]["nvidia.com/gpu"] == "2"


def test_generate_deployment_env(op):
    s = ModelDeploymentSpec(name="m", env={"FOO": "bar", "X": "1"})
    d = op.generate_deployment(s)
    env = {e["name"]: e["value"] for e in d["spec"]["template"]["spec"]["containers"][0]["env"]}
    assert env == {"FOO": "bar", "X": "1"}


def test_generate_service_clusterip(op, spec):
    svc = op.generate_service(spec)
    assert svc["kind"] == "Service"
    assert svc["spec"]["type"] == "ClusterIP"


def test_generate_service_port(op, spec):
    svc = op.generate_service(spec)
    assert svc["spec"]["ports"][0]["port"] == 9090


def test_generate_hpa_max_replicas(op, spec):
    hpa = op.generate_hpa(spec, min_replicas=2, max_replicas=15, cpu_target=80)
    assert hpa["kind"] == "HorizontalPodAutoscaler"
    assert hpa["spec"]["maxReplicas"] == 15
    assert hpa["spec"]["minReplicas"] == 2


def test_generate_hpa_cpu_target(op, spec):
    hpa = op.generate_hpa(spec, cpu_target=60)
    metric = hpa["spec"]["metrics"][0]
    assert metric["resource"]["target"]["averageUtilization"] == 60


def test_render_yaml_separator(op, spec):
    manifests = [op.generate_deployment(spec), op.generate_service(spec)]
    rendered = op.render_yaml(manifests)
    assert rendered.startswith("---\n")
    assert rendered.count("---\n") == 2


def test_render_yaml_parseable(op, spec):
    manifests = [op.generate_deployment(spec)]
    rendered = op.render_yaml(manifests)
    body = rendered.replace("---\n", "", 1)
    parsed = json.loads(body)
    assert parsed["kind"] == "Deployment"


def test_registry_key():
    assert "default" in K8S_OPERATOR_REGISTRY
    assert K8S_OPERATOR_REGISTRY["default"] is KubernetesOperator
