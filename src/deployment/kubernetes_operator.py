from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class ModelDeploymentSpec:
    name: str
    image: str = "aurelius:latest"
    replicas: int = 1
    gpu_count: int = 0
    memory_mb: int = 4096
    cpu_cores: float = 1.0
    port: int = 8080
    env: dict[str, str] = field(default_factory=dict)


class KubernetesOperator:
    def __init__(self, namespace: str = "default") -> None:
        self.namespace = namespace

    def generate_deployment(self, spec: ModelDeploymentSpec) -> dict:
        env_list = [{"name": k, "value": v} for k, v in spec.env.items()]
        resources: dict = {
            "requests": {
                "cpu": str(spec.cpu_cores),
                "memory": f"{spec.memory_mb}Mi",
            },
            "limits": {
                "cpu": str(spec.cpu_cores),
                "memory": f"{spec.memory_mb}Mi",
            },
        }
        if spec.gpu_count > 0:
            resources["requests"]["nvidia.com/gpu"] = str(spec.gpu_count)
            resources["limits"]["nvidia.com/gpu"] = str(spec.gpu_count)

        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": spec.name, "namespace": self.namespace},
            "spec": {
                "replicas": spec.replicas,
                "selector": {"matchLabels": {"app": spec.name}},
                "template": {
                    "metadata": {"labels": {"app": spec.name}},
                    "spec": {
                        "containers": [
                            {
                                "name": spec.name,
                                "image": spec.image,
                                "ports": [{"containerPort": spec.port}],
                                "resources": resources,
                                "env": env_list,
                            }
                        ]
                    },
                },
            },
        }

    def generate_service(self, spec: ModelDeploymentSpec) -> dict:
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": spec.name, "namespace": self.namespace},
            "spec": {
                "type": "ClusterIP",
                "selector": {"app": spec.name},
                "ports": [{"port": spec.port, "targetPort": spec.port, "protocol": "TCP"}],
            },
        }

    def generate_hpa(
        self,
        spec: ModelDeploymentSpec,
        min_replicas: int = 1,
        max_replicas: int = 10,
        cpu_target: int = 70,
    ) -> dict:
        return {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {"name": spec.name, "namespace": self.namespace},
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": spec.name,
                },
                "minReplicas": min_replicas,
                "maxReplicas": max_replicas,
                "metrics": [
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": cpu_target,
                            },
                        },
                    }
                ],
            },
        }

    def render_yaml(self, manifests: list[dict]) -> str:
        return "---\n" + "---\n".join(json.dumps(m, indent=2) for m in manifests)


K8S_OPERATOR_REGISTRY: dict[str, type[KubernetesOperator]] = {"default": KubernetesOperator}
