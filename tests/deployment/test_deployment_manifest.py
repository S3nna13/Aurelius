"""
test_deployment_manifest.py — Tests for src/deployment/deployment_manifest.py
Aurelius LLM Project — stdlib only.
"""

import dataclasses
import unittest

from src.deployment.deployment_manifest import (
    DEPLOYMENT_MANIFEST_REGISTRY,
    REGISTRY,
    ContainerSpec,
    DeploymentManifest,
    ResourceSpec,
)


class TestResourceSpec(unittest.TestCase):
    def test_stores_fields(self):
        rs = ResourceSpec(cpu_millicores=500, memory_mb=1024)
        self.assertEqual(rs.cpu_millicores, 500)
        self.assertEqual(rs.memory_mb, 1024)
        self.assertEqual(rs.gpu_count, 0)

    def test_gpu_count_field(self):
        rs = ResourceSpec(cpu_millicores=1000, memory_mb=2048, gpu_count=2)
        self.assertEqual(rs.gpu_count, 2)

    def test_frozen(self):
        rs = ResourceSpec(cpu_millicores=100, memory_mb=256)
        with self.assertRaises((dataclasses.FrozenInstanceError, AttributeError)):
            rs.cpu_millicores = 999  # type: ignore[misc]


class TestContainerSpec(unittest.TestCase):
    def setUp(self):
        self.rs = ResourceSpec(cpu_millicores=250, memory_mb=512)

    def test_stores_image_tag(self):
        cs = ContainerSpec(image="nginx", tag="latest", resources=self.rs)
        self.assertEqual(cs.image, "nginx")
        self.assertEqual(cs.tag, "latest")

    def test_stores_resources(self):
        cs = ContainerSpec(image="app", tag="v1", resources=self.rs)
        self.assertIs(cs.resources, self.rs)

    def test_default_env_vars(self):
        cs = ContainerSpec(image="app", tag="v1", resources=self.rs)
        self.assertEqual(cs.env_vars, {})

    def test_default_ports(self):
        cs = ContainerSpec(image="app", tag="v1", resources=self.rs)
        self.assertEqual(cs.ports, [])

    def test_env_vars_stored(self):
        cs = ContainerSpec(
            image="app", tag="v1", resources=self.rs, env_vars={"KEY": "val"}
        )
        self.assertEqual(cs.env_vars["KEY"], "val")

    def test_ports_stored(self):
        cs = ContainerSpec(image="app", tag="v1", resources=self.rs, ports=[80, 443])
        self.assertIn(80, cs.ports)

    def test_mutable_env_vars_independent(self):
        cs1 = ContainerSpec(image="a", tag="1", resources=self.rs)
        cs2 = ContainerSpec(image="b", tag="2", resources=self.rs)
        cs1.env_vars["X"] = "y"
        self.assertNotIn("X", cs2.env_vars)


class TestDeploymentManifestAddContainer(unittest.TestCase):
    def _make_cs(self):
        rs = ResourceSpec(cpu_millicores=200, memory_mb=400)
        return ContainerSpec(image="myapp", tag="1.0", resources=rs)

    def test_add_container(self):
        dm = DeploymentManifest(name="app")
        dm.add_container(self._make_cs())
        self.assertEqual(len(dm._containers), 1)

    def test_add_multiple_containers(self):
        dm = DeploymentManifest(name="app")
        dm.add_container(self._make_cs())
        dm.add_container(self._make_cs())
        self.assertEqual(len(dm._containers), 2)


class TestDeploymentManifestLabels(unittest.TestCase):
    def test_set_label(self):
        dm = DeploymentManifest(name="app")
        dm.set_label("env", "prod")
        self.assertEqual(dm._labels["env"], "prod")

    def test_overwrite_label(self):
        dm = DeploymentManifest(name="app")
        dm.set_label("env", "dev")
        dm.set_label("env", "prod")
        self.assertEqual(dm._labels["env"], "prod")


class TestDeploymentManifestValidate(unittest.TestCase):
    def _valid_cs(self):
        rs = ResourceSpec(cpu_millicores=100, memory_mb=256)
        return ContainerSpec(image="img", tag="v1", resources=rs)

    def test_validate_empty_name_error(self):
        dm = DeploymentManifest(name="")
        dm.add_container(self._valid_cs())
        errors = dm.validate()
        self.assertTrue(any("name" in e.lower() for e in errors))

    def test_validate_no_containers_error(self):
        dm = DeploymentManifest(name="myapp")
        errors = dm.validate()
        self.assertTrue(any("container" in e.lower() for e in errors))

    def test_validate_replicas_less_than_1(self):
        dm = DeploymentManifest(name="myapp", replicas=0)
        dm.add_container(self._valid_cs())
        errors = dm.validate()
        self.assertTrue(any("replica" in e.lower() for e in errors))

    def test_validate_cpu_zero_error(self):
        rs = ResourceSpec(cpu_millicores=0, memory_mb=256)
        cs = ContainerSpec(image="img", tag="v1", resources=rs)
        dm = DeploymentManifest(name="myapp")
        dm.add_container(cs)
        errors = dm.validate()
        self.assertTrue(any("cpu" in e.lower() for e in errors))

    def test_validate_memory_zero_error(self):
        rs = ResourceSpec(cpu_millicores=100, memory_mb=0)
        cs = ContainerSpec(image="img", tag="v1", resources=rs)
        dm = DeploymentManifest(name="myapp")
        dm.add_container(cs)
        errors = dm.validate()
        self.assertTrue(any("memory" in e.lower() for e in errors))

    def test_validate_empty_image_error(self):
        rs = ResourceSpec(cpu_millicores=100, memory_mb=256)
        cs = ContainerSpec(image="", tag="v1", resources=rs)
        dm = DeploymentManifest(name="myapp")
        dm.add_container(cs)
        errors = dm.validate()
        self.assertTrue(any("image" in e.lower() for e in errors))

    def test_validate_passes_valid_manifest(self):
        dm = DeploymentManifest(name="myapp", replicas=2)
        dm.add_container(self._valid_cs())
        errors = dm.validate()
        self.assertEqual(errors, [])

    def test_validate_returns_list(self):
        dm = DeploymentManifest(name="myapp")
        dm.add_container(self._valid_cs())
        self.assertIsInstance(dm.validate(), list)


class TestDeploymentManifestToDict(unittest.TestCase):
    def setUp(self):
        rs = ResourceSpec(cpu_millicores=500, memory_mb=1024, gpu_count=1)
        cs = ContainerSpec(image="model", tag="v2", resources=rs, ports=[8080])
        self.dm = DeploymentManifest(name="llm", namespace="ml", replicas=3)
        self.dm.add_container(cs)
        self.dm.set_label("team", "ai")
        self.d = self.dm.to_dict()

    def test_to_dict_name(self):
        self.assertEqual(self.d["name"], "llm")

    def test_to_dict_namespace(self):
        self.assertEqual(self.d["namespace"], "ml")

    def test_to_dict_replicas(self):
        self.assertEqual(self.d["replicas"], 3)

    def test_to_dict_labels(self):
        self.assertEqual(self.d["labels"]["team"], "ai")

    def test_to_dict_containers_list(self):
        self.assertIsInstance(self.d["containers"], list)
        self.assertEqual(len(self.d["containers"]), 1)

    def test_to_dict_container_image(self):
        self.assertEqual(self.d["containers"][0]["image"], "model")


class TestDeploymentManifestToYamlStub(unittest.TestCase):
    def test_yaml_stub_contains_name(self):
        dm = DeploymentManifest(name="aurelius")
        rs = ResourceSpec(cpu_millicores=100, memory_mb=256)
        dm.add_container(ContainerSpec(image="img", tag="1", resources=rs))
        stub = dm.to_yaml_stub()
        self.assertIn("aurelius", stub)

    def test_yaml_stub_is_string(self):
        dm = DeploymentManifest(name="test")
        stub = dm.to_yaml_stub()
        self.assertIsInstance(stub, str)

    def test_yaml_stub_contains_namespace(self):
        dm = DeploymentManifest(name="test", namespace="staging")
        stub = dm.to_yaml_stub()
        self.assertIn("staging", stub)


class TestRegistry(unittest.TestCase):
    def test_registry_key(self):
        self.assertIn("default", DEPLOYMENT_MANIFEST_REGISTRY)

    def test_registry_value(self):
        self.assertIs(DEPLOYMENT_MANIFEST_REGISTRY["default"], DeploymentManifest)

    def test_registry_alias(self):
        self.assertIs(REGISTRY, DEPLOYMENT_MANIFEST_REGISTRY)


if __name__ == "__main__":
    unittest.main()
