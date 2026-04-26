from src.serving.load_balancer import (
    LOAD_BALANCER_REGISTRY,
    BackendNode,
    LBStrategy,
    LoadBalancer,
)


def make_node(name, weight=1):
    return BackendNode(name=name, url=f"http://{name}", weight=weight)


def test_round_robin_cycles():
    lb = LoadBalancer(LBStrategy.ROUND_ROBIN)
    lb.add_backend(make_node("a"))
    lb.add_backend(make_node("b"))
    lb.add_backend(make_node("c"))
    results = [lb.next_backend().name for _ in range(6)]
    assert results == ["a", "b", "c", "a", "b", "c"]


def test_round_robin_skips_unhealthy():
    lb = LoadBalancer(LBStrategy.ROUND_ROBIN)
    lb.add_backend(make_node("a"))
    lb.add_backend(make_node("b"))
    lb.mark_unhealthy("a")
    for _ in range(4):
        assert lb.next_backend().name == "b"


def test_least_connections_picks_min():
    lb = LoadBalancer(LBStrategy.LEAST_CONNECTIONS)
    lb.add_backend(make_node("a"))
    lb.add_backend(make_node("b"))
    lb.mark_connection_open("a")
    lb.mark_connection_open("a")
    lb.mark_connection_open("b")
    assert lb.next_backend().name == "b"


def test_mark_connection_open_closed():
    lb = LoadBalancer(LBStrategy.ROUND_ROBIN)
    lb.add_backend(make_node("a"))
    lb.mark_connection_open("a")
    lb.mark_connection_open("a")
    assert lb._nodes[0].active_connections == 2
    lb.mark_connection_closed("a")
    assert lb._nodes[0].active_connections == 1
    lb.mark_connection_closed("a")
    lb.mark_connection_closed("a")
    assert lb._nodes[0].active_connections == 0


def test_mark_unhealthy_and_healthy():
    lb = LoadBalancer(LBStrategy.ROUND_ROBIN)
    lb.add_backend(make_node("a"))
    lb.mark_unhealthy("a")
    assert lb.next_backend() is None
    lb.mark_healthy("a")
    assert lb.next_backend().name == "a"


def test_healthy_count():
    lb = LoadBalancer(LBStrategy.ROUND_ROBIN)
    lb.add_backend(make_node("a"))
    lb.add_backend(make_node("b"))
    lb.add_backend(make_node("c"))
    lb.mark_unhealthy("b")
    assert lb.healthy_count() == 2


def test_returns_none_when_no_healthy():
    lb = LoadBalancer(LBStrategy.RANDOM)
    lb.add_backend(make_node("a"))
    lb.mark_unhealthy("a")
    assert lb.next_backend() is None


def test_weighted_round_robin_respects_weights():
    lb = LoadBalancer(LBStrategy.WEIGHTED_ROUND_ROBIN)
    lb.add_backend(make_node("heavy", weight=100))
    lb.add_backend(make_node("light", weight=1))
    counts = {"heavy": 0, "light": 0}
    for _ in range(200):
        counts[lb.next_backend().name] += 1
    assert counts["heavy"] > counts["light"]


def test_random_strategy_returns_healthy():
    lb = LoadBalancer(LBStrategy.RANDOM)
    lb.add_backend(make_node("a"))
    lb.add_backend(make_node("b"))
    lb.mark_unhealthy("a")
    for _ in range(10):
        assert lb.next_backend().name == "b"


def test_remove_backend():
    lb = LoadBalancer(LBStrategy.ROUND_ROBIN)
    lb.add_backend(make_node("a"))
    lb.add_backend(make_node("b"))
    lb.remove_backend("a")
    assert all(n.name != "a" for n in lb._nodes)


def test_registry_keys():
    assert "round_robin" in LOAD_BALANCER_REGISTRY
    assert "least_connections" in LOAD_BALANCER_REGISTRY
    assert "weighted_round_robin" in LOAD_BALANCER_REGISTRY
    assert "random" in LOAD_BALANCER_REGISTRY
    assert LOAD_BALANCER_REGISTRY["round_robin"] == LBStrategy.ROUND_ROBIN
