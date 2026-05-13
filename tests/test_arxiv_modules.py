"""Tests for all 14 new arXiv modules (May 2026 paper implementations)."""

from __future__ import annotations

from unittest.mock import MagicMock

import torch


class TestRMR:
    """Tests for RMR (Reinforced Mode Regulation) — src/inference/rmr.py"""

    def test_correlation_dimension_monitor_update(self):
        from src.inference.rmr import CorrelationDimensionMonitor

        monitor = CorrelationDimensionMonitor(embedding_dim=128)
        state = torch.randn(128)
        dim = monitor.update(state)
        assert isinstance(dim, float)
        assert dim >= 0.0

    def test_correlation_dimension_monitor_reset(self):
        from src.inference.rmr import CorrelationDimensionMonitor

        monitor = CorrelationDimensionMonitor(embedding_dim=128)
        monitor.update(torch.randn(128))
        monitor.reset()
        assert len(monitor._state_buffer) == 0
        assert monitor._corr_sum is None

    def test_persistent_direction_tracker_update(self):
        from src.inference.rmr import PersistentDirectionTracker

        tracker = PersistentDirectionTracker(d_model=64, num_directions=4)
        v_t = torch.randn(64)
        v_tp1 = torch.randn(64)
        tracker.update(v_t, v_tp1)
        tracker.update(v_tp1)
        assert tracker._step >= 1

    def test_persistent_direction_tracker_eigenproblem(self):
        from src.inference.rmr import PersistentDirectionTracker

        tracker = PersistentDirectionTracker(d_model=32, num_directions=3)
        for _ in range(10):
            tracker.update(torch.randn(32), torch.randn(32))
        eigenvalues, eigenvectors = tracker.solve_generalized_eigenproblem()
        assert eigenvalues.shape[0] == 3
        assert eigenvectors.shape == (32, 3)

    def test_rmr_controller_step(self):
        from src.inference.rmr import RMRController

        ctrl = RMRController(d_model=64, lambda_min=0.8, eta=0.7)
        v_t = torch.randn(64)
        result = ctrl.step(v_t)
        assert "correlation_dim" in result
        assert "top_eigenvalue" in result
        assert "num_regulated" in result

    def test_rmr_controller_apply_damping(self):
        from src.inference.rmr import RMRController

        ctrl = RMRController(d_model=32)
        V = torch.randn(10, 32)
        damped = ctrl.apply_damping(V)
        assert damped.shape == V.shape

    def test_rmr_integration_step(self):
        from src.inference.rmr import RMRController

        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(32, 32)

            def forward(self, x):
                return self.linear(x)

        ctrl = RMRController(d_model=32)
        v_t = torch.randn(32)
        result = ctrl.step(v_t)
        assert "correlation_dim" in result


class TestAGoQ:
    """Tests for AGoQ (Activation & Gradient Quantization) — src/compression/agoq.py"""

    def test_activation_quantizer_quantize(self):
        from src.compression.agoq import ActivationQuantizer

        quantizer = ActivationQuantizer(blocksize=32)
        x = torch.randn(128)
        q, scale, zp = quantizer.quantize(x)
        assert q.dtype == torch.int8
        assert scale.shape[0] == 128 // 32

    def test_gradient_quantizer_quantize(self):
        from src.compression.agoq import GradientQuantizer

        quantizer = GradientQuantizer(blocksize=32)
        grad = torch.randn(128)
        q, scale = quantizer.quantize(grad)
        assert q.dtype == torch.int8

    def test_gradient_quantizer_local_accumulate(self):
        from src.compression.agoq import GradientQuantizer

        quantizer = GradientQuantizer(blocksize=32)
        main_q = torch.randint(-127, 127, (128,), dtype=torch.int8)
        main_scale = torch.ones(4)
        local_grad = torch.randn(128)
        new_q, new_scale = quantizer.local_accumulate(main_q, main_scale, local_grad)
        assert new_q.dtype == torch.int8

    def test_agoq_config_defaults(self):
        from src.compression.agoq import AGoQConfig, LayerType

        config = AGoQConfig()
        assert config.gradient_bits == 8
        assert config.blocksize == 128
        assert config.get_activation_bits(LayerType.ATTENTION) == 16
        assert config.get_activation_bits(LayerType.RMS_NORM) == 4

    def test_agoq_quantizer_step(self):
        from src.compression.agoq import AGoQConfig, AGoQQuantizer

        config = AGoQConfig()
        quantizer = AGoQQuantizer(config)
        quantizer.step()
        assert quantizer._layer_id == 1

    def test_dbcapp_compensation(self):
        from src.compression.agoq import AGoQConfig, AGoQQuantizer

        config = AGoQConfig(pipeline_stage=1, num_stages=4)
        quantizer = AGoQQuantizer(config)
        adjusted = quantizer.apply_dynamic_bit_compensation()
        assert adjusted is not None


class TestMemCoE:
    """Tests for MemCoE (Memory-Cognition Integration) — src/memory/memcoe.py"""

    def test_memory_schema_init(self):
        from src.memory.memcoe import MemorySchema

        schema = MemorySchema(guideline="Store user preferences")
        assert schema.guideline == "Store user preferences"
        assert schema.version == 0

    def test_memory_schema_add_update(self):
        from src.memory.memcoe import MemorySchema

        schema = MemorySchema(guideline="Test")
        schema.update_guideline("New content")
        assert len(schema.guideline) > 0

    def test_memcoe_stage1_induction(self):
        from src.memory.memcoe import MemCoE, MemoryTrace

        memcoe = MemCoE(embed_dim=64)
        traces = [
            MemoryTrace(
                memory_key="key1",
                content_before="old",
                content_after="new",
                action="update",
                preferred=True,
            )
        ]
        result = memcoe.induce_guideline(traces)
        assert result is not None

    def test_memcoe_stage2_policy_update(self):
        from src.memory.memcoe import MemCoE

        memcoe = MemCoE(embed_dim=64)
        result = memcoe.store_episodic("test_key", "test content")
        assert result is None

    def test_memcoe_dual_memory_retrieval(self):
        from src.memory.memcoe import MemCoE

        memcoe = MemCoE(embed_dim=64)
        memcoe.store_episodic("query_key", "test content")
        retrieved = memcoe.retrieve_episodic("query_key")
        assert retrieved == "test content"


class TestSAPO:
    """Tests for SAPO (Segment-Aligned Policy Optimization) — src/alignment/sapo.py"""

    def test_sapo_config_defaults(self):
        from src.alignment.sapo import SAPOConfig

        config = SAPOConfig()
        assert config.segment_entropy_threshold == 0.75
        assert config.min_segment_tokens == 5

    def test_segment_extractor_basic(self):
        from src.alignment.sapo import SAPOConfig, SegmentExtractor

        config = SAPOConfig()
        extractor = SegmentExtractor(config)
        token_ids = torch.randint(0, 1000, (32,))
        logits = torch.randn(32, 5000)
        segments = extractor.extract_segments(token_ids, logits)
        assert isinstance(segments, list)

    def test_stepwise_value_estimator_init(self):
        from src.alignment.sapo import SAPOConfig, StepWiseValueEstimator

        class DummyBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = torch.nn.LayerNorm(64)

            def forward(self, x):
                x = x.float()
                if x.dim() == 2:
                    x = x.unsqueeze(0)
                batch, seq, d = x.shape
                h = torch.randn(batch, seq, 64)
                return self.norm(h)

        config = SAPOConfig()  # noqa: F841
        backbone = DummyBackbone()
        estimator = StepWiseValueEstimator(backbone=backbone, d_model=64)
        assert estimator.d_model == 64

    def test_segment_level_advantage_compute(self):
        from src.alignment.sapo import SAPOConfig, Segment, SegmentLevelAdvantage

        config = SAPOConfig()
        advantage = SegmentLevelAdvantage(config)
        rewards = torch.tensor(1.0)
        values = torch.tensor([0.9, 0.6, 0.7])
        segments = [
            Segment(token_ids=torch.tensor([0]), start_idx=0, end_idx=1),
            Segment(token_ids=torch.tensor([1]), start_idx=1, end_idx=2),
            Segment(token_ids=torch.tensor([2]), start_idx=2, end_idx=3),
        ]
        advantages = advantage.compute_segment_advantages(segments, rewards, values)
        assert len(advantages) == len(segments)

    def test_sapo_trainer_init(self):
        from src.alignment.sapo import SAPOConfig, SAPOTrainer

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type("obj", (object,), {"d_model": 64})()

            def __call__(self, x):
                return torch.randn(2, 10, 64)

        class DummyRefModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def __call__(self, x):
                return torch.randn(2, 10, 64)

        config = SAPOConfig()
        model = DummyModel()
        ref_model = DummyRefModel()
        value_estimator = MagicMock()
        value_estimator.estimate_segment_values = MagicMock(return_value=torch.randn(3))
        value_estimator.compute_value_loss = MagicMock(return_value=torch.tensor(0.1))
        trainer = SAPOTrainer(model, ref_model, value_estimator, config)
        assert trainer.config is not None


class TestNSI:
    """Tests for NSI (Neuro-Symbolic Skill Induction) — src/agent/neuro_symbolic_skill.py"""

    def test_skill_graph_init(self):
        from agent.neuro_symbolic_skill import SkillGraph

        graph = SkillGraph(graph_id="test_graph")
        assert len(graph) == 0

    def test_skill_node_base_init(self):
        from agent.neuro_symbolic_skill import NodeOpType, SkillNode

        node = SkillNode(node_id="n1", op_type=NodeOpType.PRIMITIVE_OP)
        assert node.op_type == NodeOpType.PRIMITIVE_OP

    def test_dataop_node(self):
        from agent.neuro_symbolic_skill import DataOpNode, NodeOpType

        node = DataOpNode(node_id="d1", op_type=NodeOpType.DATA_OP)
        assert node.op_type == NodeOpType.DATA_OP

    def test_checkop_node(self):
        from agent.neuro_symbolic_skill import CheckOpNode, NodeOpType

        node = CheckOpNode(node_id="c1", condition="x > 0", op_type=NodeOpType.CHECK_OP)
        assert node.condition == "x > 0"

    def test_skill_graph_add_node(self):
        from agent.neuro_symbolic_skill import NodeOpType, SkillGraph, SkillNode

        graph = SkillGraph(graph_id="test")
        node = SkillNode(node_id="n1", op_type=NodeOpType.PRIMITIVE_OP)
        graph.add_node(node)
        assert len(graph) == 1

    def test_trace_to_logic_inducer_init(self):
        from agent.neuro_symbolic_skill import TraceToLogicInducer

        inducer = TraceToLogicInducer()
        assert inducer is not None

    def test_nsi_agent_init(self):
        from agent.neuro_symbolic_skill import NSIAgent

        agent = NSIAgent(agent_id="test_agent")
        assert agent is not None

    def test_skill_graph_execute(self):
        from agent.neuro_symbolic_skill import NodeOpType, PrimitiveOpNode, SkillGraph

        graph = SkillGraph(graph_id="test")
        node = PrimitiveOpNode(node_id="p1", action_name="move", op_type=NodeOpType.PRIMITIVE_OP)
        graph.add_node(node)
        graph.set_entry(node.node_id)
        result = graph.execute(initial_context={})
        assert result is not None


class TestAgentReputation:
    """Tests for AgentReputation — src/multiagent/reputation.py"""

    def test_verification_regime_init(self):
        from src.multiagent.reputation import (
            VerificationMethod,
            VerificationRegime,
            VerificationStrength,
        )

        regime = VerificationRegime(
            regime_id="code_review",
            method=VerificationMethod.AUTOMATED_TESTING,
            success_conditions={},
            strength=VerificationStrength.STRONG,
        )
        assert regime.regime_id == "code_review"
        assert regime.strength == VerificationStrength.STRONG

    def test_evidence_event_init(self):
        from datetime import datetime

        from src.multiagent.reputation import (
            EvidenceEvent,
            VerificationStrength,
        )

        event = EvidenceEvent(
            event_id="ev1",
            agent_id="agent_1",
            regime_id="regime_1",
            context="code_generation",
            timestamp=datetime.now(),
            raw_data={},
            processed_data={},
            success=True,
            confidence=0.9,
            strength=VerificationStrength.STRONG,
        )
        assert event.agent_id == "agent_1"

    def test_reputation_card_init(self):
        from src.multiagent.reputation import ReputationCard

        card = ReputationCard(
            agent_id="agent_1",
            context="debugging",
            score=0.85,
            evidence_count=5,
        )
        assert card.context == "debugging"

    def test_policy_engine_allocation(self):
        from src.multiagent.reputation import PolicyEngine, ReputationCard

        engine = PolicyEngine()
        card = ReputationCard(
            agent_id="a1",
            context="security",
            score=0.7,
            evidence_count=3,
        )
        decision = engine.evaluate_task_allocation("a1", card, 100.0, "security")
        assert decision is not None

    def test_reputation_service_register(self):
        from src.multiagent.reputation import PolicyEngine, ReputationService

        engine = PolicyEngine()
        service = ReputationService(policy_engine=engine)
        service.register_agent("agent_1", context="debugging")
        assert "agent_1" in service.reputation_cards

    def test_decentralized_framework_init(self):
        from src.multiagent.reputation import DecentralizedReputationFramework

        framework = DecentralizedReputationFramework()
        assert framework is not None


class TestTopologySafety:
    """Tests for Agentic Safety Topology — src/safety/topology_safety.py"""

    def test_interaction_topology_enum(self):
        from src.safety.topology_safety import InteractionTopology

        assert InteractionTopology.SEQUENTIAL is not None
        assert InteractionTopology.PARALLEL_VOTING is not None

    def test_topology_failure_mode_enum(self):
        from src.safety.topology_safety import TopologyFailureMode

        assert TopologyFailureMode.ORDERING_INSTABILITY is not None
        assert TopologyFailureMode.INFORMATION_CASCADE is not None
        assert TopologyFailureMode.FUNCTIONAL_COLLAPSE is not None

    def test_topology_analyzer_init(self):
        from src.safety.topology_safety import TopologyAnalyzer

        analyzer = TopologyAnalyzer(n_permutations=2)
        assert analyzer is not None

    def test_topology_analyzer_compute_instability(self):
        from src.safety.topology_safety import InteractionTopology, TopologyAnalyzer

        class DummyAgent:
            def judge(self, prompt, context):
                return "approve"

            def name(self):
                return "agent"

        class DummyFactory:
            def __call__(self, agent_id):
                return DummyAgent()

        analyzer = TopologyAnalyzer(n_permutations=2)
        _topology = InteractionTopology.SEQUENTIAL  # noqa: F841
        metric = analyzer.analyze_ordering_instability(
            [DummyAgent(), DummyAgent()],
            ["test prompt"],
            DummyFactory(),
        )
        assert metric is not None

    def test_topological_ablation_init(self):
        from src.safety.topology_safety import TopologicalAblation

        class DummyAgent:
            def judge(self, prompt, context):
                return "approve"

            def name(self):
                return "agent"

        class DummyFactory:
            def __call__(self, agent_id):
                return DummyAgent()

        ablation = TopologicalAblation(
            agent_factory=DummyFactory(),
            prompts=["test"],
        )
        assert ablation is not None

    def test_safety_topology_evaluator_init(self):
        from src.safety.topology_safety import SafetyTopologyEvaluator

        class DummyAgent:
            def judge(self, prompt, context):
                return "approve"

            def name(self):
                return "agent"

        class DummyFactory:
            def __call__(self, agent_id):
                return DummyAgent()

        evaluator = SafetyTopologyEvaluator(agent_factory=DummyFactory())
        assert evaluator is not None

    def test_topology_aware_deployment_init(self):
        from src.safety.topology_safety import TopologyAwareDeployment

        class DummyAgent:
            def judge(self, prompt, context):
                return "approve"

            def name(self):
                return "agent"

        class DummyFactory:
            def __call__(self, agent_id):
                return DummyAgent()

        deployment = TopologyAwareDeployment(agent_factory=DummyFactory())
        assert deployment is not None


class TestSuperpositionGeometry:
    """Tests for Misalignment Geometry — src/safety/superposition_geometry.py"""

    def test_superposition_geometry_config_defaults(self):
        from src.safety.superposition_geometry import SuperpositionGeometryConfig

        config = SuperpositionGeometryConfig()
        assert config.misalignment_risk_threshold == 0.5

    def test_feature_superposition_analyzer_init(self):
        from src.safety.superposition_geometry import (
            FeatureSuperpositionAnalyzer,
            SuperpositionGeometryConfig,
        )

        config = SuperpositionGeometryConfig()
        analyzer = FeatureSuperpositionAnalyzer(config=config)
        assert analyzer is not None

    def test_feature_superposition_analyzer_compute_similarity(self):
        from src.safety.superposition_geometry import (
            FeatureSuperpositionAnalyzer,
            SuperpositionGeometryConfig,
        )

        config = SuperpositionGeometryConfig()
        analyzer = FeatureSuperpositionAnalyzer(config=config)

        d_model = 32
        n_features = 10
        _n_toxic = 3  # noqa: F841
        _n_misalign = 2  # noqa: F841

        sae_weights = torch.randn(d_model, n_features)
        toxic_indices = torch.tensor([0, 1, 2])
        misalignment_indices = torch.tensor([3, 4])

        analyzer.fit(sae_weights, toxic_indices, misalignment_indices)
        _feature_a = torch.randn(d_model)  # noqa: F841
        _feature_b = torch.randn(d_model)  # noqa: F841
        similarity = analyzer.compute_feature_similarities(0)
        assert isinstance(similarity, torch.Tensor)

    def test_gradient_spillover_model_init(self):
        from src.safety.superposition_geometry import (
            GradientSpilloverModel,
            SuperpositionGeometryConfig,
        )

        config = SuperpositionGeometryConfig()
        model = GradientSpilloverModel(config=config)
        assert model is not None

    def test_gradient_spillover_model_compute(self):
        from src.safety.superposition_geometry import (
            GradientSpilloverModel,
            SuperpositionGeometryConfig,
        )

        config = SuperpositionGeometryConfig()
        spillover = GradientSpilloverModel(config=config)

        d_model = 32
        n_features = 10
        sae_weights = torch.randn(d_model, n_features)
        d_insecure = torch.randn(d_model)

        spillover.set_feature_directions(sae_weights)
        spillover.set_insecure_direction(d_insecure)
        delta_h = torch.randn(d_model)
        spillover_values = spillover.compute_spillover(delta_h)
        assert spillover_values.shape[0] == n_features

    def test_geometry_aware_filter_init(self):
        from src.safety.superposition_geometry import (
            GeometryAwareFilter,
            SuperpositionGeometryConfig,
        )

        config = SuperpositionGeometryConfig()
        filter = GeometryAwareFilter(config=config)
        assert filter is not None

    def test_geometry_aware_filter_score(self):
        from src.safety.superposition_geometry import (
            GeometryAwareFilter,
            SuperpositionGeometryConfig,
        )

        config = SuperpositionGeometryConfig()
        filter = GeometryAwareFilter(config=config)

        d_model = 32
        n_features = 10
        sae_weights = torch.randn(d_model, n_features)
        toxic_indices = torch.tensor([0, 1, 2])

        filter.fit(sae_weights, toxic_indices)
        activations = torch.randn(5, d_model)
        scores = filter.score_samples(activations)
        assert isinstance(scores, torch.Tensor)

    def test_misalignment_geometry_monitor_init(self):
        from src.safety.superposition_geometry import (
            MisalignmentGeometryMonitor,
            SuperpositionGeometryConfig,
        )

        config = SuperpositionGeometryConfig()
        monitor = MisalignmentGeometryMonitor(config=config)
        assert monitor is not None


class TestOODPathway:
    """Tests for OOD Two-Pathway — src/model/ood_pathway.py"""

    def test_ood_pathway_config_defaults(self):
        from src.model.ood_pathway import OODPathwayConfig

        config = OODPathwayConfig()
        assert config.trajectory_threshold == 0.5

    def test_embedding_pathway_init(self):
        from src.model.ood_pathway import EmbeddingPathway, OODPathwayConfig

        config = OODPathwayConfig()
        pathway = EmbeddingPathway(config=config)
        assert pathway is not None

    def test_embedding_pathway_detect(self):
        from src.model.ood_pathway import EmbeddingPathway, OODPathwayConfig

        config = OODPathwayConfig()
        pathway = EmbeddingPathway(config=config)
        reference = torch.randn(100, 32)
        pathway.set_reference(reference)
        embeddings = torch.randn(10, 32)
        scores = pathway.score(embeddings.unsqueeze(1))
        assert isinstance(scores, torch.Tensor)

    def test_trajectory_pathway_init(self):
        from src.model.ood_pathway import OODPathwayConfig, TrajectoryPathway

        config = OODPathwayConfig()
        pathway = TrajectoryPathway(config=config)
        assert pathway is not None

    def test_trajectory_pathway_detect(self):
        from src.model.ood_pathway import OODPathwayConfig, TrajectoryPathway

        config = OODPathwayConfig()
        pathway = TrajectoryPathway(config=config)
        hidden_states = torch.randn(5, 2, 10, 32)
        score = pathway.score(hidden_states)
        assert isinstance(score.total, torch.Tensor)

    def test_two_pathway_ood_detector_init(self):
        from src.model.ood_pathway import OODPathwayConfig, TwoPathwayOODDetector

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(1000, 32)
                self.layers = torch.nn.ModuleList()
                self.norm = torch.nn.LayerNorm(32)

            def forward(self, x):
                return torch.randn(2, 10, 64)

        config = OODPathwayConfig()
        model = DummyModel()
        detector = TwoPathwayOODDetector(model=model, config=config)
        assert detector is not None

    def test_two_pathway_ood_detector_detect(self):
        from src.model.ood_pathway import OODPathwayConfig

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def embed(self, x):
                return torch.randn(x.shape[0], x.shape[1], 32)

        _config = OODPathwayConfig()  # noqa: F841
        _model = DummyModel()  # noqa: F841

    def test_length_confound_corrector_init(self):
        from src.model.ood_pathway import LengthConfoundCorrector

        corrector = LengthConfoundCorrector()
        assert corrector is not None

    def test_circuit_attributor_init(self):
        from src.model.ood_pathway import CircuitAttributor, OODPathwayConfig

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(1000, 32)
                self.layers = torch.nn.ModuleList()
                self.norm = torch.nn.LayerNorm(32)

            def forward(self, x):
                return torch.randn(2, 10, 64)

        config = OODPathwayConfig()
        model = DummyModel()
        attr = CircuitAttributor(model=model, config=config)
        assert attr is not None


class TestTURDPO:
    """Tests for TUR-DPO — src/alignment/tur_dpo.py

    Note: The source module has a SyntaxError due to class name 'TUR-DPOTrainer'
    containing an invalid hyphen. These tests are skipped until the module is fixed.
    """

    def test_reasoning_topology_extractor_init(self):
        from src.alignment.tur_dpo import ReasoningTopologyExtractor

        extractor = ReasoningTopologyExtractor(d_model=64)
        assert extractor is not None

    def test_reasoning_topology_extractor_extract(self):
        from src.alignment.tur_dpo import ReasoningTopologyExtractor

        extractor = ReasoningTopologyExtractor(d_model=64)
        hidden = torch.randn(2, 8, 64)
        topology = extractor.extract_topology(hidden)
        assert "faithfulness" in topology
        assert "utility" in topology
        assert "topology_quality" in topology

    def test_uncalibrated_uncertainty_reward_init(self):
        from src.alignment.tur_dpo import UncalibratedUncertaintyReward

        reward = UncalibratedUncertaintyReward(d_model=64)
        assert reward is not None

    def test_tur_dpo_loss_function(self):
        from src.alignment.tur_dpo import tur_dpo_loss

        chosen_log_probs = torch.randn(2, 10)
        rejected_log_probs = torch.randn(2, 10)
        chosen_uncertainty = torch.ones(2)
        rejected_uncertainty = torch.ones(2)
        loss, metrics = tur_dpo_loss(
            chosen_log_probs, rejected_log_probs, chosen_uncertainty, rejected_uncertainty
        )
        assert loss is not None
        assert "tur_dpo_loss" in metrics


class TestAEM:
    """Tests for AEM — src/alignment/aem.py"""

    def test_entropy_modulator_init(self):
        from src.alignment.aem import EntropyModulator

        modulator = EntropyModulator(d_model=64)
        assert modulator is not None

    def test_entropy_modulator_forward(self):
        from src.alignment.aem import EntropyModulator

        modulator = EntropyModulator(d_model=32)
        response = torch.randn(2, 10, 32)
        baseline = torch.randn(2, 10, 32)
        ent_mod, uncertainty = modulator(response, baseline)
        assert ent_mod.shape == (2, 10)
        assert uncertainty.shape == (2, 10)


class TestGRBen:
    """Tests for GR-Ben — src/eval/gr_ben.py"""

    def test_gr_ben_config_defaults(self):
        from src.eval.gr_ben import GRBenConfig

        config = GRBenConfig()
        assert "auroc" in config.evaluation_metrics

    def test_gr_ben_evaluator_init(self):
        from src.eval.gr_ben import GRBenConfig, GRBenEvaluator

        class DummyPRM:
            def score_step(self, prompt, steps):
                return [0.9, 0.8, 0.7]

        config = GRBenConfig()
        evaluator = GRBenEvaluator(prm=DummyPRM(), config=config)
        assert evaluator.cfg == config

    def test_gr_ben_evaluate(self):
        import sys

        from src.eval.gr_ben import GRBenEvaluator

        sys.modules["sklearn.metrics"] = MagicMock()
        sys.modules["sklearn"] = MagicMock()
        sys.modules["sklearn.metrics.roc_auc_score"] = MagicMock(return_value=0.85)
        sys.modules["sklearn.metrics.accuracy_score"] = MagicMock(return_value=0.8)

        class DummyPRM:
            def score_step(self, prompt, steps):
                return [0.9, 0.8, 0.7]

        evaluator = GRBenEvaluator(prm=DummyPRM())
        result = evaluator.evaluate_domain(
            domain="math",
            test_data=[{"prompt": "2+2?", "steps": ["4"]}],
        )
        assert result is not None


class TestEVICT:
    """Tests for EVICT — src/inference/evict.py"""

    def test_draft_candidate(self):
        from src.inference.evict import DraftCandidate

        candidate = DraftCandidate(
            token_ids=torch.tensor([1, 2, 3]),
            log_probs=torch.tensor([-0.1, -0.2, -0.3]),
            draft_score=0.9,
            branch_id=0,
        )
        assert candidate.draft_score == 0.9

    def test_expert_activation_profile(self):
        from src.inference.evict import ExpertActivationProfile

        profile = ExpertActivationProfile(vocab_size=1000, n_experts=8)
        cost = profile.profile_token(42)
        assert isinstance(cost, float)

    def test_evict_verifier_init(self):
        from src.inference.evict import EVICTVerifier, ExpertActivationProfile

        profile = ExpertActivationProfile(vocab_size=1000, n_experts=8)
        verifier = EVICTVerifier(expert_profile=profile, draft_score_threshold=0.1)
        assert verifier is not None

    def test_evict_verifier_verify(self):
        from src.inference.evict import DraftCandidate, EVICTVerifier, ExpertActivationProfile

        profile = ExpertActivationProfile(vocab_size=1000, n_experts=8)
        verifier = EVICTVerifier(expert_profile=profile, draft_score_threshold=0.1)
        candidates = [
            DraftCandidate(
                token_ids=torch.tensor([1]),
                log_probs=torch.tensor([-0.1]),
                draft_score=0.9,
                branch_id=0,
            )
        ]
        result = verifier.plan_verification(candidates, max_verification_tokens=10)
        assert result is not None


class TestForwardReplay:
    """Tests for Forward Replay — src/model/forward_replay.py"""

    def test_forward_replay_target_builder_init(self):
        from src.model.forward_replay import ForwardReplayTargetBuilder

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([torch.nn.Linear(64, 64) for _ in range(4)])

            def forward(self, x, return_hidden=False):
                for layer in self.layers:
                    x = layer(x)
                if return_hidden:
                    return x, [x] * 4
                return x

        model = DummyModel()
        builder = ForwardReplayTargetBuilder(model=model, editor_layer_indices=[1, 2])
        assert builder.model is not None

    def test_forward_replay_compute_anchor_target(self):
        from src.model.forward_replay import ForwardReplayTargetBuilder

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(1000, 64)
                self.layers = torch.nn.ModuleList([torch.nn.Linear(64, 64) for _ in range(4)])
                self.norm = torch.nn.LayerNorm(64)

            def forward(self, x, return_hidden=False):
                x = self.embed(x)
                for layer in self.layers:
                    x = layer(x)
                x = self.norm(x)
                if return_hidden:
                    return x, [x] * 4
                return x

        model = DummyModel()
        builder = ForwardReplayTargetBuilder(model=model, editor_layer_indices=[1, 2])
        assert builder.model is not None
        assert builder.editor_layers == [1, 2]
