from .benchmark_config import (
    BenchmarkSpec,
    ALL_BENCHMARKS,
    BENCHMARK_BY_NAME,
    MMLU,
    HELLASWAG,
    ARC_CHALLENGE,
    TRUTHFULQA,
    GSM8K,
    HUMANEVAL,
    MATH500,
    GPQA_DIAMOND,
    LIVECODEBENCH,
)

__all__ = [
    "BenchmarkSpec",
    "ALL_BENCHMARKS",
    "BENCHMARK_BY_NAME",
    "MMLU",
    "HELLASWAG",
    "ARC_CHALLENGE",
    "TRUTHFULQA",
    "GSM8K",
    "HUMANEVAL",
    "MATH500",
    "GPQA_DIAMOND",
    "LIVECODEBENCH",
]

# --- Additive registration for Needle-in-a-Haystack (NIAH) -------------------
# Safe to import lazily; does not remove or override any existing symbols.
from .needle_in_haystack import (
    NeedleInHaystackBenchmark as _NeedleInHaystackBenchmark,
    default_haystack_filler as _default_haystack_filler,
)

NeedleInHaystackBenchmark = _NeedleInHaystackBenchmark
default_haystack_filler = _default_haystack_filler

METRIC_REGISTRY = globals().setdefault("METRIC_REGISTRY", {})
BENCHMARK_REGISTRY = globals().setdefault("BENCHMARK_REGISTRY", {})
METRIC_REGISTRY.setdefault("niah", NeedleInHaystackBenchmark)
BENCHMARK_REGISTRY.setdefault("niah", NeedleInHaystackBenchmark)

__all__ = list(__all__) + [
    "NeedleInHaystackBenchmark",
    "default_haystack_filler",
    "METRIC_REGISTRY",
    "BENCHMARK_REGISTRY",
]

# --- Additive registration for RULER long-context benchmark ------------------
# Additive only; reuses the METRIC_REGISTRY / BENCHMARK_REGISTRY set above
# and leaves the "niah" entries untouched.
from .ruler_benchmark import RULERBenchmark as _RULERBenchmark

RULERBenchmark = _RULERBenchmark

METRIC_REGISTRY = globals().setdefault("METRIC_REGISTRY", {})
BENCHMARK_REGISTRY = globals().setdefault("BENCHMARK_REGISTRY", {})
METRIC_REGISTRY.setdefault("ruler", RULERBenchmark)
BENCHMARK_REGISTRY.setdefault("ruler", RULERBenchmark)

__all__ = list(__all__) + ["RULERBenchmark"]

# --- Additive registration for HumanEval functional-correctness benchmark ----
# Additive only; reuses the METRIC_REGISTRY / BENCHMARK_REGISTRY set above
# and leaves the "niah" and "ruler" entries untouched.
from .humaneval_scorer import (
    HumanEvalProblem as _HumanEvalProblem,
    SampleResult as _HumanEvalSampleResult,
    score_single as _humaneval_score_single,
    score_problems as _humaneval_score_problems,
    pass_at_k as _humaneval_pass_at_k,
)

HumanEvalProblem = _HumanEvalProblem
HumanEvalSampleResult = _HumanEvalSampleResult
humaneval_score_single = _humaneval_score_single
humaneval_score_problems = _humaneval_score_problems
humaneval_pass_at_k = _humaneval_pass_at_k

METRIC_REGISTRY = globals().setdefault("METRIC_REGISTRY", {})
BENCHMARK_REGISTRY = globals().setdefault("BENCHMARK_REGISTRY", {})
METRIC_REGISTRY.setdefault("humaneval", _humaneval_score_problems)
BENCHMARK_REGISTRY.setdefault("humaneval", _HumanEvalProblem)

__all__ = list(__all__) + [
    "HumanEvalProblem",
    "HumanEvalSampleResult",
    "humaneval_score_single",
    "humaneval_score_problems",
    "humaneval_pass_at_k",
]

# --- Additive registration for MBPP functional-correctness benchmark ---------
# Additive only; reuses the METRIC_REGISTRY / BENCHMARK_REGISTRY set above and
# leaves the "niah", "ruler", and "humaneval" entries untouched.
from .mbpp_scorer import (
    MBPPProblem as _MBPPProblem,
    MBPPSampleResult as _MBPPSampleResult,
    score_single as _mbpp_score_single,
    score_problems as _mbpp_score_problems,
)

MBPPProblem = _MBPPProblem
MBPPSampleResult = _MBPPSampleResult
mbpp_score_single = _mbpp_score_single
mbpp_score_problems = _mbpp_score_problems

METRIC_REGISTRY = globals().setdefault("METRIC_REGISTRY", {})
BENCHMARK_REGISTRY = globals().setdefault("BENCHMARK_REGISTRY", {})
METRIC_REGISTRY.setdefault("mbpp", _mbpp_score_problems)
BENCHMARK_REGISTRY.setdefault("mbpp", _MBPPProblem)

__all__ = list(__all__) + [
    "MBPPProblem",
    "MBPPSampleResult",
    "mbpp_score_single",
    "mbpp_score_problems",
]

# --- Additive registration for SWE-bench-lite agentic coding benchmark -------
# Additive only; reuses the METRIC_REGISTRY / BENCHMARK_REGISTRY set above and
# leaves the "niah", "ruler", "humaneval", and "mbpp" entries untouched.
from .swebench_lite_scorer import (
    SWEProblem as _SWEProblem,
    SWEResult as _SWEResult,
    score_single as _swebench_score_single,
    score_problems as _swebench_score_problems,
    apply_patch_via_python as _swebench_apply_patch_via_python,
    materialize_repo as _swebench_materialize_repo,
    run_tests as _swebench_run_tests,
)

SWEProblem = _SWEProblem
SWEResult = _SWEResult
swebench_score_single = _swebench_score_single
swebench_score_problems = _swebench_score_problems
swebench_apply_patch_via_python = _swebench_apply_patch_via_python
swebench_materialize_repo = _swebench_materialize_repo
swebench_run_tests = _swebench_run_tests

METRIC_REGISTRY = globals().setdefault("METRIC_REGISTRY", {})
BENCHMARK_REGISTRY = globals().setdefault("BENCHMARK_REGISTRY", {})
METRIC_REGISTRY.setdefault("swebench_lite", _swebench_score_problems)
BENCHMARK_REGISTRY.setdefault("swebench_lite", _SWEProblem)

__all__ = list(__all__) + [
    "SWEProblem",
    "SWEResult",
    "swebench_score_single",
    "swebench_score_problems",
    "swebench_apply_patch_via_python",
    "swebench_materialize_repo",
    "swebench_run_tests",
]

# --- Additive registration for IFEval instruction-following benchmark --------
# Additive only; reuses the METRIC_REGISTRY / BENCHMARK_REGISTRY set above and
# leaves the "niah", "ruler", "humaneval", "mbpp", and "swebench_lite" entries
# untouched.
from .ifeval_scorer import (
    IFEvalConstraint as _IFEvalConstraint,
    IFEvalProblem as _IFEvalProblem,
    IFEvalResult as _IFEvalResult,
    IFEvalScorer as _IFEvalScorer,
)

IFEvalConstraint = _IFEvalConstraint
IFEvalProblem = _IFEvalProblem
IFEvalResult = _IFEvalResult
IFEvalScorer = _IFEvalScorer

METRIC_REGISTRY = globals().setdefault("METRIC_REGISTRY", {})
BENCHMARK_REGISTRY = globals().setdefault("BENCHMARK_REGISTRY", {})
METRIC_REGISTRY.setdefault("ifeval", _IFEvalScorer)
BENCHMARK_REGISTRY.setdefault("ifeval", _IFEvalProblem)

__all__ = list(__all__) + [
    "IFEvalConstraint",
    "IFEvalProblem",
    "IFEvalResult",
    "IFEvalScorer",
]

# --- Additive registration for MT-Bench LLM-as-judge harness -----------------
# Additive only; reuses the METRIC_REGISTRY / BENCHMARK_REGISTRY set above and
# leaves the "niah", "ruler", "humaneval", "mbpp", "swebench_lite", and
# "ifeval" entries untouched.
from .mtbench_judge import (
    MTBenchJudge as _MTBenchJudge,
    MTBenchQuestion as _MTBenchQuestion,
    SingleAnswerScore as _MTBenchSingleAnswerScore,
    PairwiseResult as _MTBenchPairwiseResult,
)

MTBenchJudge = _MTBenchJudge
MTBenchQuestion = _MTBenchQuestion
SingleAnswerScore = _MTBenchSingleAnswerScore
PairwiseResult = _MTBenchPairwiseResult

METRIC_REGISTRY = globals().setdefault("METRIC_REGISTRY", {})
BENCHMARK_REGISTRY = globals().setdefault("BENCHMARK_REGISTRY", {})
METRIC_REGISTRY.setdefault("mtbench", _MTBenchJudge)
BENCHMARK_REGISTRY.setdefault("mtbench", _MTBenchQuestion)

__all__ = list(__all__) + [
    "MTBenchJudge",
    "MTBenchQuestion",
    "SingleAnswerScore",
    "PairwiseResult",
]

# --- Additive registration for AlpacaEval LLM-as-judge harness ---------------
# Additive only; reuses the METRIC_REGISTRY / BENCHMARK_REGISTRY set above and
# leaves the "niah", "ruler", "humaneval", "mbpp", "swebench_lite", "ifeval",
# and "mtbench" entries untouched.
from .alpacaeval_scorer import (
    AlpacaEvalScorer as _AlpacaEvalScorer,
    AlpacaProblem as _AlpacaProblem,
    AlpacaComparison as _AlpacaComparison,
)

AlpacaEvalScorer = _AlpacaEvalScorer
AlpacaProblem = _AlpacaProblem
AlpacaComparison = _AlpacaComparison

METRIC_REGISTRY = globals().setdefault("METRIC_REGISTRY", {})
BENCHMARK_REGISTRY = globals().setdefault("BENCHMARK_REGISTRY", {})
METRIC_REGISTRY.setdefault("alpacaeval", _AlpacaEvalScorer)
BENCHMARK_REGISTRY.setdefault("alpacaeval", _AlpacaProblem)

__all__ = list(__all__) + [
    "AlpacaEvalScorer",
    "AlpacaProblem",
    "AlpacaComparison",
]

# --- Additive registration for Arena-Hard LLM-as-judge harness ---------------
# Additive only; reuses the METRIC_REGISTRY / BENCHMARK_REGISTRY set above and
# leaves the "niah", "ruler", "humaneval", "mbpp", "swebench_lite", "ifeval",
# "mtbench", and "alpacaeval" entries untouched.
from .arena_hard_scorer import (
    ArenaHardScorer as _ArenaHardScorer,
    ArenaProblem as _ArenaProblem,
    ArenaComparison as _ArenaComparison,
)

ArenaHardScorer = _ArenaHardScorer
ArenaProblem = _ArenaProblem
ArenaComparison = _ArenaComparison

METRIC_REGISTRY = globals().setdefault("METRIC_REGISTRY", {})
BENCHMARK_REGISTRY = globals().setdefault("BENCHMARK_REGISTRY", {})
METRIC_REGISTRY.setdefault("arena_hard", _ArenaHardScorer)
BENCHMARK_REGISTRY.setdefault("arena_hard", _ArenaProblem)

__all__ = list(__all__) + [
    "ArenaHardScorer",
    "ArenaProblem",
    "ArenaComparison",
]

# --- Additive registration for GPQA graduate-level MC benchmark --------------
# Additive only; reuses the METRIC_REGISTRY / BENCHMARK_REGISTRY set above and
# leaves the "niah", "ruler", "humaneval", "mbpp", "swebench_lite", "ifeval",
# "mtbench", "alpacaeval", and "arena_hard" entries untouched.
from .gpqa_scorer import (
    GPQAProblem as _GPQAProblem,
    GPQAResult as _GPQAResult,
    GPQAScorer as _GPQAScorer,
    parse_answer_letter as _gpqa_parse_answer_letter,
)

GPQAProblem = _GPQAProblem
GPQAResult = _GPQAResult
GPQAScorer = _GPQAScorer
gpqa_parse_answer_letter = _gpqa_parse_answer_letter

METRIC_REGISTRY = globals().setdefault("METRIC_REGISTRY", {})
BENCHMARK_REGISTRY = globals().setdefault("BENCHMARK_REGISTRY", {})
METRIC_REGISTRY.setdefault("gpqa", _GPQAScorer)
BENCHMARK_REGISTRY.setdefault("gpqa", _GPQAProblem)

__all__ = list(__all__) + [
    "GPQAProblem",
    "GPQAResult",
    "GPQAScorer",
    "gpqa_parse_answer_letter",
]

# --- Additive registration for LiveCodeBench contamination-free coding ------
# Additive only; reuses the METRIC_REGISTRY / BENCHMARK_REGISTRY set above and
# leaves the "niah", "ruler", "humaneval", "mbpp", "swebench_lite", "ifeval",
# "mtbench", "alpacaeval", "arena_hard", and "gpqa" entries untouched.
from .livecodebench_scorer import (
    LiveCodeProblem as _LiveCodeProblem,
    LiveCodeResult as _LiveCodeResult,
    score_single as _livecodebench_score_single,
    score_problems as _livecodebench_score_problems,
    parse_date_filter as _livecodebench_parse_date_filter,
)

LiveCodeProblem = _LiveCodeProblem
LiveCodeResult = _LiveCodeResult
livecodebench_score_single = _livecodebench_score_single
livecodebench_score_problems = _livecodebench_score_problems
livecodebench_parse_date_filter = _livecodebench_parse_date_filter

METRIC_REGISTRY = globals().setdefault("METRIC_REGISTRY", {})
BENCHMARK_REGISTRY = globals().setdefault("BENCHMARK_REGISTRY", {})
METRIC_REGISTRY.setdefault("livecodebench", _livecodebench_score_problems)
BENCHMARK_REGISTRY.setdefault("livecodebench", _LiveCodeProblem)

__all__ = list(__all__) + [
    "LiveCodeProblem",
    "LiveCodeResult",
    "livecodebench_score_single",
    "livecodebench_score_problems",
    "livecodebench_parse_date_filter",
]

# --- Additive registration for MMLU 57-subject MC benchmark -----------------
# Additive only; reuses the METRIC_REGISTRY / BENCHMARK_REGISTRY set above and
# leaves prior entries untouched.
from .mmlu_scorer import (
    MMLUProblem as _MMLUProblem,
    MMLUResult as _MMLUResult,
    MMLUScorer as _MMLUScorer,
    parse_answer_letter as _mmlu_parse_answer_letter,
)

MMLUProblem = _MMLUProblem
MMLUResult = _MMLUResult
MMLUScorer = _MMLUScorer
mmlu_parse_answer_letter = _mmlu_parse_answer_letter

METRIC_REGISTRY = globals().setdefault("METRIC_REGISTRY", {})
BENCHMARK_REGISTRY = globals().setdefault("BENCHMARK_REGISTRY", {})
METRIC_REGISTRY.setdefault("mmlu", _MMLUScorer)
BENCHMARK_REGISTRY.setdefault("mmlu", _MMLUProblem)

__all__ = list(__all__) + [
    "MMLUProblem",
    "MMLUResult",
    "MMLUScorer",
    "mmlu_parse_answer_letter",
]

# --- Additive registration for HumanEval+ augmented benchmark ---------------
# Additive only; reuses the METRIC_REGISTRY / BENCHMARK_REGISTRY set above and
# leaves all prior entries untouched.
from .humaneval_plus_scorer import (
    HumanEvalPlusProblem as _HumanEvalPlusProblem,
    HumanEvalPlusResult as _HumanEvalPlusResult,
    score_single as _humaneval_plus_score_single,
    score_problems as _humaneval_plus_score_problems,
)

HumanEvalPlusProblem = _HumanEvalPlusProblem
HumanEvalPlusResult = _HumanEvalPlusResult
humaneval_plus_score_single = _humaneval_plus_score_single
humaneval_plus_score_problems = _humaneval_plus_score_problems

METRIC_REGISTRY = globals().setdefault("METRIC_REGISTRY", {})
BENCHMARK_REGISTRY = globals().setdefault("BENCHMARK_REGISTRY", {})
METRIC_REGISTRY.setdefault("humaneval_plus", _humaneval_plus_score_problems)
BENCHMARK_REGISTRY.setdefault("humaneval_plus", _HumanEvalPlusProblem)

__all__ = list(__all__) + [
    "HumanEvalPlusProblem",
    "HumanEvalPlusResult",
    "humaneval_plus_score_single",
    "humaneval_plus_score_problems",
]

# --- Additive registration for τ-bench agentic tool-use benchmark -----------
# Additive only; reuses the METRIC_REGISTRY / BENCHMARK_REGISTRY set above and
# leaves all prior entries untouched.
from .taubench_scorer import (
    TauBenchTask as _TauBenchTask,
    TauBenchTrajectory as _TauBenchTrajectory,
    TauBenchScorer as _TauBenchScorer,
    TauBenchDataset as _TauBenchDataset,
)

TauBenchTask = _TauBenchTask
TauBenchTrajectory = _TauBenchTrajectory
TauBenchScorer = _TauBenchScorer
TauBenchDataset = _TauBenchDataset

METRIC_REGISTRY = globals().setdefault("METRIC_REGISTRY", {})
BENCHMARK_REGISTRY = globals().setdefault("BENCHMARK_REGISTRY", {})
METRIC_REGISTRY.setdefault("taubench", _TauBenchScorer)
BENCHMARK_REGISTRY.setdefault("taubench", _TauBenchScorer)

__all__ = list(__all__) + [
    "TauBenchTask",
    "TauBenchTrajectory",
    "TauBenchScorer",
    "TauBenchDataset",
]

# --- Additive registration for Mythos agentic-coding rubric ------------------
# Additive only; reuses the METRIC_REGISTRY / BENCHMARK_REGISTRY set above and
# leaves all prior entries untouched. Default config flag is OFF.
from .mythos_coding_rubric import (
    MYTHOS_GUIDANCE_SYSTEM_PROMPT as _MYTHOS_GUIDANCE_SYSTEM_PROMPT,
    DIMENSIONS as _MYTHOS_DIMENSIONS,
    DimensionScore as _MythosDimensionScore,
    RubricResult as _MythosRubricResult,
    MythosCodingRubric as _MythosCodingRubric,
    heuristic_judge as _mythos_heuristic_judge,
    format_trajectory as _mythos_format_trajectory,
)

MYTHOS_GUIDANCE_SYSTEM_PROMPT = _MYTHOS_GUIDANCE_SYSTEM_PROMPT
MYTHOS_DIMENSIONS = _MYTHOS_DIMENSIONS
MythosDimensionScore = _MythosDimensionScore
MythosRubricResult = _MythosRubricResult
MythosCodingRubric = _MythosCodingRubric
mythos_heuristic_judge = _mythos_heuristic_judge
mythos_format_trajectory = _mythos_format_trajectory

# Config flag, default OFF.
eval_mythos_coding_rubric_enabled: bool = False

METRIC_REGISTRY = globals().setdefault("METRIC_REGISTRY", {})
BENCHMARK_REGISTRY = globals().setdefault("BENCHMARK_REGISTRY", {})
METRIC_REGISTRY.setdefault("mythos_coding_rubric", _MythosCodingRubric)
BENCHMARK_REGISTRY.setdefault("mythos_coding_rubric", _MythosCodingRubric)

__all__ = list(__all__) + [
    "MYTHOS_GUIDANCE_SYSTEM_PROMPT",
    "MYTHOS_DIMENSIONS",
    "MythosDimensionScore",
    "MythosRubricResult",
    "MythosCodingRubric",
    "mythos_heuristic_judge",
    "mythos_format_trajectory",
    "eval_mythos_coding_rubric_enabled",
]

# --- Additive registration for Crescendo multi-turn jailbreak probe ----------
# Additive only; reuses METRIC_REGISTRY / BENCHMARK_REGISTRY above and leaves
# all prior entries untouched. Default config flag is OFF.
from .crescendo_probe import (
    CrescendoProbe as _CrescendoProbe,
    CrescendoResult as _CrescendoResult,
    CrescendoTurn as _CrescendoTurn,
    ESCALATION_TEMPLATES as _CRESCENDO_ESCALATION_TEMPLATES,
    stub_target_fn as _crescendo_stub_target_fn,
    stub_judge_fn as _crescendo_stub_judge_fn,
)

CrescendoProbe = _CrescendoProbe
CrescendoResult = _CrescendoResult
CrescendoTurn = _CrescendoTurn
CRESCENDO_ESCALATION_TEMPLATES = _CRESCENDO_ESCALATION_TEMPLATES
crescendo_stub_target_fn = _crescendo_stub_target_fn
crescendo_stub_judge_fn = _crescendo_stub_judge_fn

# Config flag, default OFF.
eval_crescendo_probe_enabled: bool = False

METRIC_REGISTRY = globals().setdefault("METRIC_REGISTRY", {})
BENCHMARK_REGISTRY = globals().setdefault("BENCHMARK_REGISTRY", {})
METRIC_REGISTRY.setdefault("crescendo_probe", _CrescendoProbe)
BENCHMARK_REGISTRY.setdefault("crescendo_probe", _CrescendoProbe)

__all__ = list(__all__) + [
    "CrescendoProbe",
    "CrescendoResult",
    "CrescendoTurn",
    "CRESCENDO_ESCALATION_TEMPLATES",
    "crescendo_stub_target_fn",
    "crescendo_stub_judge_fn",
    "eval_crescendo_probe_enabled",
]

# --- Additive registration for 40-dim behavioral-audit taxonomy --------------
# Additive only; reuses METRIC_REGISTRY / BENCHMARK_REGISTRY above and leaves
# all prior entries untouched. Default config flag is OFF.
from .behavioral_audit_taxonomy import (
    BehaviorCategory as _BehaviorCategory,
    BehaviorDimension as _BehaviorDimension,
    BEHAVIORAL_AUDIT_TAXONOMY as _BEHAVIORAL_AUDIT_TAXONOMY,
    DimensionScore as _BATDimensionScore,
    AuditResult as _BATAuditResult,
    BehavioralAuditRegistry as _BehavioralAuditRegistry,
    by_id as _bat_by_id,
    by_category as _bat_by_category,
)

BehaviorCategory = _BehaviorCategory
BehaviorDimension = _BehaviorDimension
BEHAVIORAL_AUDIT_TAXONOMY = _BEHAVIORAL_AUDIT_TAXONOMY
BATDimensionScore = _BATDimensionScore
BATAuditResult = _BATAuditResult
BehavioralAuditRegistry = _BehavioralAuditRegistry
bat_by_id = _bat_by_id
bat_by_category = _bat_by_category

# Config flag, default OFF.
eval_behavioral_audit_taxonomy_enabled: bool = False

METRIC_REGISTRY = globals().setdefault("METRIC_REGISTRY", {})
BENCHMARK_REGISTRY = globals().setdefault("BENCHMARK_REGISTRY", {})
METRIC_REGISTRY.setdefault(
    "behavioral_audit_taxonomy", _BehavioralAuditRegistry
)
BENCHMARK_REGISTRY.setdefault(
    "behavioral_audit_taxonomy", _BehavioralAuditRegistry
)

__all__ = list(__all__) + [
    "BehaviorCategory",
    "BehaviorDimension",
    "BEHAVIORAL_AUDIT_TAXONOMY",
    "BATDimensionScore",
    "BATAuditResult",
    "BehavioralAuditRegistry",
    "bat_by_id",
    "bat_by_category",
    "eval_behavioral_audit_taxonomy_enabled",
]

# --- Additive registration for synthetic jailbreak probe generator ----------
from .synthetic_jailbreak_generator import (  # noqa: E402
    SyntheticJailbreakGenerator as _SyntheticJailbreakGenerator,
    SyntheticProbe as _SyntheticProbe,
    score_probe_naive_refusal as _score_probe_naive_refusal,
)

SyntheticJailbreakGenerator = _SyntheticJailbreakGenerator
SyntheticProbe = _SyntheticProbe
score_probe_naive_refusal = _score_probe_naive_refusal

eval_synthetic_jailbreak_generator_enabled: bool = False

METRIC_REGISTRY = globals().setdefault("METRIC_REGISTRY", {})
BENCHMARK_REGISTRY = globals().setdefault("BENCHMARK_REGISTRY", {})
METRIC_REGISTRY.setdefault("synthetic_jailbreak", _SyntheticJailbreakGenerator)
BENCHMARK_REGISTRY.setdefault("synthetic_jailbreak", _SyntheticJailbreakGenerator)

__all__ = list(__all__) + [
    "SyntheticJailbreakGenerator",
    "SyntheticProbe",
    "score_probe_naive_refusal",
    "eval_synthetic_jailbreak_generator_enabled",
]

# --- Additive registration for SHADE-Arena sabotage-vs-monitor eval ---------
# Additive only; reuses METRIC_REGISTRY / BENCHMARK_REGISTRY above and leaves
# all prior entries untouched. Default config flag is OFF.
from .shade_arena import (  # noqa: E402
    MainTaskSpec as _ShadeMainTaskSpec,
    SideTaskSpec as _ShadeSideTaskSpec,
    MonitorVerdict as _ShadeMonitorVerdict,
    ShadeArenaResult as _ShadeArenaResult,
    ShadeArenaHarness as _ShadeArenaHarness,
    DEFAULT_SHADE_SCENARIOS as _DEFAULT_SHADE_SCENARIOS,
    VALID_LABELS as _SHADE_VALID_LABELS,
    stub_agent_fn as _shade_stub_agent_fn,
    stub_monitor_fn as _shade_stub_monitor_fn,
)

MainTaskSpec = _ShadeMainTaskSpec
SideTaskSpec = _ShadeSideTaskSpec
MonitorVerdict = _ShadeMonitorVerdict
ShadeArenaResult = _ShadeArenaResult
ShadeArenaHarness = _ShadeArenaHarness
DEFAULT_SHADE_SCENARIOS = _DEFAULT_SHADE_SCENARIOS
SHADE_VALID_LABELS = _SHADE_VALID_LABELS
stub_agent_fn = _shade_stub_agent_fn
stub_monitor_fn = _shade_stub_monitor_fn

# Config flag, default OFF.
eval_shade_arena_enabled: bool = False

METRIC_REGISTRY = globals().setdefault("METRIC_REGISTRY", {})
BENCHMARK_REGISTRY = globals().setdefault("BENCHMARK_REGISTRY", {})
METRIC_REGISTRY.setdefault("shade_arena", _ShadeArenaHarness)
BENCHMARK_REGISTRY.setdefault("shade_arena", _ShadeArenaHarness)

__all__ = list(__all__) + [
    "MainTaskSpec",
    "SideTaskSpec",
    "MonitorVerdict",
    "ShadeArenaResult",
    "ShadeArenaHarness",
    "DEFAULT_SHADE_SCENARIOS",
    "SHADE_VALID_LABELS",
    "stub_agent_fn",
    "stub_monitor_fn",
    "eval_shade_arena_enabled",
]

# --- Additive registration for intent-level vibe code reviewer --------------
# Additive only; reuses METRIC_REGISTRY / BENCHMARK_REGISTRY above and leaves
# all prior entries untouched. Default config flag is OFF.
from .vibe_code_reviewer import (  # noqa: E402
    VULNHUNTER_SYSTEM_PROMPT as _VULNHUNTER_SYSTEM_PROMPT,
    NEGATIVE_EXAMPLES as _VIBE_NEGATIVE_EXAMPLES,
    REQUIRED_FIELDS as _VIBE_REQUIRED_FIELDS,
    VibeFinding as _VibeFinding,
    VibeReviewReport as _VibeReviewReport,
    VibeCodeReviewer as _VibeCodeReviewer,
    stub_judge_fn as _vibe_stub_judge_fn,
)

VULNHUNTER_SYSTEM_PROMPT = _VULNHUNTER_SYSTEM_PROMPT
VIBE_NEGATIVE_EXAMPLES = _VIBE_NEGATIVE_EXAMPLES
VIBE_REQUIRED_FIELDS = _VIBE_REQUIRED_FIELDS
VibeFinding = _VibeFinding
VibeReviewReport = _VibeReviewReport
VibeCodeReviewer = _VibeCodeReviewer
vibe_stub_judge_fn = _vibe_stub_judge_fn

# Config flag, default OFF.
eval_vibe_code_reviewer_enabled: bool = False

METRIC_REGISTRY = globals().setdefault("METRIC_REGISTRY", {})
BENCHMARK_REGISTRY = globals().setdefault("BENCHMARK_REGISTRY", {})
METRIC_REGISTRY.setdefault("vibe_code_reviewer", _VibeCodeReviewer)
BENCHMARK_REGISTRY.setdefault("vibe_code_reviewer", _VibeCodeReviewer)

__all__ = list(__all__) + [
    "VULNHUNTER_SYSTEM_PROMPT",
    "VIBE_NEGATIVE_EXAMPLES",
    "VIBE_REQUIRED_FIELDS",
    "VibeFinding",
    "VibeReviewReport",
    "VibeCodeReviewer",
    "vibe_stub_judge_fn",
    "eval_vibe_code_reviewer_enabled",
]
