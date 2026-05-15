# DAIES v2 Validation Gates

## Overview

DAIES (Decide, Adapt, Integrate, Evaluate, Scale) is the governing process for every Aurelius feature. No feature ships to production or scales to Atlas without passing DAIES gates.

## Gate Categories

### CLI Gates

| Gate | Validates |
|---|---|
| `CLIHelpGate` | `aurelius --help` shows accurate commands |
| `CLIHardwareDetectGate` | Hardware detection produces valid profile |
| `CLIModelLoadGate` | Requested model loads or reports fallback truthfully |
| `CLIBackendSelectGate` | Backend selection produces valid capability report |
| `CLICapabilityReportGate` | Capability report shows all metadata fields |
| `CLIDAIESQuickGate` | Quick gate check runs successfully |
| `CLICUACaptureGate` | CUA capture works in available mode |
| `CLINativeSkillListGate` | Native skills list returns >0 skills |
| `CLINativeSkillRunGate` | Native skill dry-run produces valid output |
| `CLINoMockAsLiveGate` | No stub/mock model is reported as live |

### UI Gates

| Gate | Validates |
|---|---|
| `UIModelHubTruthGate` | Model hub shows requested vs actual, backend, quantization |
| `UIHealthMemoryGate` | Health page shows RAM/VRAM/unified memory usage |
| `UIDAIESSummaryGate` | DAIES dashboard shows gate results |
| `UICUATraceGate` | CUA trace replay shows actions + screenshots |
| `UISkillCatalogGate` | Native skill catalog shows permissions, tests, telemetry |
| `UIAccessibilityGate` | Screen reader keyboard navigation works |
| `UINoSecretExposureGate` | API responses do not contain secrets |

### Skill Gates

| Gate | Validates |
|---|---|
| `SkillManifestValidGate` | Every skill has valid manifest with all required fields |
| `SkillPermissionBoundaryGate` | Skills only use declared permissions |
| `SkillDryRunGate` | All mutating skills support dry-run mode |
| `SkillExecutionGate` | Skill executes within timeout and produces output |
| `SkillTelemetryGate` | Usage counter increments after skill run |
| `SkillSafetyGate` | No credentials/secrets in skill output |
| `SkillNoSecretExfiltrationGate` | Skill does not leak memory or credentials |

### RAM/VRAM Gates

| Gate | Validates |
|---|---|
| `MemoryBudgetEstimatorGate` | Budget report shows all consumer breakdowns |
| `KVCacheBudgetGate` | KV cache stays within allocated budget |
| `QuantizedArtifactLoadGate` | Quantized artifacts load without corruption |
| `MacUnifiedMemoryGate` | Mac profile respects unified memory limits |
| `JetsonMemoryGate` | Jetson profile stays within unified memory budget |
| `BlackwellFP4ProfileGate` | Blackwell profile uses FP4 artifacts |
| `NoOOMFirstResponseGate` | Degradation ladder applies BEFORE OOM |

### Frontier Gates

| Gate | Validates |
|---|---|
| `CodeRepairGate` | Model identifies and fixes failing test |
| `LongContextRecallGate` | Model retrieves fact from long document |
| `ToolReliabilityGate` | Tool calls match schema 100% |
| `CUACompletionGate` | CUA completes task without unsafe actions |
| `DeepResearchGate` | Multi-step research with provenance |
| `ReasoningBudgetGate` | Model uses reasoning steps efficiently |
| `AgentNoLoopGate` | Agent completes without infinite loops |
| `SafetyTrajectoryGate` | No unsafe actions in agent trajectory |

## Existing Benchmarks

- CrossSessionRecall — memory retrieval across sessions
- SurprisePrioritization — identify unexpected but important info
- RelationalGraph — understand entity relationships
- ForgetGate — memory removal does not corrupt valid state
- LongRangeCoherence — maintain consistency across long context
