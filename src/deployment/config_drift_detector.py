"""Detects drift between expected and actual deployment configuration dicts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfigField:
    path: str
    expected: object
    actual: object
    drifted: bool


@dataclass(frozen=True)
class DriftReport:
    total_fields: int
    drifted_count: int
    drifted_fields: list[ConfigField]
    drift_pct: float


class ConfigDriftDetector:
    """Recursively compares two configuration dicts and reports field-level drift."""

    def compare(
        self,
        expected: dict,
        actual: dict,
        prefix: str = "",
    ) -> DriftReport:
        """Walk *expected* and *actual* recursively, collecting a :class:`ConfigField`
        for every leaf value encountered.  Nested keys are joined with ``"."``.
        """
        fields: list[ConfigField] = []
        self._recurse(expected, actual, prefix, fields)
        drifted = [f for f in fields if f.drifted]
        total = len(fields)
        drift_pct = (len(drifted) / total * 100.0) if total > 0 else 0.0
        return DriftReport(
            total_fields=total,
            drifted_count=len(drifted),
            drifted_fields=drifted,
            drift_pct=drift_pct,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _recurse(
        self,
        expected: object,
        actual: object,
        prefix: str,
        fields: list[ConfigField],
    ) -> None:
        if isinstance(expected, dict) and isinstance(actual, dict):
            all_keys = set(expected) | set(actual)
            for key in sorted(all_keys):
                child_path = f"{prefix}.{key}" if prefix else key
                exp_val = expected.get(key)
                act_val = actual.get(key)
                if isinstance(exp_val, dict) and isinstance(act_val, dict):
                    self._recurse(exp_val, act_val, child_path, fields)
                else:
                    drifted = exp_val != act_val
                    fields.append(
                        ConfigField(
                            path=child_path,
                            expected=exp_val,
                            actual=act_val,
                            drifted=drifted,
                        )
                    )
        else:
            # Leaf comparison at current prefix
            drifted = expected != actual
            fields.append(
                ConfigField(
                    path=prefix,
                    expected=expected,
                    actual=actual,
                    drifted=drifted,
                )
            )

    # ------------------------------------------------------------------
    # Report helpers
    # ------------------------------------------------------------------

    @staticmethod
    def is_clean(report: DriftReport) -> bool:
        """Return True when no fields have drifted."""
        return report.drifted_count == 0

    @staticmethod
    def summary(report: DriftReport) -> str:
        """Return a human-readable drift summary string."""
        pct = report.drift_pct
        # Format percentage: drop the decimal if it is a whole number
        if pct == int(pct):
            pct_str = f"{int(pct)}%"
        else:
            pct_str = f"{pct:.2f}%"
        return f"{report.drifted_count}/{report.total_fields} fields drifted ({pct_str})"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CONFIG_DRIFT_DETECTOR_REGISTRY: dict[str, type[ConfigDriftDetector]] = {
    "default": ConfigDriftDetector,
}
