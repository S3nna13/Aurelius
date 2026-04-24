"""Aurelius calibration scorer: Expected Calibration Error and reliability diagram data."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CalibrationBin:
    bin_lower: float
    bin_upper: float
    count: int
    mean_confidence: float
    mean_accuracy: float

    @property
    def calibration_error(self) -> float:
        """Absolute difference between mean_confidence and mean_accuracy."""
        return abs(self.mean_confidence - self.mean_accuracy)


@dataclass(frozen=True)
class CalibrationResult:
    ece: float
    mce: float
    bins: list[CalibrationBin]
    num_samples: int


class CalibrationScorer:
    """Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)."""

    def __init__(self, num_bins: int = 10) -> None:
        self.num_bins = num_bins

    def score(
        self,
        confidences: list[float],
        correct: list[bool],
    ) -> CalibrationResult:
        """Compute ECE and MCE by binning predictions into num_bins equal-width bins.

        ECE = sum over bins of (bin_count / total) * |mean_confidence - mean_accuracy|.
        MCE = max calibration error across non-empty bins.
        Returns a CalibrationResult with all bin statistics.
        """
        if len(confidences) != len(correct):
            raise ValueError(
                f"confidences and correct must have the same length, "
                f"got {len(confidences)} and {len(correct)}"
            )

        n_samples = len(confidences)
        bin_width = 1.0 / self.num_bins

        # Initialise bins
        bin_confidences: list[list[float]] = [[] for _ in range(self.num_bins)]
        bin_accuracies: list[list[float]] = [[] for _ in range(self.num_bins)]

        for conf, corr in zip(confidences, correct):
            # Clamp to [0, 1] and assign to bin; confidence==1.0 goes to last bin
            bin_idx = min(int(conf / bin_width), self.num_bins - 1)
            bin_confidences[bin_idx].append(conf)
            bin_accuracies[bin_idx].append(1.0 if corr else 0.0)

        bins: list[CalibrationBin] = []
        for b in range(self.num_bins):
            lower = b * bin_width
            upper = lower + bin_width
            count = len(bin_confidences[b])
            if count > 0:
                mean_conf = sum(bin_confidences[b]) / count
                mean_acc = sum(bin_accuracies[b]) / count
            else:
                mean_conf = 0.0
                mean_acc = 0.0
            bins.append(
                CalibrationBin(
                    bin_lower=lower,
                    bin_upper=upper,
                    count=count,
                    mean_confidence=mean_conf,
                    mean_accuracy=mean_acc,
                )
            )

        # ECE: weighted mean of calibration error over all bins
        if n_samples == 0:
            ece = 0.0
            mce = 0.0
        else:
            ece = sum(
                (b.count / n_samples) * b.calibration_error
                for b in bins
                if b.count > 0
            )
            mce = max(
                (b.calibration_error for b in bins if b.count > 0),
                default=0.0,
            )

        return CalibrationResult(
            ece=ece,
            mce=mce,
            bins=bins,
            num_samples=n_samples,
        )

    def reliability_diagram_data(
        self,
        result: CalibrationResult,
    ) -> list[dict]:
        """Return per-bin data suitable for plotting a reliability diagram.

        Each dict has keys: bin_center, confidence, accuracy, count.
        """
        output: list[dict] = []
        for b in result.bins:
            bin_center = (b.bin_lower + b.bin_upper) / 2.0
            output.append(
                {
                    "bin_center": bin_center,
                    "confidence": b.mean_confidence,
                    "accuracy": b.mean_accuracy,
                    "count": b.count,
                }
            )
        return output


CALIBRATION_SCORER_REGISTRY = {"default": CalibrationScorer}
