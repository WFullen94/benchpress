"""Quality result dataclass and combined composite score."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchpress.quality.tasks import TaskResult


@dataclass
class QualityResult:
    """Aggregated quality metrics for a single model evaluation."""

    model: str
    backend: str
    hardware: str
    timestamp: str
    perplexity: float | None  # WikiText-2 perplexity; None if not computed
    tasks: list["TaskResult"] = field(default_factory=list)

    @property
    def mean_accuracy(self) -> float | None:
        if not self.tasks:
            return None
        return sum(t.accuracy for t in self.tasks) / len(self.tasks)

    @property
    def quality_score(self) -> float | None:
        """Composite score in [0, 1]; higher = better.

        Combines task accuracy (50 %) with an inverse log-perplexity term (50 %).
        Falls back to whichever signal is available if one is missing.
        """
        acc = self.mean_accuracy

        ppl_score: float | None = None
        if self.perplexity is not None and self.perplexity > 1.0:
            # Map perplexity → [0, 1]: ppl=1 → score=1, ppl=1000 → score≈0
            ppl_score = max(0.0, 1.0 - math.log(self.perplexity) / math.log(1000))

        if acc is not None and ppl_score is not None:
            return 0.5 * acc + 0.5 * ppl_score
        if acc is not None:
            return acc
        return ppl_score
