"""Rolling calibration tracking.

If you can't show Brier < 0.25 (better than coin flip) and a stable log loss
over a meaningful sample, the model probability is noise and Kelly sizing on
it is leverage on noise. The engine exposes these metrics so the offline
analyst (LLM loop) and the human can gate parameter changes on them.
"""
from __future__ import annotations

import math
from collections import deque


class CalibrationTracker:
    def __init__(self, window: int = 500) -> None:
        self._records: deque[tuple[float, int]] = deque(maxlen=window)

    def record(self, predicted_prob: float, outcome: int) -> None:
        if outcome not in (0, 1):
            raise ValueError("outcome must be 0 or 1")
        p = min(max(predicted_prob, 1e-6), 1 - 1e-6)
        self._records.append((p, outcome))

    @property
    def n(self) -> int:
        return len(self._records)

    def brier(self) -> float | None:
        if not self._records:
            return None
        return sum((p - o) ** 2 for p, o in self._records) / len(self._records)

    def log_loss(self) -> float | None:
        if not self._records:
            return None
        total = 0.0
        for p, o in self._records:
            total += -(o * math.log(p) + (1 - o) * math.log(1 - p))
        return total / len(self._records)

    def hit_rate(self) -> float | None:
        if not self._records:
            return None
        hits = sum(1 for p, o in self._records if (p >= 0.5) == (o == 1))
        return hits / len(self._records)

    def summary(self) -> dict:
        return {
            "n": self.n,
            "brier": self.brier(),
            "log_loss": self.log_loss(),
            "hit_rate": self.hit_rate(),
        }
