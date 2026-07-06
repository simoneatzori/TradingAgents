"""Strategies. Each returns Signal(s) or nothing. Fee gate enforced HERE,
before sizing and before the risk gate ever sees the order.

PairCostArb is the only strategy with bounded downside; go live with it first.
BrownianDirectional is included for DRY_RUN calibration data collection — do
not enable it live until CalibrationTracker proves the model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from models import BookTop, Outcome, Signal
from fees import pair_cost_net_edge, directional_net_edge


@dataclass
class PairCostArb:
    min_edge: float
    fee_rate_bps: int
    name: str = "pair_cost_arb"

    def evaluate(self, yes_book: BookTop, no_book: BookTop) -> list[Signal]:
        if yes_book.ask <= 0 or no_book.ask <= 0:
            return []
        if yes_book.ask_size <= 0 or no_book.ask_size <= 0:
            return []
        net = pair_cost_net_edge(yes_book.ask, no_book.ask, self.fee_rate_bps)
        if net < self.min_edge:
            return []
        note = f"pair ask {yes_book.ask:.2f}+{no_book.ask:.2f}, net {net:.4f}/pair"
        return [
            Signal(self.name, Outcome.YES, 1.0, yes_book.ask, net, paired=True, note=note),
            Signal(self.name, Outcome.NO, 1.0, no_book.ask, net, paired=True, note=note),
        ]


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass
class BrownianDirectional:
    """P(close > open) under driftless Brownian motion with EWMA volatility.

    z = log(spot/window_open) / (sigma_per_sqrt_s * sqrt(seconds_remaining))
    p_up = Phi(z)

    KNOWN LIMITS (why shrinkage + calibration gating exist):
    - 5-min BTC returns are fat-tailed and autocorrelated, not Gaussian
    - EWMA vol lags regime changes exactly when it matters most
    """
    min_edge: float
    min_prob: float
    fee_rate_bps: int
    name: str = "brownian_dir"

    def probability_up(self, spot: float, window_open: float,
                       sigma_per_sqrt_s: float, seconds_remaining: float) -> float:
        if spot <= 0 or window_open <= 0 or sigma_per_sqrt_s <= 0 or seconds_remaining <= 0:
            return 0.5
        z = math.log(spot / window_open) / (sigma_per_sqrt_s * math.sqrt(seconds_remaining))
        return _norm_cdf(z)

    def evaluate(self, yes_book: BookTop, no_book: BookTop, p_up: float) -> list[Signal]:
        candidates: list[tuple[Outcome, float, float]] = [
            (Outcome.YES, p_up, yes_book.ask),
            (Outcome.NO, 1.0 - p_up, no_book.ask),
        ]
        out: list[Signal] = []
        for outcome, p, ask in candidates:
            if p < self.min_prob or not 0.0 < ask < 1.0:
                continue
            net = directional_net_edge(p, ask, self.fee_rate_bps)
            if net < self.min_edge:
                continue
            out.append(Signal(self.name, outcome, p, ask, net,
                              note=f"p={p:.3f} ask={ask:.2f} net={net:.4f}"))
        # never take both sides directionally; keep the better edge
        out.sort(key=lambda s: s.net_edge, reverse=True)
        return out[:1]


class EwmaVol:
    """EWMA volatility of log returns, normalized per sqrt(second)."""

    def __init__(self, halflife_s: float = 120.0) -> None:
        self.alpha = 1 - math.exp(math.log(0.5) / max(halflife_s, 1.0))
        self._var: float | None = None
        self._last_price: float | None = None
        self._last_ts: float | None = None
        self.n = 0                     # samples seen; engine gates on warm-up

    def update(self, price: float, ts: float) -> None:
        self.n += 1
        if self._last_price is not None and self._last_ts is not None:
            dt_s = max(ts - self._last_ts, 1e-3)
            r = math.log(price / self._last_price)
            inst_var = (r * r) / dt_s            # variance per second
            if self._var is None:
                self._var = inst_var
            else:
                self._var = (1 - self.alpha) * self._var + self.alpha * inst_var
        self._last_price = price
        self._last_ts = ts

    @property
    def sigma_per_sqrt_s(self) -> float:
        if self._var is None or self._var <= 0:
            return 0.0
        return math.sqrt(self._var)
