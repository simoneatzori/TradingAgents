"""Position sizing: fractional Kelly with shrinkage and hard caps.

Why shrinkage: Kelly amplifies probability-estimation error. A Brownian-motion
model on 5-minute BTC is poorly calibrated out of the box, so the model
probability is pulled toward the 0.5 prior before sizing. Increase the
shrinkage weight only when rolling calibration (Brier / log loss) supports it.
"""
from __future__ import annotations

import math


def shrink_probability(p_model: float, weight: float, prior: float = 0.5) -> float:
    if not 0.0 <= weight <= 1.0:
        raise ValueError("shrinkage weight must be in [0, 1]")
    return weight * p_model + (1.0 - weight) * prior


def kelly_fraction(probability: float, price: float) -> float:
    """Kelly fraction for a binary contract bought at `price` paying 1.

    Odds b = (1 - price) / price; f* = (b*p - q) / b. Clamped at 0.
    """
    if not 0.0 < price < 1.0:
        return 0.0
    b = (1.0 - price) / price
    q = 1.0 - probability
    f = (b * probability - q) / b
    return max(f, 0.0)


def position_notional(
    bankroll: float,
    probability: float,
    price: float,
    kelly_multiplier: float = 0.25,
    min_bet: float = 1.0,
    max_bet: float = 2.0,
    max_bankroll_fraction: float = 0.05,
) -> float:
    """Returns USDC notional to deploy, or 0.0 if below min_bet.

    Caps applied in order: fractional Kelly -> bankroll fraction -> max_bet.
    """
    f = kelly_fraction(probability, price) * kelly_multiplier
    stake = bankroll * f
    stake = min(stake, bankroll * max_bankroll_fraction, max_bet)
    if stake < min_bet:
        return 0.0
    return round(stake, 2)


def shares_for_notional(notional: float, price: float, min_order_size: float) -> float:
    """Floor (never round up): rounding up could exceed the notional the risk
    gate approved or the displayed liquidity the size was derived from."""
    if price <= 0:
        return 0.0
    shares = math.floor((notional / price) * 100) / 100
    if shares < min_order_size:
        return 0.0
    return shares
