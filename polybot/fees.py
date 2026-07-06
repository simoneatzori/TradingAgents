"""Fee-aware edge math.

Polymarket CLOB taker fee model (verify rate per-market from metadata):
    fee = fee_rate * shares * min(price, 1 - price)

This is a HARD GATE: a signal whose net edge after fees is below MIN_EDGE
must be rejected, not just logged.
"""
from __future__ import annotations


def taker_fee_per_share(price: float, fee_rate_bps: int) -> float:
    rate = fee_rate_bps / 10_000.0
    return rate * min(price, 1.0 - price)


def taker_fee(price: float, size: float, fee_rate_bps: int) -> float:
    return taker_fee_per_share(price, fee_rate_bps) * size


def directional_net_edge(probability: float, price: float, fee_rate_bps: int) -> float:
    """Expected value per share of buying one outcome token at `price`,
    given model probability that it settles at 1. Net of taker fee."""
    gross = probability - price
    return gross - taker_fee_per_share(price, fee_rate_bps)


def pair_cost_net_edge(ask_yes: float, ask_no: float, fee_rate_bps: int) -> float:
    """Risk-free edge per pair of buying YES and NO together.

    Gross = 1 - (ask_yes + ask_no). Fees are paid on both legs.
    Positive net edge -> guaranteed profit at settlement (one leg pays 1).
    """
    gross = 1.0 - (ask_yes + ask_no)
    fees = taker_fee_per_share(ask_yes, fee_rate_bps) + taker_fee_per_share(ask_no, fee_rate_bps)
    return gross - fees


def round_to_tick(price: float, tick: float) -> float:
    """Round DOWN for buys: never cross your own limit by rounding up."""
    if tick <= 0:
        raise ValueError("tick must be > 0")
    ticks = int(price / tick + 1e-9)
    return round(ticks * tick, 10)
