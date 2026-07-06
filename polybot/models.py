"""Domain models shared across the engine."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class Outcome(str, Enum):
    YES = "YES"   # UP
    NO = "NO"     # DOWN


@dataclass(frozen=True)
class BookTop:
    """Top of book for one outcome token."""
    token_id: str
    bid: float
    ask: float
    bid_size: float
    ask_size: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0


@dataclass(frozen=True)
class MarketWindow:
    slug: str
    condition_id: str
    yes_token: str
    no_token: str
    window_start: int          # unix seconds, aligned
    window_end: int
    fee_rate_bps: int
    tick_size: float
    min_order_size: float


@dataclass(frozen=True)
class Signal:
    strategy: str
    outcome: Outcome
    probability: float          # model probability that this outcome wins
    price: float                # limit price we are willing to pay
    net_edge: float             # expected value per share AFTER fees
    paired: bool = False        # True for pair-cost arb (buy both legs)
    note: str = ""


@dataclass
class Order:
    market: MarketWindow
    outcome: Outcome
    side: Side
    price: float
    size: float                 # shares
    strategy: str
    client_id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.client_id:
            self.client_id = deterministic_client_id(
                self.market.slug, self.outcome.value, self.side.value,
                self.market.window_start, self.strategy,
            )

    @property
    def notional(self) -> float:
        return self.price * self.size


def deterministic_client_id(*parts: object) -> str:
    """Idempotency key: same (market, window, outcome, side, strategy) -> same id.

    A retry after a network error reuses the id, so the exchange/executor can
    dedupe instead of double-firing.
    """
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


@dataclass
class Fill:
    order_client_id: str
    price: float
    size: float
    fee_paid: float
    ts: int
