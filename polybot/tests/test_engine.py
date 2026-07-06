"""Integration test: full tick path — strategy -> fee gate -> sizing ->
risk gate -> idempotent persistence -> dry-run fill -> settlement -> breaker.
"""
from __future__ import annotations

import os
import sys
import time
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Settings
from engine import Engine
from models import BookTop, MarketWindow
from notifier import TelegramNotifier
from store import Store


def mk_engine(tmp_path) -> Engine:
    s = Settings(db_path=str(tmp_path / "t.sqlite3"),
                 kill_switch_file=str(tmp_path / "KILL"),
                 bankroll=100, min_bet=1, max_bet=2,
                 max_daily_loss=5, max_daily_loss_pct=0.05,
                 max_consecutive_losses=3, max_open_exposure=10)
    return Engine(s, store=Store(s.db_path),
                  notifier=TelegramNotifier("", ""))   # disabled notifier


def mk_market(window_start: int) -> MarketWindow:
    return MarketWindow("btc-test", "cid", "ytok", "ntok",
                        window_start, window_start + 300,
                        fee_rate_bps=200, tick_size=0.01, min_order_size=1)


def aligned_now() -> float:
    """A timestamp 100s into the current 5-min window (inside entry zone)."""
    now = time.time()
    return now - (now % 300) + 100


def test_full_arb_tick_and_idempotency(tmp_path):
    eng = mk_engine(tmp_path)
    t = aligned_now()
    market = mk_market(int(t - (t % 300)))
    yes = BookTop("ytok", 0.44, 0.45, 100, 100)
    no = BookTop("ntok", 0.44, 0.45, 100, 100)

    with patch.object(eng.clock, "now", return_value=t):
        results = eng.tick(market, yes, no)
        assert len(results) == 2 and all(r.accepted for r in results)
        assert eng.gate.open_exposure > 0
        # same window, same books -> duplicate orders suppressed, no double-fire
        results2 = eng.tick(market, yes, no)
        assert results2 == []

    assert eng.store.stats()["orders"] == 2


def test_no_entry_near_settlement(tmp_path):
    eng = mk_engine(tmp_path)
    now = time.time()
    t = now - (now % 300) + 290          # 10s to close
    market = mk_market(int(t - (t % 300)))
    yes = BookTop("ytok", 0.40, 0.41, 100, 100)
    no = BookTop("ntok", 0.40, 0.41, 100, 100)
    with patch.object(eng.clock, "now", return_value=t):
        assert eng.tick(market, yes, no) == []


def test_thin_edge_rejected_by_fee_gate(tmp_path):
    eng = mk_engine(tmp_path)
    t = aligned_now()
    market = mk_market(int(t - (t % 300)))
    yes = BookTop("ytok", 0.48, 0.49, 100, 100)   # gross 2c, net < 1c
    no = BookTop("ntok", 0.48, 0.49, 100, 100)
    with patch.object(eng.clock, "now", return_value=t):
        assert eng.tick(market, yes, no) == []
    assert eng.store.stats()["orders"] == 0


def test_settlement_breaker_halts_engine(tmp_path):
    eng = mk_engine(tmp_path)
    t = aligned_now()
    market = mk_market(int(t - (t % 300)))
    yes = BookTop("ytok", 0.44, 0.45, 100, 100)
    no = BookTop("ntok", 0.44, 0.45, 100, 100)
    with patch.object(eng.clock, "now", return_value=t):
        results = eng.tick(market, yes, no)
    assert results
    # settle everything as a big loss -> daily loss breaker
    rows = eng.store.open_orders()
    for row in rows:
        eng.settle(row["client_id"], won=False,
                   notional=row["price"] * row["size"],
                   payout=0.0, fee_paid=0.01, predicted_prob=0.6)
    # force a loss bigger than the 5.0 limit
    eng.gate.daily_pnl = -6.0
    d = eng.gate.check_order(1.0)
    assert not d.allowed and eng.gate.halted
    assert eng.calibration.n == len(rows)
