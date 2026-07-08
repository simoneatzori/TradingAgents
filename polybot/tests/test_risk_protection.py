"""Capital-protection suite: the 1%-per-day budget, pair-leg integrity,
settlement loop, drawdown/profit-lock breakers, and restart persistence.
Each test encodes a way the previous version could lose more than intended.
"""
from __future__ import annotations

import os
import sys
import time
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Settings
from engine import HALT_KV_KEY, Engine
from executor import DryRunExecutor, ExecutionResult
from models import BookTop, MarketWindow
from notifier import TelegramNotifier
from risk_gate import RiskGate
from store import Store


def mk_settings(tmp_path, **kw) -> Settings:
    defaults = dict(db_path=str(tmp_path / "t.sqlite3"),
                    kill_switch_file=str(tmp_path / "KILL"),
                    bankroll=100, min_bet=1, max_bet=2,
                    max_daily_loss=5, max_daily_loss_pct=0.05,
                    max_consecutive_losses=3, max_open_exposure=10,
                    maker_enabled=False)   # maker paths tested in test_maker.py
    defaults.update(kw)
    return Settings(**defaults)


def mk_engine(tmp_path, **kw) -> Engine:
    s = mk_settings(tmp_path, **kw)
    return Engine(s, store=Store(s.db_path), notifier=TelegramNotifier("", ""))


def mk_market(window_start: int) -> MarketWindow:
    return MarketWindow("btc-test", "cid", "ytok", "ntok",
                        window_start, window_start + 300,
                        fee_rate_bps=200, tick_size=0.01, min_order_size=1)


def aligned_now() -> float:
    now = time.time()
    return now - (now % 300) + 100


# ------------------------------------------------------- daily risk budget
def mk_gate(**kw):
    defaults = dict(max_daily_loss=5, max_consecutive_losses=5,
                    max_open_exposure=100, kill_switch_file="/tmp/__nope__",
                    bankroll=100, max_daily_loss_pct=0.01)
    defaults.update(kw)
    return RiskGate(**defaults)


def test_budget_is_min_of_abs_and_pct():
    assert mk_gate().daily_loss_budget == pytest.approx(1.0)          # 1% of 100
    assert mk_gate(max_daily_loss=0.5).daily_loss_budget == pytest.approx(0.5)


def test_worst_case_budget_blocks_before_the_loss_happens():
    g = mk_gate()                                   # budget = 1.0
    assert g.check_order(0.9).allowed               # wc defaults to notional
    g.on_order_filled(0.9)                          # open risk 0.9
    d = g.check_order(0.5)
    assert not d.allowed and d.reason == "daily_risk_budget"
    # a completed pair (worst case ~0) still passes
    assert g.check_order(0.5, worst_case_loss=0.0).allowed


def test_realized_loss_plus_open_risk_share_one_budget():
    g = mk_gate()
    g.on_order_filled(0.6)
    g.on_position_settled(0.6, pnl=-0.6)            # realized -0.6, risk freed
    d = g.check_order(0.6)                          # 0.6 + 0 + 0.6 > 1.0
    assert not d.allowed and d.reason == "daily_risk_budget"
    assert g.check_order(0.3).allowed


def test_pair_risk_release():
    g = mk_gate()
    g.on_order_filled(0.45, worst_case_loss=0.45)   # leg 1
    g.on_order_filled(0.45, worst_case_loss=0.0)    # leg 2
    g.release_risk(0.45)                            # pair completed
    assert g.open_risk == pytest.approx(0.0)
    assert g.open_exposure == pytest.approx(0.9)
    # settlement of the legs releases exposure, not risk (already zero)
    g.on_position_settled(0.45, pnl=+0.5, risk_release=0.0, counts_streak=False)
    g.on_position_settled(0.45, pnl=-0.44, risk_release=0.0, counts_streak=False)
    assert g.consecutive_losses == 0                # pair legs never poison streak
    assert g.open_exposure == pytest.approx(0.0)


def test_drawdown_breaker_halts():
    g = mk_gate(max_daily_loss=1000, max_daily_loss_pct=0.05, max_drawdown_pct=0.03)
    g.on_order_filled(4.0)
    g.on_position_settled(4.0, pnl=-3.5)            # equity 96.5, HWM 100 -> 3.5%
    assert g.halted and "drawdown" in g.halt_reason


def test_daily_profit_lock_denies_without_halting():
    g = mk_gate(daily_profit_lock_pct=0.02)
    g.on_order_filled(0.5)
    g.on_position_settled(0.5, pnl=+2.5)            # +2.5% >= 2% lock
    d = g.check_order(0.5)
    assert not d.allowed and d.reason == "daily_profit_lock"
    assert not g.halted


# ------------------------------------------------------- pair-leg integrity
def test_pair_legs_have_equal_shares_despite_asymmetric_books(tmp_path):
    eng = mk_engine(tmp_path)
    t = aligned_now()
    market = mk_market(int(t - (t % 300)))
    yes = BookTop("ytok", 0.43, 0.44, 100, 8.0)     # thin YES ask
    no = BookTop("ntok", 0.44, 0.45, 100, 3.0)      # thinner NO ask
    with patch.object(eng.clock, "now", return_value=t):
        results = eng.tick(market, yes, no)
    assert len(results) == 2
    assert results[0].fill.size == pytest.approx(results[1].fill.size)
    # sized by the thinner leg (3.0 shares), not max_bet (2/0.89 = 2.24)
    assert results[0].fill.size <= 3.0


class SecondLegFails:
    """First submit succeeds via the real simulator, second is rejected."""

    def __init__(self, settings: Settings) -> None:
        self.inner = DryRunExecutor(settings)
        self.calls = 0

    def submit(self, order, book):
        self.calls += 1
        if self.calls >= 2:
            return ExecutionResult(False, reason="fok_rejected")
        return self.inner.submit(order, book)


def test_unhedged_pair_leg_halts_and_persists(tmp_path):
    s = mk_settings(tmp_path)
    store = Store(s.db_path)
    eng = Engine(s, store=store, executor=SecondLegFails(s),
                 notifier=TelegramNotifier("", ""))
    t = aligned_now()
    market = mk_market(int(t - (t % 300)))
    yes = BookTop("ytok", 0.44, 0.45, 100, 100)
    no = BookTop("ntok", 0.44, 0.45, 100, 100)
    with patch.object(eng.clock, "now", return_value=t):
        results = eng.tick(market, yes, no)
    assert len(results) == 1                        # only the filled leg
    assert eng.gate.halted and eng.gate.halt_reason == "unhedged_pair_leg"
    assert store.kv_get(HALT_KV_KEY) == "unhedged_pair_leg"
    statuses = {r["status"] for r in store.open_orders()}
    assert "unhedged" in statuses
    # the naked leg's premium is still counted as open risk
    assert eng.gate.open_risk > 0

    # a restarted engine comes back halted with the risk still on the books
    eng2 = Engine(s, store=Store(s.db_path), executor=DryRunExecutor(s),
                  notifier=TelegramNotifier("", ""))
    assert eng2.gate.halted
    assert eng2.gate.open_risk > 0


# ----------------------------------------------------------- settlement loop
def test_settle_expired_pairs_and_streak(tmp_path):
    eng = mk_engine(tmp_path)
    t = aligned_now()
    w = int(t - (t % 300))
    market = mk_market(w)
    yes = BookTop("ytok", 0.44, 0.45, 100, 100)
    no = BookTop("ntok", 0.44, 0.45, 100, 100)
    with patch.object(eng.clock, "now", return_value=t):
        results = eng.tick(market, yes, no, spot=100.0)   # records window open
    assert len(results) == 2

    # next window's first spot records the previous close (up move)
    eng._track_window(w + 300 + 1, 101.0)
    n = eng.settle_expired(now=w + 300 + 60)
    assert n == 2
    assert eng.store.stats()["settled"] == 2
    # pair as a whole is profitable and must not touch the loss streak
    assert eng.gate.daily_pnl > 0
    assert eng.gate.consecutive_losses == 0
    assert eng.gate.open_exposure == pytest.approx(0.0, abs=1e-9)
    assert eng.gate.open_risk == pytest.approx(0.0, abs=1e-9)


def test_settle_waits_for_close_price(tmp_path):
    eng = mk_engine(tmp_path)
    t = aligned_now()
    w = int(t - (t % 300))
    market = mk_market(w)
    yes = BookTop("ytok", 0.44, 0.45, 100, 100)
    no = BookTop("ntok", 0.44, 0.45, 100, 100)
    with patch.object(eng.clock, "now", return_value=t):
        eng.tick(market, yes, no, spot=100.0)
    # window expired but no close recorded yet -> nothing settles
    assert eng.settle_expired(now=w + 300 + 60) == 0


# ------------------------------------------------------ restart persistence
def test_daily_pnl_and_streak_survive_restart(tmp_path):
    s = mk_settings(tmp_path)
    eng = Engine(s, store=Store(s.db_path), notifier=TelegramNotifier("", ""))
    # record a real directional order + loss so the streak query sees it
    eng.store.record_order(client_id="c1", market_slug="m", window_start=0,
                           outcome="YES", side="BUY", price=0.5, size=2,
                           strategy="brownian_dir", dry_run=True, prob=0.6)
    eng.store.record_fill("c1", 0.5, 2, 0.01)
    eng.gate.on_order_filled(1.0, 1.0)
    eng.settle("c1", won=False, notional=1.0, payout=0.0, fee_paid=0.01,
               predicted_prob=0.6)
    assert eng.gate.daily_pnl == pytest.approx(-1.01)

    eng2 = Engine(s, store=Store(s.db_path), notifier=TelegramNotifier("", ""))
    assert eng2.gate.daily_pnl == pytest.approx(-1.01)
    assert eng2.gate.consecutive_losses == 1
    assert eng2.gate.cumulative_pnl == pytest.approx(-1.01)
    assert eng2.gate.high_water_mark == pytest.approx(100.0)


# ------------------------------------------------------- directional guards
def warm_engine_for_directional(eng: Engine, t: float, open_price: float) -> None:
    w = int(t - (t % 300))
    eng._current_window_start = w
    eng._window_open_price = open_price
    eng.store.record_window_open(w, open_price)
    eng.vol._var = 1e-8            # sigma_per_sqrt_s = 1e-4
    eng.vol._last_price = open_price
    eng.vol._last_ts = t - 1
    eng.vol.n = 100


def test_directional_blocked_until_vol_warmup(tmp_path):
    eng = mk_engine(tmp_path)
    t = aligned_now()
    market = mk_market(int(t - (t % 300)))
    yes = BookTop("ytok", 0.49, 0.50, 100, 100)
    no = BookTop("ntok", 0.51, 0.52, 100, 100)      # 1.02 sum: no pair arb
    warm_engine_for_directional(eng, t, 100.0)
    eng.vol.n = 3                                   # below vol_min_samples
    with patch.object(eng.clock, "now", return_value=t):
        assert eng.tick(market, yes, no, spot=101.0) == []


def test_directional_fires_after_warmup_and_stores_prob(tmp_path):
    eng = mk_engine(tmp_path)
    t = aligned_now()
    market = mk_market(int(t - (t % 300)))
    yes = BookTop("ytok", 0.49, 0.50, 100, 100)
    no = BookTop("ntok", 0.51, 0.52, 100, 100)
    warm_engine_for_directional(eng, t, 100.0)
    with patch.object(eng.clock, "now", return_value=t):
        results = eng.tick(market, yes, no, spot=101.0)
    assert len(results) == 1 and results[0].accepted
    rows = eng.store.unsettled_filled_orders()
    assert rows[0]["strategy"] == "brownian_dir"
    assert rows[0]["prob"] is not None and rows[0]["prob"] > 0.55


def test_directional_skips_wide_spread(tmp_path):
    eng = mk_engine(tmp_path, max_spread=0.03)
    t = aligned_now()
    market = mk_market(int(t - (t % 300)))
    yes = BookTop("ytok", 0.40, 0.50, 100, 100)     # 10-cent spread
    no = BookTop("ntok", 0.51, 0.52, 100, 100)
    warm_engine_for_directional(eng, t, 100.0)
    with patch.object(eng.clock, "now", return_value=t):
        assert eng.tick(market, yes, no, spot=101.0) == []


def test_directional_sizing_respects_remaining_budget(tmp_path):
    # bankroll 100, 1% budget -> 1.0 total worst-case per day
    eng = mk_engine(tmp_path, max_daily_loss_pct=0.01, min_bet=0.5)
    t = aligned_now()
    market = mk_market(int(t - (t % 300)))
    yes = BookTop("ytok", 0.49, 0.50, 100, 100)
    no = BookTop("ntok", 0.51, 0.52, 100, 100)
    warm_engine_for_directional(eng, t, 100.0)
    with patch.object(eng.clock, "now", return_value=t):
        results = eng.tick(market, yes, no, spot=101.0)
    assert len(results) == 1
    fill = results[0].fill
    # worst case (premium AND entry fee) never exceeds the 1% budget
    assert fill.price * fill.size + fill.fee_paid <= 1.0 + 1e-9
    assert eng.gate.remaining_risk_budget < 0.5     # budget consumed
