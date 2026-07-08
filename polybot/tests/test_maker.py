"""Maker pair-quoting suite. Every test encodes one way passive quoting
can hurt you: one-sided fills, stale quotes near settlement, budget leaks,
crashes with orders resting, repricing churn.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Settings
from engine import Engine
from executor import DryRunExecutor
from maker import MAKER_STRATEGY, MakerPairQuoter, QState
from models import BookTop, MarketWindow, Outcome
from notifier import TelegramNotifier
from risk_gate import RiskGate
from store import Store

W = 600_000                      # aligned window start used across tests


class SpyNotifier(TelegramNotifier):
    def __init__(self):
        super().__init__("", "")
        self.sent: list[str] = []

    def send(self, text: str) -> None:
        self.sent.append(text)


def mk_settings(tmp_path, **kw) -> Settings:
    defaults = dict(db_path=str(tmp_path / "t.sqlite3"),
                    kill_switch_file=str(tmp_path / "KILL"),
                    bankroll=100, min_bet=1, max_bet=2,
                    max_daily_loss=5, max_daily_loss_pct=0.05,
                    max_consecutive_losses=5, max_open_exposure=10)
    defaults.update(kw)
    return Settings(**defaults)


def mk_quoter(tmp_path, **kw):
    s = mk_settings(tmp_path, **kw)
    store = Store(s.db_path)
    gate = RiskGate(bankroll=s.bankroll, max_daily_loss=s.max_daily_loss,
                    max_daily_loss_pct=s.max_daily_loss_pct,
                    max_consecutive_losses=s.max_consecutive_losses,
                    max_open_exposure=s.max_open_exposure,
                    kill_switch_file=s.kill_switch_file)
    notifier = SpyNotifier()
    q = MakerPairQuoter(s, store, gate, DryRunExecutor(s), notifier)
    return q, store, gate, notifier


def mk_market(window_start: int = W) -> MarketWindow:
    return MarketWindow("btc-test", "cid", "ytok", "ntok",
                        window_start, window_start + 300,
                        fee_rate_bps=200, tick_size=0.01, min_order_size=1)


def book(token, bid, ask, bid_sz=100.0, ask_sz=100.0):
    return BookTop(token, bid, ask, bid_sz, ask_sz)


# ---------------------------------------------------------------- placement
def test_quotes_join_bid_and_lock_min_edge(tmp_path):
    q, store, gate, _ = mk_quoter(tmp_path)
    m = mk_market()
    q.tick(W + 60, m, book("y", 0.44, 0.55), book("n", 0.44, 0.55))
    assert q.state is QState.QUOTED
    py, pn = q.legs[Outcome.YES].price, q.legs[Outcome.NO].price
    assert py == pytest.approx(0.45) and pn == pytest.approx(0.45)
    assert py + pn <= 1.0 - q.s.min_edge + 1e-9
    rows = store.open_orders()
    assert len(rows) == 2 and all(r["status"] == "open" for r in rows)
    assert gate.open_risk > 0                     # reserved before any fill
    assert gate.open_exposure == 0                # nothing filled yet


def test_quotes_scaled_down_when_bids_too_rich(tmp_path):
    q, *_ = mk_quoter(tmp_path)
    m = mk_market()
    # bid+tick sums to 1.22 -> must be pushed under 1 - min_edge = 0.99
    q.tick(W + 60, m, book("y", 0.60, 0.70), book("n", 0.60, 0.70))
    assert q.state is QState.QUOTED
    total = q.legs[Outcome.YES].price + q.legs[Outcome.NO].price
    assert total <= 1.0 - q.s.min_edge + 1e-9


def test_no_new_quotes_close_to_settlement(tmp_path):
    q, *_ = mk_quoter(tmp_path)
    m = mk_market()
    q.tick(W + 300 - 50, m, book("y", 0.44, 0.55), book("n", 0.44, 0.55))
    assert q.state is QState.IDLE                 # < quote_min_remaining_s


def test_reservation_respects_daily_budget(tmp_path):
    q, _, gate, _ = mk_quoter(tmp_path, max_daily_loss_pct=0.01)  # budget 1.0
    m = mk_market()
    q.tick(W + 60, m, book("y", 0.44, 0.55), book("n", 0.44, 0.55))
    assert q.state is QState.QUOTED
    assert gate.open_risk <= 1.0 + 1e-9           # shares scaled to fit


# -------------------------------------------------------------------- fills
def test_both_legs_fill_releases_risk(tmp_path):
    q, store, gate, _ = mk_quoter(tmp_path)
    m = mk_market()
    q.tick(W + 60, m, book("y", 0.44, 0.55), book("n", 0.44, 0.55))
    # sellers cross both quotes
    q.tick(W + 70, m, book("y", 0.40, 0.45), book("n", 0.40, 0.45))
    assert q.state is QState.DONE
    assert gate.open_risk == pytest.approx(0.0)   # pair locked
    assert gate.open_exposure > 0
    fills = store.conn.execute("SELECT COUNT(*) c FROM fills").fetchone()["c"]
    assert fills == 2
    sizes = [r["size"] for r in store.unsettled_filled_orders()]
    assert sizes[0] == pytest.approx(sizes[1])    # legs always equal


def test_one_sided_fill_then_taker_hedge(tmp_path):
    q, store, gate, _ = mk_quoter(tmp_path)
    m = mk_market()
    q.tick(W + 60, m, book("y", 0.44, 0.55), book("n", 0.44, 0.55))
    # only YES gets crossed; NO ask cheap enough for a break-even-or-better
    # taker hedge: 0.45 + 0.54 + fee(0.0092) = 0.9992 <= 1.01
    q.tick(W + 70, m, book("y", 0.40, 0.45), book("n", 0.44, 0.54))
    assert q.state is QState.PARTIAL
    # after hedge_wait_s the quoter takes the other side
    q.tick(W + 70 + q.s.hedge_wait_s + 1, m,
           book("y", 0.40, 0.45), book("n", 0.44, 0.54))
    assert q.state is QState.DONE
    assert gate.open_risk == pytest.approx(0.0)
    rows = store.unsettled_filled_orders()
    assert len(rows) == 2
    assert {r["outcome"] for r in rows} == {"YES", "NO"}


def test_one_sided_fill_hedge_impossible_holds_naked(tmp_path):
    q, store, gate, notifier = mk_quoter(tmp_path)
    m = mk_market()
    q.tick(W + 60, m, book("y", 0.44, 0.55), book("n", 0.44, 0.55))
    # YES fills; NO ask far too expensive to hedge (0.45+0.70 > 1.01)
    q.tick(W + 70, m, book("y", 0.40, 0.45), book("n", 0.60, 0.70))
    assert q.state is QState.PARTIAL
    # cancel deadline arrives with the hedge still impossible
    q.tick(W + 300 - q.s.quote_cancel_remaining_s + 1, m,
           book("y", 0.40, 0.45), book("n", 0.60, 0.70))
    assert q.state is QState.HOLD
    rows = {r["outcome"]: r for r in store.open_orders()}
    assert rows["YES"]["status"] == "unhedged"
    # naked premium (+ maker fee 0) stays reserved against the daily budget
    fill_notional = q.legs[Outcome.YES].fill.price * q.legs[Outcome.YES].fill.size
    assert gate.open_risk == pytest.approx(fill_notional)
    assert any("NAKED" in s for s in notifier.sent)


def test_partial_repricing_improves_hedge_bid(tmp_path):
    q, store, _, _ = mk_quoter(tmp_path)
    m = mk_market()
    q.tick(W + 60, m, book("y", 0.44, 0.55), book("n", 0.44, 0.55))
    old_id = q.legs[Outcome.NO].client_id
    old_price = q.legs[Outcome.NO].price
    # YES fills; NO bid has risen -> reprice remaining leg up (still < hedge_wait)
    q.tick(W + 70, m, book("y", 0.40, 0.45), book("n", 0.50, 0.60))
    assert q.state is QState.PARTIAL
    new_leg = q.legs[Outcome.NO]
    assert new_leg.client_id != old_id
    assert new_leg.price > old_price
    assert new_leg.price <= 1.0 - q.legs[Outcome.YES].fill.price - m.tick_size + 1e-9
    old_row = store.conn.execute(
        "SELECT status FROM orders WHERE client_id=?", (old_id,)).fetchone()
    assert old_row["status"] == "cancelled"


# ---------------------------------------------------------------- teardown
def test_unfilled_quotes_cancelled_at_deadline(tmp_path):
    q, store, gate, _ = mk_quoter(tmp_path)
    m = mk_market()
    q.tick(W + 60, m, book("y", 0.44, 0.55), book("n", 0.44, 0.55))
    assert gate.open_risk > 0
    q.tick(W + 300 - q.s.quote_cancel_remaining_s + 1, m,
           book("y", 0.44, 0.55), book("n", 0.44, 0.55))
    assert q.state is QState.DONE
    assert gate.open_risk == pytest.approx(0.0)
    assert all(r["status"] == "cancelled" for r in store.conn.execute(
        "SELECT status FROM orders").fetchall())


def test_go_flat_on_halt_cancels_quotes(tmp_path):
    q, store, gate, _ = mk_quoter(tmp_path)
    m = mk_market()
    q.tick(W + 60, m, book("y", 0.44, 0.55), book("n", 0.44, 0.55))
    q.go_flat("test halt")
    assert q.state is QState.DONE
    assert gate.open_risk == pytest.approx(0.0)
    assert not store.open_orders()


def test_window_roll_with_partial_marks_unhedged(tmp_path):
    q, store, gate, notifier = mk_quoter(tmp_path)
    m1 = mk_market(W)
    q.tick(W + 60, m1, book("y", 0.44, 0.55), book("n", 0.44, 0.55))
    q.tick(W + 70, m1, book("y", 0.40, 0.45), book("n", 0.60, 0.70))
    assert q.state is QState.PARTIAL
    m2 = mk_market(W + 300)
    q.tick(W + 310, m2, book("y", 0.44, 0.55), book("n", 0.44, 0.55))
    unhedged = store.conn.execute(
        "SELECT COUNT(*) c FROM orders WHERE status='unhedged'").fetchone()["c"]
    assert unhedged == 1
    assert any("NAKED" in s for s in notifier.sent)


# --------------------------------------------------------- engine integration
def test_engine_wires_maker_and_settles_pair(tmp_path):
    s = mk_settings(tmp_path)
    eng = Engine(s, store=Store(s.db_path), notifier=SpyNotifier())
    assert eng.maker is not None
    m = mk_market(W)
    from unittest.mock import patch
    with patch.object(eng.clock, "now", return_value=W + 60):
        eng.tick(m, book("ytok", 0.44, 0.60), book("ntok", 0.44, 0.60),
                 spot=100.0)                       # asks 1.20: no taker arb
    with patch.object(eng.clock, "now", return_value=W + 70):
        eng.tick(m, book("ytok", 0.40, 0.45), book("ntok", 0.40, 0.45),
                 spot=100.2)
    maker_fills = eng.store.conn.execute(
        "SELECT COUNT(*) c FROM orders o JOIN fills f ON f.client_id=o.client_id "
        "WHERE o.strategy=?", (MAKER_STRATEGY,)).fetchone()["c"]
    assert maker_fills == 2
    # settle: window closes UP
    eng._track_window(W + 301, 101.0)
    n = eng.settle_expired(now=W + 360)
    assert n >= 2
    assert eng.gate.daily_pnl > 0                  # locked pair profit realized
    assert eng.gate.consecutive_losses == 0        # completed pair: no streak


def test_restart_with_resting_quotes_cancels_them(tmp_path):
    s = mk_settings(tmp_path)
    eng = Engine(s, store=Store(s.db_path), notifier=SpyNotifier())
    m = mk_market(W)
    from unittest.mock import patch
    with patch.object(eng.clock, "now", return_value=W + 60):
        eng.tick(m, book("ytok", 0.44, 0.60), book("ntok", 0.44, 0.60))
    assert any(r["status"] == "open" for r in eng.store.open_orders())

    eng2 = Engine(s, store=Store(s.db_path), notifier=SpyNotifier())
    statuses = [r["status"] for r in eng2.store.conn.execute(
        "SELECT status FROM orders").fetchall()]
    assert statuses and all(st == "cancelled" for st in statuses)
    assert eng2.gate.open_risk == pytest.approx(0.0)   # nothing filled = no risk


def test_naked_maker_leg_counts_in_streak_after_settlement(tmp_path):
    q, store, gate, _ = mk_quoter(tmp_path)
    m = mk_market()
    q.tick(W + 60, m, book("y", 0.44, 0.55), book("n", 0.44, 0.55))
    q.tick(W + 70, m, book("y", 0.40, 0.45), book("n", 0.60, 0.70))
    q.tick(W + 300 - q.s.quote_cancel_remaining_s + 1, m,
           book("y", 0.40, 0.45), book("n", 0.60, 0.70))
    assert q.state is QState.HOLD
    # settle the naked YES leg as a loss
    leg = q.legs[Outcome.YES]
    store.record_settlement(leg.client_id, outcome_won=False,
                            pnl=-(leg.fill.price * leg.fill.size))
    assert store.consecutive_losses() == 1         # unhedged leg is a real loss
