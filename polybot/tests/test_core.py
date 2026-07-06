"""Test suite. Each test maps to a failure mode from the production review:
window boundary off-by-one, fee gate, Kelly caps, breaker trips, idempotency,
pessimistic fills, calibration math, live-mode safety gate.
"""
from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calibration import CalibrationTracker
from config import LIVE_ACK_PHRASE, Settings, load_settings
from executor import DryRunExecutor
from fees import (directional_net_edge, pair_cost_net_edge, round_to_tick,
                  taker_fee_per_share)
from models import BookTop, MarketWindow, Order, Outcome, Side, deterministic_client_id
from risk_gate import RiskGate
from sizing import kelly_fraction, position_notional, shrink_probability
from strategy import BrownianDirectional, EwmaVol, PairCostArb
from timeutil import in_entry_zone, seconds_to_close, window_bounds, window_start


# --------------------------------------------------------------- time/window
def test_window_alignment_boundary():
    # exactly on a boundary belongs to the window that STARTS there
    assert window_start(600) == 600
    assert window_bounds(600) == (600, 900)
    assert window_bounds(599.999) == (300, 600)
    assert window_bounds(899) == (600, 900)


def test_seconds_to_close():
    assert seconds_to_close(600) == 300
    assert seconds_to_close(899) == 1
    assert math.isclose(seconds_to_close(600.5), 299.5)


def test_entry_zone_blocks_near_settlement():
    assert in_entry_zone(600, min_remaining_s=20)            # 300s left
    assert not in_entry_zone(885, min_remaining_s=20)        # 15s left
    assert not in_entry_zone(600, min_remaining_s=20, max_remaining_s=200)


# ----------------------------------------------------------------------- fees
def test_fee_formula_uses_min_side():
    # fee = rate * min(p, 1-p): symmetric around 0.5
    assert taker_fee_per_share(0.30, 200) == pytest.approx(0.02 * 0.30)
    assert taker_fee_per_share(0.70, 200) == pytest.approx(0.02 * 0.30)


def test_pair_cost_edge_net_of_fees():
    # gross 3 cents, fees eat 2*0.02*~0.49 ≈ 1.96 cents -> ~1 cent net
    net = pair_cost_net_edge(0.48, 0.49, 200)
    assert 0.005 < net < 0.015


def test_fee_gate_kills_thin_arb():
    """The article's $0.02 'risk-free' edge dies to a 200bps fee."""
    net = pair_cost_net_edge(0.49, 0.49, 200)   # gross 0.02
    assert net < 0.01                            # below a 1-cent MIN_EDGE


def test_round_to_tick_rounds_down():
    assert round_to_tick(0.4799, 0.01) == pytest.approx(0.47)
    assert round_to_tick(0.48, 0.01) == pytest.approx(0.48)


def test_directional_net_edge():
    e = directional_net_edge(0.60, 0.50, 200)
    assert e == pytest.approx(0.10 - 0.01)


# --------------------------------------------------------------------- sizing
def test_kelly_basics():
    assert kelly_fraction(0.5, 0.5) == 0.0           # no edge, no bet
    assert kelly_fraction(0.6, 0.5) == pytest.approx(0.2)
    assert kelly_fraction(0.4, 0.5) == 0.0           # negative edge clamped
    assert kelly_fraction(0.9, 1.0) == 0.0           # degenerate price


def test_position_caps():
    # quarter-Kelly of f*=0.2 on 100 = 5, but max_bet caps at 2
    n = position_notional(100, 0.6, 0.5, kelly_multiplier=0.25,
                          min_bet=1, max_bet=2, max_bankroll_fraction=0.05)
    assert n == 2.0
    # below min_bet -> 0 (no dust orders)
    n = position_notional(10, 0.52, 0.5, kelly_multiplier=0.25,
                          min_bet=1, max_bet=2, max_bankroll_fraction=0.05)
    assert n == 0.0


def test_shrinkage_pulls_to_prior():
    assert shrink_probability(0.9, 0.5) == pytest.approx(0.7)
    assert shrink_probability(0.9, 0.0) == 0.5
    with pytest.raises(ValueError):
        shrink_probability(0.9, 1.5)


# ------------------------------------------------------------------ risk gate
def make_gate(**kw):
    defaults = dict(max_daily_loss=5, max_consecutive_losses=3,
                    max_open_exposure=10, kill_switch_file="/tmp/__no_such_file__")
    defaults.update(kw)
    return RiskGate(**defaults)


def test_gate_allows_normal_order():
    assert make_gate().check_order(2.0).allowed


def test_gate_blocks_exposure():
    g = make_gate()
    g.on_order_filled(9.5)
    d = g.check_order(1.0)
    assert not d.allowed and d.reason == "max_open_exposure"


def test_gate_daily_loss_halts():
    g = make_gate()
    g.on_order_filled(2.0)
    g.on_position_settled(2.0, pnl=-6.0)
    assert g.halted
    assert not g.check_order(1.0).allowed


def test_gate_consecutive_losses_halt_and_reset_on_win():
    g = make_gate()
    for _ in range(2):
        g.on_order_filled(1.0)
        g.on_position_settled(1.0, pnl=-0.5)
    assert not g.halted
    g.on_order_filled(1.0)
    g.on_position_settled(1.0, pnl=+0.5)     # win resets the streak
    assert g.consecutive_losses == 0
    for _ in range(3):
        g.on_order_filled(1.0)
        g.on_position_settled(1.0, pnl=-0.1)
    assert g.halted


def test_kill_switch_file(tmp_path):
    ks = tmp_path / "KILL"
    g = make_gate(kill_switch_file=str(ks))
    assert g.check_order(1.0).allowed
    ks.write_text("stop")
    d = g.check_order(1.0)
    assert not d.allowed and d.reason == "kill_switch"
    assert g.halted


# ------------------------------------------------------------------ strategy
def book(token, bid, ask, bid_sz=100, ask_sz=100):
    return BookTop(token, bid, ask, bid_sz, ask_sz)


def test_pair_arb_fires_only_above_net_edge():
    arb = PairCostArb(min_edge=0.01, fee_rate_bps=200)
    # fat edge: 0.45 + 0.45 = 0.90 -> fires both legs
    sigs = arb.evaluate(book("y", 0.44, 0.45), book("n", 0.44, 0.45))
    assert len(sigs) == 2 and all(s.paired for s in sigs)
    # thin edge: fee gate rejects
    assert arb.evaluate(book("y", 0.48, 0.49), book("n", 0.48, 0.49)) == []
    # empty displayed size rejects
    assert arb.evaluate(book("y", 0.44, 0.45, ask_sz=0), book("n", 0.44, 0.45)) == []


def test_brownian_probability_sane():
    d = BrownianDirectional(min_edge=0.01, min_prob=0.55, fee_rate_bps=200)
    assert d.probability_up(100, 100, 0.001, 60) == pytest.approx(0.5)
    assert d.probability_up(101, 100, 0.001, 60) > 0.5
    assert d.probability_up(99, 100, 0.001, 60) < 0.5
    # degenerate inputs -> neutral, never crash
    assert d.probability_up(0, 100, 0.001, 60) == 0.5
    assert d.probability_up(100, 100, 0.0, 60) == 0.5


def test_directional_takes_one_side_max():
    d = BrownianDirectional(min_edge=0.01, min_prob=0.55, fee_rate_bps=200)
    sigs = d.evaluate(book("y", 0.49, 0.50), book("n", 0.49, 0.50), p_up=0.70)
    assert len(sigs) == 1 and sigs[0].outcome is Outcome.YES


def test_ewma_vol_updates():
    v = EwmaVol(halflife_s=60)
    v.update(100.0, 0.0)
    assert v.sigma_per_sqrt_s == 0.0
    v.update(100.5, 1.0)
    assert v.sigma_per_sqrt_s > 0


# ------------------------------------------------------------------ executor
def mk_market():
    return MarketWindow("slug", "cid", "ytok", "ntok", 600, 900,
                        fee_rate_bps=200, tick_size=0.01, min_order_size=1)


def test_dryrun_idempotency():
    ex = DryRunExecutor(Settings())
    m = mk_market()
    o1 = Order(market=m, outcome=Outcome.YES, side=Side.BUY,
               price=0.45, size=2, strategy="t")
    o2 = Order(market=m, outcome=Outcome.YES, side=Side.BUY,
               price=0.45, size=2, strategy="t")
    assert o1.client_id == o2.client_id          # deterministic id
    b = book("ytok", 0.44, 0.45)
    assert ex.submit(o1, b).accepted
    r2 = ex.submit(o2, b)
    assert not r2.accepted and r2.reason == "duplicate_client_id"


def test_dryrun_pessimistic_fill():
    ex = DryRunExecutor(Settings())
    m = mk_market()
    o = Order(market=m, outcome=Outcome.YES, side=Side.BUY,
              price=0.45, size=50, strategy="t")
    # only 10 displayed -> partial fill at displayed size
    r = ex.submit(o, book("ytok", 0.44, 0.45, ask_sz=10))
    assert r.accepted and r.fill.size == 10
    # ask moved above our limit -> reject, no chase
    o2 = Order(market=m, outcome=Outcome.NO, side=Side.BUY,
               price=0.45, size=5, strategy="t")
    r2 = ex.submit(o2, book("ntok", 0.45, 0.47))
    assert not r2.accepted and r2.reason == "ask_moved_above_limit"


def test_client_id_changes_with_window():
    a = deterministic_client_id("slug", "YES", "BUY", 600, "s")
    b = deterministic_client_id("slug", "YES", "BUY", 900, "s")
    assert a != b


# --------------------------------------------------------------- calibration
def test_calibration_metrics():
    c = CalibrationTracker()
    assert c.brier() is None
    for _ in range(10):
        c.record(0.8, 1)
    assert c.brier() == pytest.approx(0.04)
    assert c.hit_rate() == 1.0
    assert c.log_loss() == pytest.approx(-math.log(0.8))


# --------------------------------------------------------- config / live gate
def test_dry_run_is_default():
    assert load_settings({}).dry_run is True


def test_live_requires_ack_and_key():
    s = load_settings({"DRY_RUN": "false"})
    with pytest.raises(RuntimeError):
        s.assert_live_allowed()
    s = load_settings({"DRY_RUN": "false", "LIVE_TRADING_ACK": LIVE_ACK_PHRASE})
    with pytest.raises(RuntimeError):                 # still no key
        s.assert_live_allowed()
    s = load_settings({"DRY_RUN": "false", "LIVE_TRADING_ACK": LIVE_ACK_PHRASE,
                       "PRIVATE_KEY": "0x" + "a" * 64})
    s.assert_live_allowed()                           # passes


def test_full_kelly_forbidden():
    with pytest.raises(Exception):
        Settings(kelly_multiplier=1.0)


def test_private_key_redacted_in_repr():
    s = Settings(private_key="0x" + "a" * 64)
    assert "aaaa" not in repr(s)
