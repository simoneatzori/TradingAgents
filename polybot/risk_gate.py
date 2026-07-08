"""Risk gate: every order passes through here. Deny by default.

Two layers of protection:

1. WORST-CASE DAILY BUDGET (preventive, the 1%-per-day rule):
   Before an order is allowed, realized daily loss + worst-case loss of all
   open positions + worst-case loss of the new order must fit inside the
   daily loss budget. The budget is min(MAX_DAILY_LOSS,
   bankroll * MAX_DAILY_LOSS_PCT). With the default 1% the engine can never
   put more than 1% of the bankroll at risk in a single UTC day, even if
   every open position settles at zero.

2. CIRCUIT BREAKERS (reactive, any one trips -> halt until human reset):
   - kill switch file exists
   - daily realized loss beyond the budget
   - N consecutive losing trades (paired arb legs excluded by the engine)
   - drawdown from the equity high-water mark beyond MAX_DRAWDOWN_PCT
   - open exposure beyond MAX_OPEN_EXPOSURE

Optional DAILY PROFIT LOCK: when daily_profit_lock_pct > 0 and the day's
realized PnL reaches bankroll * pct, new orders are denied (not halted)
until the next UTC day. Off by default.

This module is intentionally synchronous, dependency-free, and unit-tested:
it is the last line of defense and must be boring. Persistence of its state
across restarts is the engine's job (see Engine._restore_state).
"""
from __future__ import annotations

import datetime as dt
import math
import os
from dataclasses import dataclass, field


@dataclass
class RiskDecision:
    allowed: bool
    reason: str = "ok"


@dataclass
class RiskGate:
    max_daily_loss: float
    max_consecutive_losses: int
    max_open_exposure: float
    kill_switch_file: str = "KILL_SWITCH"
    bankroll: float = 0.0                  # 0 disables percentage-based limits
    max_daily_loss_pct: float = 0.0        # 0 disables (e.g. 0.01 = 1%/day)
    max_drawdown_pct: float = 0.0          # 0 disables (e.g. 0.05 = 5% from HWM)
    daily_profit_lock_pct: float = 0.0     # 0 disables

    halted: bool = False
    halt_reason: str = ""
    _day: dt.date = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).date())
    daily_pnl: float = 0.0
    consecutive_losses: int = 0
    open_exposure: float = 0.0             # capital deployed in open positions
    open_risk: float = 0.0                 # worst-case loss of open positions
    cumulative_pnl: float = 0.0            # lifetime realized PnL
    high_water_mark: float = 0.0           # peak equity (bankroll + cumulative_pnl)

    def __post_init__(self) -> None:
        if self.high_water_mark <= 0:
            self.high_water_mark = max(self.bankroll, 0.0)

    # ---- budgets ----------------------------------------------------------
    @property
    def daily_loss_budget(self) -> float:
        """The tighter of the absolute and percentage daily loss limits."""
        candidates = []
        if self.max_daily_loss > 0:
            candidates.append(abs(self.max_daily_loss))
        if self.bankroll > 0 and self.max_daily_loss_pct > 0:
            candidates.append(self.bankroll * self.max_daily_loss_pct)
        return min(candidates) if candidates else math.inf

    @property
    def remaining_risk_budget(self) -> float:
        """Worst-case loss the gate would still accept today."""
        budget = self.daily_loss_budget
        if math.isinf(budget):
            return budget
        realized_loss = max(0.0, -self.daily_pnl)
        return max(0.0, budget - realized_loss - self.open_risk)

    @property
    def equity(self) -> float:
        return self.bankroll + self.cumulative_pnl

    # ---- lifecycle -------------------------------------------------------
    def _roll_day(self) -> None:
        today = dt.datetime.now(dt.timezone.utc).date()
        if today != self._day:
            self._day = today
            self.daily_pnl = 0.0
            # consecutive_losses and halt state intentionally NOT reset
            # by the calendar

    def halt(self, reason: str) -> None:
        self.halted = True
        self.halt_reason = reason

    def reset(self) -> None:
        """Manual human action only (CLI / supervisor approval)."""
        self.halted = False
        self.halt_reason = ""
        self.consecutive_losses = 0

    # ---- checks ----------------------------------------------------------
    def check_order(self, notional: float,
                    worst_case_loss: float | None = None) -> RiskDecision:
        """worst_case_loss defaults to the full notional (buyer of a binary
        contract can lose at most the premium paid). Paired-arb entries pass
        a smaller number because a completed pair has locked-in profit."""
        self._roll_day()
        wc = notional if worst_case_loss is None else worst_case_loss

        if os.path.exists(self.kill_switch_file):
            self.halt("kill switch file present")
            return RiskDecision(False, "kill_switch")

        if self.halted:
            return RiskDecision(False, f"halted: {self.halt_reason}")

        if self.daily_pnl <= -self.daily_loss_budget:
            self.halt(f"daily loss limit hit ({self.daily_pnl:.2f})")
            return RiskDecision(False, "daily_loss_limit")

        if self.consecutive_losses >= self.max_consecutive_losses:
            self.halt(f"{self.consecutive_losses} consecutive losses")
            return RiskDecision(False, "consecutive_losses")

        if self._drawdown_breached():
            self.halt(f"drawdown limit hit (equity {self.equity:.2f}, "
                      f"HWM {self.high_water_mark:.2f})")
            return RiskDecision(False, "max_drawdown")

        if notional <= 0:
            return RiskDecision(False, "non_positive_notional")

        if self.open_exposure + notional > self.max_open_exposure:
            return RiskDecision(False, "max_open_exposure")

        realized_loss = max(0.0, -self.daily_pnl)
        if realized_loss + self.open_risk + wc > self.daily_loss_budget:
            return RiskDecision(False, "daily_risk_budget")

        if (self.daily_profit_lock_pct > 0 and self.bankroll > 0
                and self.daily_pnl >= self.bankroll * self.daily_profit_lock_pct):
            return RiskDecision(False, "daily_profit_lock")

        return RiskDecision(True)

    def _drawdown_breached(self) -> bool:
        if self.max_drawdown_pct <= 0 or self.high_water_mark <= 0:
            return False
        dd = self.high_water_mark - self.equity
        return dd >= self.high_water_mark * self.max_drawdown_pct

    # ---- accounting ------------------------------------------------------
    def on_order_filled(self, notional: float,
                        worst_case_loss: float | None = None) -> None:
        self.open_exposure += notional
        self.open_risk += notional if worst_case_loss is None else worst_case_loss

    def reserve_risk(self, amount: float) -> None:
        """Reserve worst-case budget BEFORE exposure exists — e.g. resting
        maker quotes that may fill later. Pair with release_risk()."""
        self.open_risk += max(0.0, amount)

    def release_risk(self, amount: float) -> None:
        """Called when a position's worst case improves (e.g. the second leg
        of a pair fills, locking in profit regardless of settlement)."""
        self.open_risk = max(0.0, self.open_risk - amount)

    def on_position_settled(self, notional: float, pnl: float,
                            risk_release: float | None = None,
                            counts_streak: bool = True) -> None:
        self._roll_day()
        self.open_exposure = max(0.0, self.open_exposure - notional)
        release = notional if risk_release is None else risk_release
        self.open_risk = max(0.0, self.open_risk - release)
        self.daily_pnl += pnl
        self.cumulative_pnl += pnl
        self.high_water_mark = max(self.high_water_mark, self.equity)
        if counts_streak:
            if pnl < 0:
                self.consecutive_losses += 1
            elif pnl > 0:
                self.consecutive_losses = 0
            # pnl == 0 leaves the streak unchanged

        # re-evaluate breakers immediately after settlement
        if self.daily_pnl <= -self.daily_loss_budget:
            self.halt(f"daily loss limit hit ({self.daily_pnl:.2f})")
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.halt(f"{self.consecutive_losses} consecutive losses")
        if self._drawdown_breached():
            self.halt(f"drawdown limit hit (equity {self.equity:.2f}, "
                      f"HWM {self.high_water_mark:.2f})")
