"""Maker-side pair quoting: rest passive bids on BOTH outcomes priced so
their sum locks a profit if both fill (sum <= 1 - MIN_EDGE - maker fees).

Why this beats taking: taker pair-arb waits for someone else to misprice
the asks, then races every faster bot to lift them. The maker variant
*provides* the prices instead — resting bids on YES and NO — and earns
from two-sided flow when the window whipsaws. Maker fees on Polymarket
are lower than taker (default 0 bps here; set MAKER_FEE_BPS if the
market metadata says otherwise).

The one new risk this introduces is the ONE-SIDED FILL: a trending window
hits only the side that ends up losing. Containment, in order:

  1. The worst case of a naked leg (its full premium + fee) is reserved
     against the daily risk budget at QUOTE time, before anything can fill.
  2. On a partial fill the remaining leg is repriced up to the passive
     hedge cap (still locking >= one tick of profit).
  3. After HEDGE_WAIT_S — or when the cancel deadline approaches — it
     completes the pair by TAKING the other side, if that locks at worst
     -HEDGE_MAX_LOSS_PER_SHARE per share.
  4. Failing all that, quotes are cancelled and the naked leg is held to
     settlement as 'unhedged' (its risk already sits inside the daily
     budget) with an alert. No martingale, no chasing.

Everything unfilled is cancelled QUOTE_CANCEL_REMAINING_S before the
window closes, and the engine calls go_flat() on any halt so no quote
ever rests unattended.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from config import Settings
from fees import taker_fee_per_share
from models import BookTop, Fill, MarketWindow, Order, Outcome, Side, \
    deterministic_client_id
from notifier import TelegramNotifier
from risk_gate import RiskGate
from sizing import shares_for_notional
from store import Store

log = logging.getLogger("maker")

MAKER_STRATEGY = "pair_cost_maker"


class QState(str, Enum):
    IDLE = "idle"          # nothing resting this window
    QUOTED = "quoted"      # both legs resting
    PARTIAL = "partial"    # one leg filled, other still working
    HOLD = "hold"          # naked leg held to settlement (hedge failed)
    DONE = "done"          # pair complete or window abandoned


@dataclass
class Leg:
    client_id: str
    outcome: Outcome
    price: float
    size: float
    resting: bool = True
    fill: Fill | None = None


class MakerPairQuoter:
    def __init__(self, settings: Settings, store: Store, gate: RiskGate,
                 executor, notifier: TelegramNotifier) -> None:
        self.s = settings
        self.store = store
        self.gate = gate
        self.executor = executor
        self.notifier = notifier

        self.state = QState.IDLE
        self.window_start: int | None = None
        self.seq = 0
        self.legs: dict[Outcome, Leg] = {}
        self.reserved = 0.0            # risk currently reserved at the gate
        self.placed_at: float | None = None
        self.partial_at: float | None = None

    # ------------------------------------------------------------------ tick
    def tick(self, now: float, market: MarketWindow,
             yes_book: BookTop | None, no_book: BookTop | None) -> None:
        if market.window_start != self.window_start:
            self._roll_window(market)

        remaining = market.window_end - now

        if self.state in (QState.QUOTED, QState.PARTIAL):
            self._poll_fills(market, yes_book, no_book, now)

        if self.state is QState.QUOTED:
            if remaining < self.s.quote_cancel_remaining_s:
                self._cancel_unfilled("cancel deadline")
                self._release_all()
                self.state = QState.DONE
            elif self._should_reprice(now, market, yes_book, no_book):
                self._cancel_unfilled("reprice")
                self._release_all()
                self.state = QState.IDLE

        if self.state is QState.PARTIAL:
            self._manage_partial(now, market, yes_book, no_book, remaining)

        if (self.state is QState.IDLE and yes_book and no_book
                and remaining >= self.s.quote_min_remaining_s):
            self._try_place(now, market, yes_book, no_book)

    # ---------------------------------------------------------- window roll
    def _roll_window(self, market: MarketWindow) -> None:
        if self.state is QState.QUOTED:
            self._cancel_unfilled("window rolled")
            self._release_all()
        elif self.state is QState.PARTIAL:
            self._cancel_unfilled("window rolled")
            self._hold_naked_leg("window rolled with an unhedged leg")
        self.window_start = market.window_start
        self.seq = 0
        self.legs = {}
        self.placed_at = None
        self.partial_at = None
        self.state = QState.IDLE

    def go_flat(self, reason: str) -> None:
        """Called by the engine on halt: nothing may rest unattended."""
        if self.state is QState.QUOTED:
            self._cancel_unfilled(reason)
            self._release_all()
            self.state = QState.DONE
        elif self.state is QState.PARTIAL:
            self._cancel_unfilled(reason)
            self._hold_naked_leg(f"go_flat: {reason}")

    # ------------------------------------------------------------ placement
    def _quote_prices(self, market: MarketWindow, yes_book: BookTop,
                      no_book: BookTop) -> tuple[float, float] | None:
        tick = market.tick_size
        target_sum = 1.0 - self.s.min_edge  # maker fees subtracted per-leg below
        py = min(yes_book.bid + tick, yes_book.ask - tick)
        pn = min(no_book.bid + tick, no_book.ask - tick)
        fee = (taker_fee_per_share(py, self.s.maker_fee_bps)
               + taker_fee_per_share(pn, self.s.maker_fee_bps))
        excess = (py + pn) - (target_sum - fee)
        if excess > 0:
            py -= excess / 2.0
            pn -= excess / 2.0
        # round DOWN to tick; floor keeps the sum under target
        py = int(py / tick + 1e-9) * tick
        pn = int(pn / tick + 1e-9) * tick
        if py < tick or pn < tick:
            return None
        if py >= yes_book.ask or pn >= no_book.ask:
            return None                # would cross: taker strategy's job
        return round(py, 6), round(pn, 6)

    def _try_place(self, now: float, market: MarketWindow,
                   yes_book: BookTop, no_book: BookTop) -> None:
        prices = self._quote_prices(market, yes_book, no_book)
        if prices is None:
            return
        py, pn = prices
        pair_sum = py + pn
        max_leg = max(py, pn)
        wc_per_share = max_leg + taker_fee_per_share(max_leg, self.s.maker_fee_bps)
        budget_shares = (self.gate.remaining_risk_budget / wc_per_share
                         if wc_per_share > 0 else 0.0)
        shares = min(self.s.max_bet / pair_sum, budget_shares)
        shares = shares_for_notional(shares * pair_sum, pair_sum,
                                     market.min_order_size)
        if shares <= 0:
            return
        notional = round(shares * pair_sum, 6)
        if notional < self.s.min_bet:
            return
        worst_case = shares * wc_per_share

        decision = self.gate.check_order(notional, worst_case)
        if not decision.allowed:
            log.debug("maker quote denied: %s", decision.reason)
            return

        leg_y = self._place_leg(market, Outcome.YES, py, shares, yes_book)
        if leg_y is None:
            return
        leg_n = self._place_leg(market, Outcome.NO, pn, shares, no_book)
        if leg_n is None:
            self._cancel_leg(leg_y, "second leg failed to post")
            return

        self.legs = {Outcome.YES: leg_y, Outcome.NO: leg_n}
        self.gate.reserve_risk(worst_case)
        self.reserved = worst_case
        self.placed_at = now
        self.state = QState.QUOTED
        log.info("maker quoted %s: YES %.2f / NO %.2f x %.2f (sum %.2f)",
                 market.slug, py, pn, shares, pair_sum)

    def _place_leg(self, market: MarketWindow, outcome: Outcome, price: float,
                   size: float, book: BookTop) -> Leg | None:
        self.seq += 1
        client_id = deterministic_client_id(
            market.slug, outcome.value, "BUY", market.window_start,
            MAKER_STRATEGY, self.seq)
        order = Order(market=market, outcome=outcome, side=Side.BUY,
                      price=price, size=size, strategy=MAKER_STRATEGY,
                      client_id=client_id)
        is_new = self.store.record_order(
            client_id=client_id, market_slug=market.slug,
            window_start=market.window_start, outcome=outcome.value,
            side="BUY", price=price, size=size, strategy=MAKER_STRATEGY,
            dry_run=self.s.dry_run, prob=None)
        if not is_new:
            # id already used (restart inside this window): stay out rather
            # than risk double-quoting — idempotency beats fill rate
            log.info("maker quote suppressed, id seen (%s)", client_id[:8])
            return None
        res = self.executor.place_resting(order, book)
        if not res.accepted:
            self.store.set_order_status(client_id, "rejected")
            log.debug("maker leg rejected: %s", res.reason)
            return None
        self.store.set_order_status(client_id, "open")
        return Leg(client_id, outcome, price, size)

    # ----------------------------------------------------------------- fills
    def _poll_fills(self, market: MarketWindow, yes_book: BookTop | None,
                    no_book: BookTop | None, now: float) -> None:
        books = {Outcome.YES: yes_book, Outcome.NO: no_book}
        for outcome, leg in self.legs.items():
            book = books[outcome]
            if not leg.resting or leg.fill is not None or book is None:
                continue
            fill = self.executor.poll_resting(leg.client_id, book)
            if fill is None:
                continue
            leg.fill = fill
            leg.resting = False
            self.store.record_fill(leg.client_id, fill.price, fill.size,
                                   fill.fee_paid)
            # risk was reserved at quote time; only exposure is new
            self.gate.on_order_filled(fill.price * fill.size, 0.0)
            log.info("maker leg filled: %s %.2f x %.2f", outcome.value,
                     fill.price, fill.size)

        fills = [l for l in self.legs.values() if l.fill is not None]
        if len(fills) == 2:
            self._release_all()
            self.state = QState.DONE
            cost = sum(l.fill.price * l.fill.size + l.fill.fee_paid for l in fills)
            log.info("maker pair complete on %s: locked %.4f",
                     market.slug, fills[0].fill.size - cost)
        elif len(fills) == 1 and self.state is not QState.PARTIAL:
            self.state = QState.PARTIAL
            self.partial_at = now

    # --------------------------------------------------------------- partial
    def _manage_partial(self, now: float, market: MarketWindow,
                        yes_book: BookTop | None, no_book: BookTop | None,
                        remaining: float) -> None:
        filled = next(l for l in self.legs.values() if l.fill is not None)
        open_leg = next((l for l in self.legs.values() if l.fill is None), None)
        books = {Outcome.YES: yes_book, Outcome.NO: no_book}
        book = books[open_leg.outcome] if open_leg is not None else None

        hedge_due = (now - (self.partial_at or now) >= self.s.hedge_wait_s
                     or remaining < self.s.quote_cancel_remaining_s + 10)

        if hedge_due and book is not None:
            if self._try_taker_hedge(market, filled, open_leg, book):
                return
        if remaining < self.s.quote_cancel_remaining_s:
            self._cancel_unfilled("cancel deadline with naked leg")
            self._hold_naked_leg("hedge failed before cancel deadline")
            return
        if not hedge_due and book is not None and open_leg is not None:
            self._reprice_open_leg(market, filled, open_leg, book)

    def _reprice_open_leg(self, market: MarketWindow, filled: Leg,
                          open_leg: Leg, book: BookTop) -> None:
        """Improve the remaining bid up to the passive-hedge cap: complete
        the pair while still locking at least one tick per share."""
        tick = market.tick_size
        cap = 1.0 - filled.fill.price - tick
        target = min(book.bid + tick, cap, book.ask - tick)
        target = int(target / tick + 1e-9) * tick
        if target < tick or target <= open_leg.price + tick / 2:
            return
        self._cancel_leg(open_leg, "reprice naked hedge")
        new_leg = self._place_leg(market, open_leg.outcome, round(target, 6),
                                  filled.fill.size, book)
        if new_leg is not None:
            self.legs[open_leg.outcome] = new_leg
            log.info("maker hedge repriced %s -> %.2f", open_leg.outcome.value,
                     target)

    def _try_taker_hedge(self, market: MarketWindow, filled: Leg,
                         open_leg: Leg | None, book: BookTop) -> bool:
        """Complete the pair by taking the other side's ask, if the total
        cost locks at worst -hedge_max_loss_per_share per share."""
        outcome = open_leg.outcome if open_leg else (
            Outcome.NO if filled.outcome is Outcome.YES else Outcome.YES)
        ask = book.ask
        total = (filled.fill.price + ask
                 + taker_fee_per_share(ask, market.fee_rate_bps)
                 + taker_fee_per_share(filled.fill.price, self.s.maker_fee_bps))
        if not 0.0 < ask < 1.0 or total > 1.0 + self.s.hedge_max_loss_per_share:
            return False
        if book.ask_size < filled.fill.size:
            return False
        if open_leg is not None:
            self._cancel_leg(open_leg, "switching to taker hedge")

        self.seq += 1
        client_id = deterministic_client_id(
            market.slug, outcome.value, "BUY", market.window_start,
            MAKER_STRATEGY, self.seq)
        order = Order(market=market, outcome=outcome, side=Side.BUY,
                      price=ask, size=filled.fill.size, strategy=MAKER_STRATEGY,
                      client_id=client_id)
        is_new = self.store.record_order(
            client_id=client_id, market_slug=market.slug,
            window_start=market.window_start, outcome=outcome.value,
            side="BUY", price=ask, size=filled.fill.size,
            strategy=MAKER_STRATEGY, dry_run=self.s.dry_run, prob=None)
        if not is_new:
            return False
        res = self.executor.submit(order, book)
        if not (res.accepted and res.fill):
            self.store.set_order_status(client_id, "rejected")
            return False
        self.store.record_fill(client_id, res.fill.price, res.fill.size,
                               res.fill.fee_paid)
        self.gate.on_order_filled(res.fill.price * res.fill.size, 0.0)
        self.legs[outcome] = Leg(client_id, outcome, res.fill.price,
                                 res.fill.size, resting=False, fill=res.fill)
        self._release_all()
        self.state = QState.DONE
        log.info("maker pair completed via taker hedge at %.2f (total %.4f)",
                 ask, total)
        return True

    # ------------------------------------------------------------- teardown
    def _cancel_unfilled(self, reason: str) -> None:
        for leg in self.legs.values():
            if leg.resting and leg.fill is None:
                self._cancel_leg(leg, reason)

    def _cancel_leg(self, leg: Leg, reason: str) -> None:
        self.executor.cancel_resting(leg.client_id)
        self.store.set_order_status(leg.client_id, "cancelled")
        leg.resting = False
        log.debug("maker leg cancelled (%s): %s", reason, leg.client_id[:8])

    def _release_all(self) -> None:
        if self.reserved > 0:
            self.gate.release_risk(self.reserved)
            self.reserved = 0.0

    def _hold_naked_leg(self, reason: str) -> None:
        """Hedge failed: hold the filled leg to settlement. Its worst case
        stays reserved (re-shaped to the exact naked premium + fee) and is
        released by settle_expired via the 'unhedged' status."""
        filled = next((l for l in self.legs.values() if l.fill is not None), None)
        if filled is None:
            self.state = QState.DONE
            return
        naked = filled.fill.price * filled.fill.size + filled.fill.fee_paid
        self._release_all()
        self.gate.reserve_risk(naked)
        self.store.set_order_status(filled.client_id, "unhedged")
        self.state = QState.HOLD
        msg = (f"⚠️ MAKER NAKED LEG held to settlement: {filled.outcome.value} "
               f"{filled.fill.size:.2f} @ {filled.fill.price:.2f} ({reason}). "
               f"Worst case {naked:.2f} already inside the daily budget.")
        log.warning(msg)
        self.notifier.send(msg)

    # -------------------------------------------------------------- reprice
    def _should_reprice(self, now: float, market: MarketWindow,
                        yes_book: BookTop | None, no_book: BookTop | None) -> bool:
        if yes_book is None or no_book is None or self.placed_at is None:
            return False
        if now - self.placed_at < self.s.quote_reprice_interval_s:
            return False
        prices = self._quote_prices(market, yes_book, no_book)
        if prices is None:
            return False
        py, pn = prices
        tick = market.tick_size
        cur_y = self.legs[Outcome.YES].price
        cur_n = self.legs[Outcome.NO].price
        return abs(py - cur_y) > tick + 1e-9 or abs(pn - cur_n) > tick + 1e-9
