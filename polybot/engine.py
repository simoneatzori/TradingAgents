"""Engine: deterministic orchestration of one trading tick.

No LLM anywhere in this path. The sequence per tick:

  clock -> window/vol tracking -> settlement of expired windows ->
  entry-zone check -> book sanity -> strategy -> fee gate (inside strategy) ->
  sizing (capped by the remaining daily risk budget) -> RISK GATE ->
  execute -> persist

Capital-protection invariants enforced here:
- Pair-arb legs are sized to EQUAL share counts (a mismatch is directional
  exposure, not arbitrage) and submitted leg-by-leg; a failed second leg
  halts the engine and alerts (unhedged position needs a human).
- Every order carries a worst-case loss to the risk gate, so realized loss
  plus open worst-case can never exceed the daily budget (default 1%).
- Risk-gate state (daily PnL, streak, exposure, halt) is rebuilt from the
  SQLite store at startup; restarting the process cannot bypass a limit.
- Settlements are processed every loop iteration from recorded window
  open/close prices, so breakers trip when they should.
"""
from __future__ import annotations

import datetime as dt
import logging

from calibration import CalibrationTracker
from config import Settings
from executor import ExecutionResult, build_executor
from feeds import BinanceSpotFeed, ClobBookFeed
from fees import taker_fee, taker_fee_per_share
from models import BookTop, MarketWindow, Order, Outcome, Side, Signal
from notifier import TelegramNotifier
from reconciler import Reconciler
from risk_gate import RiskGate
from sizing import position_notional, shares_for_notional, shrink_probability
from store import Store
from strategy import BrownianDirectional, EwmaVol, PairCostArb
from timeutil import Clock, in_entry_zone, seconds_to_close, window_bounds

from maker import MAKER_STRATEGY, MakerPairQuoter

log = logging.getLogger("engine")

HALT_KV_KEY = "halt_reason"
HWM_KV_KEY = "high_water_mark"
PAIR_STRATEGY = "pair_cost_arb"
PAIR_STRATEGIES = frozenset({PAIR_STRATEGY, MAKER_STRATEGY})


class Engine:
    def __init__(self, settings: Settings, *,
                 store: Store | None = None,
                 executor=None,
                 notifier: TelegramNotifier | None = None) -> None:
        self.s = settings
        self.clock = Clock()
        self.store = store or Store(settings.db_path)
        self.gate = RiskGate(
            bankroll=settings.bankroll,
            max_daily_loss=settings.max_daily_loss,
            max_daily_loss_pct=settings.max_daily_loss_pct,
            max_drawdown_pct=settings.max_drawdown_pct,
            daily_profit_lock_pct=settings.daily_profit_lock_pct,
            max_consecutive_losses=settings.max_consecutive_losses,
            max_open_exposure=settings.max_open_exposure,
            kill_switch_file=settings.kill_switch_file,
        )
        self.executor = executor or build_executor(settings)
        self.notifier = notifier or TelegramNotifier(
            settings.telegram_bot_token, settings.telegram_chat_id)
        self.reconciler = Reconciler(self.store, self.gate, settings.dry_run)
        self.calibration = CalibrationTracker()

        self.arb = PairCostArb(settings.min_edge, settings.fee_rate_bps)
        self.directional = BrownianDirectional(
            settings.min_edge, settings.min_prob, settings.fee_rate_bps)
        self.vol = EwmaVol()

        self.spot_feed = BinanceSpotFeed()
        self.book_feed = ClobBookFeed(settings.clob_host)
        self._window_open_price: float | None = None
        self._current_window_start: int | None = None
        # directional trading is data-collection only until calibration proves it
        self.enable_directional_live = False

        self.maker = (MakerPairQuoter(settings, self.store, self.gate,
                                      self.executor, self.notifier)
                      if settings.maker_enabled else None)

        self._restore_state()

    # ------------------------------------------------------- state restore
    def _restore_state(self) -> None:
        """Rebuild risk-gate state from the store. A process restart must
        never reset the daily loss counter, the loss streak, or a halt."""
        stale = self.store.cancel_stale_open_orders()
        if stale:
            log.warning("cancelled %d stale resting orders from previous run", stale)
        reason = self.store.kv_get(HALT_KV_KEY)
        if reason:
            self.gate.halt(reason)
            log.warning("restored HALT from store: %s", reason)

        midnight = int(dt.datetime.now(dt.timezone.utc)
                       .replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
        self.gate.daily_pnl = self.store.daily_pnl(midnight)
        self.gate.cumulative_pnl = self.store.total_pnl()
        hwm_stored = self.store.kv_get(HWM_KV_KEY)
        self.gate.high_water_mark = max(
            self.s.bankroll,
            self.gate.equity,
            float(hwm_stored) if hwm_stored else 0.0,
        )
        self.gate.consecutive_losses = self.store.consecutive_losses()

        for row in self.store.unsettled_filled_orders():
            fill = self.store.fill_for_order(row["client_id"])
            if fill is None:
                continue
            notional = float(fill["price"]) * float(fill["size"])
            self.gate.open_exposure += notional
            # completed pairs have locked-in profit -> zero residual risk;
            # anything else (directional, unhedged/unmatched leg) can lose
            # the premium plus the entry fee already paid
            is_safe_pair = (row["strategy"] in PAIR_STRATEGIES
                            and row["status"] != "unhedged"
                            and self.store.matched_pair_fill(
                                row["window_start"], row["strategy"],
                                row["outcome"], float(fill["size"])))
            if not is_safe_pair:
                self.gate.open_risk += notional + float(fill["fee_paid"])
        log.info("state restored: daily_pnl=%.2f streak=%d exposure=%.2f "
                 "risk=%.2f budget_left=%.2f",
                 self.gate.daily_pnl, self.gate.consecutive_losses,
                 self.gate.open_exposure, self.gate.open_risk,
                 self.gate.remaining_risk_budget)

    def _persist_risk_state(self) -> None:
        if self.gate.halted and self.store.kv_get(HALT_KV_KEY) != self.gate.halt_reason:
            self.store.kv_set(HALT_KV_KEY, self.gate.halt_reason)
        self.store.kv_set(HWM_KV_KEY, f"{self.gate.high_water_mark:.6f}")

    # ------------------------------------------------------------------ tick
    def tick(self, market: MarketWindow, yes_book: BookTop, no_book: BookTop,
             spot: float | None = None) -> list[ExecutionResult]:
        now = self.clock.now()
        results: list[ExecutionResult] = []

        # Always track vol and window open/close, even outside the entry zone:
        # settlement and the next window's signal depend on it.
        if spot is not None:
            self.vol.update(spot, now)
            self._track_window(now, spot)

        if self.gate.halted:
            self.on_halt_maintenance()
            return results

        books_ok = self._book_sane(yes_book) and self._book_sane(no_book)

        # The maker quoter manages its own timers (place/reprice/hedge/cancel)
        # and must run every tick — including outside the entry zone, where
        # its job is tearing quotes down, not putting them up.
        if self.maker is not None:
            self.maker.tick(now, market,
                            yes_book if books_ok else None,
                            no_book if books_ok else None)

        if not in_entry_zone(now, self.s.window_seconds,
                             min_remaining_s=self.s.entry_min_remaining_s):
            self._persist_risk_state()
            return results
        if not books_ok:
            log.debug("degenerate book, skipping tick")
            return results

        pair_signals = self.arb.evaluate(yes_book, no_book)
        if pair_signals:
            results.extend(self._execute_pair(pair_signals, market, yes_book, no_book))

        if spot is not None and self._directional_ready(now):
            p_up_raw = self.directional.probability_up(
                spot, self._window_open_price,
                self.vol.sigma_per_sqrt_s, seconds_to_close(now, self.s.window_seconds))
            p_up = shrink_probability(p_up_raw, self.s.prob_shrinkage)
            for sig in self.directional.evaluate(yes_book, no_book, p_up):
                book = yes_book if sig.outcome is Outcome.YES else no_book
                if book.ask - book.bid > self.s.max_spread:
                    log.debug("spread too wide for directional (%s)", sig.note)
                    continue
                res = self._execute_directional(sig, market, book)
                if res:
                    results.append(res)

        self._persist_risk_state()
        return results

    def _book_sane(self, book: BookTop) -> bool:
        return (0.0 < book.bid < 1.0 and 0.0 < book.ask < 1.0
                and book.ask >= book.bid and book.ask_size >= 0)

    def _directional_ready(self, now: float) -> bool:
        if not (self.s.dry_run or self.enable_directional_live):
            return False
        if self._window_open_price is None or self._current_window_start is None:
            return False
        if self.vol.n < self.s.vol_min_samples or self.vol.sigma_per_sqrt_s <= 0:
            return False
        # the log-return signal is noise until some of the window has elapsed
        return now - self._current_window_start >= self.s.dir_min_elapsed_s

    def _track_window(self, now: float, spot: float) -> None:
        start, _ = window_bounds(now, self.s.window_seconds)
        if start != self._current_window_start:
            if self._current_window_start is not None:
                # first spot of the new window is the best available proxy
                # for the previous window's close
                self.store.record_window_close(self._current_window_start, spot)
            self._current_window_start = start
            self._window_open_price = spot
            self.store.record_window_open(start, spot)

    # ----------------------------------------------------------- pair legs
    def _execute_pair(self, signals: list[Signal], market: MarketWindow,
                      yes_book: BookTop, no_book: BookTop) -> list[ExecutionResult]:
        """Both legs of a pair MUST have the same share count. Sized by the
        thinner displayed ask, max_bet, and the remaining daily risk budget;
        risk is reserved for one leg (the unhedged worst case) and released
        once the pair completes."""
        sig_yes = next(s for s in signals if s.outcome is Outcome.YES)
        sig_no = next(s for s in signals if s.outcome is Outcome.NO)
        pair_cost = sig_yes.price + sig_no.price
        if pair_cost <= 0:
            return []

        # worst case per share: one leg fills, the other fails -> we hold a
        # naked binary that can expire worthless, and its entry fee is sunk.
        # Sized so even that transient risk fits the remaining daily budget.
        max_leg = max(sig_yes.price, sig_no.price)
        wc_per_share = max_leg + taker_fee_per_share(max_leg, market.fee_rate_bps)
        budget_shares = (self.gate.remaining_risk_budget / wc_per_share
                         if wc_per_share > 0 else 0.0)
        shares = min(yes_book.ask_size, no_book.ask_size,
                     self.s.max_bet / pair_cost, budget_shares)
        shares = shares_for_notional(shares * pair_cost, pair_cost,
                                     market.min_order_size)
        if shares <= 0:
            return []
        notional = round(shares * pair_cost, 6)
        if notional < self.s.min_bet:
            return []
        worst_case = shares * wc_per_share

        decision = self.gate.check_order(notional, worst_case)
        if not decision.allowed:
            self._log_denial(decision.reason, sig_yes.note)
            return []

        res1 = self._submit_leg(sig_yes, market, yes_book, shares,
                                worst_case_override=None)
        if res1 is None or not (res1.accepted and res1.fill):
            return []            # nothing at risk, pair abandoned atomically

        # second leg matches the FILLED size of the first, never the intended
        res2 = self._submit_leg(sig_no, market, no_book, res1.fill.size,
                                worst_case_override=0.0)
        if res2 is not None and res2.accepted and res2.fill:
            # pair complete: settlement pays 1 per share regardless of
            # direction -> locked-in profit, release the reserved risk
            self.gate.release_risk(res1.fill.price * res1.fill.size
                                   + res1.fill.fee_paid)
            return [res1, res2]

        # UNHEDGED: first leg filled, second failed. This is a real position
        # with real downside. Stop trading, tell the human.
        self.store.set_order_status(res1.fill.order_client_id, "unhedged")
        self.gate.halt("unhedged_pair_leg")
        self._persist_risk_state()
        msg = (f"🚨 UNHEDGED PAIR LEG on {market.slug}: "
               f"{sig_yes.outcome.value} filled, {sig_no.outcome.value} failed "
               f"({res2.reason if res2 else 'duplicate/rejected'}). Engine halted.")
        log.error(msg)
        self.notifier.send(msg)
        return [res1]

    def _submit_leg(self, sig: Signal, market: MarketWindow, book: BookTop,
                    size: float, worst_case_override: float | None) -> ExecutionResult | None:
        order = Order(market=market, outcome=sig.outcome, side=Side.BUY,
                      price=sig.price, size=size, strategy=sig.strategy)
        is_new = self.store.record_order(
            client_id=order.client_id, market_slug=market.slug,
            window_start=market.window_start, outcome=sig.outcome.value,
            side="BUY", price=sig.price, size=size, strategy=sig.strategy,
            dry_run=self.s.dry_run, prob=None)
        if not is_new:
            log.info("duplicate order suppressed (%s)", order.client_id[:8])
            return None
        res = self.executor.submit(order, book)
        if res.accepted and res.fill:
            self.store.record_fill(order.client_id, res.fill.price,
                                   res.fill.size, res.fill.fee_paid)
            filled_notional = res.fill.price * res.fill.size
            wc = (filled_notional + res.fill.fee_paid
                  if worst_case_override is None else worst_case_override)
            self.gate.on_order_filled(filled_notional, wc)
        else:
            self.store.set_order_status(order.client_id, "rejected")
        return res

    # ---------------------------------------------------------- directional
    def _execute_directional(self, sig: Signal, market: MarketWindow,
                             book: BookTop) -> ExecutionResult | None:
        notional = position_notional(
            self.s.bankroll, sig.probability, sig.price,
            kelly_multiplier=self.s.kelly_multiplier,
            min_bet=self.s.min_bet, max_bet=self.s.max_bet,
            max_bankroll_fraction=self.s.max_bankroll_fraction)
        # never size past the remaining daily risk budget (premium + entry fee
        # both burn budget on a loss) or the displayed liquidity
        fee_ps = taker_fee_per_share(sig.price, market.fee_rate_bps)
        budget_cap = (self.gate.remaining_risk_budget
                      * sig.price / (sig.price + fee_ps))
        notional = min(notional, budget_cap, book.ask_size * sig.price)
        if notional < self.s.min_bet:
            return None

        size = shares_for_notional(notional, sig.price, market.min_order_size)
        if size <= 0:
            return None
        notional = round(size * sig.price, 6)
        worst_case = notional + taker_fee(sig.price, size, market.fee_rate_bps)

        decision = self.gate.check_order(notional, worst_case)
        if not decision.allowed:
            self._log_denial(decision.reason, sig.note)
            return None

        order = Order(market=market, outcome=sig.outcome, side=Side.BUY,
                      price=sig.price, size=size, strategy=sig.strategy)
        is_new = self.store.record_order(
            client_id=order.client_id, market_slug=market.slug,
            window_start=market.window_start, outcome=sig.outcome.value,
            side="BUY", price=sig.price, size=size, strategy=sig.strategy,
            dry_run=self.s.dry_run, prob=sig.probability)
        if not is_new:
            log.info("duplicate order suppressed (%s)", order.client_id[:8])
            return None

        res = self.executor.submit(order, book)
        if res.accepted and res.fill:
            self.store.record_fill(order.client_id, res.fill.price,
                                   res.fill.size, res.fill.fee_paid)
            filled_notional = res.fill.price * res.fill.size
            self.gate.on_order_filled(filled_notional,
                                      filled_notional + res.fill.fee_paid)
        else:
            self.store.set_order_status(order.client_id, "rejected")
        return res

    def _log_denial(self, reason: str, note: str) -> None:
        log.warning("risk gate denied: %s (%s)", reason, note)
        if reason in ("kill_switch", "daily_loss_limit", "consecutive_losses",
                      "max_drawdown"):
            self.notifier.send(f"⛔ ENGINE HALTED: {reason}")

    # ------------------------------------------------------------ settlement
    def settle_expired(self, now: float | None = None) -> int:
        """Settle filled orders whose window has closed, using the recorded
        window open/close spot prices. Boundary rule: close must be STRICTLY
        above open for UP to win (flat = DOWN), matching the market's
        convention — verify per-market before going live.

        In live mode this is the accounting view; the reconciler remains the
        source of truth against the exchange."""
        now = self.clock.now() if now is None else now
        settled = 0
        for row in self.store.unsettled_filled_orders():
            window_end = row["window_start"] + self.s.window_seconds
            if now < window_end + self.s.settle_grace_s:
                continue
            win = self.store.get_window(row["window_start"])
            if win is None or win["close_price"] is None:
                log.debug("no close price yet for window %s", row["window_start"])
                continue
            fill = self.store.fill_for_order(row["client_id"])
            if fill is None:
                continue
            up = float(win["close_price"]) > float(win["open_price"])
            won = up if row["outcome"] == Outcome.YES.value else not up
            size = float(fill["size"])
            notional = float(fill["price"]) * size
            payout = size if won else 0.0
            is_pair = row["strategy"] in PAIR_STRATEGIES and row["status"] != "unhedged"
            fee_paid = float(fill["fee_paid"])
            self.settle(row["client_id"], won=won, notional=notional,
                        payout=payout, fee_paid=fee_paid,
                        predicted_prob=row["prob"],
                        counts_streak=not is_pair,
                        risk_release=0.0 if is_pair else notional + fee_paid)
            settled += 1
        if settled:
            self._persist_risk_state()
        return settled

    def settle(self, client_id: str, *, won: bool, notional: float,
               payout: float, fee_paid: float, predicted_prob: float | None,
               counts_streak: bool = True, risk_release: float | None = None) -> None:
        pnl = payout - notional - fee_paid
        self.store.record_settlement(client_id, won, pnl, predicted_prob)
        self.gate.on_position_settled(notional, pnl,
                                      risk_release=risk_release,
                                      counts_streak=counts_streak)
        if predicted_prob is not None:
            self.calibration.record(predicted_prob, int(won))
        if self.gate.halted:
            self._persist_risk_state()
            self.notifier.send(f"⛔ ENGINE HALTED after settlement: {self.gate.halt_reason}")

    def on_halt_maintenance(self) -> None:
        """A halted engine must not leave maker quotes resting: they could
        fill while nobody is managing them. Idempotent, called every loop
        iteration while halted."""
        if self.maker is not None:
            self.maker.go_flat(f"halted: {self.gate.halt_reason}")
        self._persist_risk_state()

    # ------------------------------------------------------------------ run
    def startup_checks(self) -> None:
        self.s.assert_live_allowed()
        if not self.s.dry_run and hasattr(self.executor, "cancel_all_open"):
            # crash recovery: no order may rest on the exchange unattended
            self.executor.cancel_all_open()
        if not self.reconciler.run():
            raise RuntimeError("reconciliation failed at startup")
        self._sync_live_bankroll()
        mode = "DRY_RUN" if self.s.dry_run else "LIVE"
        msg = (f"🤖 PolyBot engine starting [{mode}] bankroll={self.s.bankroll} "
               f"max_bet={self.s.max_bet} "
               f"daily_risk_budget={self.gate.daily_loss_budget:.2f} "
               f"(remaining today: {self.gate.remaining_risk_budget:.2f})")
        log.info(msg)
        self.notifier.send(msg)

    def _sync_live_bankroll(self) -> None:
        """Live mode: never size off a configured bankroll larger than the
        wallet actually holds."""
        if self.s.dry_run or not hasattr(self.executor, "collateral_balance"):
            return
        try:
            balance = float(self.executor.collateral_balance())
        except Exception as e:  # noqa: BLE001
            log.warning("could not fetch collateral balance: %s", e)
            return
        if balance < self.s.bankroll:
            log.warning("configured bankroll %.2f > wallet balance %.2f — "
                        "sizing off the wallet balance", self.s.bankroll, balance)
            self.s.bankroll = balance
            self.gate.bankroll = balance
