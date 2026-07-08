"""Order execution.

Two order modes:
- submit(): immediate taker order (FOK live, ask-crossing simulation dry).
- place_resting()/poll_resting()/cancel_resting(): passive maker orders
  (GTC live). The dry-run simulation is pessimistic AND all-or-nothing:
  a resting buy fills only when the observed best ask crosses down to our
  price with enough displayed size for the FULL order — partial maker
  fills would desynchronize pair legs, so the sim never produces them.
  (Live GTC orders CAN partially fill; poll_resting reports a fill only
  once size_matched covers the full size. Keep MAX_BET small.)

DryRunExecutor: simulates fills against the provided book (fills at the ask
up to displayed size, applies the fee model). Default.

LiveClobExecutor: thin adapter over py-clob-client. Imported lazily so the
whole engine (and the test suite) runs without the dependency. Before any
live order it verifies: ack phrase, key present, allowances approved.

NOTE on allowances: live trading on Polygon requires one-time ERC-20 (USDC)
and ERC-1155 (CTF) approvals to the exchange contracts. The article's prompts
omit this entirely — without it the first live order fails. Run
`python bot.py approve` once per wallet (live mode only).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from config import Settings
from fees import taker_fee, round_to_tick
from models import BookTop, Fill, Order, Side

log = logging.getLogger("executor")


class ExecutionError(Exception):
    pass


@dataclass
class ExecutionResult:
    accepted: bool
    fill: Fill | None = None
    reason: str = ""
    resting: bool = False       # True when accepted as a passive (maker) order


class DryRunExecutor:
    """Pessimistic fill simulation: only fills up to displayed ask size,
    never assumes price improvement."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._seen_ids: set[str] = set()
        self._resting: dict[str, Order] = {}

    def submit(self, order: Order, book: BookTop) -> ExecutionResult:
        if order.client_id in self._seen_ids:
            return ExecutionResult(False, reason="duplicate_client_id")
        self._seen_ids.add(order.client_id)

        if order.side is not Side.BUY:
            return ExecutionResult(False, reason="sim_supports_buy_only")

        price = round_to_tick(order.price, order.market.tick_size)
        if book.ask > price:
            return ExecutionResult(False, reason="ask_moved_above_limit")

        fill_size = min(order.size, book.ask_size)
        if fill_size < order.market.min_order_size:
            return ExecutionResult(False, reason="insufficient_displayed_size")

        fee = taker_fee(book.ask, fill_size, order.market.fee_rate_bps)
        fill = Fill(order.client_id, book.ask, fill_size, fee, ts=0)
        log.info("DRY_RUN fill %s %s %.2f @ %.2f fee=%.4f",
                 order.market.slug, order.outcome.value, fill_size, book.ask, fee)
        return ExecutionResult(True, fill=fill)

    # ---- passive (maker) orders ------------------------------------------
    def place_resting(self, order: Order, book: BookTop) -> ExecutionResult:
        if order.client_id in self._seen_ids:
            return ExecutionResult(False, reason="duplicate_client_id")
        if order.side is not Side.BUY:
            return ExecutionResult(False, reason="sim_supports_buy_only")
        if book.ask <= order.price:
            # post-only semantics: a bid at/above the ask would take, not make
            return ExecutionResult(False, reason="post_only_would_cross")
        self._seen_ids.add(order.client_id)
        self._resting[order.client_id] = order
        log.info("DRY_RUN resting %s %s %.2f @ %.2f",
                 order.market.slug, order.outcome.value, order.size, order.price)
        return ExecutionResult(True, resting=True)

    def poll_resting(self, client_id: str, book: BookTop) -> Fill | None:
        order = self._resting.get(client_id)
        if order is None:
            return None
        # pessimistic maker fill: a seller must cross down to our level with
        # enough size for the whole order; we fill at OUR price
        if book.ask <= order.price and book.ask_size >= order.size:
            fee = taker_fee(order.price, order.size, self.settings.maker_fee_bps)
            del self._resting[client_id]
            log.info("DRY_RUN maker fill %s %.2f @ %.2f fee=%.4f",
                     order.outcome.value, order.size, order.price, fee)
            return Fill(client_id, order.price, order.size, fee, ts=0)
        return None

    def cancel_resting(self, client_id: str) -> bool:
        return self._resting.pop(client_id, None) is not None


class LiveClobExecutor:
    def __init__(self, settings: Settings) -> None:
        settings.assert_live_allowed()
        self.settings = settings
        self._client = self._build_client()
        self._resting_ids: dict[str, tuple[str, Order]] = {}

    def _build_client(self):
        try:
            from py_clob_client.client import ClobClient            # type: ignore
            from py_clob_client.clob_types import ApiCreds          # noqa: F401
        except ImportError as e:
            raise ExecutionError(
                "py-clob-client not installed. `pip install py-clob-client` "
                "and pin the version in requirements.txt after auditing it."
            ) from e

        kwargs: dict = dict(
            host=self.settings.clob_host,
            key=self.settings.private_key,
            chain_id=self.settings.chain_id,
        )
        # Safe/proxy wallet: funder + signature_type must move TOGETHER.
        # signature_type=2 with no funder (or vice versa) silently produces
        # orders the operator did not intend.
        if self.settings.safe_address:
            kwargs["funder"] = self.settings.safe_address
            kwargs["signature_type"] = 2
        client = ClobClient(**kwargs)
        creds = client.create_or_derive_api_creds()
        client.set_api_creds(creds)
        return client

    def server_time(self) -> float:
        return float(self._client.get_server_time())

    def collateral_balance(self) -> float:
        from py_clob_client.clob_types import AssetType, BalanceAllowanceParams  # type: ignore
        res = self._client.get_balance_allowance(
            BalanceAllowanceParams(asset_type=AssetType.COLLATERAL))
        return float(res["balance"]) / 1e6  # USDC has 6 decimals

    def submit(self, order: Order, book: BookTop) -> ExecutionResult:
        from py_clob_client.clob_types import OrderArgs, OrderType   # type: ignore

        price = round_to_tick(order.price, order.market.tick_size)
        token_id = (order.market.yes_token if order.outcome.value == "YES"
                    else order.market.no_token)
        args = OrderArgs(price=price, size=order.size, side=order.side.value,
                         token_id=token_id)
        try:
            signed = self._client.create_order(args)
            # FOK on 5-minute markets: a resting partial fill near settlement
            # is unmanaged risk.
            resp = self._client.post_order(signed, OrderType.FOK)
        except Exception as e:  # noqa: BLE001 — surface everything to the engine
            return ExecutionResult(False, reason=f"clob_error: {e}")

        if not resp.get("success"):
            return ExecutionResult(False, reason=str(resp.get("errorMsg", "rejected")))

        fee = taker_fee(price, order.size, order.market.fee_rate_bps)
        return ExecutionResult(True, fill=Fill(order.client_id, price, order.size, fee, ts=0))

    # ---- passive (maker) orders ------------------------------------------
    # NOTE: integration-test these three against the real CLOB in DRY-RUN-
    # adjacent conditions before first live maker session. The mapping
    # client_id -> exchange order id lives in memory; after a crash the
    # startup path cancels ALL open orders (cancel_all_open) so nothing
    # rests unattended.
    def place_resting(self, order: Order, book: BookTop) -> ExecutionResult:
        from py_clob_client.clob_types import OrderArgs, OrderType   # type: ignore

        price = round_to_tick(order.price, order.market.tick_size)
        token_id = (order.market.yes_token if order.outcome.value == "YES"
                    else order.market.no_token)
        args = OrderArgs(price=price, size=order.size, side=order.side.value,
                         token_id=token_id)
        try:
            signed = self._client.create_order(args)
            resp = self._client.post_order(signed, OrderType.GTC)
        except Exception as e:  # noqa: BLE001
            return ExecutionResult(False, reason=f"clob_error: {e}")
        if not resp.get("success"):
            return ExecutionResult(False, reason=str(resp.get("errorMsg", "rejected")))
        self._resting_ids[order.client_id] = (resp.get("orderID", ""), order)
        return ExecutionResult(True, resting=True)

    def poll_resting(self, client_id: str, book: BookTop) -> Fill | None:
        entry = self._resting_ids.get(client_id)
        if entry is None:
            return None
        exchange_id, order = entry
        try:
            data = self._client.get_order(exchange_id)
        except Exception as e:  # noqa: BLE001
            log.warning("poll_resting failed: %s", e)
            return None
        matched = float(data.get("size_matched", 0) or 0)
        # all-or-nothing view: report the fill only once fully matched, so
        # pair legs stay size-synchronized (see module docstring)
        if matched + 1e-9 < order.size:
            return None
        del self._resting_ids[client_id]
        fee = taker_fee(order.price, order.size, self.settings.maker_fee_bps)
        return Fill(client_id, order.price, order.size, fee, ts=0)

    def cancel_resting(self, client_id: str) -> bool:
        entry = self._resting_ids.pop(client_id, None)
        if entry is None:
            return False
        try:
            self._client.cancel(entry[0])
            return True
        except Exception as e:  # noqa: BLE001
            log.error("cancel_resting failed for %s: %s", client_id[:8], e)
            return False

    def cancel_all_open(self) -> None:
        """Crash-recovery safety net: no order may rest unattended."""
        try:
            self._client.cancel_all()
            self._resting_ids.clear()
        except Exception as e:  # noqa: BLE001
            log.error("cancel_all failed: %s", e)
            raise


def build_executor(settings: Settings):
    if settings.dry_run:
        log.info("Executor: DRY_RUN simulation")
        return DryRunExecutor(settings)
    log.warning("Executor: LIVE trading enabled")
    return LiveClobExecutor(settings)
