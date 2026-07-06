"""Order execution.

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


class DryRunExecutor:
    """Pessimistic fill simulation: only fills up to displayed ask size,
    never assumes price improvement."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._seen_ids: set[str] = set()

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


class LiveClobExecutor:
    def __init__(self, settings: Settings) -> None:
        settings.assert_live_allowed()
        self.settings = settings
        self._client = self._build_client()

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


def build_executor(settings: Settings):
    if settings.dry_run:
        log.info("Executor: DRY_RUN simulation")
        return DryRunExecutor(settings)
    log.warning("Executor: LIVE trading enabled")
    return LiveClobExecutor(settings)
