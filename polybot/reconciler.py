"""Crash recovery / reconciliation.

On startup and periodically, compare the store's idea of open positions with
the exchange's. Any divergence -> halt via the risk gate and notify; a human
resolves it. The engine never "guesses" its own position.
"""
from __future__ import annotations

import logging

from risk_gate import RiskGate
from store import Store

log = logging.getLogger("reconciler")


class Reconciler:
    def __init__(self, store: Store, risk_gate: RiskGate, dry_run: bool) -> None:
        self.store = store
        self.gate = risk_gate
        self.dry_run = dry_run

    def exchange_open_positions(self) -> dict[str, float] | None:
        """Live: query positions via CLOB / data API. Dry-run: None (skip)."""
        if self.dry_run:
            return None
        raise NotImplementedError(
            "Wire to CLOB positions endpoint before live trading; "
            "reconciliation is a P1 launch blocker, not a nice-to-have.")

    def run(self) -> bool:
        """Returns True if state is consistent."""
        local_open = self.store.open_orders()
        remote = self.exchange_open_positions()
        if remote is None:
            log.info("Reconciler: dry-run, %d local open orders", len(local_open))
            return True

        local_ids = {row["client_id"] for row in local_open}
        remote_ids = set(remote.keys())
        if local_ids != remote_ids:
            diff = local_ids.symmetric_difference(remote_ids)
            self.gate.halt(f"reconciliation mismatch: {sorted(diff)[:5]}")
            log.error("RECONCILIATION FAILED, halting. diff=%s", diff)
            return False
        return True
