"""Entrypoint.

Commands:
  python bot.py run        # start the engine (DRY_RUN by default)
  python bot.py status     # print store stats, risk state, calibration
  python bot.py reset      # human reset of a persisted halt (after review)
  python bot.py approve    # one-time USDC/CTF allowance approvals (LIVE only)

The live market-discovery loop resolves the current 5-minute window's slug
against CLOB metadata each window (fetch_market_window) — token ids and fee
rate are taken from the exchange, never hardcoded.
"""
from __future__ import annotations

import datetime as dt
import logging
import sys
import time

from config import load_settings
from engine import HALT_KV_KEY, Engine
from store import Store

logging.basicConfig(
    level="INFO",
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("bot")


def cmd_run() -> None:
    settings = load_settings()
    engine = Engine(settings)
    engine.startup_checks()

    log.info("Entering main loop (poll every %.1fs). Ctrl-C to stop.",
             settings.poll_interval_s)
    try:
        while True:
            if engine.gate.halted:
                engine.on_halt_maintenance()   # tear down any resting quotes
                log.error("HALTED: %s — run `python bot.py reset` after review",
                          engine.gate.halt_reason)
                time.sleep(10)
                continue
            try:
                market = discover_current_market(settings, engine)
                if market is not None:
                    yes = engine.book_feed.top(market.yes_token)
                    no = engine.book_feed.top(market.no_token)
                    spot = engine.spot_feed.poll()
                    engine.tick(market, yes, no, spot)
                else:
                    # keep window open/close tracking alive even when the
                    # market isn't tradeable, so settlement never starves
                    spot = engine.spot_feed.poll()
                    engine._track_window(engine.clock.now(), spot)
                engine.settle_expired()
            except Exception as e:  # noqa: BLE001
                log.warning("tick error: %s", e)
            time.sleep(settings.poll_interval_s)
    except KeyboardInterrupt:
        log.info("stopped by user")


def discover_current_market(settings, engine):
    """Resolve the current window's market via CLOB metadata.

    Returns None when the market for this window isn't tradeable yet.
    Wire fetch_market_window() + slug template verification here; kept
    separate so it can be integration-tested against the real API.
    """
    from feeds import fetch_market_window
    from models import MarketWindow
    from timeutil import market_slug, window_bounds

    start, end = window_bounds(engine.clock.now(), settings.window_seconds)
    slug = market_slug(settings.market_slug_template, start)
    try:
        meta = fetch_market_window(settings.clob_host, slug)
    except Exception as e:  # noqa: BLE001
        log.debug("market discovery failed for %s: %s", slug, e)
        return None
    data = meta.get("data") or []
    if not data:
        return None
    m = data[0]
    tokens = m.get("tokens", [])
    if len(tokens) < 2:
        return None
    yes = next((t for t in tokens if t.get("outcome", "").upper() in ("YES", "UP")), tokens[0])
    no = next((t for t in tokens if t.get("outcome", "").upper() in ("NO", "DOWN")), tokens[1])
    return MarketWindow(
        slug=slug,
        condition_id=m.get("condition_id", ""),
        yes_token=yes["token_id"], no_token=no["token_id"],
        window_start=start, window_end=end,
        fee_rate_bps=int(m.get("taker_base_fee", settings.fee_rate_bps)),
        tick_size=float(m.get("minimum_tick_size", settings.tick_size)),
        min_order_size=float(m.get("minimum_order_size", settings.min_order_size)),
    )


def cmd_status() -> None:
    settings = load_settings()
    store = Store(settings.db_path)
    midnight = int(dt.datetime.now(dt.timezone.utc)
                   .replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
    print("store:", store.stats())
    print(f"today pnl: {store.daily_pnl(midnight):+.2f} "
          f"(budget {min(settings.max_daily_loss, settings.bankroll * settings.max_daily_loss_pct):.2f})")
    print(f"loss streak: {store.consecutive_losses()}")
    halt = store.kv_get(HALT_KV_KEY)
    print(f"halt: {halt or 'none'}")


def cmd_reset() -> None:
    """Clear a persisted halt. Deliberately requires the engine to be
    stopped: the in-process gate is rebuilt from the store at startup."""
    settings = load_settings()
    store = Store(settings.db_path)
    reason = store.kv_get(HALT_KV_KEY)
    if reason is None:
        print("No persisted halt. If the running engine is halted, restart it.")
        return
    print(f"Persisted halt: {reason}")
    print("Review the logs and the store BEFORE resetting. Remove any")
    print("KILL_SWITCH file yourself — this command does not touch it.")
    confirm = input("Type RESET to clear the halt: ").strip()
    if confirm == "RESET":
        store.kv_delete(HALT_KV_KEY)
        print("Halt cleared. Restart the engine.")
    else:
        print("Aborted.")


def cmd_approve() -> None:
    settings = load_settings()
    if settings.dry_run:
        print("DRY_RUN=true — approvals are a live-mode operation. Nothing to do.")
        return
    settings.assert_live_allowed()
    print("Run USDC (ERC-20) and CTF (ERC-1155) approvals to the Polymarket")
    print("exchange contracts for this wallet. Implement with web3.py against")
    print("audited contract addresses from the official docs before first live run.")
    raise SystemExit(1)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    {
        "run": cmd_run,
        "status": cmd_status,
        "reset": cmd_reset,
        "approve": cmd_approve,
    }.get(cmd, cmd_run)()
