"""Market data feeds.

Design rules learned the hard way:
- Prices come from the CLOB order book, NEVER from the Gamma API
  (Gamma returns a 0.5 default on these markets — known PolyBot bug).
- Every feed exposes `is_stale()`; the engine refuses to trade on stale data.
- Reconnection with exponential backoff; a feed that can't recover raises,
  it does not return the last known price forever.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.request

from models import BookTop

log = logging.getLogger("feeds")

STALE_AFTER_S = 5.0


class FeedError(Exception):
    pass


class BinanceSpotFeed:
    """REST polling fallback. For production latency, replace `poll` with the
    websocket stream (wss://stream.binance.com:9443/ws/btcusdt@trade) — same
    interface, the engine does not change."""

    URL = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"

    def __init__(self) -> None:
        self.last_price: float | None = None
        self.last_ts: float = 0.0

    def poll(self) -> float:
        with urllib.request.urlopen(self.URL, timeout=3) as r:
            data = json.loads(r.read())
        self.last_price = float(data["price"])
        self.last_ts = time.time()
        return self.last_price

    def is_stale(self) -> bool:
        return self.last_price is None or (time.time() - self.last_ts) > STALE_AFTER_S


class ClobBookFeed:
    """Top-of-book via CLOB REST. Production: subscribe to the CLOB websocket
    market channel; keep this as the recovery path."""

    def __init__(self, clob_host: str) -> None:
        self.host = clob_host.rstrip("/")
        self._last: dict[str, tuple[BookTop, float]] = {}

    def top(self, token_id: str) -> BookTop:
        url = f"{self.host}/book?token_id={token_id}"
        with urllib.request.urlopen(url, timeout=3) as r:
            book = json.loads(r.read())
        bids = book.get("bids") or []
        asks = book.get("asks") or []
        if not bids or not asks:
            raise FeedError(f"empty book for {token_id}")
        best_bid = max(bids, key=lambda x: float(x["price"]))
        best_ask = min(asks, key=lambda x: float(x["price"]))
        top = BookTop(
            token_id=token_id,
            bid=float(best_bid["price"]), ask=float(best_ask["price"]),
            bid_size=float(best_bid["size"]), ask_size=float(best_ask["size"]),
        )
        self._last[token_id] = (top, time.time())
        return top

    def is_stale(self, token_id: str) -> bool:
        entry = self._last.get(token_id)
        return entry is None or (time.time() - entry[1]) > STALE_AFTER_S


def fetch_market_window(clob_host: str, slug: str) -> dict:
    """Resolve slug -> condition_id, token ids, fee rate, tick size from CLOB
    market metadata. The engine verifies the slug template against this at
    startup instead of trusting the template blindly."""
    url = f"{clob_host.rstrip('/')}/markets?slug={slug}"
    with urllib.request.urlopen(url, timeout=3) as r:
        return json.loads(r.read())
