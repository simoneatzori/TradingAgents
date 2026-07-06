"""5-minute window alignment with server-clock offset.

Never trust the local clock alone: the engine keeps an offset against the
CLOB server time and aligns windows on server time. Boundary rule: a
timestamp exactly on a boundary belongs to the window that STARTS there.
"""
from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class Clock:
    server_offset_s: float = 0.0   # server_time - local_time

    def now(self) -> float:
        return time.time() + self.server_offset_s

    def sync(self, server_unix_time: float, local_unix_time: float | None = None) -> None:
        local = local_unix_time if local_unix_time is not None else time.time()
        self.server_offset_s = server_unix_time - local


def window_start(ts: float, window_seconds: int = 300) -> int:
    return int(ts) - (int(ts) % window_seconds)


def window_bounds(ts: float, window_seconds: int = 300) -> tuple[int, int]:
    start = window_start(ts, window_seconds)
    return start, start + window_seconds


def seconds_to_close(ts: float, window_seconds: int = 300) -> float:
    _, end = window_bounds(ts, window_seconds)
    return end - ts


def in_entry_zone(ts: float, window_seconds: int = 300,
                  min_remaining_s: float = 20.0,
                  max_remaining_s: float | None = None) -> bool:
    """Refuse entries too close to settlement (can't exit, adverse selection)
    and optionally too early (no information yet)."""
    rem = seconds_to_close(ts, window_seconds)
    if rem < min_remaining_s:
        return False
    if max_remaining_s is not None and rem > max_remaining_s:
        return False
    return True


def market_slug(template: str, start_ts: int) -> str:
    """Render a market slug for the window starting at start_ts (UTC).

    The exact slug format is exchange-defined and changes; keep it in config
    and verify against live market metadata at startup (engine does this).
    """
    return template.format(ts=start_ts)
