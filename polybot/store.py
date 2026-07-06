"""SQLite persistence: orders, fills, settlements, window prices, engine state.

WAL mode so the engine and the offline analyst can read concurrently.
This file is the single source of truth for reconciliation after a crash —
the risk gate's daily PnL, loss streak, open exposure and halt state are all
rebuilt from here at startup so a restart can never bypass a limit.
"""
from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager

SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    client_id   TEXT PRIMARY KEY,
    ts          INTEGER NOT NULL,
    market_slug TEXT NOT NULL,
    window_start INTEGER NOT NULL,
    outcome     TEXT NOT NULL,
    side        TEXT NOT NULL,
    price       REAL NOT NULL,
    size        REAL NOT NULL,
    strategy    TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'submitted',  -- submitted|filled|cancelled|rejected|unhedged
    dry_run     INTEGER NOT NULL,
    prob        REAL                                 -- model probability at entry (directional)
);
CREATE TABLE IF NOT EXISTS fills (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id   TEXT NOT NULL REFERENCES orders(client_id),
    ts          INTEGER NOT NULL,
    price       REAL NOT NULL,
    size        REAL NOT NULL,
    fee_paid    REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS settlements (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id   TEXT NOT NULL REFERENCES orders(client_id),
    ts          INTEGER NOT NULL,
    outcome_won INTEGER NOT NULL,           -- 1 if our token settled at 1
    pnl         REAL NOT NULL,
    predicted_prob REAL
);
CREATE TABLE IF NOT EXISTS windows (
    window_start INTEGER PRIMARY KEY,
    open_price  REAL NOT NULL,
    close_price REAL
);
CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


class Store:
    def __init__(self, path: str) -> None:
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(orders)")}
        if "prob" not in cols:
            self.conn.execute("ALTER TABLE orders ADD COLUMN prob REAL")

    @contextmanager
    def tx(self):
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # ---- orders ----------------------------------------------------------
    def record_order(self, *, client_id: str, market_slug: str, window_start: int,
                     outcome: str, side: str, price: float, size: float,
                     strategy: str, dry_run: bool, prob: float | None = None) -> bool:
        """Returns False if the client_id already exists (idempotent dedupe)."""
        with self.tx() as c:
            cur = c.execute(
                "INSERT OR IGNORE INTO orders "
                "(client_id, ts, market_slug, window_start, outcome, side, price, size, strategy, dry_run, prob) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (client_id, int(time.time()), market_slug, window_start,
                 outcome, side, price, size, strategy, int(dry_run), prob),
            )
            return cur.rowcount == 1

    def set_order_status(self, client_id: str, status: str) -> None:
        with self.tx() as c:
            c.execute("UPDATE orders SET status=? WHERE client_id=?", (status, client_id))

    def record_fill(self, client_id: str, price: float, size: float, fee_paid: float) -> None:
        with self.tx() as c:
            c.execute("INSERT INTO fills (client_id, ts, price, size, fee_paid) VALUES (?,?,?,?,?)",
                      (client_id, int(time.time()), price, size, fee_paid))
            c.execute("UPDATE orders SET status='filled' WHERE client_id=?", (client_id,))

    def record_settlement(self, client_id: str, outcome_won: bool, pnl: float,
                          predicted_prob: float | None = None) -> None:
        with self.tx() as c:
            c.execute("INSERT INTO settlements (client_id, ts, outcome_won, pnl, predicted_prob) "
                      "VALUES (?,?,?,?,?)",
                      (client_id, int(time.time()), int(outcome_won), pnl, predicted_prob))

    # ---- windows (spot open/close per 5-min window) -----------------------
    def record_window_open(self, window_start: int, open_price: float) -> None:
        with self.tx() as c:
            c.execute("INSERT OR IGNORE INTO windows (window_start, open_price) VALUES (?,?)",
                      (window_start, open_price))

    def record_window_close(self, window_start: int, close_price: float) -> None:
        with self.tx() as c:
            c.execute("UPDATE windows SET close_price=? WHERE window_start=? AND close_price IS NULL",
                      (close_price, window_start))

    def get_window(self, window_start: int) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM windows WHERE window_start=?", (window_start,)).fetchone()

    # ---- kv (engine state that must survive restarts) ---------------------
    def kv_get(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def kv_set(self, key: str, value: str) -> None:
        with self.tx() as c:
            c.execute("INSERT INTO kv (key, value) VALUES (?,?) "
                      "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))

    def kv_delete(self, key: str) -> None:
        with self.tx() as c:
            c.execute("DELETE FROM kv WHERE key=?", (key,))

    # ---- queries for reconciliation / reporting --------------------------
    def open_orders(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM orders WHERE status IN ('submitted','filled','unhedged') "
            "AND client_id NOT IN (SELECT client_id FROM settlements)").fetchall()

    def unsettled_filled_orders(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM orders WHERE status IN ('filled','unhedged') "
            "AND client_id NOT IN (SELECT client_id FROM settlements) "
            "ORDER BY window_start").fetchall()

    def fill_for_order(self, client_id: str) -> sqlite3.Row | None:
        """Aggregate fills for an order (this engine emits one fill per order)."""
        return self.conn.execute(
            "SELECT SUM(size) AS size, SUM(price*size)/SUM(size) AS price, "
            "SUM(fee_paid) AS fee_paid FROM fills WHERE client_id=? HAVING SUM(size) > 0",
            (client_id,)).fetchone()

    def daily_pnl(self, day_start_ts: int) -> float:
        row = self.conn.execute(
            "SELECT COALESCE(SUM(pnl), 0) AS p FROM settlements WHERE ts >= ?",
            (day_start_ts,)).fetchone()
        return float(row["p"])

    def total_pnl(self) -> float:
        row = self.conn.execute("SELECT COALESCE(SUM(pnl), 0) AS p FROM settlements").fetchone()
        return float(row["p"])

    def consecutive_losses(self, exclude_strategy: str = "pair_cost_arb") -> int:
        """Length of the current losing streak, newest settlement first.
        Paired-arb legs are excluded: one leg of a profitable pair always
        settles negative and must not poison the streak."""
        rows = self.conn.execute(
            "SELECT s.pnl FROM settlements s JOIN orders o ON o.client_id = s.client_id "
            "WHERE o.strategy != ? ORDER BY s.ts DESC, s.id DESC LIMIT 100",
            (exclude_strategy,)).fetchall()
        streak = 0
        for r in rows:
            if r["pnl"] < 0:
                streak += 1
            elif r["pnl"] > 0:
                break
        return streak

    def pnl_by_strategy(self) -> dict[str, float]:
        rows = self.conn.execute(
            "SELECT o.strategy AS s, COALESCE(SUM(st.pnl),0) AS p FROM settlements st "
            "JOIN orders o ON o.client_id = st.client_id GROUP BY o.strategy").fetchall()
        return {r["s"]: float(r["p"]) for r in rows}

    def stats(self) -> dict:
        n_orders = self.conn.execute("SELECT COUNT(*) c FROM orders").fetchone()["c"]
        n_settled = self.conn.execute("SELECT COUNT(*) c FROM settlements").fetchone()["c"]
        pnl = self.conn.execute("SELECT COALESCE(SUM(pnl),0) p FROM settlements").fetchone()["p"]
        fees = self.conn.execute("SELECT COALESCE(SUM(fee_paid),0) f FROM fills").fetchone()["f"]
        return {"orders": n_orders, "settled": n_settled, "pnl": pnl, "fees": fees,
                "pnl_by_strategy": self.pnl_by_strategy()}

    def close(self) -> None:
        self.conn.close()
