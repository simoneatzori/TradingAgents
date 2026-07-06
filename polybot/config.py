"""Central configuration. Loaded from environment / .env file.

Safety invariants:
- DRY_RUN defaults to True. Live trading additionally requires
  LIVE_TRADING_ACK="I_UNDERSTAND_THE_RISKS".
- PRIVATE_KEY is never logged. repr/str of Settings redacts secrets.
- Full Kelly is forbidden; the daily loss budget defaults to 1% of bankroll
  and cannot be configured above 5%.
"""
from __future__ import annotations

import os
from pydantic import BaseModel, Field, field_validator

LIVE_ACK_PHRASE = "I_UNDERSTAND_THE_RISKS"


class Settings(BaseModel):
    # --- Wallet / API ---
    private_key: str = Field(default="", repr=False)
    safe_address: str = ""                      # Polymarket Safe/proxy wallet (optional)
    clob_host: str = "https://clob.polymarket.com"
    chain_id: int = 137                         # Polygon

    # --- Safety ---
    dry_run: bool = True
    live_trading_ack: str = ""
    kill_switch_file: str = "KILL_SWITCH"       # touch this file to halt instantly

    # --- Strategy thresholds ---
    min_edge: float = 0.01                      # net-of-fee edge required (1 cent / share)
    min_prob: float = 0.55                      # min model probability for directional trades
    fee_rate_bps: int = 200                     # taker fee in bps, refreshed from market metadata
    prob_shrinkage: float = 0.6                 # weight on model prob vs 0.5 prior
    max_spread: float = 0.06                    # skip directional entries on books wider than this
    dir_min_elapsed_s: float = 60.0             # directional needs elapsed time inside the window
    vol_min_samples: int = 30                   # EWMA vol warm-up before directional trades

    # --- Sizing ---
    bankroll: float = 100.0
    kelly_multiplier: float = 0.25              # quarter-Kelly
    min_bet: float = 1.0
    max_bet: float = 2.0
    max_bankroll_fraction: float = 0.05         # per-trade hard cap

    # --- Risk gate (circuit breakers) ---
    max_daily_loss: float = 5.0                 # absolute backstop (USDC)
    max_daily_loss_pct: float = 0.01            # 1% of bankroll per UTC day (the binding limit)
    max_drawdown_pct: float = 0.05              # halt at 5% drawdown from equity high-water mark
    daily_profit_lock_pct: float = 0.0          # stop for the day after +X% (0 = disabled)
    max_consecutive_losses: int = 5
    max_open_exposure: float = 10.0

    # --- Market ---
    window_seconds: int = 300                   # 5-minute up/down markets
    market_slug_template: str = "bitcoin-up-or-down-{ts}"
    tick_size: float = 0.01
    min_order_size: float = 1.0                 # shares
    entry_min_remaining_s: float = 20.0         # no entries closer than this to settlement
    settle_grace_s: float = 15.0                # wait after window close before settling

    # --- Notifications ---
    telegram_bot_token: str = Field(default="", repr=False)
    telegram_chat_id: str = ""

    # --- Ops ---
    db_path: str = "polybot.sqlite3"
    poll_interval_s: float = 1.0
    log_level: str = "INFO"

    @field_validator("kelly_multiplier")
    @classmethod
    def _kelly_sane(cls, v: float) -> float:
        if not 0 < v <= 0.5:
            raise ValueError("kelly_multiplier must be in (0, 0.5] — full Kelly is not allowed")
        return v

    @field_validator("max_daily_loss_pct")
    @classmethod
    def _daily_loss_pct_sane(cls, v: float) -> float:
        if not 0 < v <= 0.05:
            raise ValueError(
                "max_daily_loss_pct must be in (0, 0.05] — the engine is built "
                "around a hard daily loss cap; disabling it is not supported")
        return v

    @field_validator("max_drawdown_pct")
    @classmethod
    def _drawdown_pct_sane(cls, v: float) -> float:
        if not 0 <= v <= 0.25:
            raise ValueError("max_drawdown_pct must be in [0, 0.25]")
        return v

    @field_validator("daily_profit_lock_pct")
    @classmethod
    def _profit_lock_sane(cls, v: float) -> float:
        if v < 0:
            raise ValueError("daily_profit_lock_pct must be >= 0")
        return v

    @property
    def is_live(self) -> bool:
        return (not self.dry_run) and self.live_trading_ack == LIVE_ACK_PHRASE

    def assert_live_allowed(self) -> None:
        if self.dry_run:
            return
        if self.live_trading_ack != LIVE_ACK_PHRASE:
            raise RuntimeError(
                "DRY_RUN=false but LIVE_TRADING_ACK is not set to the ack phrase. "
                "Refusing to start in live mode."
            )
        if not self.private_key:
            raise RuntimeError("Live mode requires PRIVATE_KEY in the environment.")


def _bool(v: str) -> bool:
    return v.strip().lower() in ("1", "true", "yes", "on")


def load_settings(env: dict | None = None) -> Settings:
    e = env if env is not None else os.environ
    kw: dict = {}
    mapping = {
        "PRIVATE_KEY": ("private_key", str),
        "SAFE_ADDRESS": ("safe_address", str),
        "CLOB_HOST": ("clob_host", str),
        "DRY_RUN": ("dry_run", _bool),
        "LIVE_TRADING_ACK": ("live_trading_ack", str),
        "MIN_EDGE": ("min_edge", float),
        "MIN_PROB": ("min_prob", float),
        "FEE_RATE_BPS": ("fee_rate_bps", int),
        "PROB_SHRINKAGE": ("prob_shrinkage", float),
        "MAX_SPREAD": ("max_spread", float),
        "DIR_MIN_ELAPSED_S": ("dir_min_elapsed_s", float),
        "VOL_MIN_SAMPLES": ("vol_min_samples", int),
        "BANKROLL": ("bankroll", float),
        "KELLY_MULTIPLIER": ("kelly_multiplier", float),
        "MIN_BET": ("min_bet", float),
        "MAX_BET": ("max_bet", float),
        "MAX_BANKROLL_FRACTION": ("max_bankroll_fraction", float),
        "MAX_DAILY_LOSS": ("max_daily_loss", float),
        "MAX_DAILY_LOSS_PCT": ("max_daily_loss_pct", float),
        "MAX_DRAWDOWN_PCT": ("max_drawdown_pct", float),
        "DAILY_PROFIT_LOCK_PCT": ("daily_profit_lock_pct", float),
        "MAX_CONSECUTIVE_LOSSES": ("max_consecutive_losses", int),
        "MAX_OPEN_EXPOSURE": ("max_open_exposure", float),
        "ENTRY_MIN_REMAINING_S": ("entry_min_remaining_s", float),
        "SETTLE_GRACE_S": ("settle_grace_s", float),
        "POLL_INTERVAL_S": ("poll_interval_s", float),
        "TELEGRAM_BOT_TOKEN": ("telegram_bot_token", str),
        "TELEGRAM_CHAT_ID": ("telegram_chat_id", str),
        "DB_PATH": ("db_path", str),
        "KILL_SWITCH_FILE": ("kill_switch_file", str),
    }
    for env_key, (attr, cast) in mapping.items():
        if env_key in e and e[env_key] != "":
            kw[attr] = cast(e[env_key])
    return Settings(**kw)
