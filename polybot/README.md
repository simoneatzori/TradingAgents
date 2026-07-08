# PolyBot Production — BTC 5-Minute Up/Down Engine

Deterministic trading engine for Polymarket BTC 5-minute markets.
**No LLM in the execution path.** The agent's job (Hermes / Claude Code) is to
build, audit, and tune this code offline — see `ANALYST_PROMPT.md`.

Design goal: **capital preservation first**. The engine is built so that no
single UTC day can lose more than `MAX_DAILY_LOSS_PCT` of the bankroll
(default **1%**), even counting open positions at their worst case, even
across process restarts.

## Architecture

```
Binance spot ──┐
               ├─> Engine.tick() ─> Strategy (fee gate) ─> Sizing (¼-Kelly,
CLOB book ─────┘        │            capped by remaining daily budget)
                        │                     │
                  Clock (server-         RISK GATE ─> Executor
                  synced windows)             │            │
                        │                     │       DryRun (default)
                        │                     │       or Live CLOB v2
                        └──> settle_expired() ┴──> SQLite store (WAL)
                                    │
                       Reconciler (halt on mismatch)
                       Offline analyst (read-only, nightly)
```

| Module | Responsibility |
|---|---|
| `config.py` | Pydantic v2 settings; DRY_RUN default; live-mode ack gate; full Kelly forbidden; daily loss % hard-capped at 5% |
| `timeutil.py` | Server-synced clock, 5-min window alignment, entry-zone rule (no entries < 20s to close) |
| `fees.py` | CLOB fee model `rate * min(p, 1-p)`; **hard** fee gate; tick rounding (down) |
| `strategy.py` | `PairCostArb` (bounded downside — go live with this first) and `BrownianDirectional` (DRY_RUN data collection only) + EWMA vol with warm-up counter |
| `maker.py` | `MakerPairQuoter`: passive two-sided pair quotes (sum ≤ 1 − MIN_EDGE) with reprice → taker-hedge → hold-with-alert escalation on one-sided fills |
| `sizing.py` | Quarter-Kelly with probability shrinkage toward 0.5, min/max/bankroll-fraction caps, floor rounding |
| `risk_gate.py` | Worst-case daily budget (the 1% rule) + circuit breakers: kill-switch file, daily loss, loss streak, drawdown from high-water mark, open exposure. Deny by default; halts need human reset |
| `engine.py` | Tick orchestration, equal-shares pair execution with unhedged-leg halt, settlement loop, risk-state persistence/restore |
| `executor.py` | DryRun simulator (pessimistic fills) / Live CLOB adapter (FOK orders, Safe `signature_type=2` paired with `funder`) |
| `store.py` | SQLite (WAL): orders (idempotent client IDs), fills, settlements, window open/close prices, persisted risk state |
| `reconciler.py` | Startup + periodic state-vs-exchange check; mismatch = halt |
| `calibration.py` | Rolling Brier / log loss / hit rate |
| `notifier.py` | Telegram alerts with secret redaction (alerts only — control belongs to the external supervisor bot) |

## Capital-protection invariants

1. **1%-per-day worst case.** Every order carries its worst-case loss to the
   risk gate. Realized daily loss + worst-case of open positions + the new
   order must fit in `min(MAX_DAILY_LOSS, bankroll * MAX_DAILY_LOSS_PCT)`.
   Directional sizes are trimmed to the *remaining* budget, so the engine
   uses the budget efficiently but can never overshoot it.
2. **Restart-proof.** Daily PnL, loss streak, open exposure/risk, halt state
   and the equity high-water mark are rebuilt from SQLite at startup.
   Restarting the process cannot reset a limit or clear a halt
   (`python bot.py reset` is the only way, and it asks for confirmation).
3. **Pairs are pairs.** Both arb legs are sized to the SAME share count
   (bounded by the thinner ask). If the second leg fails after the first
   fills, the engine marks the position `unhedged`, **halts**, and alerts —
   a naked binary near settlement is not something to retry programmatically.
4. **Settlement actually happens.** `settle_expired()` runs every loop
   iteration, resolving filled orders against recorded window open/close
   prices, so the daily-loss and streak breakers trip when they should.
   Losing legs of a profitable pair never count toward the loss streak.
5. **Drawdown breaker.** Equity dropping `MAX_DRAWDOWN_PCT` (default 5%)
   below its high-water mark halts the engine for human review.
6. **Optional profit lock.** `DAILY_PROFIT_LOCK_PCT` stops trading for the
   rest of the day once the target is hit (off by default).
7. **Directional guards.** No directional entries until the EWMA vol has
   `VOL_MIN_SAMPLES` observations, `DIR_MIN_ELAPSED_S` of the window has
   elapsed, and the book spread is within `MAX_SPREAD`. Size is also capped
   by displayed liquidity, so FOK orders aren't submitted into thin books.
8. **Live bankroll sync.** In live mode the engine sizes off
   `min(configured bankroll, actual wallet balance)`.
9. **Maker quotes can never rest unattended.** The naked-leg worst case is
   reserved against the daily budget at *quote* time (before anything can
   fill); on a one-sided fill the quoter escalates reprice → taker hedge
   (bounded by `HEDGE_MAX_LOSS_PER_SHARE`) → hold-to-settlement with alert;
   all unfilled quotes are torn down at `QUOTE_CANCEL_REMAINING_S` before
   close, on any halt (`go_flat`), and on restart (stale `open` orders are
   cancelled locally, plus `cancel_all` on-exchange in live mode).

## Quick start (DRY_RUN)

```bash
pip install -r requirements.txt
cp .env.example .env          # edit thresholds if you want
python -m pytest tests/ -q    # all tests must pass
python bot.py run             # paper-trades the live order books
python bot.py status          # PnL, budget left, streak, halt state
```

Run DRY_RUN for **2–4 weeks**. Go-live criteria (all of them):
- positive net expectancy after fees over n ≥ 500 simulated trades
- `pair_cost_arb` only; `brownian_dir` stays in data-collection mode
- measured signal→order latency within budget on your VPS
- circuit breakers verified by actually tripping them

## Going live (deliberately annoying)

1. **Dedicated wallet** funded with bankroll only. Key in env at runtime,
   never in files the agent can read, never in chat with any agent.
2. Audit + pin `py-clob-client` in `requirements.txt`.
3. One-time USDC (ERC-20) + CTF (ERC-1155) allowance approvals
   (`python bot.py approve` — wire to audited contract addresses first).
4. `.env`: `DRY_RUN=false` **and** `LIVE_TRADING_ACK=I_UNDERSTAND_THE_RISKS`.
   Either one alone refuses to start.
5. Verify the market's tie rule (flat close). The engine treats
   close == open as DOWN; confirm against the live market's resolution
   source before trusting DRY_RUN accounting.
6. Run on a VPS, not a laptop. Keep `MAX_BET=2` until the analyst report
   says otherwise.

**Emergency stop:** `touch KILL_SWITCH` in the working directory — next
risk-gate check halts the engine. Restart requires removing the file,
`python bot.py reset`, and a human review of the halt reason.

## What this deliberately does NOT do

- No LLM calls at trade time (latency + nondeterminism = exit liquidity)
- No martingale / DCA-into-losers (averaging down on a 5-minute binary is a
  tail-risk machine, not a strategy)
- No chasing: limit orders at the observed ask, FOK, no repricing loop
- No directional trading live until calibration proves the model
- No "recover today's loss" logic of any kind: when the budget is spent,
  the day is over

## Known limitations (accepted, documented)

- Window open/close prices come from the Binance spot feed sampled at the
  window boundary — an approximation of the market's official resolution
  source. Good enough for DRY_RUN accounting; in live mode the reconciler
  against the exchange is the source of truth.
- DRY_RUN fills are pessimistic (displayed size only, no price improvement)
  but cannot model queue position or being quoted against.

## Compliance note

Polymarket geo-blocks several jurisdictions (Italy included). Operating
through a blocked region risks account restriction and frozen funds.
That risk is yours and no engineering mitigates it.
