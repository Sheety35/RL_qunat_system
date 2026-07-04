# NSE Intraday Trading System

Intraday algorithmic trading system for NSE indices (NIFTY 50 and NIFTY BANK) using a two-phase ML pipeline — feature engineering followed by a SAC reinforcement learning agent — with hard guardrails and an ORB signal engine.

---

## Architecture

```
NIFTY 50 + NIFTY BANK + 9 sector index CSVs (5-min OHLCV)
                        |
              Phase 0: Feature Engineering
              ATR, RSI, MACD, Bollinger, ADX, EMA,
              TWAP, sector breadth, ORB signals
                        |
              +----------+-----------+
              |                      |
        ORB Engine               Phase 3: SAC
        Signal 1 (~17/yr)        Continuous sizing: -1.0 to +1.0
        standalone rule          Two action heads:
        (no SAC involved)          action[0] = NIFTY 50 size
                                   action[1] = NIFTY BANK size
        Signal 2 (~200/yr)       Shared capital pool
        passed as features
        to SAC
              |                      |
              +----------+-----------+
                         |
               Hard Guardrail Layer
               Pure Python rules — overrides all model output
                         |
               Upstox Execution API
```

---

## Pipeline

| Phase | Script | What it does | Status |
|---|---|---|---|
| 0 | `feature_engineering.py` | Reads 17 raw CSVs, computes all indicators, adds ORB signals, saves two enriched parquets to `features/` | Built |
| 3 | `sac_trainer.py` | Trains single multi-output SAC agent (NIFTY 50 + NIFTY BANK) from the parquets | Built |

Run everything:
```
./venv/Scripts/python.exe train.py
```

Run a single phase:
```
./venv/Scripts/python.exe train.py --phase 0   # feature engineering only
./venv/Scripts/python.exe train.py --phase 3   # SAC training only
./venv/Scripts/python.exe train.py --phase 3 --timesteps 200000
```

---

## SAC Agent

**Environment:** `NiftyTradingEnv` (gymnasium.Env)

One episode = one trading day. `reset()` picks the next sequential day; `step(action)` advances one 5-min bar and returns the ATR-normalised P&L reward.

**Observation space (52 features per bar):**

| Group | Features | Count |
|---|---|---|
| NIFTY 50 per-bar | `close_return`, `ema_spread_pct`, `price_vs_sma20`, `rsi14`, `macd_pct`, `atr14_pct`, `bb_width`, `bb_position`, `price_vs_vwap`, `volume_ratio` | 10 |
| NIFTY BANK per-bar | same 10 columns from NBANK parquet | 10 |
| Shared market context | `sector_breadth_norm`, `bank_vs_n50_spread`, `orb_range_pct`, `daily_trend`, `is_high_vol_day`, `atr_percentile_20d`, `monthly_trend`, `dist_from_52w_high`, `orb_signal2_active`, `orb_signal2_dir`, `slot_0`–`slot_5`, `day_of_week`, `minutes_to_close` | 18 |
| Portfolio state | Current N50 position, NBANK position, unrealised P&L, daily P&L, trades today, vol ratio, and 2 spare | 8 |
| Regime placeholder | Zeroed — regime classifier removed | 4 |

**Action space:** `Box([-1,-1], [1,1])` — target allocation fraction per instrument. Magnitudes below dead-band `0.30` are treated as flat.

**Reward:** `pnl_pts / (daily_atr × 0.5)` clipped to `[-3, +3]`. Regime-direction nudge `±0.2` applied when the agent trades against the daily trend.

**Training split (walk-forward, never mixed):**
- Train: `2019-01-01 → 2023-06-30` (includes first-half 2023 calm market alongside 2019–2022 high-vol)
- Validate: `2023-07-01 → 2023-12-31`
- Test: `2024-01-01 → 2024-12-31` — touch once only at the very end

**Key hyperparameters:**

| Param | Value | Reason |
|---|---|---|
| `gamma` | 0.95 | Short episodes (~70 bars/day); 0.99 made early and late bars nearly equal weight |
| `ent_coef` | 0.2 (fixed) | Auto-tuning collapsed to 0.003 by 40k steps every run |
| `learning_rate` | 3e-5 | Smaller LR needed after critic loss oscillation at 3e-4 |
| `learning_starts` | 20,000 | Buffer needs diverse transitions before gradient updates |
| `net_arch` | [128, 128] | Simpler network adequate for single-instrument mode |
| `TOTAL_TIMESTEPS` | 500,000 | 1M caused overfitting to 2019–2022 high-vol regime |

---

## ORB Engine

Computes Opening Range Breakout signals from the first 8 bars of each day (9:15–9:54).

**Signal 1 (~17 trades/year):** 11:05 AM high-conviction breakout. Fires as a standalone rule — SAC is not involved. Two tiers:
- Tier 1: ORB move ≥ 0.70%, ADX ≥ 55, TWAP/ATR ≥ 2.0 → 2% risk
- Tier 2: ORB move ≥ 0.45%, ADX ≥ 45, TWAP/ATR ≥ 1.5 → 1.5% risk

**Signal 2 (~200 trades/year):** Afternoon ORB retest in the 12:00–14:25 window. The flag columns `orb_signal2_active` and `orb_signal2_dir` are passed to the SAC as input features. The agent decides whether and how large to size the trade.

---

## Feature Engineering (Phase 0)

Reads 17 raw CSVs from `data/`, computes the following for each bar, and saves to `features/NIFTY_50_features.parquet` and `features/NIFTY_BANK_features.parquet`.

| Feature | Description |
|---|---|
| `close_return` | 1-bar log return |
| `ema_spread_pct` | (EMA9 − EMA21) / close |
| `price_vs_sma20` | Distance from 20-day SMA |
| `rsi14` | RSI (0–100) |
| `macd_pct` | MACD / close |
| `atr14_pct` | ATR(14) / close |
| `bb_width` | Bollinger Band width / close |
| `bb_position` | (close − lower) / (upper − lower) |
| `price_vs_vwap` | (close − TWAP) / TWAP — uses TWAP since NIFTY is an index with no volume |
| `volume_ratio` | Bar range / rolling 20-bar mean range |
| `adx14`, `di_plus`, `di_minus` | ADX trend strength |
| `daily_trend` | Rule-based: +1 UP / −1 DOWN / 0 FLAT (EMA + ADX) |
| `monthly_trend` | 20-day EMA direction |
| `dist_from_52w_high` | % below rolling 52-week high |
| `atr_percentile_20d` | ATR rank vs 20-day window |
| `is_high_vol_day` | 1 if ATR > mean + 0.5×std (20d) |
| `sector_breadth_norm` | Fraction of 9 sector indices above their own 20-day SMA |
| `bank_vs_n50_spread` | NIFTY BANK return − NIFTY 50 return |
| ORB columns | `orb_high`, `orb_low`, `orb_range_pct`, `orb_valid`, `orb_signal2_active`, `orb_signal2_dir`, `sig1_tier` |
| Time features | `slot_0`–`slot_5` (one-hot 5-min time slot), `day_of_week`, `minutes_to_close` |

---

## Hard Guardrails

All rules live in `guardrails.py`. They override every model output — the SAC cannot bypass them.

| # | Rule | Trigger | Action |
|---|---|---|---|
| HG1 | Time fence | Before 09:55 or at/after 15:00 | Block all new entries |
| HG2 | Daily loss halt | Drawdown > 3% of capital in a day | Exit all positions, halt for the day |
| HG3 | Max trade risk | SL distance > 2% of capital | Reduce lot size |
| HG4 | Position cap | Single instrument > 30% of capital | Block new buy |
| HG5 | Total exposure cap | Sum of positions > 80% of capital | Scale both SAC outputs proportionally |
| HG6 | ORB range filter | ORB range > 2% or < 0.1% | Skip ORB signals for the day |
| HG7 | Conflict filter | Signal 1 fired opposite to Signal 2 direction | Skip Signal 2 |
| HG8 | 15:00 hard exit | Clock hits 15:00 | Market-sell everything, no exceptions |
| HG9 | Stop loss required | Any new entry | Reject if no SL attached |
| HG10 | Extreme volatility | ATR > mean + 0.5×std (20d) | Halve all new position sizes |
| HG11 | Regime uncertainty gate | `regime_max_prob < 0.5` | Cap SAC action to ±0.3 — currently a no-op (regime removed) |

---

## Live Trader

`live_trader.py` implements real-time Signal 2 paper/live trading for NIFTY 50 via the Upstox API.

**How it works:**
1. Every 5 minutes, polls `GET /v2/historical-candle/intraday/NSE_INDEX|Nifty 50/5minute`
2. Computes incremental ATR(14) and TWAP on each new bar
3. Locks the ORB range from 9:15–9:54 bars
4. In the 12:00–14:30 window: fires a trade if close breaks above ORB high (long) or below ORB low (short), confirmed by TWAP
5. One trade per day (`signal_fired` flag prevents re-entry)
6. All entries checked through `guardrails.check_entry()` (HG1, HG2 enforced)
7. SL = entry ± 2×ATR, TP = entry ± 4×ATR, forced EOD exit at 15:00
8. `PAPER_MODE = True` logs to `logs/paper_trades_YYYY-MM-DD.csv`; `False` sends orders via Upstox

**Token setup (each morning):**
```
./venv/Scripts/python.exe get_token.py
# Opens browser → paste redirect URL → token saved to .env automatically
./venv/Scripts/python.exe live_trader.py
```

---

## Data

All training uses only the 17 CSVs in `data/`. No additional data will be pulled.

| File | Rows | Range | Role |
|---|---|---|---|
| `NIFTY 50_5minute.csv` | 204,167 | 2015–2026 | Primary trading instrument |
| `NIFTY BANK_5minute.csv` | 204,157 | 2015–2026 | Secondary trading instrument |
| NIFTY AUTO, ENERGY, FIN SERVICE, FMCG, INFRA, IT, COMMODITIES, CONSUMPTION, BANK | 147k–204k each | 2015–2026 | **Sector breadth** (9 long-history indices used) |
| NIFTY ALPHA 50, CPSE, GS COMPSITE, CONSR DURBL, HEALTHCARE, IND DIGITAL, INDIA MFG | 63k–121k | varies | Not used — shorter history causes NaN gaps before 2022 |

---

## Capital and Risk

- Starting capital: ₹50,000 (paper)
- Max position per instrument: 30% of capital
- Max risk per trade: 2% of capital
- Max daily loss: 3% of capital → halt for the day
- No overnight positions — all squared off by 15:00

**Transaction costs (applied on every order, training and backtest):**

| Cost | Rate |
|---|---|
| Brokerage | ₹20 flat per order |
| STT | 0.01% of turnover, sell side only |
| Exchange charges | 0.005% of turnover |
| Slippage | 1 pt (NIFTY 50) / 2 pts (NIFTY BANK) |
| Lot sizes | NIFTY 50 = 75, NIFTY BANK = 35 |

---

## Files

| File | Purpose | Status |
|---|---|---|
| `train.py` | Master pipeline runner — chains Phase 0 → Phase 3 | Built |
| `feature_engineering.py` | Phase 0: compute all indicators + ORB signals, save parquets | Built |
| `sac_trainer.py` | Phase 3: train multi-output SAC agent | Built |
| `guardrails.py` | Hard guardrail rules HG1–HG11, transaction cost formula | Built |
| `orb_engine.py` | ORB signal computation (Signal 1 and Signal 2 columns) | Built |
| `live_trader.py` | Real-time Signal 2 paper/live trading via Upstox | Built |
| `get_token.py` | Daily Upstox OAuth token refresh | Built |
| `data_puller.py` | Historical data fetch via Upstox (data collection complete, not needed again) | Built |
| `backtest.py` | Walk-forward evaluation on 2024 test set | **Not built** |
| `TRAINING_PLAN.md` | Full architecture notes and training decisions log | Reference |

---

## Next Steps

1. Validate current SAC model: Sharpe > 1.5 and MaxDD < 15% on the 2023-H2 validation set
2. Build `backtest.py` — walk-forward evaluation on 2024 test data (touch once only)
3. Paper trade ≥ 1 month before committing live capital
