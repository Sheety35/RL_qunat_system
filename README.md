# NSE Intraday Trading System

Intraday algorithmic trading system for NSE indices (NIFTY 50 and NIFTY BANK) using a 3-layer ML architecture with hard guardrails and an ORB signal engine.

Full details: [TRAINING_PLAN.md](TRAINING_PLAN.md)

---

## Architecture

```
NIFTY 50 + NIFTY BANK + 15 sector index CSVs (5-min OHLCV)
                        |
          +-------------+-------------+
          |                           |
  Layer 1: LSTM Classifier     ORB Engine (rules)
  (supervised, softmax)        Signal 1 (11AM) → standalone rule
  Bull / Bear / Flat           Signal 2 (afternoon) → SAC feature
          |                           |
  Layer 2: Trend Detector            |
  Rule-based: UP/DOWN/FLAT           |
          |                           |
          +-------------+-------------+
                        |
                Layer 3: SAC
                Continuous sizing: -1.0 to +1.0
                (short 100% <-> long 100%)
                        |
              Hard Guardrail Layer
              Pure Python rules — overrides model output
                        |
                Upstox Execution API
```

---

## Model summary

| Layer | Type | What it does |
|---|---|---|
| Layer 1 | LSTM Classifier (supervised) | Classifies current market regime (Bull/Bear/Flat) from last 50 candles; softmax head; also outputs confidence probability used by HG11 |
| Layer 2 | Rule-based | Detects trend direction and strength (ADX, EMA, SMA) |
| ORB Engine | Rule-based | Signal 1 (~17/yr) executes as a standalone rule, NOT routed through SAC. Signal 2 (~200/yr) passes `orb_signal2_active` and `orb_signal2_direction` to SAC as features |
| Layer 3 | SAC — single multi-output agent | Two action heads: `action[0]` = NIFTY 50 size, `action[1]` = NIFTY BANK size (each −1.0 to +1.0). Shared capital pool — if total exposure > 80%, both outputs scaled proportionally |
| Guardrails | Pure Python | Hard rules that override any model output (see below) |

---

## Hard guardrails (model cannot override these)

| # | Rule | Trigger | Action |
|---|---|---|---|
| HG1 | Time fence | Before 09:55 or after 15:00 | Block all new entries |
| HG2 | Daily loss halt | Drawdown > 3% in a day | Exit all, halt trading for the day |
| HG3 | Max trade risk | SL distance > 2% of capital | Reduce lot size |
| HG4 | Position cap | Any single instrument > 30% of capital | Block buy |
| HG5 | Total exposure cap | Sum of positions > 80% of capital | Scale both SAC outputs proportionally |
| HG6 | ORB range filter | ORB range > 2% or < 0.1% | Skip ORB signals for the day |
| HG7 | Conflict filter | Signal 1 fired opposite to Signal 2 direction | Skip Signal 2 |
| HG8 | 15:00 hard exit | Clock hits 15:00 | Market-sell everything, no exceptions |
| HG9 | Stop loss required | Any new entry | Reject order if no SL attached |
| HG10 | Extreme volatility scaling | ATR > 3× rolling 20d mean ATR | Halve all new position sizes; disable Signal 2 for the day |
| HG11 | Regime uncertainty gate | `max(regime_softmax_probs) < 0.5` | Cap SAC action magnitude at ±0.3 |

---

## Data

All training uses only the 17 CSVs already in `data/`. No additional data will be pulled.

| File | Rows | Range | Role |
|---|---|---|---|
| NIFTY 50_5minute.csv | 204,167 | 2015–2026 | Primary trading instrument + regime training |
| NIFTY BANK_5minute.csv | 204,157 | 2015–2026 | Secondary trading instrument |
| NIFTY AUTO, ENERGY, FIN SERVICE, FMCG, INFRA, IT, COMMODITIES, CONSUMPTION (+ BANK already above) | 147k–204k each | 2015–2026 | **Sector breadth** (9 long-history indices only — all start 2015) |
| NIFTY ALPHA 50, CPSE, GS COMPSITE, CONSR DURBL, HEALTHCARE, IND DIGITAL, INDIA MFG | 63k–121k | varies | NOT used for breadth (shorter history causes NaNs before 2022) |

Training split (walk-forward, never mixed):
- **Train:** 2019–2022 (bull run, COVID crash, 2022 rate-shock bear)
- **Validate:** 2023 (out-of-sample)
- **Test:** 2024 (touch once at the very end only)

---

## Capital and risk rules

- Starting capital: ₹50,000 (paper money)
- Max position per instrument: 30% of capital
- Max risk per trade: 2% of capital
- Max daily loss: 3% of capital → halt for the day
- No overnight positions — all squared off by 15:00

**Transaction costs and slippage (applied on every order, training and backtest):**
- Brokerage: ₹20 flat per order (Upstox)
- STT: 0.01% of turnover, sell side only
- Exchange charges: 0.005% of turnover
- Slippage: 1 pt average for NIFTY 50, 2 pts for NIFTY BANK
- Formula: `cost = 20 + (value × 0.0001 if SELL) + (value × 0.00005) + (slip_pts × lot_size × lots)`
- Lot sizes: NIFTY 50 = 75, NIFTY BANK = 35

---

## Files

| File | Purpose |
|---|---|
| `data_puller.py` | Upstox OAuth + historical data fetch (data collection complete, not needed again) |
| `orb_engine.py` | ORB signal computation — to be built |
| `guardrails.py` | Hard guardrail rules — to be built |
| `feature_engineering.py` | Compute all indicators + sector breadth, save `features/` — to be built |
| `regime_trainer.py` | Train supervised LSTM Classifier on NIFTY 50 — to be built |
| `sac_trainer.py` | Train single multi-output SAC (NIFTY 50 + NIFTY BANK) — to be built |
| `backtest.py` | Walk-forward evaluation, Sharpe + Sortino output — to be built |
| `live_trader.py` | Paper → live trading via Upstox API — to be built |
| `TRAINING_PLAN.md` | Full architecture, guardrails, all model questions, training pipeline |
| `STOCK_TRADING_GUIDE.md` | Feasibility assessment and 6-month roadmap |

---

## Next steps

1. `orb_engine.py`
2. `guardrails.py`
3. `feature_engineering.py`
4. `regime_trainer.py`
5. `sac_trainer.py`
6. `backtest.py`
7. Paper trade 1 month minimum before live capital
