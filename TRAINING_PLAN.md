# ML Trading System — Architecture, Guardrails & Training Plan

*Last updated: 2026-05-21*

---

## What data we have (final — no additional data will be pulled)

All data: 5-min OHLCV, columns = `[date, open, high, low, close, volume]`, zero nulls, already in `data/`.

| File | Rows | Range | Role |
|---|---|---|---|
| NIFTY 50_5minute.csv | 204,167 | 2015–2026 | **Primary trading instrument** + regime training |
| NIFTY BANK_5minute.csv | 204,157 | 2015–2026 | **Secondary trading instrument** (most liquid after Nifty 50) + sector feature |
| NIFTY IT_5minute.csv | 204,161 | 2015–2026 | Sector breadth / rotation feature |
| NIFTY AUTO_5minute.csv | 204,087 | 2015–2026 | Sector breadth feature |
| NIFTY FIN SERVICE_5minute.csv | 204,085 | 2015–2026 | Sector breadth feature |
| NIFTY ENERGY_5minute.csv | 204,087 | 2015–2026 | Sector breadth feature |
| NIFTY FMCG_5minute.csv | 204,086 | 2015–2026 | Sector breadth feature |
| NIFTY INFRA_5minute.csv | 204,086 | 2015–2026 | Sector breadth feature |
| NIFTY COMMODITIES_5minute.csv | 191,856 | 2015–2026 | Sector breadth feature |
| NIFTY CONSUMPTION_5minute.csv | 147,045 | 2015–2026 | Sector breadth feature |
| NIFTY ALPHA 50, CPSE, GS COMPSITE | ~121k | 2015–2026 | Secondary breadth features |
| NIFTY CONSR DURBL, HEALTHCARE, IND DIGITAL, INDIA MFG | ~63k | 2022–2026 | Breadth features (shorter history, use from 2022 only) |

**Trading universe (what the model actually trades):** NIFTY 50 and NIFTY BANK — both are highly liquid, have futures available on NSE, and we have 11 years of 5-min data for both. No individual stocks — we train and trade purely on indices with the data we already have.

---

## Full system architecture

```
                    ┌─────────────────────────────────┐
                    │   DATA LAYER (5-min OHLCV)      │
                    │   NIFTY 50 + NIFTY BANK          │
                    │   + 15 sector indices            │
                    └──────────────┬──────────────────┘
                                   │
               ┌───────────────────┼───────────────────┐
               ▼                   ▼                   ▼
    ┌──────────────────┐  ┌─────────────────┐  ┌──────────────────┐
    │  LAYER 1         │  │  LAYER 2        │  │  ORB ENGINE      │
    │  LSTM Classifier │  │  Trend Detector │  │  (Rule-based)    │
    │  Regime          │  │  (Rule-based)   │  │  Signal 1: runs  │
    │  (supervised,    │  │                 │  │  standalone —    │
    │  softmax head)   │  │                 │  │  NOT via SAC     │
    │  Bull/Bear/Flat  │  │  UP/DOWN/FLAT   │  │  Signal 2 → SAC  │
    └────────┬─────────┘  └────────┬────────┘  └───────┬──────────┘
             │                     │                   │
             └─────────────────────┼───────────────────┘
                                   ▼
                    ┌──────────────────────────┐
                    │  LAYER 3: SAC            │
                    │  Single multi-output     │
                    │  action[0]: NIFTY 50     │
                    │  action[1]: NIFTY BANK   │
                    │  Each: -1.0 to +1.0      │
                    │  Shared capital pool     │
                    └──────────────┬───────────┘
                                   │
                    ┌──────────────▼───────────┐
                    │  HARD GUARDRAIL LAYER    │  ← cannot be overridden by any model
                    │  (pure Python rules)     │
                    └──────────────┬───────────┘
                                   │
                    ┌──────────────▼───────────┐
                    │  EXECUTION LAYER         │
                    │  Position sizer          │
                    │  Order to Upstox API     │
                    └──────────────────────────┘
```

---

## Layer 1: LSTM Regime Classifier

### What it answers
*"Who is ruling the market right now? Is it bulls, bears, or is nobody in control?"*

### Inputs (last 50 candles = ~4 hours of 5-min data)
- NIFTY 50 OHLCV (normalised)
- ADX (14-period) — trend strength
- ATR (14-period) — volatility
- Distance of close from 20d SMA (%)
- EMA9 vs EMA21 spread
- Sector breadth: count of 9 long-history sector indices above their 20d SMA (normalised 0–1)
- Volume ratio: current candle volume / 20-period avg volume

### Output (discrete — softmax, 3 classes)
| Class | Meaning | Effect on SAC |
|---|---|---|
| 0 — High Volatility Bear | Strong downtrend, high fear | SAC can only go short or flat |
| 1 — Flat / Choppy | No clear direction, low momentum | SAC uses tighter sizing, ORB Signal 2 required |
| 2 — High Volatility Bull | Strong uptrend, high momentum | SAC can go long or flat |

The classifier also outputs raw softmax probabilities. If `max(probabilities) < 0.5`, the regime is treated as uncertain — see HG11 in guardrails.

### Why supervised LSTM, not PPO
Regime classification is a **3-class supervised prediction task**, not an RL problem. PPO would be the wrong algorithm here — there is no reward signal to optimise, only labelled training data. Using:
- LSTM backbone (retains memory across the 50-candle sequence window)
- Softmax output head (3 classes)
- CrossEntropyLoss
- Adam optimiser
- Standard train/val loop with early stopping

### Training data
NIFTY 50 from 2015 (204,167 bars). Covers 2015–2016 chop, 2017 bull run, 2018 correction, 2020 COVID crash, 2021 recovery, 2022 rate-shock bear, 2023–2025 recovery. Excellent multi-regime coverage.

Label creation:
- Rule-based seed: ADX > 30 + close > 20d SMA = Bull; ADX > 30 + close < 20d SMA = Bear; ADX < 20 = Flat
- Train LSTM classifier on these seed labels with cross-entropy loss

---

## Layer 2: Trend Detector (Mostly Rule-based)

### What it answers
*"Is this stock/market in an up cycle, down cycle, or going sideways? How strong is the trend?"*

This is intentionally kept **rule-based** rather than ML — it's cheaper, faster, and the rules are well-tested by decades of technical analysis. The output becomes a feature for SAC.

### Rules
```python
# Daily trend (computed once pre-market)
TREND = "UP"   if yesterday_close > sma_20d else "DOWN"

# Intraday trend (updated every bar)
EMA9  = EMA(close, 9)
EMA21 = EMA(close, 21)
intraday_trend = "UP"   if EMA9 > EMA21 else "DOWN"

# Trend strength
ADX = ADX(14)
strength = "STRONG"   if ADX >= 35 else "MODERATE" if ADX >= 20 else "CHOPPY"

# Cycle position (where are we in the move?)
rsi_14 = RSI(14)
cycle  = "OVERBOUGHT" if rsi_14 > 70 else "OVERSOLD" if rsi_14 < 30 else "NEUTRAL"
```

### Output features fed to SAC
`[daily_trend_encoded, intraday_trend_encoded, adx, ema_spread_pct, rsi, cycle_encoded]`

---

## Layer 3: SAC — Single Multi-Output Agent

One SAC agent with **two continuous action outputs**:
- `action[0]` = NIFTY 50 position size (−1.0 to +1.0)
- `action[1]` = NIFTY BANK position size (−1.0 to +1.0)

Both outputs share the same capital pool. If `|action[0]| + |action[1]|` would exceed 80% total exposure, both are scaled down proportionally before execution. This forces the agent to compete for capital across instruments rather than treating them independently.

### What it answers — the 5 core questions

#### Q1: When to enter the market?
Entry is valid when ALL of:
1. Time is in an allowed window (see Q4)
2. ORB signal is present OR technical setup meets minimum thresholds (ADX ≥ 20, VWAP alignment)
3. Regime (Layer 1) is not contradicting the direction (no longs in Bear regime)
4. Trend (Layer 2) aligns with the signal direction
5. SAC outputs |action| > 0.10 (dead-band filter — ignore tiny signals)
6. No hard guardrail is blocking (daily loss limit not hit, no existing position in same direction)

#### Q2: When to exit?
Exit triggers (first one that fires wins):
1. SAC outputs an action in the opposite direction → flip/reduce
2. Hard stop loss hit (set at entry)
3. Trailing stop loss triggered
4. Profit target reached (ORB formula: TP = entry + 1.2 × SL_distance for Signal 1, 3.0 × ATR for Signal 2)
5. **Hard time exit at 15:00** — all positions squared off, no exceptions
6. Regime flips from Bull → Bear mid-trade → immediate exit
7. Daily loss limit hit → exit all and halt

#### Q3: Stop loss and trailing stop loss

**Initial SL (set at entry, never moved against you):**
```
Normal day (ATR ≤ 20d mean + 0.5σ):
    SL = entry ± 1.5 × ATR(14)   [short: entry + 1.5×ATR, long: entry - 1.5×ATR]

High-vol day (ATR > 20d mean + 0.5σ):
    SL = entry ± 2.5 × ATR(14)

ORB Signal 1 (11 AM):
    SL = extreme of 10AM–11AM range ± 0.2×ATR

Never risk more than 2% of capital on a single trade:
    max_SL_distance = (capital × 0.02) / lot_size
    SL = min(ATR-based_SL, max_SL_distance)
```

**Trailing SL (activates once in profit):**
```
Once price moves +1×ATR in your favour → move SL to breakeven
Once price moves +2×ATR in your favour → trail SL at +1×ATR above entry (longs)
Once price moves +3×ATR in your favour → trail SL at +2×ATR above entry
```

**Hard daily stop:**
```
If today's realised + unrealised loss > 3% of capital → exit everything, no new trades today
```

#### Q4: What time does the model enter?

```
09:15 – 09:55  →  FORBIDDEN. Observation only. Compute ORB_HIGH/LOW, ATR, VWAP.
09:55 – 11:00  →  No new entries unless a very strong gap-and-go setup (handled by guardrail)
11:00 – 11:05  →  Check Signal 1 (11AM high-conviction). ENTER at 11:05 if conditions met.
11:05 – 12:00  →  Hold Signal 1 position OR wait for Signal 2 window
12:00 – 14:30  →  Signal 2 window. Enter ORB afternoon breakout if Signal 2 conditions met.
14:30 – 15:00  →  No new entries. Only exits allowed.
15:00           →  HARD EXIT. All positions closed.
15:00 – 15:30  →  Forbidden entirely.
```

SAC receives a `time_slot` feature (one-hot encoded 0-5 for the windows above). The reward function gives heavy negative reward for any action in forbidden windows. The model learns this hard.

#### Q5: What time does the model exit?

Exit can happen any time after a position is open, driven by:
- SL/TSL hit → immediate exit (any time during 9:15–15:00)
- TP hit → immediate exit
- Regime flip → immediate exit
- Daily loss limit → immediate exit
- **Mandatory 15:00 hard exit** — this is a rule, not a model decision

If position is still open at 14:55, a warning is logged. At 15:00, a hard exit fires regardless.

---

## ORB (Opening Range Breakout) Engine

This is a **deterministic rule layer** that runs in parallel, not inside the SAC model.

**Signal 1 executes as a fully standalone rule — it is NOT routed through SAC.** Signal 1 fires only ~17 times per year (~68 training examples over 4 years), which is far too sparse for SAC to learn from. It has its own entry/exit logic, uses fixed risk sizing, and calls the guardrail-checked order function directly. `orb_signal1_active` is NOT in the SAC feature vector.

Signal 2 provides binary signals (`orb_signal2_active`, `orb_signal2_direction`) that ARE passed to SAC as features.

### Pre-market setup (before 9:15, daily)
```
ORB_HIGH = max(high) of 9:15–9:55 candles
ORB_LOW  = min(low)  of 9:15–9:55 candles
ORB_RANGE_PCT = (ORB_HIGH - ORB_LOW) / open_9:15 × 100

Normal day: skip if ORB_RANGE_PCT > 1% or < 0.1%
High-vol day (ATR > 20d_mean_ATR + 0.5σ): allow ORB_RANGE_PCT up to 2%, use 2.5×ATR stops
DAILY_TREND = UP if yesterday_close > 20d_SMA else DOWN
```

### Signal 1 — 11AM High-Conviction (~17 trades/year) — STANDALONE, NOT via SAC
```
At 11:00 candle:
    MOVE_PCT = |close - open| / open × 100

Tier 1 (risk 2% capital):
    MOVE_PCT ≥ 0.70% AND ADX ≥ 55 AND VWAP_dist ≥ 2.0×ATR

Tier 2 (risk 1.5% capital):
    MOVE_PCT ≥ 0.45% AND ADX ≥ 45 AND VWAP_dist ≥ 1.5×ATR

Enter at 11:05  |  SL = extreme of 10AM–11AM range ± 0.2×ATR
TP = entry + 1.2 × SL_distance  |  Hard exit: 15:00

Execution: orb_engine.py calls the guardrail-checked order function directly.
SAC is not involved. orb_signal1_active is NOT a SAC input feature.
```

### Signal 2 — ORB Afternoon (~200 trades/year)
```
Entry window: 12:00 – 14:30

LONG  when: DAILY_TREND=UP  AND close > ORB_HIGH AND EMA9 > EMA21 AND ADX ≥ 20 AND close > VWAP
SHORT when: DAILY_TREND=DOWN AND close < ORB_LOW  AND EMA9 < EMA21 AND ADX ≥ 20 AND close < VWAP

Conflict filter: if Signal 1 fired today in OPPOSITE direction → skip Signal 2

SL = entry ± 1.5×ATR (normal) or ±2.5×ATR (high-vol)
TP = entry + 3.0×ATR (normal) or 4.0×ATR (high-vol)
Hard exit: 15:00
```

### Position sizing (both signals)
```python
lots = max(1, round((capital × risk_pct) / (sl_distance_pts × 25)))
# Example: ₹50,000 × 1% / (20 pts × 25) = 1 lot
```

---

## Guardrails

Guardrails are split into two tiers. **Hard guardrails** are pure Python rules — the model output is IGNORED if a hard guardrail fires. **Soft guardrails** shape the reward function so the model learns to avoid bad behaviour on its own.

### Transaction costs and slippage (applied on every order in training env and backtest)

```python
BROKERAGE_PER_ORDER    = 20       # ₹20 flat per order (Upstox)
STT_SELL_SIDE          = 0.0001   # 0.01% of turnover, sell side only
EXCHANGE_CHARGES       = 0.00005  # 0.005% of turnover
SLIPPAGE_NIFTY50_PTS   = 1.0      # 1 point average slippage
SLIPPAGE_NIFTYBANK_PTS = 2.0      # 2 points average slippage

def compute_transaction_cost(instrument, order_value_inr, lots, side):
    brokerage   = BROKERAGE_PER_ORDER
    stt         = order_value_inr * STT_SELL_SIDE if side == "SELL" else 0
    exchange    = order_value_inr * EXCHANGE_CHARGES
    lot_size    = 75 if instrument == "NIFTY50" else 35
    slip_pts    = SLIPPAGE_NIFTY50_PTS if instrument == "NIFTY50" else SLIPPAGE_NIFTYBANK_PTS
    slippage    = slip_pts * lot_size * lots
    return brokerage + stt + exchange + slippage
```

Do NOT use a percentage approximation. Use this formula in both the SAC training environment and `backtest.py`. Subtract the result from P&L on every order.

### Hard guardrails (override the model, no exceptions)

| # | Rule | Trigger | Action |
|---|---|---|---|
| HG1 | Time fence | Before 09:55 or after 15:00 | Reject any new entry order |
| HG2 | Daily loss halt | Realised + unrealised loss > 3% capital in one day | Exit all, block new entries for rest of day |
| HG3 | Single trade max loss | SL distance > 2% of capital | Reduce lot size to fit within 2% risk |
| HG4 | Position concentration | Either instrument > 30% of capital | Block buy, force partial exit if already exceeded |
| HG5 | Total portfolio exposure | Sum of all open positions > 80% of capital | Scale both SAC outputs down proportionally |
| HG6 | ORB range filter | ORB_RANGE_PCT > 2% or < 0.1% | Skip ORB signals for the day |
| HG7 | Conflict filter | Signal 1 fired opposite to Signal 2 direction | Skip Signal 2 |
| HG8 | 15:00 hard exit | Clock hits 15:00 | Market-sell all open positions immediately |
| HG9 | Stop loss mandatory | Any new entry order | Reject if no SL attached |
| HG10 | Extreme volatility scaling | Current ATR > 3 × rolling 20d mean ATR | Halve all new position sizes; disable Signal 2 for the day. Log: `OOD_HIGH_VOL: ATR {x:.1f} > 3x mean {y:.1f}` |
| HG11 | Regime uncertainty gate | `max(regime_softmax_probs) < 0.5` (no class has majority confidence) | Clip SAC action to ±0.3. Log: `OOD_REGIME_UNCERTAIN: max_prob={p:.2f}` |

HG10 and HG11 are **out-of-distribution guards** — they fire when the market is in a state the models have not seen reliably. Both must be checked BEFORE any order is sent to Upstox.

### Soft guardrails (reward function shaping)

These teach the model good habits during training. The reward function for SAC:

```python
def reward(action, outcome, context):
    r = 0.0

    # Core P&L reward (asymmetric — losses hurt more)
    pnl_pct = outcome.pnl / context.capital * 100
    if pnl_pct >= 0:
        r += pnl_pct * 1.0          # +1 point per 1% gain
    else:
        r += pnl_pct * 1.5          # -1.5 points per 1% loss

    # Catastrophic loss penalty
    if pnl_pct <= -5.0:
        r -= 100                    # -100 for a 5%+ single-trade loss

    # Realistic transaction cost (always subtracted — see formula below)
    r -= outcome.transaction_cost_inr / context.capital

    # NO inactivity penalty — SAC has enough signal to learn when to trade.
    # Forcing trades on bad days would hurt performance. Removed.

    # Overtrading penalty
    if context.trades_today > 10:
        r -= 2 * (context.trades_today - 10)

    # Time-window violation
    if context.is_forbidden_window and action != 0:
        r -= 50

    # Regime alignment bonus
    if context.regime == 2 and action[0] > 0:   # Bull regime, going long N50
        r += 1
    if context.regime == 0 and action[0] < 0:   # Bear regime, going short N50
        r += 1

    # Position concentration penalty (single multi-output SAC)
    total_exposure = abs(action[0]) + abs(action[1])
    if total_exposure > 0.80:
        r -= (total_exposure - 0.80) * 10

    # Drawdown scaling penalty
    if context.current_drawdown_pct > 2.0:
        r -= context.current_drawdown_pct * 2

    # Sharpe bonus (end-of-episode)
    if context.episode_done:
        r += context.episode_sharpe * 5
        r += context.episode_sortino * 3

    return r
```

---

## Training pipeline (step by step)

### Phase 0 — Feature engineering (before any training)

Build a single feature matrix for each stock + NIFTY 50 context:

```
Per-bar features (5-min):
  price:    open, high, low, close, volume
  trend:    ema9, ema21, ema_spread, sma20, price_vs_sma20
  momentum: rsi14, roc5, roc20, macd, macd_signal
  vol:      atr14, bollinger_width, volume_ratio (vs 20-bar avg)
  vwap:     vwap, price_vs_vwap_pct
  market:   nifty50_return_1h, nifty50_vs_sma20, bank_nifty_vs_nifty50_spread,
            sector_breadth (int 0–9, then normalised 0.0–1.0),
            nifty_bank_return_1h
            # sector_breadth uses ONLY these 9 long-history indices:
            # NIFTY AUTO, BANK, ENERGY, FIN SERVICE, FMCG, INFRA, IT,
            # COMMODITIES, CONSUMPTION  (all have data from 2015)
            # DO NOT include CONSR DURBL, HEALTHCARE, IND DIGITAL, INDIA MFG
            # (those only start 2022, causing NaNs in training data)

Daily (computed once before 9:15, repeated for all intraday bars of that day):
  orb_high, orb_low, orb_range_pct, daily_trend, is_high_vol_day
  prev_atr, atr_percentile_20d
  # Cycle position features (FIX 8):
  monthly_trend        # 1 if close > 200d SMA (200 days × 75 bars), else 0
  dist_from_52w_high   # (52w_high - close) / 52w_high  →  0=at high, higher=far below
  dist_from_52w_low    # (close - 52w_low)  / 52w_low   →  0=at low, higher=far above

Session features:
  time_slot (0–5 for the 6 trading windows defined in Q4)
  day_of_week (0–4)
  minutes_to_close (decreasing from 375 to 0)

Portfolio state:
  cash_ratio, current_position_n50, current_position_nbank, unrealised_pnl_pct
  trades_today, daily_pnl_pct, drawdown_from_peak
  regime (0/1/2 from Layer 1 LSTM), regime_max_prob (softmax confidence)
  daily_trend_encoded, intraday_trend_encoded, adx, ema_spread_pct, rsi
  orb_signal2_active, orb_signal2_direction   # Signal 1 NOT included — it's standalone
```

**Regime lag (FIX 2 — prevents leakage):**
When saving the enriched parquet, the regime column is shifted by 1 bar:
```python
features_df['regime'] = regime_predictions.shift(1)
```
The regime label at bar t must use only data available BEFORE bar t. This is applied in `feature_engineering.py` before saving both parquet files.

### Phase 1 — Train LSTM Regime Classifier (supervised)

**Data:** NIFTY 50 2015–2022 (train), 2023–2024 (validate), 2025 (test)
**Steps:**
1. Compute ADX, SMA20, ATR, EMA9/21, sector breadth (9 indices) on NIFTY 50
2. Create rule-based seed labels: Bull/Bear/Flat (ADX + SMA20 rules)
3. Train LSTM with softmax head using CrossEntropyLoss + Adam; early stopping on val loss
4. Evaluate: confusion matrix, per-class accuracy, calibration on 2023–2024
5. Save model weights + input scaler

Expected accuracy target: > 70% on the validation set

### Phase 2 — Enrich NIFTY 50 and NIFTY BANK data

For each of the two trading instruments:
1. Compute all per-bar features (indicators, VWAP, sector breadth, time features)
2. Run Phase 1 PPO-LSTM to stamp `regime` label on every bar
3. Compute ORB signals for every trading day from the same CSV
4. Save enriched parquet: `features/NIFTY_50_features.parquet`, `features/NIFTY_BANK_features.parquet`

Sector breadth feature: for each 5-min bar, look up the 9 long-history sector CSVs (NIFTY AUTO, BANK, ENERGY, FIN SERVICE, FMCG, INFRA, IT, COMMODITIES, CONSUMPTION), check if each is above its rolling 20d SMA, sum the count (0–9). This gives the SAC a "how many sectors are bullish right now" signal.

### Phase 3 — Train SAC

**What the SAC trades:** One multi-output SAC agent — `action[0]` for NIFTY 50, `action[1]` for NIFTY BANK. Both share a capital pool. If total exposure exceeds 80%, both outputs are scaled proportionally before execution.

**Data split (walk-forward — NEVER mix):**
```
Train:    2019-01-01 to 2022-12-31  (bull run, COVID crash, 2022 rate-shock bear — all regimes)
Validate: 2023-01-01 to 2023-12-31  (recovery market — out-of-sample)
Test:     2024-01-01 to 2024-12-31  (unseen — touch only once at the very end)

Note: 2015–2018 data is used only for regime classifier training (Phase 1).
      SAC training starts 2019 so there's enough pre-computed indicator history.
```

**Training process:**
1. Simulate trading day-by-day using the enriched parquet
2. SAC observes the full feature vector each 5-min bar
3. Hard guardrails enforced in the environment (bad actions penalised and blocked)
4. SAC outputs a continuous value −1.0 to +1.0 for each instrument
5. Target network updated every 1,000 steps; checkpoint every 10,000 steps

**Benchmark (must beat all three):**
- Buy-and-hold NIFTY 50 (passive)
- Buy-and-hold NIFTY BANK (passive)
- Pure ORB rule strategy (no ML, using the same signals)

**Success criteria (on validation set):**
- Sharpe ratio > 1.5
- Sortino ratio > 2.0
- Max drawdown < 15%
- Win rate > 52%
- Annual return > NIFTY 50 annual return

### Phase 4 — End-to-end evaluation and paper trading

1. Load Phase 1 (LSTM Classifier) + Phase 3 (SAC) + ORB engine + guardrails together
2. Run final evaluation on the **test set (2024)** — touch only once
3. Paper trade for 1 month on Upstox paper account before any live capital
4. Only after paper trade shows Sharpe > 1.0 and max drawdown < 15% → consider live capital

---

## Metrics to track (every training run)

| Metric | Target | Alarm if |
|---|---|---|
| Sharpe ratio | > 1.5 | < 0.8 |
| Sortino ratio | > 2.0 | < 1.0 |
| Max drawdown | < 15% | > 20% |
| Win rate | > 52% | < 45% |
| Avg trade duration | 30–120 min | < 5 min (overtrading) |
| Trades per day | 1–5 | > 10 |
| % days no trade | < 20% | > 40% |
| Regime accuracy (LSTM Classifier) | > 70% | < 60% |

Always show Sharpe AND Sortino together. Sharpe penalises all volatility; Sortino only penalises downside volatility — they tell different stories.

---

## File structure (target)

```
stocks/
├── data/                              # raw 5-min CSVs — DO NOT MODIFY
│   ├── NIFTY 50_5minute.csv           # primary trading instrument
│   ├── NIFTY BANK_5minute.csv         # secondary trading instrument
│   ├── NIFTY IT_5minute.csv           # sector breadth feature
│   ├── NIFTY AUTO_5minute.csv
│   └── ... (13 more sector CSVs)
│
├── features/                          # Phase 0 output — enriched, model-ready
│   ├── NIFTY_50_features.parquet      # all indicators + regime + ORB for NIFTY 50
│   └── NIFTY_BANK_features.parquet    # same for NIFTY BANK
│
├── models/
│   ├── regime_classifier/             # Phase 1 — LSTM Classifier (supervised)
│   │   ├── best_model.pt
│   │   └── scaler.pkl
│   └── sac_multi/                     # Phase 3 — single multi-output SAC
│       ├── best_model.zip             # action[0]=N50, action[1]=NBank
│       └── metadata.json
│
├── feature_engineering.py             # Phase 0 — build features/ with 1-bar regime lag
├── regime_trainer.py                  # Phase 1 — train LSTM Classifier (supervised)
├── sac_trainer.py                     # Phase 3 — train single multi-output SAC
├── backtest.py                        # walk-forward evaluation + metrics
├── live_trader.py                     # paper → live trading via Upstox
├── guardrails.py                      # hard guardrail rules (imported by all)
├── orb_engine.py                      # ORB signal computation (imported by all)
└── data_puller.py                     # (archived — no more data to pull)
```

---

## Immediate next steps

Data collection is done. All training uses only what is in `data/`.

1. Write `orb_engine.py` — ORB signal computation, shared by everything downstream
2. Write `guardrails.py` — hard guardrail rules as a standalone, model-independent module
3. Write `feature_engineering.py` — compute all indicators + sector breadth, save `features/`
4. Write `regime_trainer.py` — train supervised LSTM Classifier on NIFTY 50 (2015–2024)
5. Write `sac_trainer.py` — SAC environment + training loop (Stable-Baselines3)
6. Write `backtest.py` — walk-forward evaluation, Sharpe + Sortino output
7. Paper trade for at least 1 month before putting any real capital in
