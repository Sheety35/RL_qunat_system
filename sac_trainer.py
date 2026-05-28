"""
Phase 3: SAC — Single multi-output agent (NIFTY 50 + NIFTY BANK).

Environment:  NiftyTradingEnv  (gymnasium.Env)
  Observation: 52 normalised features per bar
  Action:      Box([-1,-1], [1,1])  — target allocation for each instrument
  Episode:     One trading day (9:15 → 15:00, hard exit enforced)

Training split (walk-forward, never mixed):
  Train:    2019-01-01 → 2022-12-31
  Validate: 2023-01-01 → 2023-12-31
  Test:     2024-01-01 → 2024-12-31  (touch once at the very end only)

Run standalone or imported by train.py.
"""

from __future__ import annotations
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, BaseCallback
)
from stable_baselines3.common.monitor import Monitor

from guardrails import (
    Guardrails, compute_transaction_cost,
    LOT_SIZE_N50, LOT_SIZE_NBANK
)

warnings.filterwarnings("ignore", category=UserWarning)

FEATURES_DIR = Path("features")
MODELS_DIR   = Path("models") / "sac_multi"

CAPITAL      = 50_000.0
TRAIN_START  = "2019-01-01"
TRAIN_END    = "2023-06-30"   # extended — includes first-half 2023 calm market
VAL_START    = "2023-07-01"  # second half 2023 stays out-of-sample
VAL_END      = "2023-12-31"

# SB3 SAC hyperparameters
SAC_KWARGS = dict(
    learning_rate   = 3e-5,
    buffer_size     = 100_000,
    learning_starts = 20_000,
    batch_size      = 512,
    tau             = 0.005,
    gamma           = 0.95,        # lower gamma for short episodes (~70 bars/day)
    train_freq      = 1,
    gradient_steps  = 1,
    ent_coef        = 0.2,          # fixed — auto-tuning collapses to 0.06 by 40k steps every run
    policy_kwargs   = dict(
        net_arch=[128, 128],
        optimizer_kwargs=dict(eps=1e-5),
    ),
)
TOTAL_TIMESTEPS = 500_000   # 1M caused overfitting to 2019-2022 vol regime

# Dead-band: actions below this magnitude are treated as flat
DEAD_BAND = 0.30   # raised from 0.20 — stops max-action exploit


# ── Observation feature list ──────────────────────────────────────────────────
# These columns must exist in the parquets after Phases 0–2.
OBS_COLS_N50 = [
    "close_return", "ema_spread_pct", "price_vs_sma20", "rsi14",
    "macd_pct", "atr14_pct", "bb_width", "bb_position",
    "price_vs_vwap", "volume_ratio",
]
OBS_COLS_NBANK = [
    "close_return", "ema_spread_pct", "price_vs_sma20", "rsi14",
    "macd_pct", "atr14_pct", "bb_width", "bb_position",
    "price_vs_vwap", "volume_ratio",
]
OBS_SHARED = [
    "sector_breadth_norm",
    "bank_vs_n50_spread",          # in N50 parquet
    "orb_range_pct",
    "daily_trend",
    "is_high_vol_day",
    "atr_percentile_20d",
    "monthly_trend",
    "dist_from_52w_high",
    "orb_signal2_active",
    "orb_signal2_dir",
    "slot_0", "slot_1", "slot_2", "slot_3", "slot_4", "slot_5",
    "day_of_week",
    "minutes_to_close",
]
# Portfolio state (8) + regime (4) = 12 extra features
# Total: 10 + 10 + 18 + 12 = 50  (plus we clip/pad to 52)
OBS_SIZE = len(OBS_COLS_N50) + len(OBS_COLS_NBANK) + len(OBS_SHARED) + 12


# ── Helper ────────────────────────────────────────────────────────────────────

def _safe(x, fallback=0.0):
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return fallback
    if isinstance(x, np.floating) and (np.isnan(x) or np.isinf(x)):
        return fallback
    return float(x)


# ── Environment ───────────────────────────────────────────────────────────────

class NiftyTradingEnv(gym.Env):
    """
    One episode = one trading day.
    reset() picks the next sequential day from the date list.
    step(action) advances one 5-min bar and returns reward.
    """

    metadata = {"render_modes": []}

    def __init__(self, n50_df: pd.DataFrame, nbank_df: pd.DataFrame,
                 start: str, end: str, capital: float = CAPITAL,
                 randomise_start: bool = False, single_instrument: bool = True):

        super().__init__()
        self._capital           = capital
        self._randomise         = randomise_start
        self._single_instrument = single_instrument

        # Slice to date range
        self.n50   = n50_df[(n50_df.index >= start) & (n50_df.index <= end)].copy()
        self.nbank = nbank_df[(nbank_df.index >= start) & (nbank_df.index <= end)].copy()

        # Fill missing cols with 0 to avoid KeyError at runtime
        for col in OBS_COLS_N50 + OBS_SHARED + ["close", "atr14", "regime",
                                                  "regime_max_prob", "time_slot"]:
            if col not in self.n50.columns:
                self.n50[col] = 0.0
        for col in OBS_COLS_NBANK + ["close", "atr14", "bank_vs_n50_spread"]:
            if col not in self.nbank.columns:
                self.nbank[col] = 0.0

        # Align NIFTY BANK to NIFTY 50 index (forward-fill missing bars)
        self.nbank = self.nbank.reindex(self.n50.index, method="ffill").fillna(0)

        # Pre-fill NaN features
        feat_cols = OBS_COLS_N50 + OBS_COLS_NBANK + OBS_SHARED
        self.n50[OBS_COLS_N50 + OBS_SHARED]  = (
            self.n50[OBS_COLS_N50 + OBS_SHARED].ffill().bfill().fillna(0))
        self.nbank[OBS_COLS_NBANK] = self.nbank[OBS_COLS_NBANK].ffill().bfill().fillna(0)

        # Build list of trading days
        self._all_dates = sorted(set(self.n50.index.date))
        self._day_idx   = 0

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(OBS_SIZE,), dtype=np.float32)

        # Episode state (initialised in reset)
        self._bar_idx      = 0
        self._day_bars     = None   # list of row indices for current day
        self._nbank_bars   = None
        self._capital      = capital
        self._pos_n50      = 0.0
        self._pos_nbank    = 0.0
        self._entry_n50    = 0.0
        self._entry_nbank  = 0.0
        self._sl_n50       = 0.0
        self._sl_nbank     = 0.0
        self._peak_n50     = 0.0   # peak favourable move from entry (price pts)
        self._peak_nbank   = 0.0
        self._daily_pnl    = 0.0
        self._trades_today = 0
        self._bars_since_entry_n50   = 0
        self._bars_since_entry_nbank = 0
        self._max_capital  = capital
        self._gr           = Guardrails(capital)

        # BUG 1 FIX: populate _day_bars before SB3 ever touches the env
        self._day_idx = -1   # reset() will increment to 0
        self.reset()         # populate _day_bars immediately

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Pick next (or random) day
        if self._randomise:
            self._day_idx = self.np_random.integers(0, len(self._all_dates))
        else:
            self._day_idx = (self._day_idx + 1) % len(self._all_dates)

        date = self._all_dates[self._day_idx]
        mask = self.n50.index.date == date
        self._day_bars   = self.n50.index[mask].tolist()
        self._nbank_bars = self.nbank.index[mask].tolist()
        self._bar_idx    = 0

        # Reset position state
        self._pos_n50    = 0.0
        self._pos_nbank  = 0.0
        self._entry_n50  = 0.0
        self._entry_nbank= 0.0
        self._sl_n50     = 0.0
        self._sl_nbank   = 0.0
        self._peak_n50   = 0.0
        self._peak_nbank = 0.0
        self._daily_pnl  = 0.0
        self._trades_today = 0
        self._bars_since_entry_n50   = 0
        self._bars_since_entry_nbank = 0

        # Reset guardrails for the day
        row = self.n50.loc[self._day_bars[0]]
        self._gr.reset_day(
            orb_range_pct = _safe(row.get("orb_range_pct", 0)),
            daily_atr     = _safe(row.get("prev_atr", 0)),
            atr_mean_20d  = _safe(row.get("atr_mean_20d", 1), 1),
            atr_std_20d   = _safe(row.get("atr_std_20d",  0.1), 0.1),
        )
        # Belt-and-braces: explicitly zero out guardrail state that must not bleed
        self._gr.pos_n50      = 0.0
        self._gr.pos_nbank    = 0.0
        self._gr.realized_pnl = 0.0
        self._gr.daily_loss   = 0.0
        self._gr.daily_halted = False
        self._gr.trades_today = 0

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        assert self._day_bars is not None, "Call reset() before step()"
        if self._bar_idx >= len(self._day_bars):
            # Normal end of day — signal truncated, don't silently reset
            obs = np.zeros(OBS_SIZE, dtype=np.float32)
            return obs, 0.0, False, True, {}

        idx      = self._day_bars[self._bar_idx]
        nbank_idx = (self._nbank_bars[self._bar_idx]
                     if self._bar_idx < len(self._nbank_bars)
                     else self._nbank_bars[-1])
        row_n50  = self.n50.loc[idx]
        row_nb   = self.nbank.loc[nbank_idx]

        close_n50  = _safe(row_n50["close"], 22_000)
        close_nb   = _safe(row_nb["close"],  47_000)
        atr_n50    = _safe(row_n50["atr14"], close_n50 * 0.005)
        atr_nb     = _safe(row_nb["atr14"],  close_nb  * 0.005)
        time_slot  = int(_safe(row_n50.get("time_slot", 0)))
        regime     = int(_safe(row_n50.get("regime", 1)))
        reg_prob   = _safe(row_n50.get("regime_max_prob", 0.34))
        hour       = idx.hour
        minute     = idx.minute

        # ── Guardrail modifications (before action) ───────────────────────
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        if self._single_instrument:
            action[1] = 0.0  # NIFTY BANK disabled in single-instrument mode
        action = self._gr.apply_hg11(action, reg_prob)
        action = self._gr.apply_hg10(action)
        action = self._gr.apply_hg5(action)

        # Dead-band filter
        action[0] = 0.0 if abs(action[0]) < DEAD_BAND else action[0]
        action[1] = 0.0 if abs(action[1]) < DEAD_BAND else action[1]

        # Time-window restrictions (HG1, soft)
        if time_slot == 0 or hour >= 15:
            action = np.zeros(2, dtype=np.float32)
        elif time_slot == 5:
            # Only allow moves toward flat
            if self._pos_n50  > 0 and action[0] >= 0:
                action[0] = max(0.0, min(action[0], self._pos_n50))
            elif self._pos_n50 < 0 and action[0] <= 0:
                action[0] = min(0.0, max(action[0], self._pos_n50))
            else:
                action[0] = 0.0
            action[1] = 0.0

        # ── Check SL/TSL before applying new action ───────────────────────
        sl_pnl = self._check_stop_losses(close_n50, close_nb, atr_n50, atr_nb)

        # ── Apply new target positions ────────────────────────────────────
        trade_pnl, trade_cost = 0.0, 0.0
        pnl_n50, cost_n50 = self._apply_position(
            "n50", action[0], close_n50, atr_n50, hour, minute)
        if not self._single_instrument:
            pnl_nb, cost_nb = self._apply_position(
                "nbank", action[1], close_nb, atr_nb, hour, minute)
        else:
            pnl_nb, cost_nb = 0.0, 0.0
        trade_pnl  = pnl_n50 + pnl_nb + sl_pnl
        trade_cost = cost_n50 + cost_nb

        # Unrealised P&L on held positions (mark-to-market)
        unreal = self._unrealised_pnl(close_n50, close_nb)
        self._gr.update_unrealised(unreal)

        # ── Hard exit at 15:00 ────────────────────────────────────────────
        if hour == 15 and minute == 0:
            eod_pnl, eod_cost = self._force_exit(close_n50, close_nb)
            trade_pnl  += eod_pnl
            trade_cost += eod_cost

        self._daily_pnl += trade_pnl - trade_cost

        # BUG 2 FIX: compound capital across days so model learns account growth
        self._capital = self._capital + trade_pnl - trade_cost
        self._capital = max(self._capital, 1000)  # floor to avoid division by zero
        self._max_capital = max(self._max_capital, self._capital)
        self._gr.capital = self._capital

        # ── Reward ────────────────────────────────────────────────────────
        reward = self._compute_reward(
            action, trade_pnl, trade_cost, time_slot, regime, reg_prob)

        # ── Advance ───────────────────────────────────────────────────────
        self._bar_idx += 1
        if self._pos_n50   != 0: self._bars_since_entry_n50   += 1
        else:                    self._bars_since_entry_n50    = 0
        if self._pos_nbank != 0: self._bars_since_entry_nbank += 1
        else:                    self._bars_since_entry_nbank  = 0
        terminated = self._gr.daily_halted
        truncated  = (hour == 15 and minute == 0) or \
                     (self._bar_idx >= len(self._day_bars))

        obs = self._get_obs()
        info = {
            "daily_pnl":    self._daily_pnl,
            "trades_today": self._trades_today,
            "pos_n50":      self._pos_n50,
            "pos_nbank":    self._pos_nbank,
        }
        return obs, float(reward), terminated, truncated, info

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _apply_position(self, instrument: str, target: float,
                        close: float, atr: float,
                        hour: int, minute: int) -> tuple[float, float]:
        """Change position toward target. Returns (realised_pnl, cost)."""
        # Hard daily trade cap — 6 trades max (3 round-trips), checked first
        if self._trades_today >= 6:
            return 0.0, 0.0

        if instrument == "n50":
            current = self._pos_n50
        else:
            current = self._pos_nbank

        delta = target - current
        if abs(delta) < 0.01:
            return 0.0, 0.0

        # Minimum 12-bar (1 hour) holding period — block any change while in position
        MIN_HOLD_BARS = 12
        bars_held = (self._bars_since_entry_n50 if instrument == "n50"
                     else self._bars_since_entry_nbank)
        if current != 0 and bars_held < MIN_HOLD_BARS:
            return 0.0, 0.0

        # Check entry guardrails (only for new positions / adding)
        if abs(target) > abs(current) or np.sign(target) != np.sign(current):
            allowed, reason = self._gr.check_entry(hour, minute,
                                                   atr * 1.5, instrument, has_sl=True)
            if not allowed:
                return 0.0, 0.0

        lot_size   = LOT_SIZE_N50 if instrument == "n50" else LOT_SIZE_NBANK
        inst_str   = "NIFTY50"    if instrument == "n50" else "NIFTYBANK"
        lots       = 1  # FIX 4: fixed at 1 lot — ₹50k capital cannot support multi-lot

        # Realised P&L from closed portion
        pnl = 0.0
        if np.sign(target) != np.sign(current) and current != 0:
            entry     = self._entry_n50 if instrument == "n50" else self._entry_nbank
            direction = 1 if current > 0 else -1
            pnl       = direction * (close - entry) * lot_size * lots
            pnl       = max(pnl, -(self._capital * 0.02))  # HG3: cap single-trade loss at 2%
            self._gr.record_trade_close(pnl)
            self._trades_today += 1
            if instrument == "n50":
                self._sl_n50    = 0.0
                self._peak_n50  = 0.0
            else:
                self._sl_nbank  = 0.0
                self._peak_nbank = 0.0

        # Transaction cost for the trade
        trade_value = close * lot_size * lots
        side        = "BUY" if delta > 0 else "SELL"
        cost        = compute_transaction_cost(inst_str, trade_value, lots, side)

        # Update position and entry price
        if instrument == "n50":
            if np.sign(target) != np.sign(self._pos_n50):
                self._entry_n50 = close   # new direction
            self._pos_n50 = target
            # Set initial SL — dynamic multiplier from ATR percentile
            if target != 0 and self._sl_n50 == 0.0:
                atr_pct = _safe(self.n50.loc[self._day_bars[self._bar_idx]].get(
                    'atr_percentile_20d', 0.5))
                mult = 1.0 + (1.5 * atr_pct)   # 1.0 (calm) → 2.5 (volatile)
                self._sl_n50 = close - np.sign(target) * mult * atr
        else:
            if np.sign(target) != np.sign(self._pos_nbank):
                self._entry_nbank = close
            self._pos_nbank = target
            if target != 0 and self._sl_nbank == 0.0:
                atr_pct = 0.5   # no per-bar ATR percentile for NBANK, use midpoint
                mult = 1.0 + (1.5 * atr_pct)   # 1.75x ATR
                self._sl_nbank = close - np.sign(target) * mult * atr

        return pnl, cost

    def _check_stop_losses(self, close_n50: float, close_nb: float,
                           atr_n50: float, atr_nb: float) -> float:
        """Check and fire SL/TSL. Returns realised P&L from any triggered SL."""
        pnl = 0.0

        # Dynamic ATR multiplier from current bar's ATR percentile
        if self._day_bars and self._bar_idx < len(self._day_bars):
            atr_pct = _safe(self.n50.loc[self._day_bars[self._bar_idx]].get(
                'atr_percentile_20d', 0.5))
        else:
            atr_pct = 0.5
        tsl_mult = 1.0 + (1.5 * atr_pct)   # trail trigger: 1.0–2.5x ATR

        # NIFTY 50
        if self._pos_n50 != 0 and self._sl_n50 != 0:
            sl_hit = ((self._pos_n50 > 0 and close_n50 <= self._sl_n50) or
                      (self._pos_n50 < 0 and close_n50 >= self._sl_n50))
            if sl_hit:
                direction = 1 if self._pos_n50 > 0 else -1
                sl_pnl    = direction * (self._sl_n50 - self._entry_n50) * LOT_SIZE_N50
                sl_pnl    = max(sl_pnl, -(self._capital * 0.02))
                pnl += sl_pnl
                self._gr.record_trade_close(sl_pnl)
                self._trades_today += 1
                self._pos_n50  = 0.0
                self._sl_n50   = 0.0
                self._peak_n50 = 0.0
            else:
                # Update TSL — thresholds scale with volatility regime
                move = self._pos_n50 * (close_n50 - self._entry_n50)
                self._peak_n50 = max(self._peak_n50, move)
                if self._peak_n50 >= tsl_mult * atr_n50:
                    new_sl = self._entry_n50 + np.sign(self._pos_n50) * atr_n50
                    if self._pos_n50 > 0:
                        self._sl_n50 = max(self._sl_n50, new_sl)
                    else:
                        self._sl_n50 = min(self._sl_n50, new_sl)
                elif self._peak_n50 >= (tsl_mult / 2) * atr_n50:
                    # Move to breakeven
                    if self._pos_n50 > 0:
                        self._sl_n50 = max(self._sl_n50, self._entry_n50)
                    else:
                        self._sl_n50 = min(self._sl_n50, self._entry_n50)

        # NIFTY BANK (same logic)
        if not self._single_instrument and self._pos_nbank != 0 and self._sl_nbank != 0:
            sl_hit = ((self._pos_nbank > 0 and close_nb <= self._sl_nbank) or
                      (self._pos_nbank < 0 and close_nb >= self._sl_nbank))
            if sl_hit:
                direction = 1 if self._pos_nbank > 0 else -1
                nb_pnl    = direction * (self._sl_nbank - self._entry_nbank) * LOT_SIZE_NBANK
                nb_pnl    = max(nb_pnl, -(self._capital * 0.02))
                pnl += nb_pnl
                self._gr.record_trade_close(nb_pnl)
                self._trades_today += 1
                self._pos_nbank  = 0.0
                self._sl_nbank   = 0.0
                self._peak_nbank = 0.0
            else:
                move = self._pos_nbank * (close_nb - self._entry_nbank)
                self._peak_nbank = max(self._peak_nbank, move)
                if self._peak_nbank >= tsl_mult * atr_nb:
                    new_sl = self._entry_nbank + np.sign(self._pos_nbank) * atr_nb
                    if self._pos_nbank > 0:
                        self._sl_nbank = max(self._sl_nbank, new_sl)
                    else:
                        self._sl_nbank = min(self._sl_nbank, new_sl)
                elif self._peak_nbank >= (tsl_mult / 2) * atr_nb:
                    if self._pos_nbank > 0:
                        self._sl_nbank = max(self._sl_nbank, self._entry_nbank)
                    else:
                        self._sl_nbank = min(self._sl_nbank, self._entry_nbank)

        return pnl

    def _force_exit(self, close_n50: float, close_nb: float) -> tuple[float, float]:
        """HG8: hard EOD exit at 15:00."""
        pnl, cost = 0.0, 0.0
        if self._pos_n50 != 0:
            direction = 1 if self._pos_n50 > 0 else -1
            p    = direction * (close_n50 - self._entry_n50) * LOT_SIZE_N50
            c    = compute_transaction_cost("NIFTY50",
                       close_n50 * LOT_SIZE_N50, 1, "SELL")
            pnl  += p
            cost += c
            self._gr.record_trade_close(p)
            self._trades_today += 1
            self._pos_n50 = 0.0
            self._sl_n50  = 0.0
        if not self._single_instrument and self._pos_nbank != 0:
            direction = 1 if self._pos_nbank > 0 else -1
            p    = direction * (close_nb - self._entry_nbank) * LOT_SIZE_NBANK
            c    = compute_transaction_cost("NIFTYBANK",
                       close_nb * LOT_SIZE_NBANK, 1, "SELL")
            pnl  += p
            cost += c
            self._gr.record_trade_close(p)
            self._trades_today += 1
            self._pos_nbank = 0.0
            self._sl_nbank  = 0.0
        self._gr.hard_exit_eod()
        return pnl, cost

    def _unrealised_pnl(self, close_n50: float, close_nb: float) -> float:
        unreal = 0.0
        if self._pos_n50 != 0:
            direction = 1 if self._pos_n50 > 0 else -1
            unreal += direction * (close_n50 - self._entry_n50) * LOT_SIZE_N50
        if self._pos_nbank != 0:
            direction = 1 if self._pos_nbank > 0 else -1
            unreal += direction * (close_nb - self._entry_nbank) * LOT_SIZE_NBANK
        return unreal

    def _compute_reward(self, action, trade_pnl, trade_cost,
                        time_slot, regime, reg_prob):

        net_pnl = trade_pnl - trade_cost

        # Convert to points (1 lot N50: 1 point = ₹75)
        pnl_pts = net_pnl / 75.0

        # Normalise by daily ATR so reward is regime-independent
        try:
            idx = self._day_bars[min(self._bar_idx, len(self._day_bars) - 1)]
            daily_atr = _safe(self.n50.loc[idx].get('prev_atr', 100), 100)
        except Exception:
            daily_atr = 100.0
        daily_atr = max(daily_atr, 10.0)

        # Core reward: P&L in points normalised by daily ATR
        # Capturing 0.5 × daily ATR = reward of +1.0
        r = np.clip(pnl_pts / (daily_atr * 0.5), -3.0, 3.0)

        # Penalty for being wrong direction vs regime
        if regime == 2 and self._pos_n50 < -0.1:   # Bull but short
            r -= 0.2
        if regime == 0 and self._pos_n50 > 0.1:    # Bear but long
            r -= 0.2

        # Time window violation
        if time_slot == 0 and abs(action[0]) > DEAD_BAND:
            r -= 1.0

        # Action smoothness penalty
        if self._pos_n50 != 0 and abs(action[0]) > 0.5:
            r -= (abs(action[0]) - 0.5) * 0.2

        return float(np.clip(r, -3.0, 3.0))

    def _get_obs(self) -> np.ndarray:
        """Build a normalised 52-dim observation vector."""
        if self._bar_idx >= len(self._day_bars):
            return np.zeros(OBS_SIZE, dtype=np.float32)

        idx      = self._day_bars[self._bar_idx]
        nbank_idx = (self._nbank_bars[self._bar_idx]
                     if self._bar_idx < len(self._nbank_bars)
                     else self._nbank_bars[-1])
        r50  = self.n50.loc[idx]
        rnb  = self.nbank.loc[nbank_idx]

        close_n50 = _safe(r50["close"], 22_000)
        close_nb  = _safe(rnb["close"],  47_000)

        # N50 per-bar features (clipped to [-5, 5])
        n50_feats = np.array(
            [np.clip(_safe(r50.get(c, 0)), -5, 5) for c in OBS_COLS_N50],
            dtype=np.float32)

        # NBANK per-bar features
        nb_feats = np.array(
            [np.clip(_safe(rnb.get(c, 0)), -5, 5) for c in OBS_COLS_NBANK],
            dtype=np.float32)

        # Shared / daily / session features
        shared_feats = np.array(
            [np.clip(_safe(r50.get(c, 0)), -5, 5) for c in OBS_SHARED],
            dtype=np.float32)

        # Portfolio state
        unreal = self._unrealised_pnl(close_n50, close_nb)
        dd_pct = max(0.0, -(self._daily_pnl)) / self._capital
        # Volatility regime signal: >0 = more volatile than avg, <0 = calmer
        curr_atr  = _safe(r50.get('atr14', 1))
        mean_atr  = _safe(r50.get('atr_mean_20d', 1), 1)
        vol_ratio = np.clip(curr_atr / max(mean_atr, 1e-6), 0.2, 3.0)
        vol_ratio_norm = float((vol_ratio - 1.0) / 2.0)   # centred at 0
        port = np.array([
            np.clip(self._pos_n50,   -1, 1),
            np.clip(self._pos_nbank, -1, 1),
            np.clip(unreal / self._capital, -0.2, 0.2),
            np.clip(self._daily_pnl / self._capital, -0.1, 0.1),
            np.clip(dd_pct, 0, 0.2),
            np.clip(self._trades_today / 10.0, 0, 1),
            np.clip((self._capital + self._daily_pnl) / self._capital - 1, -0.1, 0.1),
            np.clip(vol_ratio_norm, -0.4, 1.0),   # volatility regime vs training avg
        ], dtype=np.float32)

        # Regime one-hot + max prob
        regime = int(np.clip(_safe(r50.get("regime", 1)), 0, 2))
        regime_oh = np.zeros(3, dtype=np.float32)
        regime_oh[regime] = 1.0
        reg_prob = np.array([_safe(r50.get("regime_max_prob", 0.34))],
                            dtype=np.float32)

        obs = np.concatenate([n50_feats, nb_feats, shared_feats, port,
                              regime_oh, reg_prob])

        # Pad or trim to OBS_SIZE
        if len(obs) < OBS_SIZE:
            obs = np.concatenate([obs, np.zeros(OBS_SIZE - len(obs), dtype=np.float32)])
        else:
            obs = obs[:OBS_SIZE]

        obs = np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)
        return obs.astype(np.float32)


# ── Metrics callback ──────────────────────────────────────────────────────────

class MetricsCallback(BaseCallback):
    """Logs Sharpe/Sortino/max-drawdown every N episodes."""

    def __init__(self, eval_env: NiftyTradingEnv, eval_freq: int = 5_000,
                 verbose: int = 1, log_path: Path | None = None):
        super().__init__(verbose)
        self._eval_env  = eval_env
        self._eval_freq = eval_freq
        self._step      = 0
        self._log_path  = log_path
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a") as f:
                f.write(f"\n--- run started {__import__('datetime').datetime.now():%Y-%m-%d %H:%M:%S} ---\n")

    def _on_step(self) -> bool:
        self._step += 1
        if self._step % self._eval_freq == 0:
            self._run_eval()
        return True

    def _run_eval(self):
        env = self._eval_env  # raw NiftyTradingEnv directly

        daily_returns = []
        obs, _ = env.reset()
        done = False
        start_cap = env._capital

        for _ in range(len(env._all_dates)):
            obs, _ = env.reset()
            ep_pnl = 0.0
            done = trunc = False
            while not (done or trunc):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, r, done, trunc, info = env.step(action)
                ep_pnl = info.get("daily_pnl", ep_pnl)
            daily_returns.append(ep_pnl / start_cap)

        rets  = np.array(daily_returns)
        if len(rets) < 2 or rets.std() == 0:
            return
        sharpe  = rets.mean() / rets.std() * np.sqrt(252)
        neg     = rets[rets < 0]
        sortino = rets.mean() / (neg.std() if len(neg) > 1 else 1e-6) * np.sqrt(252)
        mdd     = self._max_drawdown(rets)
        wr      = float((rets > 0).mean())

        line = (f"[Eval @{self.num_timesteps}]  "
                f"Sharpe={sharpe:.2f}  Sortino={sortino:.2f}  "
                f"MaxDD={mdd:.1%}  WinRate={wr:.1%}")
        if self.verbose:
            print(f"\n{line}")
        if self._log_path is not None:
            with open(self._log_path, "a") as f:
                f.write(line + "\n")

    @staticmethod
    def _max_drawdown(returns: np.ndarray) -> float:
        cum   = np.cumprod(1 + returns)
        peaks = np.maximum.accumulate(cum)
        dds   = (cum - peaks) / peaks
        return float(abs(dds.min()))


# ── Debug helper ─────────────────────────────────────────────────────────────

def debug_env(env: NiftyTradingEnv, n_steps: int = 200):
    """
    Manually step through the env and print what's happening.
    This MUST show ep_len > 1 before we start real training.
    """
    print("\n=== DEBUG ENV ===")
    obs, _ = env.reset()
    print(f"Obs shape: {obs.shape}")
    print(f"Obs NaNs: {np.isnan(obs).sum()}")
    print(f"Day bars: {len(env._day_bars)}")
    print(f"Day date: {env._all_dates[env._day_idx]}")

    ep_lens, ep_rewards = [], []
    ep_len, ep_rew = 0, 0

    for i in range(n_steps):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        ep_len += 1
        ep_rew += reward

        if term or trunc:
            ep_lens.append(ep_len)
            ep_rewards.append(ep_rew)
            print(f"  Episode {len(ep_lens)}: len={ep_len}, "
                  f"reward={ep_rew:.4f}, trades={info.get('trades_today', 0)}, "
                  f"pnl={info.get('daily_pnl', 0):.2f}")
            ep_len, ep_rew = 0, 0
            obs, _ = env.reset()

    print(f"\nMean ep_len:    {np.mean(ep_lens):.1f}  (MUST be > 50)")
    print(f"Mean ep_reward: {np.mean(ep_rewards):.4f}")
    print(f"Reward std:     {np.std(ep_rewards):.4f}  (MUST be > 0)")
    print("=== END DEBUG ===\n")

    if np.mean(ep_lens) < 10:
        raise RuntimeError(
            "Mean episode length < 10. Fix the environment before training. "
            "Check guardrails.check_entry() — it may be blocking all entries."
        )


# ── Main training function ────────────────────────────────────────────────────

def run_sac_training() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    n50_path  = FEATURES_DIR / "NIFTY_50_features.parquet"
    nb_path   = FEATURES_DIR / "NIFTY_BANK_features.parquet"
    for p in (n50_path, nb_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found — run Phases 0–2 first.")

    print("Loading feature parquets...")
    n50_df  = pd.read_parquet(n50_path)
    nbank_df = pd.read_parquet(nb_path)
    print(f"  NIFTY 50:   {len(n50_df):,} bars")
    print(f"  NIFTY BANK: {len(nbank_df):,} bars")

    print("Building training environment (NIFTY 50 only)...")
    train_env_raw = NiftyTradingEnv(
        n50_df, nbank_df, TRAIN_START, TRAIN_END,
        capital=CAPITAL, randomise_start=False, single_instrument=True
    )
    debug_env(train_env_raw)
    train_env = Monitor(train_env_raw)

    print("Building validation environment (NIFTY 50 only)...")
    val_env = NiftyTradingEnv(
        n50_df, nbank_df, VAL_START, VAL_END,
        capital=CAPITAL, single_instrument=True
    )

    print("Running environment sanity check...")
    try:
        check_env(train_env_raw, warn=True)
        print("  Environment OK.")
    except Exception as e:
        print(f"  Warning during env check: {e}")

    print("Initialising SAC agent...")
    import shutil
    best_model_path = str(MODELS_DIR / "best_model")
    checkpoint_path = MODELS_DIR / "checkpoints"
    if checkpoint_path.exists():
        shutil.rmtree(checkpoint_path)
    checkpoint_path.mkdir(exist_ok=True)
    print("  Cleared old checkpoints.")
    checkpoint_path = str(checkpoint_path)

    model = SAC(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=str(MODELS_DIR / "tb_logs"),
        **SAC_KWARGS
    )

    callbacks = [
        CheckpointCallback(
            save_freq=10_000,
            save_path=checkpoint_path,
            name_prefix="sac_nifty",
        ),
        EvalCallback(
            val_env,
            best_model_save_path=str(MODELS_DIR),
            log_path=str(MODELS_DIR / "eval_logs"),
            eval_freq=10_000,
            n_eval_episodes=20,
            deterministic=True,
            verbose=1,
        ),
        MetricsCallback(val_env, eval_freq=5_000, verbose=1,
                        log_path=MODELS_DIR / "eval_metrics.log"),
    ]

    print(f"\nTraining SAC for {TOTAL_TIMESTEPS:,} timesteps...")
    print(f"  Train: {TRAIN_START} → {TRAIN_END}")
    print(f"  Val:   {VAL_START}   → {VAL_END}")
    print(f"  Benchmarks to beat:")
    print("    • Buy-and-hold NIFTY 50")
    print("    • Buy-and-hold NIFTY BANK")
    print("    • Pure ORB rule strategy")
    print(f"  Success targets: Sharpe>1.5, Sortino>2.0, MaxDD<15%, WinRate>52%\n")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        reset_num_timesteps=True,
    )

    # Save final model + metadata
    final_path = str(MODELS_DIR / "final_model")
    model.save(final_path)
    meta = {
        "train_start"     : TRAIN_START,
        "train_end"       : TRAIN_END,
        "val_start"       : VAL_START,
        "val_end"         : VAL_END,
        "total_timesteps" : TOTAL_TIMESTEPS,
        "capital"         : CAPITAL,
        "obs_size"        : OBS_SIZE,
        "action_space"    : "Box([-1,-1],[1,1]): action[0]=N50, action[1]=NBank",
        "obs_cols_n50"    : OBS_COLS_N50,
        "obs_cols_nbank"  : OBS_COLS_NBANK,
        "obs_shared"      : OBS_SHARED,
    }
    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSAC training complete.")
    print(f"  Final model  → {final_path}.zip")
    print(f"  Best model   → {MODELS_DIR}/best_model.zip")
    print(f"  Metadata     → {MODELS_DIR}/metadata.json")
    print("\n*** Run test-set evaluation next (touch 2024 data only once). ***")


if __name__ == "__main__":
    run_sac_training()
