"""
ORB (Opening Range Breakout) signal engine.

Signal 1 (~17/yr): 11AM high-conviction breakout. Fires as a standalone rule;
                   SAC is NOT involved. Tier 1 risks 2%, Tier 2 risks 1.5%.
Signal 2 (~200/yr): Afternoon ORB breakout in 12:00–14:25 window. Only the
                    flag columns (orb_signal2_active, orb_signal2_dir) are
                    passed to the SAC as input features.

Call compute_orb_signals() AFTER all per-bar indicators (ADX, ATR, VWAP, EMAs,
daily_trend, is_high_vol_day) have been added to the DataFrame.
"""

import numpy as np
import pandas as pd

# Signal 1 tiers: (min_move_pct, min_adx, min_vwap_atr_ratio, risk_pct)
_S1_TIERS = [
    (0.70, 55, 2.0, 0.020),
    (0.45, 45, 1.5, 0.015),
]


def compute_orb_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ORB feature columns to a 5-min OHLCV DataFrame (datetime index).

    Output columns added:
      orb_high            daily high of 9:15–9:54 candles
      orb_low             daily low  of 9:15–9:54 candles
      orb_range_pct       (orb_high - orb_low) / open_9:15 * 100
      orb_valid           1 if range within allowed band, else 0
      orb_signal2_active  1 at the first bar where Signal 2 fires, else 0
      orb_signal2_dir     +1 long / -1 short at that bar; 0 elsewhere
      sig1_tier           signed tier at 11:05 bar (e.g. +1, -1, +2, -2); 0 elsewhere
    """
    df = df.copy()
    for col in ("orb_high", "orb_low", "orb_range_pct"):
        df[col] = np.nan
    for col in ("orb_valid", "orb_signal2_active", "orb_signal2_dir", "sig1_tier"):
        df[col] = 0

    has_adx   = "adx14"       in df.columns
    has_atr   = "atr14"       in df.columns
    has_vwap  = "vwap"        in df.columns
    has_ema   = "ema9"        in df.columns and "ema21" in df.columns
    has_trend = "daily_trend" in df.columns
    has_hvol  = "is_high_vol_day" in df.columns

    for date, day_df in df.groupby(df.index.date):
        pre = day_df.between_time("09:15", "09:54")
        if len(pre) == 0:
            continue

        orb_h = pre["high"].max()
        orb_l = pre["low"].min()
        open_915 = pre.iloc[0]["open"]
        if open_915 <= 0:
            continue

        range_pct = (orb_h - orb_l) / open_915 * 100
        is_hv = bool(day_df["is_high_vol_day"].iloc[0]) if has_hvol else False
        max_range = 2.0 if is_hv else 1.0
        valid = (0.1 <= range_pct <= max_range)

        df.loc[day_df.index, "orb_high"]      = orb_h
        df.loc[day_df.index, "orb_low"]       = orb_l
        df.loc[day_df.index, "orb_range_pct"] = range_pct
        df.loc[day_df.index, "orb_valid"]     = int(valid)

        if not valid:
            continue

        # ── Signal 1 ────────────────────────────────────────────────────────
        sig1_dir = 0
        bars_1100 = day_df.between_time("11:00", "11:00")
        if has_adx and has_atr and has_vwap and len(bars_1100) > 0:
            b = bars_1100.iloc[0]
            move_pct = abs(b["close"] - b["open"]) / b["open"] * 100 if b["open"] > 0 else 0
            atr      = b["atr14"]
            adx      = b["adx14"]
            vwap_d   = abs(b["close"] - b["vwap"]) / atr if atr > 0 else 0

            if not (np.isnan(adx) or np.isnan(atr)):
                for min_mv, min_adx, min_vd, _risk in _S1_TIERS:
                    tier_idx = _S1_TIERS.index((min_mv, min_adx, min_vd, _risk)) + 1
                    if move_pct >= min_mv and adx >= min_adx and vwap_d >= min_vd:
                        sig1_dir = 1 if b["close"] > b["open"] else -1
                        bars_1105 = day_df.between_time("11:05", "11:05")
                        if len(bars_1105) > 0:
                            df.loc[bars_1105.index[0], "sig1_tier"] = tier_idx * sig1_dir
                        break

        # ── Signal 2 ────────────────────────────────────────────────────────
        if not (has_ema and has_adx and has_vwap and has_trend):
            continue

        daily_trend = int(day_df["daily_trend"].iloc[0])
        sig2_window = day_df.between_time("12:00", "14:25")

        for idx, row in sig2_window.iterrows():
            ema9  = row["ema9"]
            ema21 = row["ema21"]
            adx   = row["adx14"]
            vwap  = row["vwap"]
            if any(np.isnan(v) for v in [ema9, ema21, adx, vwap]):
                continue

            long_ok  = (daily_trend == 1  and row["close"] > orb_h
                        and ema9 > ema21 and adx >= 20 and row["close"] > vwap)
            short_ok = (daily_trend == -1 and row["close"] < orb_l
                        and ema9 < ema21 and adx >= 20 and row["close"] < vwap)

            # HG7: skip if Signal 1 fired in opposite direction
            if long_ok and (sig1_dir >= 0):
                df.loc[idx, "orb_signal2_active"] = 1
                df.loc[idx, "orb_signal2_dir"]    = 1
                break
            elif short_ok and (sig1_dir <= 0):
                df.loc[idx, "orb_signal2_active"] = 1
                df.loc[idx, "orb_signal2_dir"]    = -1
                break

    return df


def size_orb_signal1(tier: int, capital: float, sl_distance_pts: float,
                     lot_size: int = 75) -> int:
    """Return number of lots for a Signal 1 trade."""
    risk_pct = 0.020 if tier == 1 else 0.015
    if sl_distance_pts <= 0:
        return 1
    return max(1, round((capital * risk_pct) / (sl_distance_pts * lot_size)))
