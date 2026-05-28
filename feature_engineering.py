"""
Phase 0: Feature engineering.

Reads all raw CSVs from data/, computes every indicator, adds ORB signals,
adds placeholder regime columns (filled later by regime_trainer.py), and
saves two enriched parquets:
    features/NIFTY_50_features.parquet
    features/NIFTY_BANK_features.parquet

Run standalone or imported by train.py.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from pathlib import Path

from orb_engine import compute_orb_signals

DATA_DIR     = Path("data")
FEATURES_DIR = Path("features")

# 9 long-history sector indices used for breadth (all start 2015)
BREADTH_SECTORS = [
    "NIFTY AUTO", "NIFTY BANK", "NIFTY ENERGY", "NIFTY FIN SERVICE",
    "NIFTY FMCG", "NIFTY INFRA", "NIFTY IT", "NIFTY COMMODITIES",
    "NIFTY CONSUMPTION",
]

# ── Low-level indicator functions ─────────────────────────────────────────────

def _wilder_ema(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (alpha = 1/period, no adjustment)."""
    return series.ewm(alpha=1.0 / period, adjust=False).mean()


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int = 14) -> pd.Series:
    prev_c = close.shift(1)
    tr = pd.concat([high - low,
                    (high - prev_c).abs(),
                    (low  - prev_c).abs()], axis=1).max(axis=1)
    return _wilder_ema(tr, period)


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (adx, di_plus, di_minus)."""
    prev_h = high.shift(1)
    prev_l = low.shift(1)
    up   = high - prev_h
    down = prev_l - low
    dm_p = np.where((up > down) & (up > 0), up, 0.0)
    dm_m = np.where((down > up) & (down > 0), down, 0.0)
    dm_p = pd.Series(dm_p, index=close.index)
    dm_m = pd.Series(dm_m, index=close.index)

    atr     = compute_atr(high, low, close, period)
    dmp_s   = _wilder_ema(dm_p, period)
    dmm_s   = _wilder_ema(dm_m, period)
    di_p    = 100 * dmp_s / atr.replace(0, np.nan)
    di_m    = 100 * dmm_s / atr.replace(0, np.nan)
    di_sum  = (di_p + di_m).replace(0, np.nan)
    dx      = 100 * (di_p - di_m).abs() / di_sum
    adx     = _wilder_ema(dx, period)
    return adx, di_p, di_m


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = _wilder_ema(gain, period)
    avg_l = _wilder_ema(loss, period)
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26,
                 signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_f  = close.ewm(span=fast,   adjust=False).mean()
    ema_s  = close.ewm(span=slow,   adjust=False).mean()
    macd   = ema_f - ema_s
    sig    = macd.ewm(span=signal,  adjust=False).mean()
    hist   = macd - sig
    return macd, sig, hist


def compute_twap(df: pd.DataFrame) -> pd.Series:
    """
    Time-Weighted Average Price — daily resetting.
    For index data where volume=0, TWAP is the standard substitute.
    TWAP = cumulative mean of typical price (H+L+C)/3 since market open.
    """
    tp   = (df["high"] + df["low"] + df["close"]) / 3
    twap = pd.Series(np.nan, index=df.index)

    for date, grp in df.groupby(df.index.date):
        idx      = grp.index
        tp_grp   = tp.loc[idx]
        cum_twap = tp_grp.expanding().mean()
        twap.loc[idx] = cum_twap.values

    return twap


# ── Main indicator computation ────────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all per-bar technical indicators to df (in-place copy returned)."""
    df = df.copy()
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

    # EMAs / SMAs
    df["ema9"]          = c.ewm(span=9,    adjust=False).mean()
    df["ema21"]         = c.ewm(span=21,   adjust=False).mean()
    df["sma20"]         = c.rolling(20,    min_periods=10).mean()
    df["ema_spread"]    = df["ema9"] - df["ema21"]
    df["ema_spread_pct"]= df["ema_spread"] / c

    # 20-bar price position (intraday SMA)
    df["price_vs_sma20"] = (c - df["sma20"]) / df["sma20"]

    # ADX / ATR
    adx, di_p, di_m     = compute_adx(h, l, c, 14)
    df["adx14"]         = adx
    df["di_plus"]       = di_p
    df["di_minus"]      = di_m
    df["atr14"]         = compute_atr(h, l, c, 14)
    df["atr14_pct"]     = df["atr14"] / c

    # RSI
    df["rsi14"]         = compute_rsi(c, 14)

    # MACD (12/26/9)
    macd, msig, mhist   = compute_macd(c)
    df["macd"]          = macd
    df["macd_signal"]   = msig
    df["macd_hist"]     = mhist
    df["macd_pct"]      = macd / c       # normalised

    # Bollinger Bands (20-bar, 2σ)
    bb_sma              = c.rolling(20, min_periods=10).mean()
    bb_std              = c.rolling(20, min_periods=10).std()
    df["bb_upper"]      = bb_sma + 2 * bb_std
    df["bb_lower"]      = bb_sma - 2 * bb_std
    bb_range            = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    df["bb_width"]      = bb_range / bb_sma
    df["bb_position"]   = (c - df["bb_lower"]) / bb_range   # 0=lower, 1=upper

    # TWAP (time-weighted, daily-resetting) — stored as 'vwap' for compatibility
    df["vwap"]          = compute_twap(df)
    df["price_vs_vwap"] = (c - df["vwap"]) / df["vwap"].replace(0, np.nan)

    # Bar range as activity proxy — volume is always 0 for index series
    bar_range           = df["high"] - df["low"]
    range_avg20         = bar_range.rolling(20, min_periods=5).mean().replace(0, np.nan)
    df["volume_ratio"]  = bar_range / range_avg20

    # Rate of change
    df["roc5"]          = c.pct_change(5)
    df["roc20"]         = c.pct_change(20)
    df["close_return"]  = c.pct_change(1)

    twap_valid   = df["vwap"].notna().sum()
    pvsv_nonzero = (df["price_vs_vwap"] != 0).sum()
    print(f"  TWAP valid bars:          {twap_valid:,}")
    print(f"  price_vs_vwap non-zero:   {pvsv_nonzero:,}")
    assert twap_valid > len(df) * 0.9, \
        f"TWAP has too many NaN values: {twap_valid} valid out of {len(df)}"
    # First bar of each day can have close == typical price → price_vs_vwap = 0; that's correct

    return df


def compute_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily-level features (using prior-day data to avoid lookahead)
    and forward-fill them to every intraday bar.
    """
    df = df.copy()
    c = df["close"]

    # Build daily series from last bar of each day
    daily_close = df.groupby(df.index.date)["close"].last()
    daily_close.index = pd.to_datetime(daily_close.index)

    sma20d  = daily_close.rolling(20,  min_periods=10).mean()
    sma200d = daily_close.rolling(200, min_periods=100).mean()
    high52w = daily_close.rolling(252, min_periods=126).max()
    low52w  = daily_close.rolling(252, min_periods=126).min()

    # Daily ATR for volatility regime
    daily_atr  = df.groupby(df.index.date)["atr14"].last()
    daily_atr.index = pd.to_datetime(daily_atr.index)
    atr_m20    = daily_atr.rolling(20, min_periods=10).mean()
    atr_s20    = daily_atr.rolling(20, min_periods=10).std()

    # All comparisons use .shift(1) so today's bars only see yesterday's info
    daily_feats = pd.DataFrame({
        "daily_trend"       : np.where(daily_close.shift(1) > sma20d.shift(1),  1, -1),
        "monthly_trend"     : np.where(daily_close.shift(1) > sma200d.shift(1), 1,  0),
        "dist_from_52w_high": (high52w.shift(1) - daily_close.shift(1)) / high52w.shift(1).replace(0, np.nan),
        "dist_from_52w_low" : (daily_close.shift(1) - low52w.shift(1))  / low52w.shift(1).replace(0, np.nan),
        "prev_atr"          : daily_atr.shift(1),
        "atr_mean_20d"      : atr_m20.shift(1),
        "atr_std_20d"       : atr_s20.shift(1),
        "is_high_vol_day"   : (daily_atr.shift(1) > atr_m20.shift(1) + 0.5 * atr_s20.shift(1)).astype(int),
        "atr_percentile_20d": daily_atr.shift(1).rolling(20, min_periods=5).apply(
                                  lambda x: float(np.mean(x[:-1] <= x[-1])) if len(x) > 1 else 0.5,
                                  raw=True),
    }, index=daily_close.index)

    # Forward-fill to all intraday bars
    daily_feats_ffill = daily_feats.reindex(df.index, method="ffill")
    for col in daily_feats.columns:
        df[col] = daily_feats_ffill[col]

    # Intraday SMA1500 (20-day equivalent for breadth indicator)
    df["sma1500"] = c.rolling(1500, min_periods=500).mean()
    df["price_vs_sma1500"] = (c - df["sma1500"]) / df["sma1500"].replace(0, np.nan)

    return df


def compute_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-slot, day-of-week, and minutes-to-close features."""
    df = df.copy()
    t  = df.index

    # Time slot (0–5 matching the 6 trading windows in the training plan)
    # 0: 09:15–09:54  forbidden observation
    # 1: 09:55–10:59  restricted entry
    # 2: 11:00–11:04  Signal 1 window
    # 3: 11:05–11:59  hold / wait
    # 4: 12:00–14:29  Signal 2 window
    # 5: 14:30–14:59  exits only
    minutes = pd.Series(t.hour * 60 + t.minute, index=df.index)
    slot = pd.cut(
        minutes,
        bins=[0, 9*60+55, 11*60, 11*60+5, 12*60, 14*60+30, 24*60],
        labels=[0, 1, 2, 3, 4, 5],
        right=False
    ).astype(float).fillna(0).astype(int)
    df["time_slot"] = slot.values

    # One-hot (for model input)
    for s in range(6):
        df[f"slot_{s}"] = (df["time_slot"] == s).astype(int)

    df["day_of_week"]        = t.dayofweek              # 0=Mon, 4=Fri
    df["minutes_to_close"]   = (15 * 60 - minutes).clip(lower=0)

    return df


def compute_sector_breadth(base_df: pd.DataFrame,
                           sector_dfs: list[pd.DataFrame]) -> pd.Series:
    """
    For each 5-min bar in base_df, count how many sector indices have
    close > rolling 20-day SMA (1500 bars). Returns a Series (0–9).
    """
    breadth = pd.Series(0.0, index=base_df.index)
    for sec in sector_dfs:
        sma  = sec["close"].rolling(1500, min_periods=500).mean()
        above = (sec["close"] > sma).astype(float)
        above = above.reindex(base_df.index, method="ffill").fillna(0)
        breadth += above
    return breadth


# ── File loading ──────────────────────────────────────────────────────────────

def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    df.index.name = "datetime"

    # Ensure proper DatetimeIndex (parse_dates may leave object dtype on some pandas versions)
    df.index = pd.to_datetime(df.index)

    # Remove timezone info if present — causes comparison issues with date slicing
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def _load_sector_csvs() -> list[pd.DataFrame]:
    dfs = []
    for name in BREADTH_SECTORS:
        path = DATA_DIR / f"{name}_5minute.csv"
        if path.exists():
            dfs.append(_load_csv(path))
        else:
            print(f"  WARNING: sector file not found: {path}")
    return dfs


# ── Main entry point ──────────────────────────────────────────────────────────

def run_feature_engineering() -> None:
    FEATURES_DIR.mkdir(exist_ok=True)

    print("Loading NIFTY 50...")
    n50   = _load_csv(DATA_DIR / "NIFTY 50_5minute.csv")
    print("Loading NIFTY BANK...")
    nbank = _load_csv(DATA_DIR / "NIFTY BANK_5minute.csv")

    print("Loading sector CSVs for breadth...")
    sector_dfs = _load_sector_csvs()
    print(f"  Loaded {len(sector_dfs)}/{len(BREADTH_SECTORS)} sector indices.")

    # Validate index types before any computation
    assert isinstance(n50.index, pd.DatetimeIndex), \
        f"n50 index is {type(n50.index)}, expected DatetimeIndex"
    assert n50.index.tz is None, "n50 index has timezone — remove it"
    print(f"n50 index type: OK — {n50.index.dtype}")
    print(f"n50 date range: {n50.index.min()} to {n50.index.max()}")

    # ── NIFTY 50 features ────────────────────────────────────────────────────
    print("Computing NIFTY 50 indicators...")
    n50 = compute_indicators(n50)

    # Spot-check VWAP on first trading day
    first_date = n50.index.date[0]
    first_day  = n50[n50.index.date == first_date]
    print(f"First day VWAP sample: {first_day['vwap'].values[:5]}")
    assert not first_day["vwap"].isna().all(), \
        "VWAP is all NaN on first day — index type problem"
    print("Computing NIFTY 50 daily features...")
    n50 = compute_daily_features(n50)
    print("Computing ORB signals for NIFTY 50...")
    n50 = compute_orb_signals(n50)
    print("Computing session features for NIFTY 50...")
    n50 = compute_session_features(n50)

    # ── NIFTY BANK features ──────────────────────────────────────────────────
    print("Computing NIFTY BANK indicators...")
    nbank = compute_indicators(nbank)
    print("Computing NIFTY BANK daily features...")
    nbank = compute_daily_features(nbank)
    print("Computing ORB signals for NIFTY BANK...")
    nbank = compute_orb_signals(nbank)
    print("Computing session features for NIFTY BANK...")
    nbank = compute_session_features(nbank)

    # ── Sector breadth (shared, uses NIFTY 50 timestamps as base) ────────────
    print("Computing sector breadth...")
    breadth = compute_sector_breadth(n50, sector_dfs)
    n50["sector_breadth"]      = breadth
    n50["sector_breadth_norm"] = breadth / max(len(sector_dfs), 1)
    # reindex breadth to NIFTY BANK timestamps
    breadth_nbank = breadth.reindex(nbank.index, method="ffill").fillna(0)
    nbank["sector_breadth"]      = breadth_nbank
    nbank["sector_breadth_norm"] = breadth_nbank / max(len(sector_dfs), 1)

    # ── Cross-instrument features ────────────────────────────────────────────
    print("Computing cross-instrument features...")
    # NIFTY 50 1-hour (12-bar) return → added to NIFTY BANK features
    n50_return_1h = n50["close"].pct_change(12)
    n50_vs_sma20  = n50["price_vs_sma20"]
    nbank["nifty50_return_1h"]   = n50_return_1h.reindex(nbank.index, method="ffill")
    nbank["nifty50_vs_sma20"]    = n50_vs_sma20.reindex(nbank.index,  method="ffill")

    # NIFTY BANK / NIFTY 50 spread (ratio - 1)
    n50_close_aligned   = n50["close"].reindex(nbank.index, method="ffill")
    nbank_close         = nbank["close"]
    spread              = (nbank_close / n50_close_aligned.replace(0, np.nan) - 1)
    n50["bank_vs_n50_spread"]   = spread.reindex(n50.index, method="ffill")
    nbank["bank_vs_n50_spread"] = spread

    # NIFTY BANK 1-hour return → added to NIFTY 50 features
    nbank_return_1h = nbank["close"].pct_change(12)
    n50["nifty_bank_return_1h"] = nbank_return_1h.reindex(n50.index, method="ffill")

    # ── Placeholder regime columns (filled by regime_trainer.py) ─────────────
    for df in (n50, nbank):
        df["regime"]          = 1    # default: Flat
        df["regime_max_prob"] = 0.34  # uncertain

    # ── Drop rows with all NaN close (shouldn't happen, but safety) ───────────
    n50   = n50[n50["close"].notna()]
    nbank = nbank[nbank["close"].notna()]

    # ── Save ──────────────────────────────────────────────────────────────────
    out50   = FEATURES_DIR / "NIFTY_50_features.parquet"
    out_nb  = FEATURES_DIR / "NIFTY_BANK_features.parquet"
    print(f"Saving {out50}  ({len(n50):,} bars, {len(n50.columns)} columns)...")
    n50.to_parquet(out50)
    print(f"Saving {out_nb} ({len(nbank):,} bars, {len(nbank.columns)} columns)...")
    nbank.to_parquet(out_nb)
    print("Phase 0 complete.")


if __name__ == "__main__":
    run_feature_engineering()
