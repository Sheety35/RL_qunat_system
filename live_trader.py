"""
live_trader.py — Real-time / paper trading for NIFTY 50 Signal 2 strategy.

Set PAPER_MODE = True  to log trades to CSV without sending orders.
Set PAPER_MODE = False to send live orders via Upstox API.

Token setup (do this each morning):
    1. Run:  python get_token.py
       (opens browser, you paste the redirect URL, token is saved to .env)
    2. Run:  python live_trader.py

Prerequisites:
    - .env with api_key, api_secret, redirect_url set
    - models/regime_classifier/best_model.pt and scaler.pkl trained
"""

from __future__ import annotations

import csv
import logging
import os
import pickle
import time
from collections import deque
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import requests
import torch
from dotenv import load_dotenv

from guardrails import Guardrails, LOT_SIZE_N50
from regime_trainer import RegimeLSTM, SEQ_LEN

load_dotenv()
ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN")

# ── Configuration ─────────────────────────────────────────────────────────────

PAPER_MODE      = True    # False = live orders via Upstox

CAPITAL         = 50_000.0
LOT_SIZE        = LOT_SIZE_N50   # 75
ROUND_TRIP_COST = 250.0          # ₹ per round trip
SL_ATR_MULT     = 2.0
TP_ATR_MULT     = 4.0

IST             = ZoneInfo("Asia/Kolkata")
UPSTOX_BASE     = "https://api.upstox.com/v2"
NIFTY50_KEY     = "NSE_INDEX|Nifty 50"

ORB_START   = (9,  15)
ORB_END     = (9,  54)   # 8 bars: 9:15 9:20 9:25 9:30 9:35 9:40 9:45 9:50
SIG2_START  = (12,  0)
SIG2_END    = (14, 30)
EOD_EXIT    = (15,  0)
SESSION_END = (15,  5)

ATR_PERIOD      = 14
REGIME_SEQ_LEN  = SEQ_LEN   # 50 bars
API_RETRY_WAIT  = 30         # seconds between retry attempts
API_MAX_RETRIES = 3

MODELS_DIR = Path("models") / "regime_classifier"
LOG_DIR    = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

_REGIME_NAME = {0: "Bear", 1: "Flat", 2: "Bull"}

# ── Logging — plain message format so HH:MM prefix controls the layout ────────

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def _t(ts: datetime) -> str:
    """Return 'HH:MM' string from a datetime."""
    return ts.strftime("%H:%M")


def _now_t() -> str:
    return _t(datetime.now(tz=IST))


def _pnl_str(pnl: float) -> str:
    sign = "+" if pnl >= 0 else ""
    return f"{sign}Rs{pnl:.0f}"


# ── Token ─────────────────────────────────────────────────────────────────────

def _get_token() -> str:
    """
    Returns today's Upstox access token from .env (via ACCESS_TOKEN).
    Run get_token.py each morning to refresh — tokens expire at midnight IST.
    """
    token = (ACCESS_TOKEN or "").strip()
    if not token:
        raise SystemExit(
            "UPSTOX_ACCESS_TOKEN is not set in .env.\n"
            "Run:  python get_token.py\n"
            "This opens the browser, you paste the redirect URL, token is saved automatically."
        )
    return token


# ── Bar ───────────────────────────────────────────────────────────────────────

class Bar:
    __slots__ = ("ts", "open", "high", "low", "close")

    def __init__(self, ts: datetime, o: float, h: float, l: float, c: float):
        self.ts    = ts
        self.open  = o
        self.high  = h
        self.low   = l
        self.close = c

    def typical(self) -> float:
        return (self.high + self.low + self.close) / 3.0


# ── Upstox feed ───────────────────────────────────────────────────────────────

class UpstoxFeed:
    """Polls today's intraday 5-min candles. Retries up to 3× on failure."""

    def __init__(self, token: str):
        self._headers = {
            "Accept":        "application/json",
            "Authorization": f"Bearer {token}",
        }

    def fetch_today_bars(self, instrument_key: str = NIFTY50_KEY) -> list[Bar]:
        url = (f"{UPSTOX_BASE}/historical-candle/intraday"
               f"/{instrument_key}/5minute")

        for attempt in range(1, API_MAX_RETRIES + 1):
            try:
                resp = requests.get(url, headers=self._headers, timeout=15)
                resp.raise_for_status()
                candles = resp.json().get("data", {}).get("candles", [])
                return self._parse(candles)

            except requests.exceptions.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else "?"
                if status == 401:
                    raise SystemExit(
                        f"{_now_t()} - API returned 401 Unauthorized. "
                        "Update UPSTOX_ACCESS_TOKEN in .env with today's token."
                    ) from exc
                log.warning(
                    "%s - API error HTTP %s (attempt %d/%d) — retrying in %ds",
                    _now_t(), status, attempt, API_MAX_RETRIES, API_RETRY_WAIT,
                )

            except Exception as exc:
                log.warning(
                    "%s - API fetch failed (attempt %d/%d): %s — retrying in %ds",
                    _now_t(), attempt, API_MAX_RETRIES, exc, API_RETRY_WAIT,
                )

            if attempt < API_MAX_RETRIES:
                time.sleep(API_RETRY_WAIT)

        log.error("%s - API fetch failed after %d attempts — no new bars this cycle",
                  _now_t(), API_MAX_RETRIES)
        return []

    @staticmethod
    def _parse(candles: list) -> list[Bar]:
        bars = []
        for c in candles:
            ts = datetime.fromisoformat(c[0])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=IST)
            else:
                ts = ts.astimezone(IST)
            bars.append(Bar(ts, float(c[1]), float(c[2]), float(c[3]), float(c[4])))
        return sorted(bars, key=lambda b: b.ts)

    def place_order(self, direction: int, price: float) -> str | None:
        if PAPER_MODE:
            return "PAPER"
        side = "BUY" if direction > 0 else "SELL"
        payload = {
            "quantity":           LOT_SIZE,
            "product":            "I",
            "validity":           "DAY",
            "price":              0,
            "tag":                "live_trader",
            "instrument_token":   NIFTY50_KEY,
            "order_type":         "MARKET",
            "transaction_type":   side,
            "disclosed_quantity": 0,
            "trigger_price":      0,
            "is_amo":             False,
        }
        for attempt in range(1, API_MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    f"{UPSTOX_BASE}/order/place",
                    json=payload,
                    headers=self._headers,
                    timeout=10,
                )
                resp.raise_for_status()
                return resp.json()["data"]["order_id"]
            except Exception as exc:
                log.warning("%s - Order attempt %d/%d failed: %s",
                            _now_t(), attempt, API_MAX_RETRIES, exc)
                if attempt < API_MAX_RETRIES:
                    time.sleep(API_RETRY_WAIT)
        log.error("%s - Order placement failed after %d attempts",
                  _now_t(), API_MAX_RETRIES)
        return None


# ── Indicator state ───────────────────────────────────────────────────────────

class IndicatorState:
    """Incremental ATR(14) via Wilder's smoothing and daily-resetting TWAP."""

    def __init__(self):
        self._bars:         list[Bar]  = []
        self._atr14:        float      = float("nan")
        self._twap_day:     date | None = None
        self._twap_sum:     float      = 0.0
        self._twap_count:   int        = 0
        self._twap_current: float      = float("nan")

    def update(self, bar: Bar) -> None:
        self._bars.append(bar)
        self._update_atr(bar)
        self._update_twap(bar)

    def _update_atr(self, bar: Bar) -> None:
        n = len(self._bars)
        if n < 2:
            return
        prev_close = self._bars[-2].close
        tr = max(bar.high - bar.low,
                 abs(bar.high - prev_close),
                 abs(bar.low  - prev_close))
        if np.isnan(self._atr14):
            if n >= ATR_PERIOD + 1:
                trs = []
                for i in range(1, ATR_PERIOD + 1):
                    b  = self._bars[-i]
                    pc = self._bars[-i - 1].close
                    trs.append(max(b.high - b.low,
                                   abs(b.high - pc),
                                   abs(b.low  - pc)))
                self._atr14 = float(np.mean(trs))
        else:
            alpha       = 1.0 / ATR_PERIOD
            self._atr14 = self._atr14 * (1.0 - alpha) + tr * alpha

    def _update_twap(self, bar: Bar) -> None:
        today = bar.ts.date()
        if today != self._twap_day:
            self._twap_day   = today
            self._twap_sum   = 0.0
            self._twap_count = 0
        self._twap_sum   += bar.typical()
        self._twap_count += 1
        self._twap_current = self._twap_sum / self._twap_count

    @property
    def atr14(self) -> float:
        return self._atr14

    @property
    def twap(self) -> float:
        return self._twap_current


# ── Regime inference ──────────────────────────────────────────────────────────

class RegimeInferencer:
    """Rolls a 50-bar feature window through the saved LSTM model."""

    def __init__(self):
        self._model:     RegimeLSTM | None = None
        self._scaler                       = None
        self._device     = torch.device("cpu")
        self._feat_cols: list[str]         = []
        self._window:    deque             = deque(maxlen=REGIME_SEQ_LEN)
        self._regime:    int               = 1
        self._prob:      float             = 1 / 3
        self._load()

    def _load(self) -> None:
        ckpt_path = MODELS_DIR / "best_model.pt"
        scl_path  = MODELS_DIR / "scaler.pkl"
        if not ckpt_path.exists():
            log.warning("%s - Regime model not found — defaulting to Flat", _now_t())
            return
        ckpt = torch.load(ckpt_path, map_location=self._device)
        self._feat_cols = ckpt["feat_cols"]
        self._model     = RegimeLSTM(input_size=len(self._feat_cols)).to(self._device)
        self._model.load_state_dict(ckpt["model_state"])
        self._model.eval()
        with open(scl_path, "rb") as f:
            self._scaler = pickle.load(f)

    def push(self, bar: Bar, inds: IndicatorState) -> None:
        if self._model is None:
            return
        atr   = inds.atr14
        close = bar.close
        prev_close = self._window[-1].get("close", close) if self._window else close
        feats = {
            "close_return":        close / prev_close - 1,
            "ema_spread_pct":      0.0,
            "price_vs_sma1500":    0.0,
            "rsi14":               50.0,
            "macd_pct":            0.0,
            "atr14_pct":           (atr / close) if (close > 0 and not np.isnan(atr)) else 0.0,
            "bb_width":            0.0,
            "price_vs_vwap":       ((close - inds.twap) / inds.twap
                                    if not np.isnan(inds.twap) and inds.twap > 0 else 0.0),
            "sector_breadth_norm": 0.5,
            "volume_ratio":        1.0,
            "close":               close,
        }
        self._window.append(feats)
        if len(self._window) < REGIME_SEQ_LEN:
            return
        feat_arr = np.array(
            [[w.get(c, 0.0) for c in self._feat_cols] for w in self._window],
            dtype=np.float32,
        )
        feat_arr = self._scaler.transform(feat_arr)
        x = torch.tensor(feat_arr, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(self._model(x), dim=1).cpu().numpy()[0]
        self._regime = int(probs.argmax())
        self._prob   = float(probs.max())

    @property
    def regime(self) -> int:
        return self._regime

    @property
    def name(self) -> str:
        return _REGIME_NAME[self._regime]


# ── ORB tracker ───────────────────────────────────────────────────────────────

class ORBTracker:
    """Collects 9:15–9:54 bars and locks ORB high/low/validity."""

    def __init__(self):
        self.orb_high:  float | None = None
        self.orb_low:   float | None = None
        self.orb_valid: bool         = False
        self.range_pct: float        = 0.0
        self.locked:    bool         = False
        self._bars:     list[Bar]    = []

    def update(self, bar: Bar) -> bool:
        """Update with new bar. Returns True the first time the ORB locks."""
        if self.locked:
            return False
        t = bar.ts.hour * 60 + bar.ts.minute
        orb_s = ORB_START[0] * 60 + ORB_START[1]
        orb_e = ORB_END[0]   * 60 + ORB_END[1]
        if orb_s <= t <= orb_e:
            self._bars.append(bar)
        if t > orb_e and self._bars:
            self._lock()
            return True
        return False

    def _lock(self) -> None:
        self.locked   = True
        self.orb_high = max(b.high for b in self._bars)
        self.orb_low  = min(b.low  for b in self._bars)
        open_915      = self._bars[0].open
        self.range_pct = (self.orb_high - self.orb_low) / open_915 * 100 if open_915 > 0 else 0.0
        self.orb_valid = (0.1 <= self.range_pct <= 2.0)


# ── Signal 2 ──────────────────────────────────────────────────────────────────

def check_signal2(bar: Bar, orb: ORBTracker,
                  regime: int, twap: float) -> int:
    """Returns +1 (long), -1 (short), or 0 (no signal)."""
    if not orb.orb_valid or orb.orb_high is None:
        return 0
    t = bar.ts.hour * 60 + bar.ts.minute
    if not (SIG2_START[0] * 60 + SIG2_START[1] <= t
            <= SIG2_END[0] * 60 + SIG2_END[1]):
        return 0
    c = bar.close
    twap_ok = np.isnan(twap) or twap <= 0   # skip TWAP check if unavailable
    if regime == 2 and c > orb.orb_high and (twap_ok or c > twap):
        return 1
    if regime == 0 and c < orb.orb_low  and (twap_ok or c < twap):
        return -1
    return 0


# ── Paper logger ──────────────────────────────────────────────────────────────

class PaperLogger:
    COLUMNS = [
        "date", "time", "direction", "entry_price", "sl", "tp", "atr",
        "regime", "exit_reason", "exit_price", "pnl", "cumulative_pnl",
    ]

    def __init__(self):
        today        = date.today().isoformat()
        self._path   = LOG_DIR / f"paper_trades_{today}.csv"
        self._cum    = 0.0
        self._exists = self._path.exists()

    def record(self, row: dict) -> None:
        self._cum += row.get("pnl", 0.0)
        row["cumulative_pnl"] = round(self._cum, 2)
        write_header = not self._exists
        with open(self._path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.COLUMNS)
            if write_header:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in self.COLUMNS})
        self._exists = True

    @property
    def cumulative_pnl(self) -> float:
        return self._cum


# ── Position ──────────────────────────────────────────────────────────────────

class Position:
    def __init__(self):
        self.direction:    int            = 0
        self.entry_price:  float          = 0.0
        self.entry_time:   datetime | None = None
        self.sl:           float          = 0.0
        self.tp:           float          = 0.0
        self.atr_at_entry: float          = 0.0
        self.regime:       int            = 1

    @property
    def is_open(self) -> bool:
        return self.direction != 0

    def open(self, bar: Bar, direction: int, atr: float, regime: int) -> None:
        self.direction    = direction
        self.entry_price  = bar.close
        self.entry_time   = bar.ts
        self.atr_at_entry = atr
        self.regime       = regime
        self.sl           = bar.close - direction * SL_ATR_MULT * atr
        self.tp           = bar.close + direction * TP_ATR_MULT * atr

    def check_exit(self, bar: Bar) -> str | None:
        if not self.is_open:
            return None
        t = bar.ts.hour * 60 + bar.ts.minute
        if t >= EOD_EXIT[0] * 60 + EOD_EXIT[1]:
            return "EOD"
        if self.direction > 0:
            if bar.low  <= self.sl: return "SL"
            if bar.high >= self.tp: return "TP"
        else:
            if bar.high >= self.sl: return "SL"
            if bar.low  <= self.tp: return "TP"
        return None

    def exit_price(self, reason: str, bar: Bar) -> float:
        if reason == "SL": return self.sl
        if reason == "TP": return self.tp
        return bar.close

    def pnl(self, ex_price: float) -> float:
        return self.direction * (ex_price - self.entry_price) * LOT_SIZE - ROUND_TRIP_COST

    def close(self) -> None:
        self.direction = 0


# ── Live trader ───────────────────────────────────────────────────────────────

class LiveTrader:

    def __init__(self):
        token          = _get_token()
        self._feed     = UpstoxFeed(token)
        self._inds     = IndicatorState()
        self._regime   = RegimeInferencer()
        self._orb      = ORBTracker()
        self._pos      = Position()
        self._logger   = PaperLogger()
        self._gr       = Guardrails(CAPITAL)
        self._seen_ts: set[datetime] = set()
        self._signal_fired = False

    # ── Bar processing ────────────────────────────────────────────────────────

    def _process_bar(self, bar: Bar) -> None:
        self._inds.update(bar)
        just_locked = self._orb.update(bar)
        self._regime.push(bar, self._inds)

        atr    = self._inds.atr14
        twap   = self._inds.twap
        regime = self._regime.regime
        ts     = _t(bar.ts)

        # ── ORB lock announcement ─────────────────────────────────────────
        if just_locked:
            validity = "VALID" if self._orb.orb_valid else "INVALID (range out of bounds)"
            log.info("%s - ORB computed: HIGH=%.0f  LOW=%.0f  Range=%.2f%%  [%s]",
                     ts, self._orb.orb_high, self._orb.orb_low,
                     self._orb.range_pct, validity)

        # ── Exit check on open position ───────────────────────────────────
        if self._pos.is_open:
            reason = self._pos.check_exit(bar)
            if reason:
                self._exit(bar, reason)
            return

        # ── Signal 2 window: verbose per-bar check ────────────────────────
        t = bar.ts.hour * 60 + bar.ts.minute
        in_sig2 = (SIG2_START[0] * 60 + SIG2_START[1]
                   <= t
                   <= SIG2_END[0]  * 60 + SIG2_END[1])

        if in_sig2 and self._orb.orb_valid and not self._signal_fired:
            if bar.close > self._orb.orb_high:
                vs_orb = f"above ORB_HIGH ({self._orb.orb_high:.0f})"
            elif bar.close < self._orb.orb_low:
                vs_orb = f"below ORB_LOW ({self._orb.orb_low:.0f})"
            else:
                vs_orb = "inside ORB range"
            log.info("%s - Signal 2 check: close=%.0f, %s, regime=%s",
                     ts, bar.close, vs_orb, self._regime.name)

        if self._signal_fired:
            return
        if np.isnan(atr) or atr <= 0:
            return

        sig = check_signal2(bar, self._orb, regime, twap)
        if sig == 0:
            return

        allowed, block_reason = self._gr.check_entry(
            bar.ts.hour, bar.ts.minute, SL_ATR_MULT * atr, "NIFTY50", has_sl=True)
        if not allowed:
            log.info("%s - Entry blocked: %s", ts, block_reason)
            return

        self._enter(bar, sig, atr, regime)

    def _enter(self, bar: Bar, direction: int, atr: float, regime: int) -> None:
        order_id = self._feed.place_order(direction, bar.close)
        if order_id is None:
            log.error("%s - Order failed — position not opened", _t(bar.ts))
            return

        self._pos.open(bar, direction, atr, regime)
        self._signal_fired = True

        side = "LONG" if direction > 0 else "SHORT"
        log.info("%s - SIGNAL FIRED: %s at %.0f, SL=%.0f, TP=%.0f%s",
                 _t(bar.ts), side, bar.close,
                 self._pos.sl, self._pos.tp,
                 "" if PAPER_MODE else f"  order_id={order_id}")

    def _exit(self, bar: Bar, reason: str) -> None:
        ex_price = self._pos.exit_price(reason, bar)
        pnl      = self._pos.pnl(ex_price)
        side     = "LONG" if self._pos.direction > 0 else "SHORT"
        ts       = _t(bar.ts)

        if reason == "TP":
            log.info("%s - TP hit at %.0f, P&L=%s", ts, ex_price, _pnl_str(pnl))
        elif reason == "SL":
            log.info("%s - SL hit at %.0f, P&L=%s", ts, ex_price, _pnl_str(pnl))
        else:
            log.info("%s - EOD exit at %.0f, P&L=%s", ts, ex_price, _pnl_str(pnl))

        if not PAPER_MODE:
            self._feed.place_order(-self._pos.direction, ex_price)

        self._logger.record({
            "date":        bar.ts.date().isoformat(),
            "time":        ts,
            "direction":   side,
            "entry_price": round(self._pos.entry_price, 2),
            "sl":          round(self._pos.sl,           2),
            "tp":          round(self._pos.tp,           2),
            "atr":         round(self._pos.atr_at_entry, 2),
            "regime":      self._pos.regime,
            "exit_reason": reason,
            "exit_price":  round(ex_price, 2),
            "pnl":         round(pnl,      2),
        })

        self._gr.record_trade_close(pnl)
        self._pos.close()

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _is_session_over(self) -> bool:
        now = datetime.now(tz=IST)
        return now.hour * 60 + now.minute >= SESSION_END[0] * 60 + SESSION_END[1]

    def _new_bars(self) -> list[Bar]:
        all_bars = self._feed.fetch_today_bars()
        new = [b for b in all_bars if b.ts not in self._seen_ts]
        for b in new:
            self._seen_ts.add(b.ts)
        return new

    def run(self) -> None:
        log.info("%s - Session started  [mode=%s  capital=Rs%.0f]",
                 _now_t(), "PAPER" if PAPER_MODE else "LIVE", CAPITAL)

        self._gr.reset_day(orb_range_pct=0.5, daily_atr=100,
                           atr_mean_20d=100,   atr_std_20d=20)

        while not self._is_session_over():
            bars = self._new_bars()
            for bar in bars:
                self._process_bar(bar)
                if self._pos.is_open and self._gr.daily_halted:
                    log.warning("%s - HG2 daily loss limit hit — force-closing", _t(bar.ts))
                    self._exit(bar, "EOD")

            # Sleep until 10 seconds after the next 5-minute bar boundary
            now     = datetime.now(tz=IST)
            elapsed = now.minute % 5 * 60 + now.second
            sleep   = max(5, (5 * 60 - elapsed) + 10)
            time.sleep(sleep)

        # Ensure no open position survives past session end
        if self._pos.is_open:
            bars = self._new_bars()
            if bars:
                self._exit(bars[-1], "EOD")

        pos_status = "no open positions" if not self._pos.is_open else "position closed"
        log.info("%s - Session ended, %s  |  Day P&L: %s",
                 _now_t(), pos_status, _pnl_str(self._logger.cumulative_pnl))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    LiveTrader().run()
