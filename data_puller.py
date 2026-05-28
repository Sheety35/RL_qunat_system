"""
Upstox 5-Minute Historical Data Puller
=======================================
Pulls 5-min OHLCV candles for the configured NSE stocks via Upstox API v2.

Storage layout:
    data/
        ASIANPAINT/ASIANPAINT_5min.parquet
        TATASTEEL/TATASTEEL_5min.parquet
        ...

Run:
    python data_puller.py

First run opens a browser for Upstox login (OAuth2). The access token is
cached in auth/access_token.json and reused until it expires (~18 h).
Re-runs are incremental: already-downloaded date ranges are skipped.

Rate limiting:
    Upstox allows 50 requests/second. This script targets 25 req/s to keep
    a 2x safety margin. Exponential back-off on 429 responses.

Upstox sub-daily candle limit:
    The API returns at most 100 calendar days per request for intervals < 1day.
    Two years of data requires ~7-8 requests per stock (fine at 25 req/s).

Dependencies (add to your venv):
    pip install requests python-dotenv pyarrow pandas
"""

import gzip
import json
import logging
import os
import threading
import time
import webbrowser
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import pandas as pd
import requests
from dotenv import load_dotenv

# ── Configuration ─────────────────────────────────────────────────────────────

load_dotenv()

API_KEY      = os.getenv("api_key")
API_SECRET   = os.getenv("api_secret")
REDIRECT_URL = os.getenv("redirect_url")          # http://127.0.0.1:5000/

UPSTOX_BASE         = "https://api.upstox.com/v2"
AUTH_DIALOG_URL     = f"{UPSTOX_BASE}/login/authorization/dialog"
TOKEN_EXCHANGE_URL  = f"{UPSTOX_BASE}/login/authorization/token"
INSTRUMENTS_URL     = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"

# NSE equity symbols to fetch.

#                   company listed on NSE — VERIFY this with your broker before trading.
NSE_SYMBOLS = [
    "ADANIENT",
    "TATASTEEL",
    "HINDALCO",
    "BAJFINANCE",
    "TATAMOTORS",
    "RELIANCE",
    "DIXON",
    "SUZLON",
    "COFORGE",      # <-- verify: execution_plan.md said LOFORGE, no such NSE ticker found
]

# How many calendar days to request per API call (Upstox hard limit for sub-daily).
CHUNK_DAYS = 100

# How far back to pull history (Upstox stores ~2 years of 5-min data).
HISTORY_YEARS = 2

# Safe request rate (Upstox cap is 50/s; we run at half that).
RATE_LIMIT_RPS = 25

INTERVAL = "5minute"

DATA_DIR       = Path("data")
AUTH_DIR       = Path("auth")
TOKEN_CACHE    = AUTH_DIR / "access_token.json"
INST_CACHE     = AUTH_DIR / "nse_instruments.json"
TOKEN_MAX_AGE  = timedelta(hours=18)   # Upstox tokens expire at midnight IST

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Rate limiter ──────────────────────────────────────────────────────────────

class _TokenBucket:
    """Enforces a maximum call rate by sleeping between calls."""
    def __init__(self, rps: float):
        self._gap = 1.0 / rps
        self._last = 0.0

    def wait(self):
        now = time.monotonic()
        sleep_for = self._gap - (now - self._last)
        if sleep_for > 0:
            time.sleep(sleep_for)
        self._last = time.monotonic()

_bucket = _TokenBucket(RATE_LIMIT_RPS)

# ── OAuth 2.0 ─────────────────────────────────────────────────────────────────

_captured_code: str | None = None


class _CallbackHandler(BaseHTTPRequestHandler):
    """One-shot HTTP handler that captures the OAuth code from Upstox redirect."""
    def do_GET(self):
        global _captured_code
        qs = parse_qs(urlparse(self.path).query)
        _captured_code = qs.get("code", [None])[0]
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"<h2>Login successful. You can close this tab and return to the terminal.</h2>")

    def log_message(self, *_):
        pass  # silence default access log


def _read_cached_token() -> str | None:
    if not TOKEN_CACHE.exists():
        return None
    with TOKEN_CACHE.open() as f:
        data = json.load(f)
    issued = datetime.fromisoformat(data["issued_at"])
    if datetime.utcnow() - issued > TOKEN_MAX_AGE:
        log.info("Cached token is older than %s — will refresh.", TOKEN_MAX_AGE)
        return None
    return data["access_token"]


def _write_token(token: str):
    AUTH_DIR.mkdir(parents=True, exist_ok=True)
    with TOKEN_CACHE.open("w") as f:
        json.dump({"access_token": token, "issued_at": datetime.utcnow().isoformat()}, f)


def get_access_token() -> str:
    """Return a valid access token, either from cache or via browser OAuth flow."""
    cached = _read_cached_token()
    if cached:
        log.info("Using cached access token.")
        return cached

    log.info("Auth config  ->  api_key=%s  redirect_url=%s", API_KEY, REDIRECT_URL)

    # Spin up a local server to catch the redirect code.
    parsed   = urlparse(REDIRECT_URL)
    port     = parsed.port or 80
    server   = HTTPServer(("127.0.0.1", port), _CallbackHandler)
    listener = threading.Thread(target=server.handle_request, daemon=True)
    listener.start()

    login_url = AUTH_DIALOG_URL + "?" + urlencode({
        "response_type": "code",
        "client_id":     API_KEY,
        "redirect_uri":  REDIRECT_URL,
    })
    log.info("Opening Upstox login page in your browser...")
    webbrowser.open(login_url)
    log.info("Waiting for redirect (you have 120 seconds to log in)...")
    listener.join(timeout=120)

    if not _captured_code:
        raise RuntimeError(
            "No auth code received within 120 s. "
            "Re-run the script and complete the Upstox login in the browser."
        )

    log.info("Exchanging auth code for access token...")

    payload = {
        "code":          _captured_code,
        "client_id":     API_KEY,
        "client_secret": API_SECRET,
        "redirect_uri":  REDIRECT_URL,   # must match exactly what is registered in Upstox developer portal
        "grant_type":    "authorization_code",
    }
    headers = {
        "Accept":       "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    resp = None
    for attempt in range(1, 4):
        try:
            resp = requests.post(TOKEN_EXCHANGE_URL, data=payload, headers=headers, timeout=30)
            break
        except requests.exceptions.Timeout:
            wait = 5 * attempt
            log.warning("Token exchange timed out (attempt %d/3). Retrying in %ds...", attempt, wait)
            time.sleep(wait)
        except requests.exceptions.ConnectionError as exc:
            log.error("Network error during token exchange: %s", exc)
            raise

    if resp is None:
        raise RuntimeError("Token exchange failed after 3 timeout retries. Check your internet connection.")

    if not resp.ok:
        log.error(
            "Token exchange failed — HTTP %d\nUpstox response: %s",
            resp.status_code, resp.text,
        )
        resp.raise_for_status()

    body = resp.json()
    token = body.get("access_token")
    if not token:
        raise RuntimeError(f"No access_token in response: {body}")

    _write_token(token)
    log.info("Access token obtained and cached at %s", TOKEN_CACHE)
    return token

# ── Instrument key resolution ──────────────────────────────────────────────────

def _download_instruments() -> dict[str, str]:
    """Download and cache the NSE equity symbol -> instrument_key mapping."""
    if INST_CACHE.exists() and (time.time() - INST_CACHE.stat().st_mtime) < 86400:
        log.info("Using cached instruments master (< 24 h old).")
        with INST_CACHE.open() as f:
            return json.load(f)

    log.info("Downloading NSE instruments master from Upstox...")
    resp = requests.get(INSTRUMENTS_URL, timeout=45)
    resp.raise_for_status()

    try:
        raw = gzip.decompress(resp.content)
        instruments = json.loads(raw)
    except Exception:
        instruments = resp.json()

    mapping: dict[str, str] = {}
    for inst in instruments:
        sym   = inst.get("trading_symbol", "").strip()
        key   = inst.get("instrument_key", "").strip()
        itype = inst.get("instrument_type", "")
        if itype == "EQ" and sym and key:
            mapping[sym] = key

    AUTH_DIR.mkdir(parents=True, exist_ok=True)
    with INST_CACHE.open("w") as f:
        json.dump(mapping, f)

    log.info("Instruments master: %d NSE equity symbols cached.", len(mapping))
    return mapping


def resolve_keys(symbols: list[str]) -> dict[str, str]:
    """Return {symbol: instrument_key} for every symbol that exists on NSE."""
    mapping = _download_instruments()
    resolved: dict[str, str] = {}
    log.info("Resolving instrument keys:")
    for sym in symbols:
        key = mapping.get(sym)
        if key:
            log.info("  %-16s -> %s", sym, key)
            resolved[sym] = key
        else:
            log.warning(
                "  %-16s -> NOT FOUND in NSE instruments. "
                "Check the symbol spelling and re-run.",
                sym,
            )
    return resolved

# ── API: fetch one candle chunk ───────────────────────────────────────────────

def _api_headers(token: str) -> dict:
    return {"Accept": "application/json", "Authorization": f"Bearer {token}"}


def fetch_candles_chunk(
    token: str, instrument_key: str, from_date: str, to_date: str
) -> list:
    """
    Fetch 5-minute candles for [from_date, to_date] (both YYYY-MM-DD).
    Returns raw candle list: [[ts, o, h, l, c, vol, oi], ...].
    Retries up to 3 times with exponential back-off on errors.
    """
    url = f"{UPSTOX_BASE}/historical-candle/{instrument_key}/{INTERVAL}/{to_date}/{from_date}"

    for attempt in range(1, 4):
        _bucket.wait()
        try:
            resp = requests.get(url, headers=_api_headers(token), timeout=20)

            if resp.status_code == 429:
                wait = 2 ** attempt
                log.warning("429 rate-limited. Sleeping %ds (attempt %d/3).", wait, attempt)
                time.sleep(wait)
                continue

            if resp.status_code == 200:
                return resp.json().get("data", {}).get("candles", [])

            # Non-retryable error
            log.error("HTTP %d for %s [%s->%s]: %s", resp.status_code, instrument_key, from_date, to_date, resp.text[:200])
            return []

        except requests.RequestException as exc:
            wait = 2 ** attempt
            log.warning("Request failed (attempt %d/3): %s. Retrying in %ds.", attempt, exc, wait)
            time.sleep(wait)

    log.error("Gave up after 3 attempts for %s [%s->%s].", instrument_key, from_date, to_date)
    return []

# ── Data helpers ──────────────────────────────────────────────────────────────

def candles_to_df(candles: list) -> pd.DataFrame:
    if not candles:
        return pd.DataFrame()
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata")
    df = df.drop(columns=["oi"])
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)
    df["volume"] = df["volume"].astype(int)
    return df.sort_values("timestamp").reset_index(drop=True)


def parquet_path(symbol: str) -> Path:
    return DATA_DIR / symbol / f"{symbol}_5min.parquet"


def load_existing(symbol: str) -> pd.DataFrame:
    p = parquet_path(symbol)
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame()


def save_symbol(symbol: str, df: pd.DataFrame):
    p = parquet_path(symbol)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False, engine="pyarrow", compression="snappy")


def merge_and_save(symbol: str, new_df: pd.DataFrame) -> pd.DataFrame:
    existing = load_existing(symbol)
    combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
    combined = (
        combined
        .drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    save_symbol(symbol, combined)
    return combined


def _date_chunks(start: datetime, end: datetime):
    """Yield (from_str, to_str) tuples covering [start, end] in CHUNK_DAYS windows."""
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=CHUNK_DAYS - 1), end)
        yield cursor.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        cursor = chunk_end + timedelta(days=1)

# ── Per-symbol orchestration ──────────────────────────────────────────────────

def pull_symbol(token: str, symbol: str, instrument_key: str):
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    history_start = today - timedelta(days=365 * HISTORY_YEARS)

    # Incremental: if we already have data, only pull the gap.
    existing = load_existing(symbol)
    if not existing.empty:
        last_ts = pd.to_datetime(existing["timestamp"]).max()
        # tz-strip for comparison
        last_date = last_ts.tz_localize(None) if last_ts.tzinfo else last_ts
        resume_from = last_date.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        if resume_from >= today:
            log.info("[%s] Already current (last bar: %s). Nothing to do.", symbol, last_date.date())
            return
        log.info("[%s] Incremental update from %s (existing last bar: %s).", symbol, resume_from.date(), last_date.date())
        fetch_start = resume_from
    else:
        fetch_start = history_start

    chunks = list(_date_chunks(fetch_start, today))
    log.info("[%s] Fetching %d chunk(s): %s -> %s", symbol, len(chunks), fetch_start.date(), today.date())

    frames = []
    for idx, (from_d, to_d) in enumerate(chunks, 1):
        log.info("[%s]  chunk %d/%d  (%s -> %s)", symbol, idx, len(chunks), from_d, to_d)
        raw = fetch_candles_chunk(token, instrument_key, from_d, to_d)
        df  = candles_to_df(raw)
        if not df.empty:
            frames.append(df)
            log.info("[%s]  -> %d bars", symbol, len(df))
        else:
            log.info("[%s]  -> no data (market holiday range or API gap)", symbol)

    if frames:
        new_data = pd.concat(frames, ignore_index=True)
        combined = merge_and_save(symbol, new_data)
        log.info(
            "[%s] Saved %d bars total  (%s to %s)  -> %s",
            symbol,
            len(combined),
            pd.to_datetime(combined["timestamp"]).min().date(),
            pd.to_datetime(combined["timestamp"]).max().date(),
            parquet_path(symbol),
        )
    else:
        log.warning("[%s] No new bars fetched for the requested date range.", symbol)

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("  Upstox 5-Min Data Puller  |  rate limit: %d req/s", RATE_LIMIT_RPS)
    log.info("  History: %d year(s)  |  chunk size: %d days", HISTORY_YEARS, CHUNK_DAYS)
    log.info("=" * 60)

    if not API_KEY or not API_SECRET or not REDIRECT_URL:
        raise SystemExit("Missing api_key / api_secret / redirect_url in .env")

    token    = get_access_token()
    resolved = resolve_keys(NSE_SYMBOLS)

    if not resolved:
        raise SystemExit("No valid instrument keys found. Fix the symbol list and re-run.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    success, failed = [], []
    for symbol, key in resolved.items():
        try:
            pull_symbol(token, symbol, key)
            success.append(symbol)
        except Exception as exc:
            log.error("[%s] Unhandled error: %s", symbol, exc, exc_info=True)
            failed.append(symbol)

    # ── Summary ──
    log.info("")
    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    for sym in success:
        p = parquet_path(sym)
        if p.exists():
            df = pd.read_parquet(p)
            ts = pd.to_datetime(df["timestamp"])
            log.info(
                "  %-16s  %6d bars   %s  ->  %s",
                sym, len(df), ts.min().date(), ts.max().date(),
            )
    if failed:
        log.warning("Failed symbols: %s", ", ".join(failed))
    log.info("")
    log.info("Data directory: %s", DATA_DIR.resolve())


if __name__ == "__main__":
    main()
