"""
get_token.py — Daily Upstox access token generator.

Run once each morning before live_trader.py:
    python get_token.py

What it does:
    1. Reads api_key, api_secret, redirect_url from .env
    2. Opens the Upstox login page in your browser
    3. Asks you to paste the redirect URL (contains ?code=...)
    4. Exchanges the code for an access token
    5. Writes UPSTOX_ACCESS_TOKEN to .env automatically

Tokens expire at midnight IST — run this script every morning.
"""

import os
import sys
import webbrowser
from pathlib import Path
from urllib.parse import urlencode, urlparse, parse_qs

import requests
from dotenv import load_dotenv, set_key

load_dotenv()

ENV_FILE  = Path(".env")
AUTH_URL  = "https://api.upstox.com/v2/login/authorization/dialog"
TOKEN_URL = "https://api.upstox.com/v2/login/authorization/token"


def main() -> None:
    api_key      = os.getenv("api_key",      "").strip()
    api_secret   = os.getenv("api_secret",   "").strip()
    redirect_url = os.getenv("redirect_url", "http://127.0.0.1/").strip()

    if not api_key or not api_secret:
        sys.exit("ERROR: api_key and api_secret must be set in .env")

    # ── Step 1: open login page ───────────────────────────────────────────────
    login_url = AUTH_URL + "?" + urlencode({
        "response_type": "code",
        "client_id":     api_key,
        "redirect_uri":  redirect_url,
    })

    print("\nOpening Upstox login page in your browser...")
    print(f"  {login_url}\n")
    webbrowser.open(login_url)

    print("After logging in, your browser will redirect to a URL like:")
    print(f"  {redirect_url}?code=AbCd1234&state=...\n")
    print("The page may show a connection error — that's fine.")
    print("Just copy the full URL from your browser's address bar.\n")

    # ── Step 2: paste redirect URL ────────────────────────────────────────────
    redirect = input("Paste the full redirect URL here: ").strip()
    if not redirect:
        sys.exit("No URL entered — exiting.")

    # ── Step 3: extract code ──────────────────────────────────────────────────
    parsed = urlparse(redirect)
    params = parse_qs(parsed.query)
    codes  = params.get("code") or params.get("code[]")
    if not codes:
        sys.exit(
            "ERROR: No 'code' parameter found in that URL.\n"
            "Make sure you pasted the full URL including the ?code=... part."
        )
    code = codes[0]
    print(f"\nAuth code extracted: {code[:8]}...")

    # ── Step 4: exchange code for token ───────────────────────────────────────
    print("Exchanging code for access token...")
    try:
        resp = requests.post(
            TOKEN_URL,
            data={
                "code":          code,
                "client_id":     api_key,
                "client_secret": api_secret,
                "redirect_uri":  redirect_url,
                "grant_type":    "authorization_code",
            },
            headers={
                "Accept":       "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            timeout=30,
        )
    except requests.RequestException as exc:
        sys.exit(f"Network error during token exchange: {exc}")

    if not resp.ok:
        sys.exit(
            f"ERROR: Token exchange failed (HTTP {resp.status_code})\n"
            f"Upstox response: {resp.text[:400]}"
        )

    body  = resp.json()
    token = body.get("access_token")
    if not token:
        sys.exit(f"ERROR: No access_token in response: {body}")

    # ── Step 5: save to .env ──────────────────────────────────────────────────
    set_key(str(ENV_FILE), "UPSTOX_ACCESS_TOKEN", token)

    print(f"\nToken saved to {ENV_FILE}.")
    print("Token saved. Valid until midnight.")
    print(f"  UPSTOX_ACCESS_TOKEN={token[:20]}...{token[-6:]}")
    print("\nYou can now run:  python live_trader.py")


if __name__ == "__main__":
    main()
