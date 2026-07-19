"""
universe.py
-----------
Fetches the Nifty Total Market (750-stock) constituent list and maps
each symbol to the ticker format Screener.in / Yahoo Finance expect.
"""

import logging
import io
import requests
import pandas as pd

from config import NIFTY_TOTAL_MARKET_CSV

logger = logging.getLogger("universe")

_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def fetch_nifty_total_market(cache_path: str = "nifty_total_market.csv") -> pd.DataFrame:
    """
    Downloads the official Nifty Total Market constituent CSV from
    niftyindices.com. Falls back to a local cache if the network call fails.

    Returns a DataFrame with columns: Company Name, Industry, Symbol, Series, ISIN Code
    """
    try:
        resp = requests.get(NIFTY_TOTAL_MARKET_CSV, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        df.to_csv(cache_path, index=False)
        logger.info("Fetched %d Nifty Total Market constituents.", len(df))
        return df
    except Exception as exc:
        logger.warning("Live fetch failed (%s). Trying local cache %r.", exc, cache_path)
        try:
            return pd.read_csv(cache_path)
        except FileNotFoundError:
            raise RuntimeError(
                "Could not fetch Nifty Total Market constituents online, "
                "and no local cache exists. Check your internet connection."
            )


def universe_symbols(df: pd.DataFrame) -> list[str]:
    """Plain NSE symbols (no suffix) — used for Screener DSL matching."""
    return df["Symbol"].dropna().astype(str).str.strip().str.upper().tolist()


def universe_yf_tickers(df: pd.DataFrame) -> list[str]:
    """Yahoo Finance-ready tickers with the .NS suffix."""
    return [f"{s}.NS" for s in universe_symbols(df)]
