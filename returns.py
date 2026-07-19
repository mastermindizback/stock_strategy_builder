"""
returns.py
----------
Split/bonus-adjusted return calculations.

Problem: price_fetcher.py pulls raw daily Close Price from NSE archives.
Raw close-to-close pct_change() across a stock-split or bonus-issue date
produces a spurious huge negative "return" (e.g. a 1:2 split halves the
price overnight with no economic loss). This module detects corporate
actions via nselib and builds split/bonus-adjusted price & return series
so Beta / Std Dev / any ranking factor built on returns stays correct.
"""

import logging
import re
import numpy as np
import pandas as pd
from nselib import capital_market

logger = logging.getLogger("returns")


def _parse_ratio_from_subject(subject: str) -> float | None:
    """
    NSE corporate-action 'SUBJECT' strings look like:
      "Face Value Split (Sub Division) - From Rs 10/- Per Share To Rs 2/- Per Share"
      "Bonus Issue 1:1"
      "Bonus Issue 2:3"
    Returns the multiplicative adjustment factor to apply to PRE-action
    historical prices (i.e. divide old prices by this factor), or None if
    the subject text doesn't parse.
    """
    if not isinstance(subject, str):
        return None
    s = subject.strip().lower()

    # --- Bonus issue: "bonus issue 1:1" -> holder gets 1 extra share per 1
    # held -> shares outstanding x2 -> price roughly halves.
    m = re.search(r"bonus.*?(\d+)\s*:\s*(\d+)", s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if b > 0:
            return (a + b) / b  # multiply factor for share count growth

    # --- Face value split: "...from rs 10/- per share to rs 2/- per share"
    m = re.search(r"from\s*rs\.?\s*([\d.]+).*?to\s*rs\.?\s*([\d.]+)", s)
    if m:
        old_fv, new_fv = float(m.group(1)), float(m.group(2))
        if new_fv > 0:
            return old_fv / new_fv  # e.g. 10 -> 2 gives factor 5

    # --- Simple "split 1:5" or "stock split 5:1" style text
    m = re.search(r"split.*?(\d+)\s*:\s*(\d+)", s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if b > 0:
            return a / b if a > b else b / a

    return None


def fetch_corporate_action_factors(symbol: str, start: str, end: str) -> pd.Series:
    """
    Returns a pd.Series indexed by ex-date, values = cumulative adjustment
    factor to apply to ALL prices strictly BEFORE that ex-date (standard
    backward/rolling price adjustment, same convention as yfinance
    'Adj Close').
    """
    from_date = pd.Timestamp(start).strftime("%d-%m-%Y")
    to_date = pd.Timestamp(end).strftime("%d-%m-%Y")

    try:
        df = capital_market.corporate_actions_for_equity(
            from_date=from_date, to_date=to_date, fno_only=False
        )
    except Exception as exc:
        logger.warning("Corporate action fetch failed for %s: %s", symbol, exc)
        return pd.Series(dtype=float)

    if df is None or df.empty:
        return pd.Series(dtype=float)

    sym_col = next((c for c in df.columns if c.lower() in ("symbol", "sym")), None)
    subj_col = next((c for c in df.columns if "subject" in c.lower() or "purpose" in c.lower()), None)
    date_col = next((c for c in df.columns if "exdate" in c.lower().replace(" ", "") or "ex-date" in c.lower()), None)
    if date_col is None:
        date_col = next((c for c in df.columns if "date" in c.lower()), None)

    if sym_col is None or subj_col is None or date_col is None:
        logger.warning("Unexpected corporate-actions columns: %s", list(df.columns))
        return pd.Series(dtype=float)

    rows = df[df[sym_col].astype(str).str.upper() == symbol.upper()].copy()
    if rows.empty:
        return pd.Series(dtype=float)

    rows["_ratio"] = rows[subj_col].apply(_parse_ratio_from_subject)
    rows = rows.dropna(subset=["_ratio"])
    if rows.empty:
        return pd.Series(dtype=float)

    rows["_exdate"] = pd.to_datetime(rows[date_col], errors="coerce", dayfirst=True)
    rows = rows.dropna(subset=["_exdate"]).sort_values("_exdate")

    return rows.set_index("_exdate")["_ratio"]


def adjust_prices_for_splits(prices: pd.Series, action_factors: pd.Series) -> pd.Series:
    """
    Applies backward (Yahoo-style) split/bonus adjustment to a raw close
    price series so that pct_change() across corporate-action dates is
    economically correct.

    prices: raw close prices, indexed by date, sorted ascending.
    action_factors: output of fetch_corporate_action_factors() — index =
        ex-date, value = ratio by which the share count grew (equivalently,
        by which the pre-action price should be divided).
    """
    if action_factors is None or action_factors.empty:
        return prices.copy()

    adjusted = prices.copy().astype(float)
    for ex_date, factor in action_factors.items():
        if factor is None or factor <= 0:
            continue
        mask = adjusted.index < ex_date
        adjusted.loc[mask] = adjusted.loc[mask] / factor
    return adjusted


def compute_adjusted_returns(prices: pd.Series, symbol: str, start: str, end: str) -> pd.Series:
    """
    One-call convenience: fetch corporate actions for `symbol` over the
    price window, apply split/bonus adjustment, and return daily pct
    returns computed on the adjusted series (safe for Beta/Std Dev/CAGR).
    """
    factors = fetch_corporate_action_factors(symbol, start, end)
    adj_prices = adjust_prices_for_splits(prices, factors)
    return adj_prices.pct_change().dropna()


def build_adjusted_price_frame(prices: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """
    Batch version for a full price DataFrame (columns = symbols). Fetches
    corporate actions per symbol and returns a NEW DataFrame of
    split/bonus-adjusted close prices, same shape as input.
    """
    adjusted_cols = {}
    for sym in prices.columns:
        try:
            factors = fetch_corporate_action_factors(sym, start, end)
            adjusted_cols[sym] = adjust_prices_for_splits(prices[sym].dropna(), factors)
        except Exception as exc:
            logger.warning("Split adjustment failed for %s, using raw prices: %s", sym, exc)
            adjusted_cols[sym] = prices[sym]
    return pd.DataFrame(adjusted_cols)
