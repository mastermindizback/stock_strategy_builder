"""
price_fetcher.py
----------------
Replaces yfinance with nselib for fetching daily price history — for both
individual stocks and the Nifty Total Market benchmark index.

Why: yfinance depends on curl_cffi / native extensions that have caused
segmentation faults when combined with Streamlit'''s file-watcher and thread
pool on some macOS setups (Python 3.11+/ARM). nselib is pure-Python
(requests + pandas), which avoids that entire class of crash.
"""

import logging
import time
import pandas as pd
from nselib import capital_market, indices

logger = logging.getLogger("price_fetcher")


def _to_ddmmyyyy(iso_date: str) -> str:
    return pd.Timestamp(iso_date).strftime("%d-%m-%Y")




def fetch_stock_prices(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    """
    symbols: plain NSE symbols WITHOUT .NS suffix (e.g. ["RELIANCE", "TCS"])
    Returns a DataFrame of daily Close Price, columns = symbols, index = dates.
    """
    from_date = _to_ddmmyyyy(start)
    to_date = _to_ddmmyyyy(end)

    series = {}
    for sym in symbols:
        try:
            df = capital_market.price_volume_data(symbol=sym, from_date=from_date, to_date=to_date)
            if df is None or df.empty:
                logger.warning("No price data for %s", sym)
                continue

            df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True, errors="coerce")
            df = df.dropna(subset=["Date"])
            df = df.drop_duplicates(subset="Date", keep="last").set_index("Date").sort_index()

            close_col = "Close Price" if "Close Price" in df.columns else "ClosePrice"
            s = pd.to_numeric(
                df[close_col].astype(str).str.replace(",", "", regex=False), errors="coerce"
            )

            # Hard guarantee: collapse ANY remaining duplicate index labels,
            # regardless of why the earlier dedup missed them.
            if not s.index.is_unique:
                dupe_dates = s.index[s.index.duplicated()].unique().tolist()
                logger.warning(
                    "%s still had %d duplicate date(s) after drop_duplicates: %s — collapsing with groupby.last()",
                    sym, len(dupe_dates), dupe_dates,
                )
                s = s.groupby(level=0).last()
                print("Not unique")
                print(s.shape)
            series[sym] = s
        except Exception as exc:
            logger.warning("Price fetch failed for %s: %s", sym, exc)
        time.sleep(0.3)  # be polite to NSE

    if not series:
        return pd.DataFrame()

    # Final belt-and-suspenders check before DataFrame construction, with
    # a clear error message identifying the exact offending symbol(s)
    # instead of pandas' generic ValueError.
    bad_syms = [sym for sym, s in series.items() if not s.index.is_unique]
    if bad_syms:
        raise ValueError(f"Duplicate date index still present for symbols: {bad_syms}")
    
    prices = pd.DataFrame(series).ffill().dropna(how="all")
    logger.info("Fetched prices for %d/%d symbols.", len(series), len(symbols))
    print(prices.shape)
    return prices




def fetch_benchmark_prices(index_name: str, start: str, end: str) -> pd.Series:
    """
    Fetches historical OHLC for an NSE index (e.g. "NIFTY TOTAL MARKET",
    "NIFTY 500") via nselib.capital_market.index_data.

    NSE's underlying API caps each request to ~90 days of data, so this
    chunks the requested range into <=85-day windows and concatenates.
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    chunks = []
    cur_start = start_ts
    while cur_start <= end_ts:
        cur_end = min(cur_start + pd.Timedelta(days=85), end_ts)
        from_date = cur_start.strftime("%d-%m-%Y")
        to_date = cur_end.strftime("%d-%m-%Y")

        try:
            df = capital_market.index_data(index=index_name, from_date=from_date, to_date=to_date)
            if df is not None and not df.empty:
                chunks.append(df)
                print(f"Got {len(df)} rows for {index_name}: {from_date} -> {to_date}")
            else:
                print(f"Empty chunk for {index_name}: {from_date} -> {to_date}")
        except Exception as exc:
            print(f"Chunk fetch failed ({from_date} -> {to_date}): {exc}")

        time.sleep(0.5)  # avoid NSE rate limiting
        cur_start = cur_end + pd.Timedelta(days=1)

    if not chunks:
        raise RuntimeError(f"nselib returned no data for benchmark index {index_name!r}")

    df = pd.concat(chunks, ignore_index=True)
    print("Got benchmark data for", index_name)
    print(df.columns)

    df["Date"] = pd.to_datetime(df["TIMESTAMP"], format="mixed", dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.drop_duplicates(subset="Date", keep="last")  # dedupe overlapping chunk boundaries
    df = df.set_index("Date").sort_index()

    df = df.rename(columns={
        "OPEN_INDEX_VAL": "Open",
        "HIGH_INDEX_VAL": "High",
        "LOW_INDEX_VAL": "Low",
        "CLOSE_INDEX_VAL": "Close",
    })
    close_col = "Close" if "Close" in df.columns else "Close Price"
    print(df.shape)

    series = pd.to_numeric(df[close_col].astype(str).str.replace(",", "", regex=False), errors="coerce")
    series = series.groupby(series.index).last()  # extra safety against any remaining dupes
    return series
