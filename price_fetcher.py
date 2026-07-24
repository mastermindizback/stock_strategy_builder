"""
price_fetcher.py
----------------
Fetches daily price history via yfinance instead of nselib.

Why: nselib calls NSE's endpoints directly, which block Streamlit Community
Cloud / most cloud-provider IP ranges. yfinance hits Yahoo Finance's CDN,
which works reliably from cloud environments.

Returns missing-symbol info alongside prices so the UI can warn users
which securities had no data (e.g. delisted, wrong symbol, illiquid).
"""

import logging
import pandas as pd
import yfinance as yf

logger = logging.getLogger("price_fetcher")

def _to_yf_symbol(sym: str) -> str:
    """NSE plain symbol -> Yahoo Finance .NS suffix (unless already suffixed)."""
    sym = sym.strip().upper()
    if sym.startswith("^") or sym.endswith((".NS", ".BO")):
        return sym
    return f"{sym}.NS"

def fetch_stock_prices(symbols: list[str], start: str, end: str) -> tuple[pd.DataFrame, list[str]]:
    """
    symbols: plain NSE symbols WITHOUT .NS suffix (e.g. ["RELIANCE", "TCS"])
    Returns (prices_df, missing_symbols):
      prices_df: daily Close, columns = original symbols (no suffix), index = dates
      missing_symbols: symbols yfinance returned no usable data for
    """
    if not symbols:
        return pd.DataFrame(), []

    yf_map = {_to_yf_symbol(s): s for s in symbols}
    tickers = list(yf_map.keys())

    raw = yf.download(
        tickers, start=start, end=end,
        auto_adjust=True, progress=False, threads=True, group_by="column",
    )

    if raw.empty:
        logger.warning("yfinance returned no data for any of %d symbols.", len(symbols))
        return pd.DataFrame(), symbols

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else pd.DataFrame()
    else:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]}) if "Close" in raw.columns else pd.DataFrame()

    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])

    close = close.rename(columns=yf_map)
    missing = [s for s in symbols if s not in close.columns or close[s].dropna().empty]
    close = close.drop(columns=[c for c in close.columns if c in missing], errors="ignore")
    close = close.ffill().dropna(how="all")

    if missing:
        logger.warning("No price data from yfinance for %d symbol(s): %s", len(missing), missing)

    logger.info("Fetched prices for %d/%d symbols.", close.shape[1], len(symbols))
    return close, missing

BENCHMARK_YF_SYMBOLS = {
    "NIFTY 50": "^NSEI",
    "BSE SENSEX": "^BSESN",
}

def fetch_benchmark_prices(index_name: str, start: str, end: str) -> pd.Series:
    """
    index_name: friendly label ("NIFTY 50" or "BSE SENSEX") OR a raw yfinance
    symbol like "^NSEI"/"^BSESN" directly.
    """
    symbol = BENCHMARK_YF_SYMBOLS.get(index_name, index_name)
    df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"yfinance returned no data for benchmark {symbol!r}")

    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.name = symbol
    return close.dropna()