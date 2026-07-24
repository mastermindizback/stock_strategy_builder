"""
screening.py
------------
Two-phase screening pipeline (matches the two-button Streamlit UI):

Phase 1 — run_fundamental_screen(): universe + Screener.in filters,
          sector enrichment, AND now returns BOTH the in-universe
          shortlist and the full Screener.in result set (with an
          `In_Universe` flag) so the user can see the 300+ stocks that
          passed fundamentals but fall outside the 750-stock Nifty Total
          Market universe, instead of only the ~26 that intersect it.
Phase 2 — run_risk_screen(): takes a shortlist, fetches TTM prices via
          nselib, SPLIT/BONUS-ADJUSTS them (returns.py), fetches the
          user-selected benchmark, computes Beta/Std Dev/Momentum, and
          applies the risk filters PASSED IN AS ARGUMENTS (not the module
          defaults) — fixes the bug where changing the sidebar's Beta/
          StdDev inputs had no effect on the actual screen.
"""

import logging
import pandas as pd

from config import (
    build_screener_query,
    BETA_MAX,
    ACTIVE_RISK_MAX_PCT,
    AVAILABLE_BENCHMARKS,
    BENCHMARK_INDEX_NAME,
    require_screener_session,
)
from universe import fetch_nifty_total_market, universe_symbols
from data_fetcher import fetch_screener_universe
from price_fetcher import fetch_stock_prices, fetch_benchmark_prices
from metrics import compute_risk_metrics, compute_momentum
from sector import enrich_with_sector
from returns import build_adjusted_price_frame

logger = logging.getLogger("screening")


def run_fundamental_screen(progress_cb=None) -> dict:
    """
    Returns a dict:
        {
            "in_universe": DataFrame — stocks that pass fundamentals AND
                            belong to the 750-stock Nifty Total Market universe,
            "all_screened": DataFrame — every stock that passed Screener.in
                            fundamentals, with an `In_Universe` bool column,
            "universe_size": int,
        }
    """
    def note(msg: str):
        logger.info(msg)
        if progress_cb:
            progress_cb(msg)

    require_screener_session()

    note("Fetching Nifty Total Market universe (750 stocks) …")
    universe_df = fetch_nifty_total_market()
    universe_set = set(universe_symbols(universe_df))
    note(f"Universe loaded: {len(universe_set)} symbols.")

    query = build_screener_query()
    note(f"Running Screener.in query: {query}")
    fundamentals = fetch_screener_universe(query)

    if fundamentals.empty:
        note("Screener.in returned zero rows — check SCREENER_SESSION and query syntax.")
        empty = pd.DataFrame()
        return {"in_universe": empty, "all_screened": empty, "universe_size": len(universe_set)}

    ticker_symbol = fundamentals["Ticker"].astype(str).str.replace(".NS", "", regex=False).str.upper()
    fundamentals = fundamentals.copy()
    fundamentals["In_Universe"] = ticker_symbol.isin(universe_set)
    note(
        f"Screener.in fundamentals matched {len(fundamentals)} stocks total; "
        f"{fundamentals['In_Universe'].sum()} fall inside the Nifty Total Market universe, "
        f"{(~fundamentals['In_Universe']).sum()} are outside it."
    )

    fundamentals = enrich_with_sector(fundamentals, universe_df)
    note("Sector tagging complete (source: NSE Industry classification where available).")

    in_universe = fundamentals[fundamentals["In_Universe"]].reset_index(drop=True)
    all_screened = fundamentals.reset_index(drop=True)

    return {
        "in_universe": in_universe,
        "all_screened": all_screened,
        "universe_size": len(universe_set),
    }


def run_risk_screen(
    fundamentals: pd.DataFrame,
    start_date: str,
    end_date: str,
    beta_max: float = BETA_MAX,
    active_risk_max_pct: float = ACTIVE_RISK_MAX_PCT,
    benchmark_label: str = "NIFTY TOTAL MARKET",
    adjust_for_splits: bool = True,
    progress_cb=None,
) -> tuple[pd.DataFrame, dict]:
    """
    beta_max / active_risk_max_pct: passed explicitly from the UI's current
    sidebar values (previously this function silently re-imported the
    module-level BETA_MAX / STD_DEV_MAX_PCT constants captured at import
    time, so sidebar changes were ignored — now fixed).

    The risk-screen CAP now uses Beta + Active Risk (tracking error vs. the
    chosen benchmark), NOT absolute StdDev_%. Absolute StdDev_% is still
    computed and carried through in the output for use as a RANKING factor
    (e.g. the Low Vol style) — capping on it directly would unfairly
    penalise/reward stocks depending on whether the benchmark itself is in
    a high- or low-volatility regime, since it ignores the benchmark
    entirely. Active Risk isolates how much a stock's returns diverge from
    the benchmark's, which is the statistically correct thing to cap when
    the whole screen is meant to be benchmark-relative.

    Returns (final_shortlist_df, benchmark_stats_dict). benchmark_stats
    discloses the benchmark's own daily variance and annualised Std Dev
    so Beta can be interpreted correctly in the UI.
    """
    def note(msg: str):
        logger.info(msg)
        if progress_cb:
            progress_cb(msg)

    if fundamentals.empty:
        return fundamentals, {}

    benchmark_index_name = AVAILABLE_BENCHMARKS.get(benchmark_label, BENCHMARK_INDEX_NAME)

    symbols = (
        fundamentals["Ticker"].astype(str).str.replace(".NS", "", regex=False).str.upper().dropna().unique().tolist()
    )

    note(f"Downloading TTM daily prices for {len(symbols)} shortlisted stocks via yfinance …")
    stock_prices, missing_symbols = fetch_stock_prices(symbols, start=start_date, end=end_date)

    if missing_symbols:
      note(f"⚠️ No yfinance price data for {len(missing_symbols)} symbol(s): {', '.join(missing_symbols)}")

    if adjust_for_splits and not stock_prices.empty:
      note("Applying split/bonus adjustment to raw prices …")
      stock_prices = build_adjusted_price_frame(stock_prices, start=start_date, end=end_date)

    note(f"Downloading benchmark prices for {benchmark_index_name} …")
    benchmark_prices = fetch_benchmark_prices(benchmark_index_name, start=start_date, end=end_date)

    note("Computing Beta, annualised Std Dev, and Momentum (TTM) …")
    risk, benchmark_stats = compute_risk_metrics(stock_prices, benchmark_prices)
    risk.index = [f"{s}" for s in risk.index]
    benchmark_stats["Benchmark_Label"] = benchmark_label
    benchmark_stats["Benchmark_Index_Name"] = benchmark_index_name

    momentum = compute_momentum(stock_prices)

    fundamentals = fundamentals.copy()
    fundamentals["_sym"] = fundamentals["Ticker"].astype(str).str.replace(".NS", "", regex=False).str.upper()
    merged = fundamentals.merge(risk, left_on="_sym", right_index=True, how="left")
    merged = merged.merge(momentum.rename("Momentum_%"), left_on="_sym", right_index=True, how="left")
    merged = merged.drop(columns=["_sym"])

    # final = merged[(merged["Beta"] < beta_max) & (merged["Active_Risk_%"] < active_risk_max_pct)].copy()
    final = merged.copy()
    note(
        f"Final shortlist after Beta < {beta_max} and Active Risk (tracking error) < "
        f"{active_risk_max_pct}% (benchmark: {benchmark_label}): {len(final)} stocks. "
        f"Note: absolute StdDev_% is NOT used for this cap — it's retained in the output "
        f"purely as a ranking factor."
    )

    cols = [c for c in [
        "Name", "Ticker", "Sector", "Market Cap", "P/E", "ROE", "ROCE",
        "Sales Growth", "Profit Growth", "Debt to Equity", "Piotroski score", "Promoter Holding",
        "Beta", "StdDev_%", "Active_Risk_%", "Momentum_%"
    ] if c in final.columns]
    return final[cols].sort_values("Beta").reset_index(drop=True), benchmark_stats

# artifact refresh: 1784455641.5617282
