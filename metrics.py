"""
metrics.py
----------
Computes Beta, annualised Standard Deviation (volatility), and 12-1 month
Momentum for each stock, using trailing daily returns. Also returns the
BENCHMARK's own variance/std-dev so the app can disclose exactly what
every stock's Beta was measured against (issue: "app doesn\'t disclose
benchmark beta and variance").

Both stock and benchmark returns should be SPLIT/BONUS-ADJUSTED before
being passed in (see returns.py) so corporate actions don\'t distort Beta,
Std Dev, or Momentum. The benchmark index itself is user-selectable
(config.py / app.py) rather than hard-coded.
"""

import logging
import numpy as np
import pandas as pd

from config import TTM_TRADING_DAYS

logger = logging.getLogger("metrics")


def compute_risk_metrics(
    prices: pd.DataFrame,
    benchmark_prices: pd.Series,
) -> tuple[pd.DataFrame, dict]:
    """
    prices: DataFrame of daily close prices, columns = tickers, index = dates.
        Pass split/bonus-ADJUSTED prices (see returns.build_adjusted_price_frame).
    benchmark_prices: Series of daily close prices for the user-chosen
        benchmark index (NIFTY 50 / 200 / 500 / TOTAL MARKET).

    Returns:
        out: DataFrame indexed by ticker with columns Beta, StdDev_%,
            Active_Risk_%.
            - StdDev_%: the stock's own absolute annualised volatility,
              computed in isolation (used for ranking/Low-Vol style —
              NOT benchmark-relative).
            - Active_Risk_% (tracking error): annualised std dev of
              (stock daily return - benchmark daily return). This is the
              benchmark-relative risk measure and should be used for the
              risk-screen CAP, since it doesn't unfairly penalise/reward
              a stock just because the benchmark itself is in a high- or
              low-volatility regime.
        benchmark_stats: dict disclosing the benchmark's own Beta (=1 by
            definition), daily return variance, and annualised Std Dev —
            i.e. exactly what every stock's Beta/StdDev/Active_Risk were
            computed relative to. Surface this in the UI for transparency.
    """
    prices = prices.tail(TTM_TRADING_DAYS + 1)
    benchmark_prices = benchmark_prices.tail(TTM_TRADING_DAYS + 1)

    stock_returns = prices.pct_change().dropna(how="all")
    bench_returns = benchmark_prices.pct_change().dropna()

    aligned = stock_returns.join(bench_returns.rename("__bench__"), how="inner")
    bench_col = aligned["__bench__"]
    bench_var = bench_col.var()
    bench_std_annual_pct = bench_col.std() * np.sqrt(252) * 100 if len(bench_col) > 1 else np.nan

    benchmark_stats = {
        "Benchmark_Beta": 1.0,
        "Benchmark_Daily_Variance": bench_var,
        "Benchmark_StdDev_Annual_%": bench_std_annual_pct,
        "Benchmark_Observations": int(len(bench_col)),
    }

    results = {}
    for ticker in prices.columns:
        col = aligned[ticker].dropna()
        common = aligned.loc[col.index, "__bench__"]
        if len(col) < 30 or bench_var == 0:
            beta = np.nan
        else:
            cov = np.cov(col, common)[0, 1]
            beta = cov / bench_var
        std_dev_annual_pct = col.std() * np.sqrt(252) * 100 if len(col) > 1 else np.nan

        active_returns = col - common
        active_risk_annual_pct = (
            active_returns.std() * np.sqrt(252) * 100 if len(active_returns) > 1 else np.nan
        )

        results[ticker] = {
            "Beta": beta,
            "StdDev_%": std_dev_annual_pct,
            "Active_Risk_%": active_risk_annual_pct,
        }

    out = pd.DataFrame(results).T
    logger.info("Computed risk metrics for %d tickers.", len(out))
    return out, benchmark_stats


def compute_momentum(
    prices: pd.DataFrame,
    lookback_days: int = 252,
    skip_days: int = 21,
) -> pd.Series:
    """
    Classic academic "12-1 month" momentum (Jegadeesh-Titman / Fama-French
    UMD factor convention): total return over the trailing `lookback_days`
    (~12 months) EXCLUDING the most recent `skip_days` (~1 month), which
    avoids the well-documented short-term reversal effect.

    prices: split/bonus-ADJUSTED close prices, columns = tickers.
    Returns a Series indexed by ticker, values in percent.
    """
    if prices.empty:
        return pd.Series(dtype=float)

    momentum = {}
    for ticker in prices.columns:
        s = prices[ticker].dropna()
        if len(s) < lookback_days // 2:
            momentum[ticker] = np.nan
            continue
        end_price = s.iloc[-skip_days - 1] if len(s) > skip_days else s.iloc[-1]
        start_idx = max(0, len(s) - lookback_days - skip_days - 1)
        start_price = s.iloc[start_idx]
        momentum[ticker] = (end_price / start_price - 1.0) * 100 if start_price else np.nan

    return pd.Series(momentum, name="Momentum_%")

# artifact refresh: 1784455641.5617282
