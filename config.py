"""
config.py
---------
Central configuration for the Nifty Total Market screening system.
"""

import os

NIFTY_TOTAL_MARKET_CSV = "https://niftyindices.com/IndexConstituent/ind_niftytotalmarket_list.csv"

SCREEN_CRITERIA = {
    "sales_growth_min": 20,
    "profit_growth_min": 40,
    "peg_ratio_max": 1,
    "promoter_holding_min": 60,
}


def build_screener_query() -> str:
    c = SCREEN_CRITERIA
    return (
        f"Sales growth >= {c['sales_growth_min']} AND "
        f"Profit growth >= {c['profit_growth_min']} AND "
        f"PEG Ratio <= {c['peg_ratio_max']} AND "
        f"Promoter holding >= {c['promoter_holding_min']}"
    )


BETA_MAX = 1.0
ACTIVE_RISK_MAX_PCT = 15.0  # cap on tracking error (stock vs benchmark), not absolute StdDev
TTM_TRADING_DAYS = 252

# --------------------------------------------------------------------------
# Benchmark selection — user-selectable instead of hard-coded.
# Maps a friendly UI label -> the exact index name nselib expects for
# capital_market.index_data(index=...).
# --------------------------------------------------------------------------
AVAILABLE_BENCHMARKS = {
    "NIFTY 50": "NIFTY 50",
    "NIFTY 200": "NIFTY 200",
    "NIFTY 500": "NIFTY 500",
    "NIFTY TOTAL MARKET": "NIFTY TOTAL MARKET",
}

# Default benchmark kept for backward compatibility with any code that
# still imports BENCHMARK_INDEX_NAME directly; the Streamlit UI overrides
# this at runtime via st.session_state / function args.
BENCHMARK_INDEX_NAME = "NIFTY TOTAL MARKET"


def require_screener_session() -> None:
    if not os.environ.get("SCREENER_SESSION", "").strip():
        raise RuntimeError(
            "SCREENER_SESSION environment variable not set. "
            "Log in to screener.in, copy the 'sessionid' cookie, and run:\n"
            "  export SCREENER_SESSION='paste-here'"
        )

# artifact refresh: 1784455641.5617282
