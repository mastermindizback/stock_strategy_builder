"""
app.py
------
Streamlit UI for the Nifty Total Market stock screening system.

Workflow:
  Step 1 - Fundamental screen (Screener.in) + sector tagging. Shows BOTH
           the in-universe shortlist and the full screened set (300+
           stocks) with an "outside universe" view toggle.
  Step 2 - Risk screen: user-selectable benchmark, split-adjusted Beta,
           Std Dev, and Momentum, with the benchmark own variance and
           Std Dev disclosed alongside every stock Beta for context.
  Step 3 - Style-based ranking: user picks ONE of Low Vol, Quality,
           Growth, Value, Momentum from a dropdown. The underlying
           factor weights/formulas stay in the backend (ranking.py) -
           nothing tunable is exposed in the UI.

Run with:
    export SCREENER_SESSION="your-sessionid-cookie"
    streamlit run app.py
"""

import datetime
import logging
import os

import numpy as np
import pandas as pd
import streamlit as st

import config
from screening import run_fundamental_screen, run_risk_screen
from ranking import compute_style_rank, STYLE_DESCRIPTIONS
from sector import sector_neutral_summary, sparse_sector_warning

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Nifty 750 Stock Screener", layout="wide")
st.title("Nifty Total Market (750) Stock Screener")
st.caption(
    "Universe: Nifty Total Market Index (750 stocks). "
    "Step 1: fundamental filters via Screener.in + sector tagging (shows in- and out-of-universe results). "
    "Step 2: risk filters (split-adjusted Beta, Std Dev, Momentum) vs. a benchmark of your choice; benchmark stats disclosed. "
    "Step 3: rank by a pre-built quant style factor."
)


def safe_for_arrow(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitize a DataFrame before handing it to st.dataframe()/st.download_button().
    PyArrow Rust serializer can segfault on mixed-type object columns,
    inf/-inf values, or stale indices - keep pyarrow<25 pinned in
    requirements AND keep this sanitizer as defence-in-depth.
    """
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).replace({"nan": "", "None": ""})
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype(float)
    df = df.reset_index(drop=True)
    return df


for key, default in [
    ("fundamentals_result", None), ("final", None), ("benchmark_stats", None), ("ranked", None)
]:
    if key not in st.session_state:
        st.session_state[key] = default

with st.sidebar:
    st.header("Step 1 - Fundamental Filters")
    sales_growth = st.number_input("Sales Growth % (min)", value=config.SCREEN_CRITERIA["sales_growth_min"])
    pat_growth = st.number_input("Profit Growth % (min)", value=config.SCREEN_CRITERIA["profit_growth_min"])
    peg_max = st.number_input("Max PEG Ratio", value=float(config.SCREEN_CRITERIA["peg_ratio_max"]), step=0.1)
    promoter_min = st.number_input("Min Promoter Holding (%)", value=config.SCREEN_CRITERIA["promoter_holding_min"])

    st.header("Step 2 - Risk Filters (TTM)")
    benchmark_label = st.selectbox(
        "Benchmark index",
        options=list(config.AVAILABLE_BENCHMARKS.keys()),
        index=0,
        help="Beta is computed relative to whichever benchmark you pick here.",
    )
    adjust_splits = st.checkbox(
        "Adjust for stock splits / bonus issues", value=True,
        help="Uses NSE corporate-action data to correct returns around split/bonus ex-dates.",
    )
    beta_max_input = config.BETA_MAX
    active_risk_max_input = config.ACTIVE_RISK_MAX_PCT
    # beta_max_input = st.number_input("Max Beta", value=config.BETA_MAX, step=0.05, key="beta_max_input")
    # active_risk_max_input = st.number_input(
    #     "Max Active Risk / Tracking Error (annualised %)",
    #     value=config.ACTIVE_RISK_MAX_PCT, step=1.0, key="active_risk_max_input",
    #     help=(
    #         "Caps how much a stock's returns diverge from the chosen benchmark's "
    #         "returns (tracking error), not the stock's absolute volatility. "
    #         "Absolute Std Dev is still shown and used later for ranking (e.g. Low Vol "
    #         "style), but is not used to filter here — it would unfairly penalise/reward "
    #         "stocks depending on the benchmark's own volatility regime."
    #     ),
    # )

    st.header("Session")
    session_cookie = st.text_input(
        "SCREENER_SESSION cookie", type="password",
        help="Log in to screener.in, DevTools, Cookies, sessionid",
    )

    step1_btn = st.button("Step 1: Run Fundamental Screen", width="stretch")
    step2_btn = st.button(
        "Step 2: Run Risk Screen",
        width="stretch",
        disabled=(
            st.session_state.fundamentals_result is None
            or st.session_state.fundamentals_result["in_universe"].empty
        ),
    )

config.SCREEN_CRITERIA.update({
    "sales_growth_min": sales_growth,
    "profit_growth_min": pat_growth,
    "peg_ratio_max": peg_max,
    "promoter_holding_min": promoter_min,
})

if session_cookie:
    os.environ["SCREENER_SESSION"] = session_cookie

log_box = st.empty()
logs: list[str] = []


def progress(msg: str):
    logs.append(msg)
    log_box.code("\n".join(logs[-15:]))


if step1_btn:
    with st.spinner("Running fundamental screen on Nifty Total Market universe..."):
        try:
            st.session_state.fundamentals_result = run_fundamental_screen(progress_cb=progress)
            st.session_state.final = None
            st.session_state.benchmark_stats = None
            st.session_state.ranked = None
        except Exception as exc:
            st.error(f"Fundamental screen failed: {exc}")
            st.stop()
    st.rerun()

if st.session_state.fundamentals_result is not None:
    res = st.session_state.fundamentals_result
    in_universe_df = res["in_universe"]
    all_screened_df = res["all_screened"]
    universe_size = res["universe_size"]

    st.subheader(
        f"Step 1 results - {len(all_screened_df)} stocks pass fundamental filters "
        f"({len(in_universe_df)} inside the {universe_size}-stock Nifty Total Market universe, "
        f"{len(all_screened_df) - len(in_universe_df)} outside it)"
    )

    view_choice = st.radio(
        "Show:",
        options=["In-universe only (used for Step 2)", "All screened stocks (incl. outside universe)"],
        horizontal=True,
    )
    display_df = in_universe_df if view_choice.startswith("In-universe") else all_screened_df

    if display_df.empty:
        st.warning("No stocks passed the fundamental filters. Try loosening thresholds.")
    else:
        safe_display = safe_for_arrow(display_df)
        st.dataframe(safe_display, width="stretch")
        st.download_button(
            "Download this view (CSV)",
            safe_display.to_csv(index=False).encode(),
            "nifty750_fundamental_screen.csv",
            "text/csv",
        )
        if "Sector" in display_df.columns:
            with st.expander("Sector breakdown (confirms sector-neutrality)"):
                st.dataframe(safe_for_arrow(sector_neutral_summary(display_df)), width="stretch")

if step2_btn and st.session_state.fundamentals_result is not None:
    in_universe_df = st.session_state.fundamentals_result["in_universe"]
    if not in_universe_df.empty:
        end_date = datetime.date.today().isoformat()
        start_date = (datetime.date.today() - datetime.timedelta(days=400)).isoformat()
        with st.spinner(f"Fetching prices vs {benchmark_label} and computing Beta / Std Dev / Momentum via nselib..."):
            try:
                final_df, bench_stats, missing_syms = run_risk_screen(
                    in_universe_df, start_date, end_date,
                    beta_max=beta_max_input,
                    active_risk_max_pct=active_risk_max_input,
                    benchmark_label=benchmark_label,
                    adjust_for_splits=adjust_splits,
                    progress_cb=progress,
                )
                st.session_state.final = final_df
                st.session_state.benchmark_stats = bench_stats
                st.session_state.ranked = None
                st.session_state.missing_symbols = missing_syms
                if st.session_state.get("missing_symbols"):
                    st.warning(
                        f"No price data found on Yahoo Finance for: {', '.join(st.session_state.missing_symbols)}. "
                        "These stocks are excluded from Beta/StdDev/Momentum but were still fundamentally screened."
                    )
            except Exception as exc:
                st.error(f"Risk screen failed: {exc}")
                st.stop()
        st.rerun()

if st.session_state.final is not None:
    result = st.session_state.final
    bench_stats = st.session_state.benchmark_stats or {}

    st.subheader(f"Step 2 results - final shortlist: {len(result)} stocks (benchmark: {benchmark_label})")

    if bench_stats:
        bc1, bc2, bc3, bc4 = st.columns(4)
        bc1.metric("Benchmark", bench_stats.get("Benchmark_Label", benchmark_label))
        bc2.metric("Benchmark Beta (by definition)", f"{bench_stats.get('Benchmark_Beta', 1.0):.2f}")
        bc3.metric("Benchmark daily variance", f"{bench_stats.get('Benchmark_Daily_Variance', float('nan')):.6f}")
        bc4.metric("Benchmark Std Dev (annualised %)", f"{bench_stats.get('Benchmark_StdDev_Annual_%', float('nan')):.2f}%")
        st.caption(
            f"Filters applied: Beta < {beta_max_input}, Active Risk (tracking error vs. "
            f"{bench_stats.get('Benchmark_Index_Name', benchmark_label)}) < {active_risk_max_input}%. "
            f"Absolute Std Dev_% is shown in the table below for reference and used later in "
            f"ranking, but is not part of this cap."
        )

    if result.empty:
        st.warning("No stocks passed the risk filters. Try loosening Beta / Active Risk thresholds.")
    else:
        st.success(f"{len(result)} stocks passed all filters.")
        safe_result = safe_for_arrow(result)
        st.dataframe(safe_result, width="stretch")
        st.download_button(
            "Download final shortlist (CSV)",
            safe_result.to_csv(index=False).encode(),
            "nifty750_final_screen.csv",
            "text/csv",
        )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Beta distribution")
            st.bar_chart(result.set_index("Name")["Beta"])
        with col2:
            if "Active_Risk_%" in result.columns:
                st.subheader("Active Risk / Tracking Error (annualised %) distribution")
                st.bar_chart(result.set_index("Name")["Active_Risk_%"])
            else:
                st.subheader("Std Dev (annualised %) distribution")
                st.bar_chart(result.set_index("Name")["StdDev_%"])

        st.divider()
        st.header("Step 3 - Rank Your Shortlist")
        st.caption(
            "Pick ONE strategy below. Each one ranks your shortlisted stocks by a "
            "different investing philosophy — you don't need to understand the math, "
            "just pick the style that matches how you want to invest."
        )
        style_choice = st.selectbox("Pick a ranking strategy", options=["Low Vol", "Quality", "Growth", "Value", "Momentum"])
        st.caption(STYLE_DESCRIPTIONS.get(style_choice, ""))
        
        top_n = st.number_input(
            "How many stocks do you want in your final list?",
            min_value=1, max_value=len(result), value=min(20, len(result)), step=1,
            help="Only your top-ranked stocks by this many will be shown/downloaded.",
        )

        if st.button("Compute style rank", width="stretch"):
            ranked = compute_style_rank(result, style_choice)
            st.session_state.ranked = ranked.head(int(top_n))

if st.session_state.ranked is not None:
    ranked = st.session_state.ranked
    st.subheader(f"Your final list — top {len(ranked)} stocks ({style_choice} strategy)")
    safe_ranked = safe_for_arrow(ranked)
    st.dataframe(safe_ranked, width="stretch")
    st.download_button(
        "Download your final list (CSV)",
        safe_ranked.to_csv(index=False).encode(),
        "final_stock_list.csv",
        "text/csv",
    )
    st.bar_chart(ranked.set_index("Name")["Style_Score"])
  
if st.session_state.fundamentals_result is None:
    st.info("Set your Screener.in session cookie and click Step 1: Run Fundamental Screen to begin.")

# artifact refresh: 1784455641.5617282
