"""
sector.py
---------
Reliable sector/industry tagging for screened stocks.

Screener.in's "Sector" column is often missing or inconsistent because it
depends on the raw HTML table headers actually returned for a given query.
The Nifty Total Market constituent file (universe.py) ALWAYS carries an
"Industry" column per official NSE classification, so we use it as the
single source of truth and just overlay/merge it onto the screened frame.
"""

import logging
import pandas as pd

logger = logging.getLogger("sector")

_UNIVERSE_SYMBOL_COL = "Symbol"
_UNIVERSE_INDUSTRY_COL = "Industry"


def enrich_with_sector(df: pd.DataFrame, universe_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds/overwrites a 'Sector' column on `df` using the NSE-classified
    Industry field from the Nifty Total Market universe file.

    df: fundamentals/final DataFrame with a 'Ticker' column (with or
        without the '.NS' suffix).
    universe_df: DataFrame returned by universe.fetch_nifty_total_market().
    """
    if df.empty:
        return df

    df = df.copy()
    sym = (
        df["Ticker"].astype(str)
        .str.replace(".NS", "", regex=False)
        .str.upper()
        .str.strip()
    )

    lookup = (
        universe_df[[_UNIVERSE_SYMBOL_COL, _UNIVERSE_INDUSTRY_COL]]
        .dropna(subset=[_UNIVERSE_SYMBOL_COL])
        .assign(**{_UNIVERSE_SYMBOL_COL: lambda d: d[_UNIVERSE_SYMBOL_COL].astype(str).str.upper().str.strip()})
        .drop_duplicates(subset=_UNIVERSE_SYMBOL_COL)
        .set_index(_UNIVERSE_SYMBOL_COL)[_UNIVERSE_INDUSTRY_COL]
    )

    mapped_sector = sym.map(lookup)

    # Fall back to whatever Screener.in already gave us (if any) when the
    # universe file has no match, rather than leaving it blank.
    if "Sector" in df.columns:
        df["Sector"] = mapped_sector.fillna(df["Sector"])
    else:
        df["Sector"] = mapped_sector

    n_missing = df["Sector"].isna().sum()
    if n_missing:
        logger.warning("Sector still missing for %d/%d rows after enrichment.", n_missing, len(df))
    df["Sector"] = df["Sector"].fillna("Unknown")
    return df


def sector_neutral_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Quick sector-count breakdown, handy for confirming sector-neutrality."""
    if df.empty or "Sector" not in df.columns:
        return pd.DataFrame(columns=["Sector", "Count", "Weight_%"])
    counts = df["Sector"].value_counts().rename_axis("Sector").reset_index(name="Count")
    counts["Weight_%"] = (counts["Count"] / counts["Count"].sum() * 100).round(2)
    return counts


def sparse_sector_warning(df: pd.DataFrame, min_count: int = 4) -> list[str]:
    """
    Sectors with fewer than `min_count` stocks in the given DataFrame.

    Sector-neutral z-scoring divides by within-sector std dev. With very
    few members (esp. n=1), the z-score collapses to a flat 0 regardless
    of how good/bad the stock actually is on that factor -- giving a false
    sense of a level playing field. Callers should surface this list to
    the user whenever sector-neutral ranking is enabled.
    """
    if df.empty or "Sector" not in df.columns:
        return []
    counts = df["Sector"].value_counts()
    return counts[counts < min_count].index.tolist()

# artifact refresh: 1784455641.5617282
