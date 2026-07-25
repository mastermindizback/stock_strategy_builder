"""
ranking.py
----------
Style-factor ranking for screened stocks.

Instead of exposing raw factor weights to the end user, this module ships
five pre-built "quant style" scores — Low Vol, Quality, Growth, Value,
Momentum — each a fixed-weight composite of screener columns, modelled on
the academic factor literature (Fama-French HML/value, QMJ-style quality,
UMD/momentum, and the low-volatility anomaly). The user picks ONE style
label from a dropdown; the underlying formula/weights stay in the backend.

Internally this still uses the generic z-score + weighted-sum engine
(FactorSpec / CompositeRankConfig / compute_composite_rank) — that engine
is kept for backend/programmatic use and for anyone extending the style
definitions below, but the Streamlit UI should only ever call
`compute_style_rank()`.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger("ranking")

# Sensible defaults: is a HIGHER raw value better for this factor?
DEFAULT_DIRECTION = {
    "P/E": False,             # lower P/E is better (cheaper)
    "ROE": True,
    "ROCE": True,
    "Sales Growth": True,
    "Profit Growth": True,
    "Debt to Equity": False,  # lower leverage is better
    "Piotroski score": True,
    "Promoter Holding": True,
    "Market Cap": True,
    "Beta": False,            # lower beta is better (less risk)
    "StdDev_%": False,        # lower volatility is better (used for RANKING, e.g. Low Vol style)
    "Active_Risk_%": False,   # lower tracking error vs benchmark is better (used for CAPPING, not ranking)
    "Momentum_%": True,
    "Chg in FII Hold": True,   # rising FII stake is bullish
    "Chg in DII Hold": True,   # rising DII stake is bullish
}


@dataclass
class FactorSpec:
    column: str
    weight: float = 1.0
    higher_is_better: bool = True
    winsorize_pct: float = 0.01  # clip extreme 1% tails before z-scoring


@dataclass
class CompositeRankConfig:
    factors: list[FactorSpec] = field(default_factory=list)

    def normalize_weights(self) -> "CompositeRankConfig":
        total = sum(abs(f.weight) for f in self.factors) or 1.0
        for f in self.factors:
            f.weight = f.weight / total
        return self


def _zscore(s: pd.Series, winsorize_pct: float) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if winsorize_pct and 0 < winsorize_pct < 0.5:
        lo, hi = s.quantile(winsorize_pct), s.quantile(1 - winsorize_pct)
        s = s.clip(lo, hi)
    mu, sigma = s.mean(), s.std(ddof=0)
    if not sigma or np.isnan(sigma):
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sigma


def compute_composite_rank(
    df: pd.DataFrame,
    config: CompositeRankConfig,
    score_col: str = "Composite_Score",
    rank_col: str = "Rank",
    sector_neutral: bool = False,
    sector_col: str = "Sector",
) -> pd.DataFrame:
    """
    Generic engine: adds `score_col` (weighted sum of per-factor z-scores,
    sign-flipped for lower-is-better factors) and `rank_col` (1 = best) to
    a COPY of df. Used internally by the named style presets below.
    """
    if df.empty or not config.factors:
        out = df.copy()
        out[score_col] = np.nan
        out[rank_col] = np.nan
        return out

    config = config.normalize_weights()
    out = df.copy()
    score = pd.Series(0.0, index=out.index)

    for spec in config.factors:
        if spec.column not in out.columns:
            logger.warning("Factor column %r not found — skipping.", spec.column)
            continue

        if sector_neutral and sector_col in out.columns:
            z = out.groupby(sector_col)[spec.column].transform(
                lambda s: _zscore(s, spec.winsorize_pct)
            )
        else:
            z = _zscore(out[spec.column], spec.winsorize_pct)

        z = z.fillna(0.0)
        sign = 1.0 if spec.higher_is_better else -1.0
        score = score + spec.weight * sign * z

    out[score_col] = score
    out[rank_col] = out[score_col].rank(ascending=False, method="min").astype(int)
    return out.sort_values(rank_col).reset_index(drop=True)


def available_ranking_columns(df: pd.DataFrame) -> list[str]:
    """Numeric columns from the screened DataFrame eligible for ranking."""
    exclude = {"Rank", "Composite_Score", "Style_Score"}
    return [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]


def default_factor_specs(columns: list[str]) -> list[FactorSpec]:
    """Build a starter FactorSpec list (equal weight) for the given columns,
    using DEFAULT_DIRECTION where known (defaults to higher-is-better)."""
    return [
        FactorSpec(column=c, weight=1.0, higher_is_better=DEFAULT_DIRECTION.get(c, True))
        for c in columns
    ]


# ---------------------------------------------------------------------------
# Named quant styles — Fama-French-inspired, fixed backend weights.
# Users pick the LABEL only; weights/formula are never surfaced in the UI.
# Each entry: column -> (weight, higher_is_better).
# ---------------------------------------------------------------------------
STYLE_FACTOR_WEIGHTS: dict[str, dict[str, tuple[float, bool]]] = {
    # Low-volatility anomaly (Ang et al.): rewards low Beta / low Std Dev.
    "Low Vol": {
        "StdDev_%": (0.6, False),
        "Beta": (0.4, False),
    },
    # QMJ-style quality (Asness-Frazzini-Pedersen): profitability + low
    # leverage + balance-sheet strength (Piotroski) + promoter skin-in-game.
    "Quality": {
        "ROE": (0.30, True),
        "ROCE": (0.30, True),
        "Piotroski score": (0.20, True),
        "Debt to Equity": (0.10, False),
        "Promoter Holding": (0.10, True),
    },
    # Growth: revenue momentum plus a profitability kicker.
    "Growth": {
        "Sales Growth": (0.45, True),
        "Profit Growth": (0.30, True),
        "ROE": (0.25, True),
    },
    # Fama-French HML-style value: cheapness on earnings, i.e. low P/E.
    "Value": {
        "P/E": (1.0, False),
    },
    # Fama-French/Carhart UMD momentum: trailing 12-1 month price return.
    "Momentum": {
        "Momentum_%": (0.40, True),        # trailing 12-1 month price momentum (core UMD factor)
        "Chg in FII Hold": (0.30, True),   # institutional (FII) accumulation
        "Chg in DII Hold": (0.30, True),   # institutional (DII) accumulation
    },
}

STYLE_DESCRIPTIONS: dict[str, str] = {
    "Low Vol": "Favors stocks with lower annualised volatility and lower Beta (defensive/low-vol anomaly).",
    "Quality": "Favors high ROE/ROCE, strong Piotroski score, low leverage, and high promoter holding.",
    "Growth": "Favors stocks with the strongest annual sales growth and profit growth, with a profitability kicker from ROE.",
    "Value": "Favors cheaper stocks on a P/E basis (Fama-French HML-style value tilt).",
    "Momentum": "Favors stocks with strong trailing 12-1 month price momentum (70% weight), "
    "plus rising FII and DII shareholding (15% each) as a smart-money confirmation signal.",
}


def compute_style_rank(
    df: pd.DataFrame,
    style: str,
    score_col: str = "Style_Score",
    rank_col: str = "Rank",
) -> pd.DataFrame:
    """
    Ranks `df` on one of the named quant styles (Low Vol / Quality / Growth
    / Value / Momentum). Backend fixed weights only. Ranks across the full
    list of stocks (no sector grouping) — simplest, most intuitive behavior.
    """
    if style not in STYLE_FACTOR_WEIGHTS:
        raise ValueError(f"Unknown style {style!r}. Choose from: {list(STYLE_FACTOR_WEIGHTS)}")

    weights = STYLE_FACTOR_WEIGHTS[style]
    factors = [
        FactorSpec(column=col, weight=w, higher_is_better=hib)
        for col, (w, hib) in weights.items()
    ]

    missing = [f.column for f in factors if f.column not in df.columns]
    if missing:
        logger.warning("Style %r: columns unavailable and will be skipped: %s", style, missing)

    config = CompositeRankConfig(factors=[f for f in factors if f.column in df.columns])
    return compute_composite_rank(df, config, score_col=score_col, rank_col=rank_col)
# artifact refresh: 1784455641.5617282
