# Nifty Total Market (750) Stock Screener

A Streamlit app that screens the **Nifty Total Market Index** (750 stocks —
large, mid, small & microcap) using:

**Fundamental filters (from Screener.in):**
- Sales Growth (5Y) > 20%
- PAT Growth (5Y) > 40%
- PEG Ratio < 1
- Promoter Holding > 60%

**Risk filters (calculated locally, TTM):**
- Beta < 1 (vs Nifty 500 benchmark)
- Annualised Std Dev < 20%

## Project layout
```
nifty_screener/
├── config.py        # thresholds, screener query builder, benchmark ticker
├── universe.py       # fetches Nifty Total Market 750-stock constituent list
├── data_fetcher.py    # (your existing file) Screener.in scraper + yfinance price fetcher
├── metrics.py         # Beta & Std Dev calculation
├── screening.py        # orchestrates the full pipeline
├── app.py               # Streamlit UI
└── requirements.txt
```

## Setup
```bash
pip install -r requirements.txt

# Log in to screener.in in your browser, then:
# DevTools → Application → Cookies → screener.in → copy 'sessionid'
export SCREENER_SESSION="paste-your-sessionid-here"

streamlit run app.py
```

## How it works
1. `universe.py` downloads the live 750-stock constituent CSV from niftyindices.com.
2. `screening.py` builds a Screener.in DSL query from `config.py` thresholds and
   calls your existing `fetch_screener_universe()` in `data_fetcher.py`.
3. Results are restricted to only the Nifty Total Market universe.
4. `fetch_prices()` downloads ~13 months of daily closes for shortlisted stocks
   + the Nifty 500 benchmark (`^CRSLDX`) via yfinance.
5. `metrics.py` computes Beta (covariance/variance vs benchmark) and annualised
   Std Dev on trailing 252 trading days.
6. Final shortlist = fundamental pass ∩ Beta < 1 ∩ Std Dev < 20%.

## Notes
- Verify exact Screener.in DSL field names (e.g. "PAT growth 5Years" vs
  "Profit growth 5Years") using the autosuggest on screener.in/screen/new/ —
  field names occasionally change.
- Nifty Total Market has no direct Yahoo Finance ticker, so Nifty 500
  (`^CRSLDX`) is used as the closest liquid benchmark proxy for Beta. Change
  `BENCHMARK_TICKER` in `config.py` if you have a better proxy.
- Sector enrichment in `data_fetcher.py` is capped at 200 stocks to avoid
  excessive scraping — this is fine post-filtering since final shortlists are
  typically much smaller than 750.
