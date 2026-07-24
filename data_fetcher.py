"""
data_fetcher.py
---------------
Fetches the stock universe from Screener.in and price data from Yahoo Finance.
No imports from app.py — no circular imports.
"""
from __future__ import annotations

import datetime
import html as html_module
import json
import logging
import os
import re
import time
import warnings
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
import pandas as pd
from bs4 import BeautifulSoup
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

logger = logging.getLogger("data_fetcher")

LOGIN_URL      = "https://www.screener.in/login"
RAW_SCREEN_URL = "https://www.screener.in/screen/raw/"   # trailing slash avoids 301
SCREENER_BASE  = "https://www.screener.in"
PAGE_LIMIT     = 25
MAX_PAGES      = 40


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------
class AuthenticationError(Exception):
    """Session cookie expired or SCREENER_SESSION env var not set."""


# ---------------------------------------------------------------------------
# Cookie / Session Helpers
# ---------------------------------------------------------------------------
def _parse_cookie_string(cookie_str: str) -> dict[str, str]:
    cookie_str = cookie_str.strip()
    if not cookie_str:
        return {}
    if "=" not in cookie_str:
        return {"sessionid": cookie_str}
    out: dict[str, str] = {}
    for part in cookie_str.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _get_session_cookies() -> dict[str, str]:
    val = os.environ.get("SCREENER_SESSION", "").strip()
    if not val:
        raise RuntimeError(
            "SCREENER_SESSION environment variable is not set.\n"
            "  1. Log in to screener.in in your browser\n"
            "  2. Open DevTools → Application → Cookies → screener.in\n"
            "  3. Copy the `sessionid` value\n"
            "  4. Run: export SCREENER_SESSION='paste here'"
        )
    cookies = _parse_cookie_string(val)
    logger.debug("Parsed cookies: %s", list(cookies.keys()))
    return cookies


# ---------------------------------------------------------------------------
# URL Helpers
# ---------------------------------------------------------------------------
def _urlencode(params: dict[str, str]) -> str:
    return "&".join(
        f"{quote(str(k), safe='')}={quote(str(v), safe='')}"
        for k, v in params.items()
    )


SCREENER_COLUMNS = [
    "Pe",
    "Roe",
    "SalesGrowth5Years",
    "DebtToEquity",
    "PiotroskiScore",
    "MarketCap",
    "Roce",
    "PromoterHolding",
]


def _screen_url(query: str, page: int) -> str:
    # Use trailing slash to avoid the 301 redirect that strips query params
    base_params = {"query": query, "limit": str(PAGE_LIMIT), "page": str(page)}
    base = f"{RAW_SCREEN_URL}?{_urlencode(base_params)}"
    col_str = "".join(f"&column={quote(c, safe='')}" for c in SCREENER_COLUMNS)
    return f"{base}{col_str}"


# ---------------------------------------------------------------------------
# HTTP Client
# ---------------------------------------------------------------------------
def _make_client(cookies: dict[str, str]) -> httpx.Client:
    return httpx.Client(
        cookies=cookies,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Referer": "https://www.screener.in",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "X-Requested-With": "XMLHttpRequest",
        },
        follow_redirects=True,
        timeout=60.0,
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _httpx_fetch(client: httpx.Client, url: str) -> str:
    logger.debug("GET %s", url)
    resp = client.get(url)
    resp.raise_for_status()
    final_url = str(resp.url)
    if "login" in final_url or "register" in final_url:
        raise AuthenticationError(
            f"Screener.in redirected to {final_url!r}. "
            "Your SCREENER_SESSION cookie has expired — please refresh it."
        )
    logger.debug("Response: %d bytes, status %s", resp.text.__len__(), resp.status_code)
    return resp.text


# ---------------------------------------------------------------------------
# Playwright fallback
# ---------------------------------------------------------------------------
def _playwright_fetch(url: str, cookies: dict[str, str]) -> str:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise RuntimeError(
            "playwright not installed. Run: uv add 'playwright>=1,<2' && playwright install chromium"
        )
    logger.info("Playwright fallback for %s", url)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            )
        )
        ctx.add_cookies(
            [{"name": k, "value": v, "domain": "www.screener.in", "path": "/"} for k, v in cookies.items()]
        )
        page = ctx.new_page()
        page.goto(url, wait_until="networkidle", timeout=45000)
        final_url = page.url
        if "login" in final_url or "register" in final_url:
            browser.close()
            raise AuthenticationError(
                f"Playwright: Screener.in redirected to {final_url!r} — session expired."
            )
        try:
            page.wait_for_selector("table.data-table", timeout=12000)
        except Exception:
            logger.warning("Playwright: no table.data-table within 12 s on %s", url)
        html = page.content()
        browser.close()
    logger.debug("Playwright returned %d bytes", len(html))
    return html


# ---------------------------------------------------------------------------
# HTML Parsing — BeautifulSoup primary, regex fallback
# ---------------------------------------------------------------------------
def _clean_cell(tag_or_str: Any) -> str:
    """Extract plain text from a BS4 tag or raw HTML string."""
    if hasattr(tag_or_str, "get_text"):
        return " ".join(tag_or_str.get_text(separator=" ").split())
    return " ".join(re.sub(r"<[^>]*>", "", str(tag_or_str)).split())


def _parse_results_table(html: str) -> pd.DataFrame:
    """
    Parse Screener.in's results HTML using BeautifulSoup.
    """
    soup = BeautifulSoup(html, "lxml")

    # Strategy 1: find table with "data-table" in its class list
    table = soup.find("table", class_="data-table")

    # Strategy 2: any table whose class string contains "data-table"
    if table is None:
        for t in soup.find_all("table"):
            cls = " ".join(t.get("class", []))
            if "data-table" in cls:
                table = t
                break

    # Strategy 3: first table on the page (last resort)
    if table is None:
        all_tables = soup.find_all("table")
        logger.debug(
            "_parse_results_table: no data-table found. "
            "Tables on page: %d | HTML snippet: %s",
            len(all_tables), html[:3000],
        )
        if all_tables:
            logger.info(
                "Falling back to first <table> on page (%d tables found).", len(all_tables)
            )
            table = all_tables[0]
        else:
            logger.warning("_parse_results_table: NO <table> found at all in the HTML.")
            logger.debug("Full HTML (first 5000 chars):\n%s", html[:5000])
            return pd.DataFrame()

    # Extract headers from <th>
    thead = table.find("thead")
    if thead:
        header_tags = thead.find_all("th")
    else:
        first_row = table.find("tr")
        header_tags = first_row.find_all("th") if first_row else []

    if not header_tags:
        # Some Screener layouts put headers in the first <tr> as <td>
        first_row = table.find("tr")
        header_tags = first_row.find_all(["th", "td"]) if first_row else []

    headers = []
    seen: set[str] = set()
    for th in header_tags:
        text = _clean_cell(th)
        norm = text.lower()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        headers.append(text)

    logger.debug("Parsed headers: %s", headers)
    if not headers:
        logger.warning("_parse_results_table: no headers found in table.")
        logger.debug("Table HTML snippet: %s", str(table)[:2000])
        return pd.DataFrame()

    # Extract data rows
    rows: list[dict[str, Any]] = []
    tbody = table.find("tbody")
    row_source = tbody if tbody else table
    for tr in row_source.find_all("tr"):
        # Skip header rows
        if tr.find("th"):
            continue
        cells = tr.find_all("td")
        if not cells:
            continue

        values = [_clean_cell(c) for c in cells[: len(headers)]]
        if len(values) < len(headers):
            values += [""] * (len(headers) - len(values))
        record: dict[str, Any] = dict(zip(headers, values, strict=False))

        # Extract company slug from any anchor in the row
        company_link = tr.find("a", href=re.compile(r"/company/[^/]+"))
        if company_link:
            href = company_link.get("href", "")
            slug_match = re.search(r"/company/([^/\"]+)", href)
            if slug_match:
                record["company_url"] = href
                raw_slug = slug_match.group(1)
                decoded_slug = html_module.unescape(raw_slug)
                record["company_id"] = decoded_slug.upper()
                logger.debug(
                    "Row → Name=%r  slug=%r", record.get("Name"), decoded_slug
                )
        rows.append(record)

    logger.debug("Parsed %d data rows", len(rows))
    if not rows:
        logger.warning(
            "_parse_results_table: table found but 0 data rows extracted. "
            "Table HTML snippet:\n%s",
            str(table)[:3000],
        )
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "company_id" in df.columns:
        df["symbol"] = df["company_id"]
    elif "Name" in df.columns:
        df["symbol"] = df["Name"].str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    return df


# ---------------------------------------------------------------------------
# Numeric Coercion & Column Normalisation
# ---------------------------------------------------------------------------
_NON_NUMERIC = {"Name", "Ticker", "Sector", "company_url", "company_id", "symbol", "ISIN"}
_FORCE_NUMERIC = {
    "P/E", "ROE", "ROCE", "Sales Growth", "Debt to Equity", "Piotroski score",
    "Market Cap", "ROCE %", "Div Yld %", "Promoter Holding", "Chg in FII Hold", "Chg in DII Hold", "Profit Growth"
}


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if col in _NON_NUMERIC:
            continue
        cleaned = (
            df[col].astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.strip()
        )
        coerced = pd.to_numeric(cleaned, errors="coerce")
        non_empty = df[col].astype(str).str.strip().ne("").sum()
        ratio = coerced.notna().sum() / max(non_empty, 1)
        if col in _FORCE_NUMERIC or ratio > 0.25:
            df[col] = coerced
    return df


# ---------------------------------------------------------------------------
# EXPECTED_COLUMNS — canonical name → list of known Screener header variants
# ---------------------------------------------------------------------------
EXPECTED_COLUMNS: dict[str, list[str]] = {
    "Name": ["Name", "Company"],
    "Ticker": [
        "symbol",
        "company_id",
        "Ticker",
        "NSE Symbol",
        "NSE",
        "Symbol",
    ],
    "Sector": ["Sector", "Industry"],
    "Market Cap": [
        "Mar Cap Rs.Cr.",
        "Market Cap",
        "Mar Cap",
        "Market Capitalization",
        "MarketCap",
        "market_cap",
        "Market Cap Cr.",
        "Mar Cap Cr",
    ],
    "P/E": [
        "P/E",
        "PE",
        "Price to Earnings",
        "Pe",
        "P / E",
        "Price/Earnings",
        "Price Earnings",
        "Price to Earning",
        "P/E Ratio",
    ],
    "ROE": [
        "Roe %",
        "ROE %",
        "ROE",
        "Return on Equity",
        "Roe",
        "Return on equity",
        "Return On Equity %",
        "Return on equity %",
    ],
    "ROCE": [
        "ROCE",
        "Roce",
        "return on capital",
        "return on capital employed",
        "ROCE %",
        "Roce %"
    ],
    "Profit Growth": [
        "Profit growth %", "Profit Var %", "Profit Growth", "PAT Growth %",
        "Net Profit Growth %", "ProfitGrowth", "Net profit growth %",
    ],
    "Sales Growth": [
        "Sales Var 5Yrs %",
        "Sales Var 5Yr %",
        "Rev Var 5Yrs %",
        "Sales growth 5Yrs %",
        "Sales growth 5Years %",
        "Sales growth 5Years",
        "Sales Growth",
        "Sales Gr.",
        "Revenue Growth",
        "Sales growth 5years",
        "SalesGrowth5Years",
        "Sales Growth 5Years",
        "5Yr Sales CAGR",
        "Sales CAGR 5Yrs %",
        "Revenue Growth 5Yr %",
        "Sales growth 3Years %",
        "Sales growth 3Yrs %",
        "Sales Var 3Yrs %",
        "Sales Var 3Yr %",
    ],
    "Debt to Equity": [
        "Debt / Eq",
        "Debt / Eq.",
        "Debt / Equity",
        "Debt to equity",
        "Debt to Equity",
        "D/E",
        "Debt/Equity",
        "DebtToEquity",
        "Debt To Equity",
        "Debt to equity %",
    ],
    "Piotroski score": [
        "Piotski Scr",
        "Piotroski score",
        "Piotroski Score",
        "Piotroski",
        "PiotroskiScore",
        "Piotroski F-Score",
        "F Score",
        "F-Score",
        "Piotroski F Score",
    ],
    "Promoter Holding": [
        "Promoter Holding",
        "PromoterHolding",
        "promoter",
        "promoter holding",
        "promoter holdings",
        "Promoter holding",
        "Promoter holding %",
        "Promoter Holdings",
        "prom. hold. %",
        "prom hold %",
        "prom. holding",
        "promoterholding"
    ],
    "Chg in FII Hold": [
        "Chg in FII Hold %", "Change in FII Holding", "chg in fii holding",
    ],
    "Chg in DII Hold": [
        "Chg in DII Hold %", "Change in DII Holding", "chg in dii holding",
    ],
}


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename raw Screener column headers to canonical factor names.
    Applies .strip().lower() for robust mapping of aliases to real columns.
    """
    df = df.copy()
    logger.debug("Raw columns before normalisation: %s", list(df.columns))

    # Case-insensitive mapping of the actual columns dataframe has
    raw_cols = list(df.columns)
    col_map = {str(c).strip().lower(): c for c in raw_cols}

    for canonical, aliases in EXPECTED_COLUMNS.items():
        if canonical in df.columns and df[canonical].notna().any():
            logger.debug("Column '%s' already present with data — skipping.", canonical)
            continue

        for alias in aliases:
            alias_lower = str(alias).strip().lower()
            if alias_lower not in col_map:
                continue

            actual_col = col_map[alias_lower]
            if df[actual_col].isna().all():
                continue

            if canonical in df.columns:
                logger.debug(
                    "Column conflict: '%s' all-NaN, overwriting with alias '%s'.",
                    canonical, alias,
                )
                df[canonical] = df[actual_col]
                df = df.drop(columns=[actual_col])
            else:
                logger.debug("Column rename: %r → %r", actual_col, canonical)
                df = df.rename(columns={actual_col: canonical})
            break

    # Diagnostic
    factor_cols = ["P/E", "ROE", "ROCE", "Sales Growth", "Debt to Equity", "Piotroski score", "Promoter Holding"]
    missing = [c for c in factor_cols if c not in df.columns or df[c].isna().all()]
    if missing:
        present = [c for c in df.columns if c not in {"company_url", "company_id", "symbol"}]
        logger.warning(
            "Factor column(s) MISSING after normalisation: %s\n"
            "  Raw columns were: %s\n"
            "  → Add the matching header as the first alias in EXPECTED_COLUMNS.",
            missing, present,
        )
    else:
        logger.info("All factor columns mapped successfully ✓")

    return df


# ---------------------------------------------------------------------------
# Ticker Sanitisation
# ---------------------------------------------------------------------------
def sanitize_ticker(ticker: str) -> str:
    """Map a Screener slug / symbol → Yahoo Finance .NS format."""
    original = ticker
    ticker = ticker.strip()
    ticker = html_module.unescape(ticker)
    changed = False
    if " & " in ticker:
        ticker = ticker.replace(" & ", "AND"); changed = True
    if "&" in ticker:
        ticker = ticker.replace("&", "AND"); changed = True
    if "." in ticker:
        ticker = ticker.replace(".", ""); changed = True
    if " " in ticker:
        ticker = ticker.replace(" ", ""); changed = True
    if "-" in ticker:
        ticker = ticker.replace("-", ""); changed = True
    ticker = ticker.upper() + ".NS"
    if changed:
        logger.info("Ticker mapped: %r → %r", original, ticker)
    return ticker


# ---------------------------------------------------------------------------
# Sector Enrichment
# ---------------------------------------------------------------------------
def _clean_company_slug(raw: object) -> str | None:
    """
    Return a clean, URL-safe company slug, or None for invalid values.
    Filters NaN / 'nan' / empty / HTML-entity slugs before URL construction.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    s = html_module.unescape(str(raw)).strip().lower()
    if s in {"", "nan", "none", "null"}:
        return None
    return s


def _fetch_sector(client: httpx.Client, slug: str) -> str:
    for path in [f"/company/{slug}/consolidated/", f"/company/{slug}/"]:
        url = f"{SCREENER_BASE}{path}"
        try:
            html = _httpx_fetch(client, url)
            soup = BeautifulSoup(html, "lxml")
            # Try structured sector link first
            for a in soup.find_all("a", href=True):
                if "/sector/" in a["href"] or "/industry/" in a["href"]:
                    text = a.get_text(strip=True)
                    if 2 < len(text) < 60:
                        return text
            # Fallback: regex patterns
            m = re.search(r'"sector"\s*:\s*"([^"]{2,60})"', html, re.I)
            if m:
                return m.group(1).strip()
            m = re.search(r'Sector[^A-Za-z ,]{0,10}([A-Za-z ]{2,60}?)(?:<|\n|$)', html, re.I)
            if m and m.group(1).strip():
                return m.group(1).strip()
        except Exception as exc:
            logger.debug("Sector fetch failed for slug %r at %s: %s", slug, path, exc)
            continue
    return "Unknown"


# ---------------------------------------------------------------------------
# Main Public API
# ---------------------------------------------------------------------------
def fetch_screener_universe(query: str) -> pd.DataFrame:
    """
    Run a Screener.in DSL query and return a clean DataFrame.
    """
    cookies = _get_session_cookies()
    frames: list[pd.DataFrame] = []
    pages_fetched = 0

    logger.info("━━━ Screener.in Query ━━━")
    logger.info("Query: %s", query)
    logger.info("URL: %s", _screen_url(query, 1))

    with _make_client(cookies) as client:
        for pagenum in range(1, MAX_PAGES + 1):
            url = _screen_url(query, pagenum)
            logger.info("📄 Page %d → %s", pagenum, url)
            try:
                html = _httpx_fetch(client, url)
            except AuthenticationError:
                raise
            except Exception as exc:
                logger.error("httpx failed on page %d: %s", pagenum, exc)
                break

            # Check if table is present before deciding on Playwright
            soup_check = BeautifulSoup(html, "lxml")
            has_table = soup_check.find("table", class_="data-table") is not None
            if not has_table:
                # Broader check: any table on the page
                has_any_table = bool(soup_check.find("table"))
                logger.info(
                    "No data-table in httpx response (any table: %s) — trying Playwright…",
                    has_any_table,
                )
                try:
                    html = _playwright_fetch(url, cookies)
                except Exception as exc:
                    logger.error("Playwright fallback failed: %s", exc)
                    logger.error("httpx HTML snippet (first 3000 chars):\n%s", html[:3000])
                    break

            page_df = _parse_results_table(html)
            pages_fetched += 1
            logger.info(
                "  → Page %d: %d rows parsed  (cumulative: %d)",
                pagenum,
                len(page_df),
                sum(len(f) for f in frames) + len(page_df),
            )
            if page_df.empty:
                logger.info("  Empty page — stopping pagination.")
                break
            frames.append(page_df)
            if len(page_df) < PAGE_LIMIT:
                logger.info(
                    "  Last page had %d < %d rows — end of results.", len(page_df), PAGE_LIMIT
                )
                break
            time.sleep(0.4)

    if not frames:
        logger.error(
            "fetch_screener_universe: zero rows fetched after %d pages.\n"
            "  Troubleshooting:\n"
            "  1. Is SCREENER_SESSION set to your current sessionid value?\n"
            "  2. Open %s in a browser — does it show results?\n"
            "  3. Run with DEBUG logging: logging.getLogger().setLevel(logging.DEBUG)\n"
            "  4. Check query syntax is valid on screener.in/screen\n"
            "  5. Try refreshing your SCREENER_SESSION cookie (it may have expired)",
            pages_fetched,
            _screen_url(query, 1),
        )
        return pd.DataFrame(columns=list(EXPECTED_COLUMNS.keys()))

    df = pd.concat(frames, ignore_index=True)
    logger.info("━━━ Fetched %d raw rows across %d pages ━━━", len(df), pages_fetched)
    df["Sales Growth"] = df['Sales growth %']

    df = _normalise_columns(df)
    df = _coerce_numeric(df)

    # Factor coverage log
    for fc in ["P/E", "ROE", "ROCE", "Sales Growth", "Debt to Equity", "Piotroski score", "Promoter Holding"]:
        if fc in df.columns:
            n_valid = df[fc].notna().sum()
            if n_valid == 0:
                logger.warning(
                    "Factor '%s': 0/%d rows have valid data after normalisation.", fc, len(df)
                )
            else:
                logger.info("Factor '%s': %d/%d rows have valid data.", fc, n_valid, len(df))

    # Sanitize tickers
    if "Ticker" in df.columns:
        df["Ticker"] = df["Ticker"].apply(
            lambda t: sanitize_ticker(str(t))
            if pd.notna(t) and str(t).strip() not in ("", "nan")
            else t
        )
        logger.info("Sample tickers: %s", df["Ticker"].head(5).tolist())
    else:
        logger.warning("No Ticker column after normalisation. Columns: %s", df.columns.tolist())

    # Drop duplicates
    if "Name" in df.columns and "Ticker" in df.columns:
        # Prefer the ticker ending in .NS when duplicates by Name exist
        df['_is_ns'] = df['Ticker'].astype(str).str.endswith('.NS')
        df = df.sort_values('_is_ns', ascending=False)
        df = df.drop_duplicates(subset=['Name'], keep='first')
        df = df.drop(columns=['_is_ns'])
    else:
        dedup_col = next((c for c in ["company_id", "Ticker", "Name"] if c in df.columns), None)
        if dedup_col:
            df = df.drop_duplicates(subset=dedup_col, keep="first")

    # Ensure all expected columns exist
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan")

    logger.info("━━━ Final universe: %d stocks, %d columns ━━━", len(df), len(df.columns))
    logger.info("Columns: %s", df.columns.tolist())

    df['Sector'] = 'Unknown'

    # # Sector enrichment
    # needs_sector = (
    #     "Sector" not in df.columns
    #     or df["Sector"].isna().all()
    #     or df["Sector"].astype(str).str.strip().isin({"", "Unknown", "nan"}).all()
    # )

    # if needs_sector and "company_id" in df.columns:
    #     df["_slug"] = df["company_id"].apply(_clean_company_slug)
    #     bad = df[df["_slug"].isna()][["Name", "company_id"]].drop_duplicates()
    #     if not bad.empty:
    #         logger.warning(
    #             "%d rows have invalid company_id — sector will be 'Unknown':\n%s",
    #             len(bad), bad.to_string(index=False),
    #         )

    #     if len(df) <= 200:
    #         logger.info("Fetching sector data for %d stocks …", df["_slug"].notna().sum())
            
    #         # --- Initialize Sector Cache ---
    #         cache_file = Path.home() / ".screener_sector_cache.json"
    #         cache_data = {}
    #         try:
    #             if cache_file.exists():
    #                 with open(cache_file, "r") as f:
    #                     cache_data = json.load(f)
    #         except Exception as e:
    #             logger.warning(f"Failed to read sector cache: {e}")

    #         now = datetime.datetime.now(datetime.timezone.utc)

    #         with _make_client(cookies) as client:
    #             sectors: list[str] = []
    #             for _, row in df.iterrows():
    #                 slug = row["_slug"]
    #                 if slug is None:
    #                     sectors.append("Unknown")
    #                     continue
                    
    #                 # Cache read validation
    #                 cached_val = cache_data.get(slug)
    #                 if cached_val and "sector" in cached_val and "fetched_at" in cached_val:
    #                     try:
    #                         fetched_at = datetime.datetime.fromisoformat(cached_val["fetched_at"])
    #                         if (now - fetched_at).days < 7:
    #                             sectors.append(cached_val["sector"])
    #                             continue
    #                     except Exception:
    #                         pass # Fallback to fetch on parsing failure

    #                 logger.debug("Sector fetch: Name=%r  slug=%r", row.get("Name"), slug)
    #                 sector_val = _fetch_sector(client, slug)
    #                 sectors.append(sector_val)
                    
    #                 # Update cache memory
    #                 cache_data[slug] = {
    #                     "sector": sector_val,
    #                     "fetched_at": now.isoformat()
    #                 }
    #                 time.sleep(0.3)
    #         df["Sector"] = sectors
            
    #         # --- Persist Sector Cache ---
    #         try:
    #             with open(cache_file, "w") as f:
    #                 json.dump(cache_data, f)
    #         except Exception as e:
    #             logger.warning(f"Failed to write sector cache: {e}")
                
    #     else:
    #         logger.warning(
    #             "Universe has %d > 200 stocks — sector enrichment skipped.", len(df)
    #         )
    #         df["Sector"] = "Unknown"

    #     df = df.drop(columns=["_slug"], errors="ignore")
    # elif not needs_sector:
    #     logger.info("Sector column already present — skipping enrichment.")
    # else:
    #     logger.warning("company_id column missing — cannot enrich sector.")
    #     df["Sector"] = "Unknown"

    return df


def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Download daily adjusted close prices via yfinance.
    """
    import yfinance as yf

    if not tickers:
        return pd.DataFrame()

    # Remove NaN / empty tickers before passing to yfinance
    clean_tickers = [
        t for t in tickers
        if t
        and str(t).strip().lower() not in ("nan", "none", "")
        and not (isinstance(t, float) and pd.isna(t))
    ]
    if len(clean_tickers) < len(tickers):
        logger.warning(
            "Removed %d NaN/empty tickers before yfinance download.",
            len(tickers) - len(clean_tickers),
        )

    logger.info(
        "📈 Downloading prices for %d tickers (%s → %s) …",
        len(clean_tickers), start, end,
    )
    
    # Suppress mean of empty slice runtime warnings typically thrown by yfinance on sparse dividends/splits
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        raw = yf.download(
            clean_tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=True,
        )

    if isinstance(raw.columns, pd.MultiIndex):
        prices = (
            raw["Close"]
            if "Close" in raw.columns.get_level_values(0)
            else raw[raw.columns.get_level_values(0).unique()[0]]
        )
    else:
        prices = raw["Close"] if "Close" in raw.columns else raw

    if prices.empty:
        logger.warning("yfinance returned no data.")
        return pd.DataFrame()

    # Ensure it's a DataFrame (single ticker returns a Series)
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=clean_tickers[0])

    dropped = prices.columns[prices.isna().all()].tolist()
    if dropped:
        logger.warning("Dropping %d delisted/invalid tickers: %s", len(dropped), dropped)
    prices = prices.drop(columns=dropped)
    prices = prices.ffill().dropna(how="all")

    logger.info("Prices ready: %d tickers × %d days", prices.shape[1], len(prices))
    return prices
# artifact refresh: 1784455641.5617282
