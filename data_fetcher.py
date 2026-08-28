"""
data_fetcher.py
---------------
Fetches the stock universe from Screener.in with multi-tier network fallbacks
(requests -> httpx with trust_env=False -> playwright fallback) and robust
error handling.
"""
from __future__ import annotations

import datetime
import html as html_module
import json
import logging
import os
import re
import socket
import time
import warnings
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import connection as urllib3_cn

# Force urllib3 / requests to resolve IPv4 only — fixes [Errno 111] on Streamlit Cloud / Docker
urllib3_cn.HAS_IPV6 = False

try:
    import httpx
except ImportError:
    httpx = None

try:
    from curl_cffi import requests as cffi_requests
except ImportError:
    cffi_requests = None

import pandas as pd
from bs4 import BeautifulSoup

logger = logging.getLogger("data_fetcher")

LOGIN_URL      = "https://www.screener.in/login"
RAW_SCREEN_URL = "https://www.screener.in/screen/raw/"
SCREENER_BASE  = "https://www.screener.in"
PAGE_LIMIT     = 25
MAX_PAGES      = 40

class AuthenticationError(Exception):
    """Session cookie expired or SCREENER_SESSION env var not set."""


def _parse_cookie_string(cookie_str: str) -> dict[str, str]:
    """Parse cookie string (either raw sessionid token or full 'k=v; k2=v2' format)."""
    cookie_str = cookie_str.strip().strip("'").strip('"')
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
    """Retrieve session cookies from Streamlit Secrets or Environment Variables."""
    val = ""
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "SCREENER_SESSION" in st.secrets:
            val = str(st.secrets["SCREENER_SESSION"]).strip()
    except Exception:
        pass

    if not val:
        val = os.environ.get("SCREENER_SESSION", "").strip()

    if not val:
        raise RuntimeError(
            "SCREENER_SESSION is not set.\n"
            "  1. Log in to screener.in in your browser\n"
            "  2. Open DevTools → Application → Cookies → screener.in\n"
            "  3. Copy the `sessionid` value\n"
            "  4. On Streamlit Cloud: Add `SCREENER_SESSION = \"...\"` to App Settings > Secrets\n"
            "     OR paste it directly into the app sidebar."
        )
    cookies = _parse_cookie_string(val)
    logger.debug("Parsed cookies: %s", list(cookies.keys()))
    return cookies


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
    "Change in FII holding",
    "Change in DII holding",
]


def _screen_url(query: str, page: int) -> str:
    base_params = {"query": query, "limit": str(PAGE_LIMIT), "page": str(page)}
    base = f"{RAW_SCREEN_URL}?{_urlencode(base_params)}"
    col_str = "".join(f"&column={quote(c, safe='')}" for c in SCREENER_COLUMNS)
    return f"{base}{col_str}"


_STANDARD_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.screener.in/screens/",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "DNT": "1",
    "Sec-Ch-Ua": '"Not-A.Brand";v="99", "Chromium";v="124", "Google Chrome";v="124"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Upgrade-Insecure-Requests": "1",
}


def _fetch_with_curl_cffi(cookies: dict[str, str], url: str) -> str:
    """Fetch using curl_cffi with browser impersonation to bypass Cloudflare TLS checks."""
    if cffi_requests is None:
        raise ImportError("curl_cffi not installed")
    
    resp = cffi_requests.get(
        url,
        cookies=cookies,
        headers=_STANDARD_HEADERS,
        impersonate="chrome124",
        timeout=30,
    )
    resp.raise_for_status()
    final_url = str(resp.url)
    if "login" in final_url or "register" in final_url:
        raise AuthenticationError(
            f"Screener.in redirected to {final_url!r}. "
            "Your SCREENER_SESSION cookie has expired — please refresh it."
        )
    return resp.text


def _fetch_with_requests(session: requests.Session, url: str) -> str:
    """Fetch URL using IPv4-forced requests.Session."""
    resp = session.get(url, headers=_STANDARD_HEADERS, timeout=25, allow_redirects=True)
    resp.raise_for_status()
    final_url = str(resp.url)
    if "login" in final_url or "register" in final_url:
        raise AuthenticationError(
            f"Screener.in redirected to {final_url!r}. "
            "Your SCREENER_SESSION cookie has expired — please refresh it."
        )
    return resp.text


def _fetch_with_httpx(cookies: dict[str, str], url: str) -> str:
    """Fetch URL using httpx with trust_env=False to prevent proxy traps."""
    if httpx is None:
        raise ImportError("httpx not installed")
    with httpx.Client(
        cookies=cookies,
        headers=_STANDARD_HEADERS,
        follow_redirects=True,
        timeout=30.0,
        trust_env=False,
    ) as client:
        resp = client.get(url)
        resp.raise_for_status()
        final_url = str(resp.url)
        if "login" in final_url or "register" in final_url:
            raise AuthenticationError(
                f"Screener.in redirected to {final_url!r}. "
                "Your SCREENER_SESSION cookie has expired — please refresh it."
            )
        return resp.text


def _unified_fetch(requests_session: requests.Session, cookies: dict[str, str], url: str) -> str:
    """
    Attempts download via requests (IPv4-forced) -> curl_cffi -> httpx -> Playwright.
    """
    errors = []

    # 1. requests (with IPv4 forced)
    try:
        return _fetch_with_requests(requests_session, url)
    except AuthenticationError:
        raise
    except Exception as err:
        errors.append(f"requests: {err}")
        logger.warning("Requests fetch failed for %s (%s). Trying curl_cffi / httpx...", url, err)

    # 2. curl_cffi (Chrome TLS fingerprint)
    if cffi_requests is not None:
        try:
            return _fetch_with_curl_cffi(cookies, url)
        except AuthenticationError:
            raise
        except Exception as err:
            errors.append(f"curl_cffi: {err}")
            logger.warning("curl_cffi fetch failed for %s (%s)...", url, err)

    # 3. httpx (trust_env=False)
    if httpx is not None:
        try:
            return _fetch_with_httpx(cookies, url)
        except AuthenticationError:
            raise
        except Exception as err:
            errors.append(f"httpx: {err}")
            logger.warning("httpx fetch failed for %s (%s)...", url, err)

    # 4. Playwright (optional headless fallback)
    try:
        return _playwright_fetch(url, cookies)
    except AuthenticationError:
        raise
    except Exception as err:
        errors.append(f"playwright: {err}")

    raise RuntimeError(
        f"All fetch methods failed for {url}.\n"
        f"Errors: {'; '.join(errors)}\n"
        "If you are running on Streamlit Cloud, Screener.in may be blocking cloud IP requests.\n"
        "You can use the 'Upload Screener CSV' tab in the app as a zero-setup alternative."
    )


def _clean_cell(tag_or_str: Any) -> str:
    """Extract clean whitespace-normalized plain text from a BS4 tag or raw string."""
    if hasattr(tag_or_str, "get_text"):
        return " ".join(tag_or_str.get_text(separator=" ").split())
    return " ".join(re.sub(r"<[^>]*>", "", str(tag_or_str)).split())


def _parse_results_table(html: str) -> pd.DataFrame:
    """Parse Screener.in's results table from raw HTML using BeautifulSoup."""
    soup = BeautifulSoup(html, "lxml")

    # Strategy 1: class data-table
    table = soup.find("table", class_="data-table")

    # Strategy 2: class contains data-table
    if table is None:
        for t in soup.find_all("table"):
            cls = " ".join(t.get("class", []))
            if "data-table" in cls:
                table = t
                break

    # Strategy 3: first table on page
    if table is None:
        all_tables = soup.find_all("table")
        if all_tables:
            table = all_tables[0]
        else:
            return pd.DataFrame()

    thead = table.find("thead")
    if thead:
        header_tags = thead.find_all("th")
    else:
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

    if not headers:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    tbody = table.find("tbody")
    row_source = tbody if tbody else table
    for tr in row_source.find_all("tr"):
        if tr.find("th"):
            continue
        cells = tr.find_all("td")
        if not cells:
            continue

        values = [_clean_cell(c) for c in cells[: len(headers)]]
        if len(values) < len(headers):
            values += [""] * (len(headers) - len(values))
        record: dict[str, Any] = dict(zip(headers, values, strict=False))

        company_link = tr.find("a", href=re.compile(r"/company/[^/]+"))
        if company_link:
            href = company_link.get("href", "")
            slug_match = re.search(r"/company/([^/\"]+)", href)
            if slug_match:
                record["company_url"] = href
                raw_slug = slug_match.group(1)
                decoded_slug = html_module.unescape(raw_slug)
                record["company_id"] = decoded_slug.upper()

        rows.append(record)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "company_id" in df.columns:
        df["symbol"] = df["company_id"]
    elif "Name" in df.columns:
        df["symbol"] = df["Name"].str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    return df


_NON_NUMERIC = {"Name", "Ticker", "Sector", "company_url", "company_id", "symbol", "ISIN"}
_FORCE_NUMERIC = {
    "P/E", "ROE", "ROCE", "Sales Growth", "Profit Growth", "Debt to Equity",
    "Piotroski score", "Market Cap", "ROCE %", "Div Yld %", "Promoter Holding"
}

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
        "Roce %",
    ],
    "Profit Growth": [
        "Profit growth %",
        "Profit Var %",
        "Profit Growth",
        "PAT Growth %",
        "Net Profit Growth %",
        "ProfitGrowth",
        "Net profit growth %",
        "Profit Var 5Yrs %",
        "Profit Var 3Yrs %",
    ],
    "Sales Growth": [
        "Sales growth %",
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
        "promoterholding",
    ],
}


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Safely coerce numerical columns to floats."""
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


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw column variants to standard canonical names."""
    df = df.copy()
    raw_cols = list(df.columns)
    col_map = {str(c).strip().lower(): c for c in raw_cols}

    for canonical, aliases in EXPECTED_COLUMNS.items():
        if canonical in df.columns and df[canonical].notna().any():
            continue

        for alias in aliases:
            alias_lower = str(alias).strip().lower()
            if alias_lower not in col_map:
                continue

            actual_col = col_map[alias_lower]
            if df[actual_col].isna().all():
                continue

            if canonical in df.columns:
                df[canonical] = df[actual_col]
                df = df.drop(columns=[actual_col])
            else:
                df = df.rename(columns={actual_col: canonical})
            break

    return df


def sanitize_ticker(ticker: str) -> str:
    """Map Screener company ID / symbol to Yahoo Finance .NS format."""
    ticker = ticker.strip()
    ticker = html_module.unescape(ticker)
    if " & " in ticker:
        ticker = ticker.replace(" & ", "AND")
    if "&" in ticker:
        ticker = ticker.replace("&", "AND")
    if "." in ticker:
        ticker = ticker.replace(".", "")
    if " " in ticker:
        ticker = ticker.replace(" ", "")
    if "-" in ticker:
        ticker = ticker.replace("-", "")
    return ticker.upper() + ".NS"


def parse_screener_csv(file_or_content: Any) -> pd.DataFrame:
    """Process an exported CSV from Screener.in into a normalized dataframe."""
    df = pd.read_csv(file_or_content)
    df = _normalise_columns(df)
    df = _coerce_numeric(df)

    if "Ticker" not in df.columns or df["Ticker"].isna().all():
        if "Name" in df.columns:
            df["Ticker"] = df["Name"].str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)

    if "Ticker" in df.columns:
        df["Ticker"] = df["Ticker"].apply(
            lambda t: sanitize_ticker(str(t))
            if pd.notna(t) and str(t).strip() not in ("", "nan", "none")
            else t
        )

    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan")

    if "Sector" not in df.columns or df["Sector"].isna().all():
        df["Sector"] = "Unknown"

    return df


def fetch_screener_universe(query: str) -> pd.DataFrame:
    """
    Runs a Screener.in DSL query across all result pages and returns a clean DataFrame.
    """
    cookies = _get_session_cookies()
    frames: list[pd.DataFrame] = []
    pages_fetched = 0

    logger.info("━━━ Screener.in Query ━━━")
    logger.info("Query: %s", query)
    logger.info("Base URL: %s", _screen_url(query, 1))

    # Configure session with IPv4-forced HTTPAdapter
    req_session = requests.Session()
    req_session.cookies.update(cookies)
    adapter = HTTPAdapter(max_retries=2)
    req_session.mount("https://", adapter)
    req_session.mount("http://", adapter)

    for pagenum in range(1, MAX_PAGES + 1):
        url = _screen_url(query, pagenum)
        logger.info("📄 Fetching Page %d → %s", pagenum, url)
        try:
            html = _unified_fetch(req_session, cookies, url)
        except AuthenticationError:
            raise
        except Exception as exc:
            logger.error("Failed to fetch page %d via all mechanisms: %s", pagenum, exc)
            break

        page_df = _parse_results_table(html)
        pages_fetched += 1
        logger.info(
            "  → Page %d: %d rows parsed (cumulative: %d)",
            pagenum,
            len(page_df),
            sum(len(f) for f in frames) + len(page_df),
        )

        if page_df.empty:
            logger.info("  Empty page reached — pagination complete.")
            break

        frames.append(page_df)
        if len(page_df) < PAGE_LIMIT:
            logger.info("  Last page received %d < %d rows — end of screen results.", len(page_df), PAGE_LIMIT)
            break

        time.sleep(0.3)

    if not frames:
        logger.error(
            "fetch_screener_universe: zero rows fetched after %d pages.\n"
            "  Troubleshooting on Streamlit Cloud:\n"
            "  1. Verify SCREENER_SESSION cookie in App Secrets or sidebar\n"
            "  2. If Screener is blocking the AWS Cloud IP, use the 'Upload CSV' option in the sidebar\n"
            "  3. Verify the query syntax on screener.in directly",
            pages_fetched,
        )
        return pd.DataFrame(columns=list(EXPECTED_COLUMNS.keys()))

    df = pd.concat(frames, ignore_index=True)
    logger.info("━━━ Fetched %d raw rows across %d pages ━━━", len(df), pages_fetched)

    df = _normalise_columns(df)
    df = _coerce_numeric(df)

    if "Ticker" in df.columns:
        df["Ticker"] = df["Ticker"].apply(
            lambda t: sanitize_ticker(str(t))
            if pd.notna(t) and str(t).strip() not in ("", "nan", "none")
            else t
        )
    else:
        logger.warning("Ticker column could not be derived from result set.")

    if "Name" in df.columns and "Ticker" in df.columns:
        df["_is_ns"] = df["Ticker"].astype(str).str.endswith(".NS")
        df = df.sort_values("_is_ns", ascending=False)
        df = df.drop_duplicates(subset=["Name"], keep="first")
        df = df.drop(columns=["_is_ns"])
    else:
        dedup_col = next((c for c in ["company_id", "Ticker", "Name"] if c in df.columns), None)
        if dedup_col:
            df = df.drop_duplicates(subset=dedup_col, keep="first")

    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan")

    if "Sector" not in df.columns or df["Sector"].isna().all():
        df["Sector"] = "Unknown"

    logger.info("━━━ Final universe: %d stocks, %d columns ━━━", len(df), len(df.columns))
    return df


def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download daily adjusted close prices via yfinance."""
    import yfinance as yf

    if not tickers:
        return pd.DataFrame()

    clean_tickers = [
        t for t in tickers
        if t
        and str(t).strip().lower() not in ("nan", "none", "")
        and not (isinstance(t, float) and pd.isna(t))
    ]

    logger.info("Downloading prices for %d tickers (%s → %s)...", len(clean_tickers), start, end)

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
        logger.warning("yfinance returned no price data.")
        return pd.DataFrame()

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=clean_tickers[0])

    dropped = prices.columns[prices.isna().all()].tolist()
    if dropped:
        logger.warning("Dropping %d delisted/invalid tickers: %s", len(dropped), dropped)
    prices = prices.drop(columns=dropped)
    prices = prices.ffill().dropna(how="all")

    return prices