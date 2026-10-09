#!/usr/bin/env python3
"""
Market Monitor: data updater.

GitHub Actions runs this after each U.S. trading day (see .github/workflows/update.yml).
Each section is fetched independently. If a source fails, that section keeps its
previous JSON file and the failure is recorded in data/status.json, so one broken
source never takes the whole site down.

Sources
  Yahoo Finance (via yfinance)     indices, sector ETFs, cross-asset prices, largest firms
  FRED, Federal Reserve Bank of St. Louis
                                   Treasury yields, policy rate, inflation, jobs, GDP, recessions
  Robert J. Shiller (Yale)         CAPE and trailing P/E since 1881
  Kenneth R. French Data Library   Fama-French 5 factors + momentum (monthly)

Usage
  python scripts/update_data.py                  # all sections
  python scripts/update_data.py macro factors    # only some sections
"""

from __future__ import annotations

import datetime as dt
import io
import json
import math
import os
import re
import sys
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# --------------------------------------------------------------------------------------
# What to track. Edit these lists to change what the site shows.
# --------------------------------------------------------------------------------------

INDICES = [
    ("^GSPC", "S&P 500"),
    ("^IXIC", "Nasdaq Composite"),
    ("^DJI", "Dow Jones Industrial Average"),
    ("^RUT", "Russell 2000"),
    ("^VIX", "VIX volatility index"),
]

# Select Sector SPDR ETFs: one per GICS sector of the S&P 500 (total return, incl. dividends)
SECTORS = [
    ("XLK", "Information Technology"),
    ("XLF", "Financials"),
    ("XLV", "Health Care"),
    ("XLY", "Consumer Discretionary"),
    ("XLC", "Communication Services"),
    ("XLI", "Industrials"),
    ("XLP", "Consumer Staples"),
    ("XLE", "Energy"),
    ("XLU", "Utilities"),
    ("XLRE", "Real Estate"),
    ("XLB", "Materials"),
]
BENCHMARK = ("SPY", "S&P 500 (SPY)")

CROSS_ASSET = [
    ("CL=F", "WTI crude oil", "$ per barrel"),
    ("GC=F", "Gold", "$ per troy ounce"),
    ("DX-Y.NYB", "U.S. Dollar Index", "index vs. 6 currencies"),
    ("EURUSD=X", "Euro / U.S. dollar", "$ per euro"),
    ("BTC-USD", "Bitcoin", "$ per coin"),
]

# Candidate pool for "Largest firms". The site ranks these by current market cap
# and shows the top TOP_N, so the ranking stays right as companies rise and fall.
# Add a ticker here if a company not on the list grows into the top 20.
FIRM_UNIVERSE = [
    "NVDA", "MSFT", "AAPL", "GOOGL", "AMZN", "META", "AVGO", "TSLA", "BRK-B", "LLY",
    "JPM", "WMT", "V", "ORCL", "MA", "XOM", "NFLX", "COST", "JNJ", "PLTR",
    "AMD", "HD", "ABBV", "BAC", "PG", "UNH", "KO", "CVX", "GE", "CSCO",
    "IBM", "MU", "WFC", "CRM", "MS", "GS", "PM", "TMUS", "LIN", "MRK",
    "CAT", "AXP", "RTX", "INTU", "NOW", "AMAT", "LRCX", "ISRG", "UBER", "APP",
]
TOP_N = 20

# FRED series ------------------------------------------------------------------------
# (id, label, units)
RATE_SERIES = [
    ("DFF", "Effective federal funds rate", "%"),
    ("DGS3MO", "3-month Treasury bill", "%"),
    ("DGS2", "2-year Treasury note", "%"),
    ("DGS10", "10-year Treasury note", "%"),
    ("DGS30", "30-year Treasury bond", "%"),
    ("DFII10", "10-year TIPS (real) yield", "%"),
    ("T10YIE", "10-year breakeven inflation", "%"),
    ("T10Y2Y", "10-year minus 2-year spread", "pp"),
    ("T10Y3M", "10-year minus 3-month spread", "pp"),
    ("BAMLH0A0HYM2", "High-yield bond spread", "pp"),
    ("MORTGAGE30US", "30-year fixed mortgage rate", "%"),
]
CURVE_SERIES = [
    ("DGS1MO", "1M", 1 / 12), ("DGS3MO", "3M", 0.25), ("DGS6MO", "6M", 0.5),
    ("DGS1", "1Y", 1), ("DGS2", "2Y", 2), ("DGS3", "3Y", 3), ("DGS5", "5Y", 5),
    ("DGS7", "7Y", 7), ("DGS10", "10Y", 10), ("DGS20", "20Y", 20), ("DGS30", "30Y", 30),
]
# (id, label, units, transform)  transform: level | yoy (% change vs. 12 months ago) | diff
MACRO_SERIES = [
    ("CPIAUCSL", "CPI inflation", "% y/y", "yoy"),
    ("CPILFESL", "Core CPI inflation", "% y/y", "yoy"),
    ("PCEPILFE", "Core PCE inflation", "% y/y", "yoy"),
    ("UNRATE", "Unemployment rate", "%", "level"),
    ("PAYEMS", "Payroll job gains", "thousands", "diff"),
    ("A191RL1Q225SBEA", "Real GDP growth", "% annualized", "level"),
]

YEARS_DAILY = 3        # daily detail kept for rate charts (longer ranges use monthly averages)
HISTORY_START = "1960-01-01"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0 (Market Monitor educational dashboard)"})


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[{dt.datetime.now(dt.timezone.utc):%H:%M:%S}] {msg}", flush=True)


def retry(fn, tries: int = 3, wait: float = 4.0, what: str = ""):
    last = None
    for k in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - network calls fail in many ways
            last = e
            log(f"  retry {k + 1}/{tries} {what}: {type(e).__name__}: {str(e)[:160]}")
            time.sleep(wait * (k + 1))
    raise last  # type: ignore[misc]


def sig(x, digits: int = 6):
    """Round to significant digits; NaN/inf -> None."""
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    if x == 0:
        return 0.0
    return float(f"{x:.{digits}g}")


def rnd(x, nd: int = 4):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, nd) if math.isfinite(x) else None


def clean(obj):
    """Make an object JSON-safe (NaN -> null, numpy -> python)."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return sig(obj, 7)  # 7 significant digits keeps files small without visible rounding
    if isinstance(obj, (pd.Timestamp, dt.date)):
        return obj.strftime("%Y-%m-%d")
    return obj


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(clean(obj), separators=(",", ":"), allow_nan=False))
    tmp.replace(path)


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# --------------------------------------------------------------------------------------
# Return math (pure functions, unit-tested in tests/test_parsers.py)
# --------------------------------------------------------------------------------------

def perf(s: pd.Series) -> dict | None:
    """Level and trailing returns for a daily price series (fractions, e.g. 0.012 = +1.2%)."""
    s = pd.to_numeric(s, errors="coerce").dropna()
    s = s[s > 0].sort_index()
    if len(s) < 2:
        return None
    d = s.index[-1]
    last = float(s.iloc[-1])
    prev = float(s.iloc[-2])

    def base_at(ts):
        sub = s.loc[:ts]
        # require the base date to be inside the series, not before it starts
        if len(sub) == 0 or (ts - s.index[0]).days < 0:
            return None
        return float(sub.iloc[-1])

    def ret(base):
        return None if not base else last / base - 1

    prior_year = s[s.index < pd.Timestamp(d.year, 1, 1)]
    year = s[s.index > d - pd.DateOffset(years=1)]
    return {
        "date": d.strftime("%Y-%m-%d"),
        "last": last,
        "prev": prev,
        "chg1d": last / prev - 1,
        "chg1w": ret(base_at(d - pd.Timedelta(days=7))),
        "chg1m": ret(base_at(d - pd.DateOffset(months=1))),
        "chg3m": ret(base_at(d - pd.DateOffset(months=3))),
        "ytd": ret(float(prior_year.iloc[-1])) if len(prior_year) else None,
        "chg1y": ret(base_at(d - pd.DateOffset(years=1))),
        "high52": float(year.max()),
        "low52": float(year.min()),
    }


def spark(s: pd.Series, n: int = 126) -> list:
    s = pd.to_numeric(s, errors="coerce").dropna().sort_index().iloc[-n:]
    return [sig(v, 6) for v in s.values]


def series_pairs(s: pd.Series, fmt: str = "%Y-%m-%d", nd: int = 4) -> list:
    s = s.dropna()
    return [[i.strftime(fmt), rnd(v, nd)] for i, v in s.items()]


# --------------------------------------------------------------------------------------
# Yahoo Finance
# --------------------------------------------------------------------------------------

def yahoo_history(tickers: list[str], period: str = "2y") -> dict[str, pd.DataFrame]:
    """Return {"Close": DataFrame, "Adj Close": DataFrame} with one column per ticker."""
    import yfinance as yf

    def dl(tks):
        df = yf.download(tks, period=period, interval="1d", auto_adjust=False, actions=False,
                         group_by="column", progress=False, threads=True, multi_level_index=True)
        if df is None or df.empty:
            raise RuntimeError(f"Yahoo returned no data for {tks[:5]}...")
        return df

    df = retry(lambda: dl(tickers), what=f"yahoo download ({len(tickers)} tickers)")
    out = {}
    for field in ("Close", "Adj Close"):
        if field in df.columns.get_level_values(0):
            f = df[field].copy()
        else:  # e.g. Adj Close missing for some instruments
            f = df["Close"].copy()
        if isinstance(f, pd.Series):
            f = f.to_frame(tickers[0])
        idx = pd.to_datetime(f.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        f.index = idx.normalize()
        out[field] = f

    # retry tickers that came back empty, one at a time
    missing = [t for t in tickers if t not in out["Close"] or out["Close"][t].dropna().empty]
    for t in missing:
        try:
            one = dl([t])
            for field in ("Close", "Adj Close"):
                col = one[field] if field in one.columns.get_level_values(0) else one["Close"]
                col = col.iloc[:, 0] if isinstance(col, pd.DataFrame) else col
                idx = pd.to_datetime(col.index)
                col.index = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
                out[field] = out[field].reindex(out[field].index.union(col.index))
                out[field][t] = col
            log(f"  recovered {t} on single retry")
        except Exception as e:  # noqa: BLE001
            log(f"  {t}: no Yahoo data ({e})")
    return out


def clean_name(name: str) -> str:
    """'NVIDIA Corporation' -> 'NVIDIA'; 'Berkshire Hathaway Inc. New' -> 'Berkshire Hathaway'."""
    n = re.sub(r"\s*\(The\)$", "", (name or "").strip())
    if re.search(r"&\s*Co\.?$", n):          # 'JPMorgan Chase & Co.' stays as is
        return n
    n = re.sub(r",?\s+(Inc\.?|Incorporated|Corporation|Corp\.?|plc|PLC|Ltd\.?|N\.V\.|S\.A\.)(\s+New)?$", "", n)
    return n.strip().rstrip(",") or name


def yahoo_info(sym: str) -> dict:
    import yfinance as yf

    try:
        info = retry(lambda: yf.Ticker(sym).get_info(), tries=3, wait=3, what=f"info {sym}")
        return info or {}
    except Exception:  # noqa: BLE001
        return {}


# FRED fallbacks if Yahoo fails for an index
FRED_INDEX_FALLBACK = {"^GSPC": "SP500", "^IXIC": "NASDAQCOM", "^DJI": "DJIA", "^VIX": "VIXCLS"}


def build_markets() -> dict:
    syms = [s for s, _ in INDICES] + [s for s, _ in SECTORS] + [BENCHMARK[0]] + [s for s, _, _ in CROSS_ASSET]
    h = yahoo_history(syms, "2y")
    close, adj = h["Close"], h["Adj Close"]

    def col(frame, sym):
        return frame[sym] if sym in frame else pd.Series(dtype=float)

    indices = []
    for sym, name in INDICES:
        s = col(close, sym)
        source = "Yahoo Finance"
        if s.dropna().empty and sym in FRED_INDEX_FALLBACK:
            s = fred(FRED_INDEX_FALLBACK[sym], (dt.date.today() - dt.timedelta(days=800)).isoformat())
            source = "FRED"
        p = perf(s)
        if not p:
            log(f"  index {sym}: no data")
            continue
        p.update(symbol=sym, name=name, spark=spark(s), source=source)
        indices.append(p)
    if not indices:
        raise RuntimeError("no index data")

    sectors = []
    for sym, name in SECTORS:
        p = perf(col(adj, sym))
        if p:
            p.update(symbol=sym, name=name)
            sectors.append(p)
    bench = perf(col(adj, BENCHMARK[0]))
    if bench:
        bench.update(symbol=BENCHMARK[0], name=BENCHMARK[1])

    cross = []
    for sym, name, unit in CROSS_ASSET:
        s = col(close, sym)
        p = perf(s)
        if p:
            p.update(symbol=sym, name=name, unit=unit, spark=spark(s))
            cross.append(p)

    # S&P 500 long history: all-time high, drawdown, realized volatility, 10-year chart
    spx = None
    try:
        spx = yahoo_history(["^GSPC"], "max")["Close"]["^GSPC"].dropna()
    except Exception as e:  # noqa: BLE001
        log(f"  ^GSPC max history failed ({e}); using FRED SP500 (10 years)")
        spx = fred("SP500", (dt.date.today() - dt.timedelta(days=3700)).isoformat())
    spx = spx[spx > 0].sort_index()
    last_d = spx.index[-1]
    ath_d = spx.idxmax()
    logret = np.log(spx).diff().dropna()
    ten = spx[spx.index >= last_d - pd.DateOffset(years=10)]
    spx_block = {
        "date": last_d.strftime("%Y-%m-%d"),
        "last": float(spx.iloc[-1]),
        "ath": float(spx.max()),
        "athDate": ath_d.strftime("%Y-%m-%d"),
        "drawdown": float(spx.iloc[-1] / spx.max() - 1),
        "rv21": float(logret.iloc[-21:].std() * math.sqrt(252)),
        "rv63": float(logret.iloc[-63:].std() * math.sqrt(252)),
        "history": [[i.strftime("%Y-%m-%d"), rnd(v, 2)] for i, v in ten.items()],
    }

    asof = max(p["date"] for p in indices)
    return {"asof": asof, "indices": indices, "sectors": sectors, "benchmark": bench,
            "cross": cross, "spx": spx_block}


def build_firms() -> dict:
    h = yahoo_history(FIRM_UNIVERSE, "2y")
    close = h["Close"]
    rows = []
    for sym in FIRM_UNIVERSE:
        if sym not in close:
            continue
        p = perf(close[sym])
        if not p:
            continue
        info = yahoo_info(sym)
        time.sleep(0.3)  # be polite to Yahoo
        mcap = info.get("marketCap")
        if not mcap:
            shares = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
            mcap = shares * p["last"] if shares else None
        if not mcap:
            log(f"  {sym}: no market cap, skipped")
            continue
        div_rate = info.get("dividendRate")
        div_yield = (div_rate / p["last"]) if div_rate else info.get("trailingAnnualDividendYield")
        if div_yield is not None and not (0 <= div_yield < 0.25):
            div_yield = None
        rows.append({
            "symbol": sym,
            "name": clean_name(info.get("shortName") or info.get("longName") or sym),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "price": p["last"], "date": p["date"],
            "chg1d": p["chg1d"], "chg1m": p["chg1m"], "ytd": p["ytd"], "chg1y": p["chg1y"],
            "high52": p["high52"], "low52": p["low52"],
            "marketCap": float(mcap),
            "trailingPE": info.get("trailingPE"),
            "forwardPE": info.get("forwardPE"),
            "divYield": div_yield,
            "beta": info.get("beta"),
        })
    rows.sort(key=lambda r: r["marketCap"], reverse=True)
    if len(rows) < 10:
        raise RuntimeError(f"only {len(rows)} firms with market-cap data; keeping previous file")
    top = rows[:TOP_N]
    for i, r in enumerate(top, 1):
        r["rank"] = i
    return {"asof": max(r["date"] for r in top), "universeSize": len(FIRM_UNIVERSE), "firms": top}


# --------------------------------------------------------------------------------------
# FRED
# --------------------------------------------------------------------------------------

def parse_fred_csv(text: str, sid: str = "") -> pd.Series:
    df = pd.read_csv(io.StringIO(text))
    if df.shape[1] < 2 or "date" not in str(df.columns[0]).lower():
        raise ValueError(f"unexpected FRED response for {sid}: {text[:120]!r}")
    dates = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    vals = pd.to_numeric(df.iloc[:, 1], errors="coerce")  # FRED marks gaps with "." or blanks
    s = pd.Series(vals.values, index=dates, name=sid)
    s = s[s.index.notna()].dropna().sort_index()
    if s.empty:
        raise ValueError(f"FRED series {sid} is empty")
    return s


_FRED_CACHE: dict[str, pd.Series] = {}


def fred(sid: str, start: str = HISTORY_START) -> pd.Series:
    """Download one FRED series from `start` on (cached, so a series is fetched once per run)."""
    cached = _FRED_CACHE.get(sid)
    if cached is not None and cached.index[0] <= pd.Timestamp(start) + pd.Timedelta(days=31):
        return cached[cached.index >= pd.Timestamp(start)]
    s = _fred_download(sid, start)
    _FRED_CACHE[sid] = s
    return s


def _fred_download(sid: str, start: str) -> pd.Series:
    key = os.environ.get("FRED_API_KEY", "").strip()
    if key:
        def get():
            r = SESSION.get("https://api.stlouisfed.org/fred/series/observations",
                            params={"series_id": sid, "api_key": key, "file_type": "json",
                                    "observation_start": start}, timeout=40)
            r.raise_for_status()
            obs = r.json()["observations"]
            s = pd.Series({pd.Timestamp(o["date"]): o["value"] for o in obs}, name=sid)
            s = pd.to_numeric(s, errors="coerce").dropna().sort_index()
            if s.empty:
                raise ValueError(f"FRED series {sid} is empty")
            return s
    else:
        def get():
            r = SESSION.get("https://fred.stlouisfed.org/graph/fredgraph.csv",
                            params={"id": sid, "cosd": start}, timeout=40)
            r.raise_for_status()
            return parse_fred_csv(r.text, sid)
    return retry(get, what=f"FRED {sid}")


def recession_periods(usrec: pd.Series) -> list[list[str]]:
    """USREC (1 = recession month) -> [[start 'YYYY-MM', end 'YYYY-MM'], ...]."""
    out, start, prev = [], None, None
    for d, v in usrec.sort_index().items():
        if v >= 0.5 and start is None:
            start = d
        if v < 0.5 and start is not None:
            out.append([start.strftime("%Y-%m"), prev.strftime("%Y-%m")])
            start = None
        prev = d
    if start is not None:
        out.append([start.strftime("%Y-%m"), prev.strftime("%Y-%m")])
    return out


def value_before(s: pd.Series, ts) -> float | None:
    sub = s.loc[:ts]
    return float(sub.iloc[-1]) if len(sub) else None


def build_macro() -> dict:
    today = dt.date.today()
    rates, errors = {}, []

    for sid, label, units in RATE_SERIES:
        try:
            s = fred(sid, HISTORY_START)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{sid}: {e}")
            continue
        d = s.index[-1]
        recent = s[s.index >= pd.Timestamp(today) - pd.DateOffset(years=YEARS_DAILY)]
        monthly = s.resample("MS").mean().dropna()
        last = float(s.iloc[-1])
        m1, y1 = value_before(s, d - pd.DateOffset(months=1)), value_before(s, d - pd.DateOffset(years=1))
        rates[sid] = {
            "label": label, "units": units, "last": last, "lastDate": d.strftime("%Y-%m-%d"),
            "chg1d": last - float(s.iloc[-2]) if len(s) > 1 else None,
            "chg1m": last - m1 if m1 is not None else None,
            "chg1y": last - y1 if y1 is not None else None,
            "daily": series_pairs(recent, "%Y-%m-%d", 3),
            "monthly": series_pairs(monthly, "%Y-%m", 3),
        }
        time.sleep(0.2)

    # Yield curve snapshots: latest, ~1 month ago, ~1 year ago
    curve_raw = {}
    for sid, lab, yrs in CURVE_SERIES:
        try:
            curve_raw[sid] = fred(sid, (today - dt.timedelta(days=500)).isoformat())
        except Exception as e:  # noqa: BLE001
            errors.append(f"{sid}: {e}")
        time.sleep(0.2)
    curve = None
    if "DGS10" in curve_raw:
        d0 = curve_raw["DGS10"].index[-1]
        snaps = []
        for name, ts in (("Latest", d0), ("1 month ago", d0 - pd.DateOffset(months=1)),
                         ("1 year ago", d0 - pd.DateOffset(years=1))):
            ref = curve_raw["DGS10"].loc[:ts]
            if not len(ref):
                continue
            ref_d = ref.index[-1]
            vals = [value_before(curve_raw[sid], ref_d) if sid in curve_raw else None
                    for sid, _, _ in CURVE_SERIES]
            snaps.append({"name": name, "date": ref_d.strftime("%Y-%m-%d"), "values": vals})
        curve = {"labels": [lab for _, lab, _ in CURVE_SERIES],
                 "years": [y for _, _, y in CURVE_SERIES], "snapshots": snaps}

    macro = {}
    for sid, label, units, how in MACRO_SERIES:
        try:
            raw = fred(sid, "1958-01-01")
        except Exception as e:  # noqa: BLE001
            errors.append(f"{sid}: {e}")
            continue
        if how == "yoy":
            s = (raw.pct_change(12, fill_method=None) * 100).dropna()
        elif how == "diff":
            s = raw.diff().dropna()
        else:
            s = raw
        s = s[s.index >= pd.Timestamp(HISTORY_START)]
        quarterly = sid.endswith("Q225SBEA")
        fmt = "%Y-%m"
        prev = float(s.iloc[-2]) if len(s) > 1 else None
        y1 = value_before(s, s.index[-1] - pd.DateOffset(years=1))
        macro[sid] = {
            "label": label, "units": units, "freq": "quarterly" if quarterly else "monthly",
            "last": float(s.iloc[-1]), "period": s.index[-1].strftime(fmt),
            "prev": prev, "yearAgo": y1,
            "monthly": series_pairs(s, fmt, 3),
        }
        time.sleep(0.2)

    recessions = []
    try:
        recessions = recession_periods(fred("USREC", "1948-01-01"))
    except Exception as e:  # noqa: BLE001
        errors.append(f"USREC: {e}")

    if len(rates) < 4 or len(macro) < 3:
        raise RuntimeError("too many FRED series failed: " + "; ".join(errors[:5]))
    return {"rates": rates, "curve": curve, "macro": macro, "recessions": recessions,
            "partialErrors": errors}


# --------------------------------------------------------------------------------------
# Shiller CAPE
# --------------------------------------------------------------------------------------

SHILLER_PAGES = ["https://shillerdata.com/", "http://www.econ.yale.edu/~shiller/data.htm"]
SHILLER_DIRECT = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"


def shiller_frac_to_month(x) -> str | None:
    """Shiller dates are 'year.month' floats: 1871.01 = Jan 1871, 1871.1 = Oct 1871."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x) or x < 1800:
        return None
    year = int(x)
    month = int(round((x - year) * 100))
    if not 1 <= month <= 12:
        return None
    return f"{year:04d}-{month:02d}"


def parse_shiller(raw: pd.DataFrame) -> dict:
    """raw = 'Data' sheet read with header=None. Returns monthly CAPE and trailing P/E."""
    header_row = None
    for i in range(min(40, len(raw))):
        if str(raw.iat[i, 0]).strip().lower() == "date":
            header_row = i
            break
    if header_row is None:
        raise ValueError("Shiller sheet: no 'Date' header row")

    def find_col(pred):
        for r in range(header_row, max(header_row - 4, -1), -1):
            for c in range(raw.shape[1]):
                if pred(str(raw.iat[r, c]).strip()):
                    return c
        return None

    cape_col = find_col(lambda v: v.upper() == "CAPE") or find_col(lambda v: v.upper() == "P/E10")
    p_col = find_col(lambda v: v == "P")
    e_col = find_col(lambda v: v == "E")
    cpi_col = find_col(lambda v: v.upper() == "CPI")

    body = raw.iloc[header_row + 1:]
    months = body.iloc[:, 0].map(shiller_frac_to_month)
    ok = months.notna()
    if cape_col is not None:
        cape = pd.to_numeric(body.iloc[:, cape_col], errors="coerce")[ok]
    elif None not in (p_col, e_col, cpi_col):
        # Fallback: rebuild CAPE = real price / 10-year average of real earnings
        price = pd.to_numeric(body.iloc[:, p_col], errors="coerce")[ok]
        earn = pd.to_numeric(body.iloc[:, e_col], errors="coerce")[ok]
        cpi = pd.to_numeric(body.iloc[:, cpi_col], errors="coerce")[ok].ffill()
        real_p, real_e = price / cpi, earn / cpi
        cape = real_p / real_e.rolling(120, min_periods=100).mean()
    else:
        raise ValueError("Shiller sheet: no CAPE column and no P/E/CPI columns to rebuild it")
    cape.index = months[ok]
    cape = cape[~cape.index.duplicated()].dropna()
    cape = cape[(cape > 1) & (cape < 150)]
    if len(cape) < 600:
        raise ValueError(f"Shiller CAPE series too short ({len(cape)})")

    pe = pd.Series(dtype=float)
    if p_col is not None and e_col is not None:
        price = pd.to_numeric(body.iloc[:, p_col], errors="coerce")[ok]
        earn = pd.to_numeric(body.iloc[:, e_col], errors="coerce")[ok]
        pe = (price / earn)
        pe.index = months[ok]
        pe = pe[~pe.index.duplicated()].replace([np.inf, -np.inf], np.nan).dropna()
        pe = pe[(pe > 0) & (pe < 200)]

    latest = float(cape.iloc[-1])
    return {
        "cape": [[k, rnd(v, 2)] for k, v in cape.items()],
        "capeLatest": latest,
        "capeDate": cape.index[-1],
        "capeMean": float(cape.mean()),
        "capeMedian": float(cape.median()),
        "capePercentile": float((cape <= latest).mean()),
        "capeStart": cape.index[0],
        "pe": [[k, rnd(v, 2)] for k, v in pe.items()],
        "peLatest": float(pe.iloc[-1]) if len(pe) else None,
        "peDate": pe.index[-1] if len(pe) else None,
    }


def build_valuation() -> dict:
    url = None
    for page in SHILLER_PAGES:
        try:
            html = retry(lambda: SESSION.get(page, timeout=30).text, tries=2, what=page)
            m = re.search(r'href="([^"]*ie_data\.xls[^"]*)"', html, re.I)
            if m:
                url = m.group(1)
                if url.startswith("//"):
                    url = "https:" + url
                elif url.startswith("/"):
                    url = requests.compat.urljoin(page, url)
                elif not url.startswith("http"):
                    url = requests.compat.urljoin(page, url)
                break
        except Exception as e:  # noqa: BLE001
            log(f"  Shiller page {page}: {e}")
    url = url or SHILLER_DIRECT
    log(f"  Shiller data: {url}")

    def get():
        r = SESSION.get(url, timeout=60)
        r.raise_for_status()
        return r.content
    content = retry(get, what="Shiller ie_data.xls")
    engine = "openpyxl" if content[:2] == b"PK" else "xlrd"  # .xlsx is a zip; .xls is not
    raw = pd.read_excel(io.BytesIO(content), sheet_name="Data", header=None, engine=engine)
    out = parse_shiller(raw)
    out["sourceUrl"] = url
    return out


# --------------------------------------------------------------------------------------
# Ken French Data Library
# --------------------------------------------------------------------------------------

FRENCH = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FF5_ZIP = FRENCH + "F-F_Research_Data_5_Factors_2x3_CSV.zip"
MOM_ZIP = FRENCH + "F-F_Momentum_Factor_CSV.zip"


def parse_french_monthly(text: str) -> pd.DataFrame:
    """Parse the first (monthly) block of a French library CSV. Values in percent."""
    lines = text.splitlines()
    header_idx, cols = None, None
    for i, ln in enumerate(lines):
        cells = [c.strip() for c in ln.split(",")]
        if len(cells) >= 2 and cells[0] == "" and any(cells[1:]):
            header_idx, cols = i, [c for c in cells[1:] if c != ""]
            break
    if header_idx is None:
        raise ValueError("French CSV: header row not found")
    rows, idx = [], []
    for ln in lines[header_idx + 1:]:
        cells = [c.strip() for c in ln.split(",")]
        if not cells or not re.fullmatch(r"\d{6}", cells[0]):
            if rows:
                break  # end of the monthly block (annual block follows)
            continue
        vals = []
        for c in cells[1:1 + len(cols)]:
            try:
                v = float(c)
            except ValueError:
                v = float("nan")
            vals.append(np.nan if v <= -99.99 else v)
        rows.append(vals)
        idx.append(f"{cells[0][:4]}-{cells[0][4:]}")
    if not rows:
        raise ValueError("French CSV: no monthly rows")
    return pd.DataFrame(rows, index=idx, columns=cols)


def french_zip_text(url: str) -> str:
    def get():
        r = SESSION.get(url, timeout=60)
        r.raise_for_status()
        return r.content
    z = zipfile.ZipFile(io.BytesIO(retry(get, what=url.rsplit("/", 1)[-1])))
    name = next(n for n in z.namelist() if n.lower().endswith((".csv", ".txt")))
    return z.read(name).decode("latin-1")


def factor_summary(r: pd.Series, last_month: str) -> dict:
    """r: monthly returns in percent, index 'YYYY-MM'."""
    x = (r.dropna() / 100.0)
    yr = last_month[:4]
    ytd = x[[k.startswith(yr) for k in x.index]]
    t12, t120 = x.iloc[-12:], x.iloc[-120:]
    mean_a, vol_a = x.mean() * 12, x.std() * math.sqrt(12)
    return {
        "lastMonth": float(x.iloc[-1]),
        "ytd": float((1 + ytd).prod() - 1) if len(ytd) else None,
        "trailing12m": float((1 + t12).prod() - 1) if len(t12) == 12 else None,
        "ann10y": float((1 + t120).prod() ** (12 / len(t120)) - 1) if len(t120) == 120 else None,
        "annMean": float(mean_a),
        "annVol": float(vol_a),
        "sharpe": float(mean_a / vol_a) if vol_a > 0 else None,
        "start": x.index[0],
    }


def build_factors() -> dict:
    ff5 = parse_french_monthly(french_zip_text(FF5_ZIP))
    mom = parse_french_monthly(french_zip_text(MOM_ZIP))
    mom.columns = ["Mom" if c.lower().startswith("mom") else c for c in mom.columns]
    df = ff5.join(mom[["Mom"]], how="left")
    need = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"French 5-factor file missing columns {missing}")
    df = df[need + ["Mom"]]
    last_month = df.dropna(subset=["Mkt-RF"]).index[-1]
    summary = {c: factor_summary(df[c], last_month) for c in df.columns if c != "RF"}
    summary["RF"] = factor_summary(df["RF"], last_month)
    return {
        "lastMonth": last_month,
        "months": list(df.index),
        "returns": {c: [rnd(v, 3) for v in df[c].values] for c in df.columns},
        "summary": summary,
    }


# --------------------------------------------------------------------------------------

def build_risk_model() -> dict:
    """Ryan Chen's drawdown-risk model (scripts/risk_model.py)."""
    import risk_model
    return risk_model.build_model()


SECTIONS = {
    "markets": build_markets,
    "firms": build_firms,
    "macro": build_macro,
    "valuation": build_valuation,
    "factors": build_factors,
    "model": build_risk_model,
}


def main(argv: list[str]) -> int:
    wanted = [a for a in argv if a in SECTIONS] or list(SECTIONS)
    status_path = DATA / "status.json"
    try:
        status = json.loads(status_path.read_text())
    except Exception:  # noqa: BLE001
        status = {"sections": {}}
    status.setdefault("sections", {})

    successes = 0
    for name in wanted:
        log(f"== {name}")
        t0 = time.time()
        entry = status["sections"].get(name, {})
        try:
            data = SECTIONS[name]()
            data["updated"] = now_iso()
            write_json(DATA / f"{name}.json", data)
            entry.update(ok=True, lastSuccess=data["updated"], error=None)
            successes += 1
            log(f"   ok ({time.time() - t0:.1f}s)")
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            entry.update(ok=False, lastAttempt=now_iso(), error=f"{type(e).__name__}: {str(e)[:300]}")
            log(f"   FAILED, keeping previous data/{name}.json")
        status["sections"][name] = entry

    status["updated"] = now_iso()
    write_json(status_path, status)
    if successes == 0:
        log("every section failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main(sys.argv[1:]))
