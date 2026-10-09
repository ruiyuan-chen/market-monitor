"""Offline tests for the data parsers (no network). Run: python -m pytest -q"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import update_data as u  # noqa: E402


def test_parse_fred_csv_handles_gaps_and_new_header():
    text = "observation_date,DGS10\n2026-10-01,4.12\n2026-10-02,\n2026-10-05,.\n2026-10-06,4.15\n"
    s = u.parse_fred_csv(text, "DGS10")
    assert list(s.values) == [4.12, 4.15]
    assert s.index[-1] == pd.Timestamp("2026-10-06")


def test_parse_fred_csv_old_header():
    s = u.parse_fred_csv("DATE,UNRATE\n2026-08-01,4.3\n2026-09-01,4.4\n", "UNRATE")
    assert s.iloc[-1] == 4.4


def test_parse_fred_csv_rejects_html():
    with pytest.raises(ValueError):
        u.parse_fred_csv("<html><body>Error</body></html>\n", "X")


FF5_SAMPLE = """This file was created using the 202608 CRSP database. The 1-month TBill rate data until 202405 are from Ibbotson Associates.

,Mkt-RF,SMB,HML,RMW,CMA,RF
196307,   -0.39,   -0.48,   -0.81,    0.64,   -1.15,    0.27
196308,    5.07,   -0.80,    1.70,    0.40,   -0.38,    0.25
202607,    1.00,    0.50,   -0.20,    0.10,    0.30,    0.35
202608,    2.00,   -0.50,    0.40,   -0.10,   -0.30,    0.35

 Annual Factors: January-December
,Mkt-RF,SMB,HML,RMW,CMA,RF
  1964,   12.00,    0.10,    9.00,    1.00,    2.00,    3.50
"""

MOM_SAMPLE = """This file was created by CMPT_ME_PRIOR_RETURNS using the 202608 CRSP database.
Missing data are indicated by -99.99.

,Mom
192701,    0.57
196307,    0.90
196308,    1.00
202607,  -99.99
202608,    3.00

Annual Factors:

,Mom
1927,   20.00
"""


def test_parse_french_monthly_block_only():
    df = u.parse_french_monthly(FF5_SAMPLE)
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    assert list(df.index) == ["1963-07", "1963-08", "2026-07", "2026-08"]  # annual block excluded
    assert df.loc["2026-08", "Mkt-RF"] == 2.0


def test_parse_french_momentum_missing_values():
    df = u.parse_french_monthly(MOM_SAMPLE)
    assert list(df.columns) == ["Mom"]
    assert np.isnan(df.loc["2026-07", "Mom"])
    assert len(df) == 5


def test_factor_summary_ytd_compounds_within_year():
    r = pd.Series([1.0, 2.0], index=["2026-07", "2026-08"])
    s = u.factor_summary(r, "2026-08")
    assert s["lastMonth"] == pytest.approx(0.02)
    assert s["ytd"] == pytest.approx(1.01 * 1.02 - 1)
    assert s["trailing12m"] is None  # fewer than 12 months


def test_shiller_month_parsing_october_quirk():
    assert u.shiller_frac_to_month(1871.01) == "1871-01"
    assert u.shiller_frac_to_month(1871.1) == "1871-10"   # October is written 1871.1
    assert u.shiller_frac_to_month(1871.12) == "1871-12"
    assert u.shiller_frac_to_month("Date") is None


def test_parse_shiller_sheet_layout():
    header = ["Date", "P", "D", "E", "CPI", "Fraction", "Rate GS10", "Price", "Dividend",
              "Price", "Earnings", "Earnings", "CAPE", "", "TR CAPE"]
    rows = [["Stock market data used in 'Irrational Exuberance'"] + [""] * 14,
            [""] * 15, header]
    months = pd.period_range("1871-01", periods=1800, freq="M")
    for i, p in enumerate(months):
        code = p.year + (p.month / 100 if p.month != 10 else 0.1)
        cape = 15 + 10 * np.sin(i / 100) if i >= 120 else np.nan
        rows.append([code, 100 + i, 1, 20.0, 10, 0, 4, 0, 0, 0, 0, 0, cape, "", cape])
    raw = pd.DataFrame(rows)
    out = u.parse_shiller(raw)
    assert out["capeStart"] == "1881-01"
    assert out["capeDate"] == months[-1].strftime("%Y-%m")
    assert 0 <= out["capePercentile"] <= 1
    assert out["pe"][-1][1] == pytest.approx((100 + 1799) / 20.0, rel=1e-3)


def test_perf_returns():
    idx = pd.bdate_range("2024-12-02", "2026-10-07")
    s = pd.Series(np.linspace(100, 200, len(idx)), index=idx)
    p = u.perf(s)
    last = s.iloc[-1]
    base_ytd = s[s.index < "2026-01-01"].iloc[-1]
    assert p["ytd"] == pytest.approx(last / base_ytd - 1)
    assert p["chg1d"] == pytest.approx(last / s.iloc[-2] - 1)
    assert p["chg1y"] == pytest.approx(last / s.loc[:"2025-10-07"].iloc[-1] - 1)
    assert p["high52"] == pytest.approx(last)


def test_perf_none_when_history_too_short_for_1y():
    idx = pd.bdate_range("2026-06-01", "2026-10-07")
    p = u.perf(pd.Series(np.arange(1, len(idx) + 1, dtype=float), index=idx))
    assert p["chg1y"] is None
    assert p["chg1m"] is not None


def test_recession_periods():
    idx = pd.date_range("2019-12-01", periods=8, freq="MS")
    usrec = pd.Series([0, 0, 1, 1, 0, 0, 0, 1], index=idx)
    assert u.recession_periods(usrec) == [["2020-02", "2020-03"], ["2020-07", "2020-07"]]


def test_clean_name():
    assert u.clean_name("NVIDIA Corporation") == "NVIDIA"
    assert u.clean_name("Berkshire Hathaway Inc. New") == "Berkshire Hathaway"
    assert u.clean_name("Meta Platforms, Inc.") == "Meta Platforms"
    assert u.clean_name("JPMorgan Chase & Co.") == "JPMorgan Chase & Co."
    assert u.clean_name("Eli Lilly and Company") == "Eli Lilly and Company"


def test_parse_shiller_rebuilds_cape_without_cape_column():
    header = ["Date", "P", "D", "E", "CPI"]
    rows = [header]
    months = pd.period_range("1871-01", periods=900, freq="M")
    for i, p in enumerate(months):
        code = p.year + (p.month / 100 if p.month != 10 else 0.1)
        rows.append([code, 100.0, 1, 5.0, 10.0])
    out = u.parse_shiller(pd.DataFrame(rows))
    assert out["capeLatest"] == pytest.approx(20.0)


def test_fred_cache_reuses_longer_download(monkeypatch):
    calls = []
    def fake(sid, start):
        calls.append(start)
        idx = pd.date_range("1990-01-01", "2026-10-01", freq="MS")
        return pd.Series(1.0, index=idx)
    monkeypatch.setattr(u, "_fred_download", fake)
    u._FRED_CACHE.clear()
    u.fred("DGS10", "1990-01-01")
    s = u.fred("DGS10", "2025-06-01")
    assert len(calls) == 1 and s.index[0] >= pd.Timestamp("2025-06-01")
