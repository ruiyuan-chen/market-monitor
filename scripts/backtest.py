#!/usr/bin/env python3
"""
Strategy backtest: what trading on the drawdown model's predictions would have done.

Run it from the Actions tab ("Strategy backtest" -> Run workflow). It reads the model's out-of-sample
predictions (data/model_oos.json, written by the daily update), downloads SPY, the VIX and the 3-month
T-bill rate, simulates a set of simple rules and writes the results to the run's summary page. It does
not change the website.

Every rule chooses a position at the close, using only what was known then, and earns the next day's
return. Long only, no leverage. Cash earns the T-bill rate.

  - Buy and hold SPY
  - Model 5d: cash when the probability is at least 2x the usual rate ("High" on the site)
  - Model 5d: cash when at least 1.25x the usual rate ("Elevated" or "High")
  - Model 5d: out at 2x, back in once it falls below 1x (fewer trades)
  - Model 5d: scaled exposure, 100% at the usual rate falling to 0% at 2x
  - Model 21d: cash when at least 1.25x the usual rate
  - Volatility-only model (the site's baseline): cash at 2x
  - VIX above 25 -> cash; VIX above 30 -> cash
  - VIX volatility targeting: exposure = min(1, target / VIX)
  - Realized-volatility targeting: exposure = min(1, target / RV21)

Costs: COST_BPS per dollar traded, one way (default 5 = 0.05%). Taxes: TAX_RATE on realized gains,
settled each year with losses carried forward, and the final liquidation taxed too (short-term for the
active rules, LONG_TERM_TAX_RATE for buy and hold). The pre-tax column is what a tax-free account would see.

Local run with CSVs (date + value): BACKTEST_SPY_CSV, BACKTEST_VIX_CSV, BACKTEST_IRX_CSV (percent).
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "backtest_out"

COST = float(os.environ.get("COST_BPS", "5")) / 1e4
TAX = float(os.environ.get("TAX_RATE", "30")) / 100
LT_TAX = float(os.environ.get("LONG_TERM_TAX_RATE", "15")) / 100
VOL_TARGET = float(os.environ.get("VOL_TARGET", "15")) / 100
START = os.environ.get("BACKTEST_START", "2020-01-01")
SUB_START = os.environ.get("BACKTEST_SUBPERIOD", "2021-01-01")
START_VALUE = 10_000.0
BOOT_REPS, BOOT_BLOCK = 1000, 63
SERIES = {"s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a", "s4": "#eda100", "s5": "#e87ba4"}


def log(msg: str) -> None:
    print(f"[backtest {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------

def load_predictions() -> tuple[pd.DataFrame, dict, str]:
    p = DATA / "model_oos.json"
    if p.exists():
        j = json.loads(p.read_text())
        src = f"`data/model_oos.json` (walk-forward predictions from {j['evalStart']}, as of {j['asof']})"
    else:
        m = json.loads((DATA / "model.json").read_text())
        h = m["history"]
        j = {"dates": h["dates"], "spx": h["spx"], "baseRates": {k: v["baseRate"] for k, v in m["horizons"].items()}}
        for k in m["horizons"]:
            j[f"p_{k}"] = h.get(f"p_{k}"); j[f"y_{k}"] = h.get(f"y_{k}")
        src = ("`data/model.json` (its history is trimmed to the last few years; run the daily update once "
               "to get the full record in `data/model_oos.json`)")
    cols = {k: v for k, v in j.items() if isinstance(v, list) and k != "dates"}
    df = pd.DataFrame(cols, index=pd.to_datetime(j["dates"])).astype(float)
    return df, j["baseRates"], src


def yahoo_close(ticker: str, start: str, adjust: bool) -> pd.Series:
    import yfinance as yf
    df = yf.download(ticker, start=start, auto_adjust=adjust, progress=False, multi_level_index=False)
    s = df["Close"].dropna()
    idx = pd.to_datetime(s.index)
    s.index = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
    return s.astype(float)


def market_data(start: str) -> tuple[pd.Series, pd.Series, pd.Series | None]:
    if os.environ.get("BACKTEST_SPY_CSV"):
        def rd(k):
            s = pd.read_csv(os.environ[k], index_col=0, parse_dates=True).iloc[:, 0]
            return pd.to_numeric(s, errors="coerce").dropna()
        return rd("BACKTEST_SPY_CSV"), rd("BACKTEST_VIX_CSV"), rd("BACKTEST_IRX_CSV")
    spy = yahoo_close("SPY", start, True)          # dividends reinvested
    vix = yahoo_close("^VIX", start, False)
    try:
        irx = yahoo_close("^IRX", start, False)    # 13-week T-bill, percent
    except Exception as e:  # noqa: BLE001
        log(f"^IRX failed ({e}); cash earns 0")
        irx = None
    return spy, vix, irx


def build_panel() -> tuple[pd.DataFrame, dict, str]:
    pred, base, src = load_predictions()
    spy, vix, irx = market_data((pd.Timestamp(START) - pd.Timedelta(days=120)).strftime("%Y-%m-%d"))
    px = pd.DataFrame({"spy": spy, "vix": vix}).dropna()
    px["rf"] = (irx.reindex(px.index).ffill().fillna(0) / 100 / 252) if irx is not None else 0.0
    px["ret"] = px["spy"].pct_change()
    px["rv21"] = np.log(px["spy"]).diff().rolling(21).std() * math.sqrt(252)
    df = px.join(pred.drop(columns=["spx"], errors="ignore"), how="left")
    df = df[df.index >= pd.Timestamp(START)]
    first = df["p_5d"].first_valid_index()
    df = df.loc[first:]
    pcols = [c for c in df.columns if c[:2] in ("p_", "b_")]
    df[pcols] = df[pcols].ffill(limit=5)
    return df, base, src


# --------------------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------------------

def hysteresis(p: pd.Series, out_at: float, in_at: float) -> pd.Series:
    w, state = np.ones(len(p)), 1.0
    for i, v in enumerate(p.values):
        if not np.isnan(v):
            if state == 1.0 and v >= out_at:
                state = 0.0
            elif state == 0.0 and v <= in_at:
                state = 1.0
        w[i] = state
    return pd.Series(w, index=p.index)


def rules(df: pd.DataFrame, base: dict) -> dict[str, pd.Series]:
    b5, b21 = base["5d"], base["21d"]
    p5, p21 = df["p_5d"], df["p_21d"]
    keep = lambda w, ok: w.where(ok).ffill().fillna(1.0).clip(0, 1)   # no signal -> keep the last position
    W = {
        "Buy and hold SPY": pd.Series(1.0, index=df.index),
        "Model 5d: cash when High (≥2× usual rate)": keep((p5 < 2 * b5).astype(float), p5.notna()),
        "Model 5d: cash when Elevated or High (≥1.25×)": keep((p5 < 1.25 * b5).astype(float), p5.notna()),
        "Model 5d: out at 2×, back in below 1×": hysteresis(p5, 2 * b5, b5),
        "Model 5d: scaled (100% at usual rate → 0% at 2×)": keep((2 - p5 / b5).clip(0, 1), p5.notna()),
        "Model 21d: cash when ≥1.25× usual rate": keep((p21 < 1.25 * b21).astype(float), p21.notna()),
    }
    if "b_5d" in df:
        W["Volatility-only model 5d: cash at 2×"] = keep((df["b_5d"] < 2 * b5).astype(float), df["b_5d"].notna())
    W["VIX above 25 → cash"] = (df["vix"] <= 25).astype(float)
    W["VIX above 30 → cash"] = (df["vix"] <= 30).astype(float)
    W[f"VIX volatility target {VOL_TARGET:.0%}"] = (VOL_TARGET * 100 / df["vix"]).clip(upper=1)
    W[f"Realized-vol target {VOL_TARGET:.0%} (RV21)"] = keep((VOL_TARGET / df["rv21"]).clip(upper=1), df["rv21"].notna())
    return W


# --------------------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------------------

def simulate(w: pd.Series, px: pd.Series, rf: pd.Series, cost: float, tax: float, final_tax: float) -> dict:
    """Trade to weight w[t] at the close of day t. Average-cost lots; taxes settled each January on the
    previous year's realized gains (losses carried forward) and on the final liquidation."""
    P, wv, rfv, dates = px.values, w.values, rf.values, px.index
    n = len(P)
    V = np.empty(n)
    cash, shares, basis = START_VALUE, 0.0, 0.0
    realized, carry, tax_paid, cost_paid, turnover = 0.0, 0.0, 0.0, 0.0, 0.0
    year = dates[0].year
    for t in range(n):
        p = P[t]
        if t > 0:
            cash *= 1 + rfv[t]
            if dates[t].year != year:                      # settle last year's taxes
                taxable = realized - carry
                if taxable > 0:
                    due = tax * taxable; cash -= due; tax_paid += due; carry = 0.0
                else:
                    carry = -taxable
                realized = 0.0
                year = dates[t].year
                if cash < 0 and shares > 0:                 # sell to pay the tax bill
                    sell = min(shares, -cash / p); realized += (p - basis) * sell
                    shares -= sell; cash += sell * p
        value = cash + shares * p
        target = wv[t] * value / p
        d = target - shares
        if d < -1e-12:
            realized += (p - basis) * (-d); shares += d; cash -= d * p
        elif d > 1e-12:
            basis = (basis * shares + p * d) / (shares + d); shares += d; cash -= d * p
        c = abs(d) * p * cost
        cash -= c; cost_paid += c; turnover += abs(d) * p / value
        V[t] = cash + shares * p
    # liquidate at the end
    p = P[-1]
    realized += (p - basis) * shares
    taxable = realized - carry
    final_due = final_tax * taxable if taxable > 0 else 0.0
    tax_paid += final_due
    end_after_tax = V[-1] - final_due
    return {"V": pd.Series(V, index=dates), "endAfterTax": end_after_tax, "taxPaid": tax_paid,
            "costPaid": cost_paid, "turnover": turnover}


def metrics(V: pd.Series, rf: pd.Series) -> dict:
    r = V.pct_change().dropna()
    years = len(r) / 252
    ex = r - rf.reindex(r.index).fillna(0)
    dd = (V / V.cummax() - 1).min()
    return {"end": float(V.iloc[-1]), "cagr": float((V.iloc[-1] / V.iloc[0]) ** (1 / years) - 1),
            "vol": float(r.std() * math.sqrt(252)), "sharpe": float(ex.mean() / ex.std() * math.sqrt(252)) if ex.std() > 0 else float("nan"),
            "maxdd": float(dd), "years": years}


def boot_diff(Vs: pd.Series, Vb: pd.Series, rng) -> tuple[float, float, float]:
    """Annualized return difference vs buy and hold, with a 90% moving-block bootstrap range."""
    d = (np.log(Vs).diff() - np.log(Vb).diff()).dropna().values
    n = len(d); nb = max(1, n // BOOT_BLOCK)
    est = 252 * d.mean()
    sims = []
    for _ in range(BOOT_REPS):
        starts = rng.integers(0, max(1, n - BOOT_BLOCK + 1), nb)
        idx = (starts[:, None] + np.arange(BOOT_BLOCK)[None, :]).ravel()
        sims.append(252 * d[idx[idx < n]].mean())
    return est, float(np.percentile(sims, 5)), float(np.percentile(sims, 95))


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------

def money(x):
    return f"${x:,.0f}"


def run_period(df: pd.DataFrame, W: dict, label: str, rng) -> tuple[list[str], dict]:
    px, rf = df["spy"], df["rf"]
    res, lines = {}, []
    Vb = None
    for name, w in W.items():
        w = w.reindex(df.index).ffill().fillna(1.0)
        pre = simulate(w, px, rf, COST, 0.0, 0.0)
        final_tax = LT_TAX if name.startswith("Buy and hold") else TAX
        post = simulate(w, px, rf, COST, TAX, final_tax)
        m = metrics(pre["V"], rf)
        binary = bool(np.isin(w.values, [0.0, 1.0]).all())
        entries = int(((w.shift(1) < 0.5) & (w >= 0.5)).sum())
        res[name] = {"w": w, "V": pre["V"], "m": m, "afterTax": post["endAfterTax"], "taxPaid": post["taxPaid"],
                     "binary": binary, "entries": entries, "turnover": pre["turnover"] / m["years"],
                     "inMarket": float(w.mean())}
        if Vb is None:
            Vb = pre["V"]
    for name, r in res.items():
        est, lo, hi = boot_diff(r["V"], Vb, rng) if name != list(res)[0] else (0.0, 0.0, 0.0)
        r["diff"] = (est, lo, hi)
    d0, d1 = df.index[0].date(), df.index[-1].date()
    lines += [f"### {label}: {d0} to {d1} ({res[list(res)[0]]['m']['years']:.1f} years)", "",
              "| Rule | $10,000 became | Return / yr | vs. buy & hold / yr (90% range) | Volatility | Sharpe | Worst drawdown | Time in market | Trades | After tax |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in res.items():
        m = r["m"]
        diff = "–" if name.startswith("Buy and hold") else f"{r['diff'][0] * 100:+.1f}% ({r['diff'][1] * 100:+.1f} to {r['diff'][2] * 100:+.1f})"
        trades = f"{r['entries']} round trips" if r["binary"] else f"turnover {r['turnover']:.1f}×/yr"
        lines.append(f"| {name} | {money(m['end'])} | {m['cagr'] * 100:.1f}% | {diff} | {m['vol'] * 100:.1f}% | {m['sharpe']:.2f} | "
                     f"{m['maxdd'] * 100:.1f}% | {r['inMarket'] * 100:.0f}% | {trades} | {money(r['afterTax'])} |")
    lines.append("")
    return lines, res


def out_of_market_table(df: pd.DataFrame, res: dict) -> list[str]:
    r = df["ret"]
    worst = r.nsmallest(10).index; best = r.nlargest(10).index
    lines = ["#### When the rule was in cash, what did the market do the next day?", "",
             "| Rule | Days in cash | Market return on those days (annualized) | Market return on invested days (annualized) | Of the 10 worst days, in cash | Of the 10 best days, in cash |",
             "|---|---|---|---|---|---|"]
    for name, x in res.items():
        if not x["binary"] or name.startswith("Buy and hold"):
            continue
        pos = x["w"].shift(1).reindex(df.index)           # position held during day t
        out = pos == 0
        if out.sum() == 0:
            lines.append(f"| {name} | 0 | – | – | 0 | 0 |"); continue
        lines.append(f"| {name} | {int(out.sum())} ({out.mean() * 100:.0f}%) | {r[out].mean() * 252 * 100:+.1f}% | {r[~out].mean() * 252 * 100:+.1f}% | "
                     f"{int(out.reindex(worst).sum())} | {int(out.reindex(best).sum())} |")
    lines.append("")
    return lines


def monthly_table(res: dict, names: list[str]) -> tuple[list[str], dict]:
    M = {n: res[n]["V"].resample("ME").last() for n in names if n in res}
    tbl = pd.DataFrame(M)
    lines = ["<details><summary>Month-end value of $10,000 (for charting)</summary>", "",
             "| Month | " + " | ".join(tbl.columns) + " |", "|---|" + "---|" * len(tbl.columns)]
    for d, row in tbl.iterrows():
        lines.append(f"| {d:%Y-%m} | " + " | ".join(f"{v:,.0f}" for v in row.values) + " |")
    lines += ["", "</details>", ""]
    series = {n: [round(float(v)) for v in tbl[n].values] for n in tbl.columns}
    return lines, {"months": [f"{d:%Y-%m}" for d in tbl.index], "series": series}


SHORT = {"Buy and hold SPY": "Buy and hold", "Model 5d: cash when High (≥2× usual rate)": "Model: cash when High",
         "Model 5d: scaled (100% at usual rate → 0% at 2×)": "Model: scaled exposure", "VIX above 25 → cash": "VIX > 25: cash"}


def draw(res: dict, names: list[str], path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker
    except Exception as e:  # noqa: BLE001
        log(f"matplotlib unavailable ({e}); no chart"); return
    names = [n for n in names if n in res]
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
    fig.patch.set_facecolor("#ffffff"); ax.set_facecolor("#ffffff")
    colors = list(SERIES.values())
    lo = min(res[n]["V"].min() for n in names); hi = max(res[n]["V"].max() for n in names)
    ends = []
    for i, n in enumerate(names):
        V = res[n]["V"]
        short = SHORT.get(n, n.split("(")[0].strip())
        ax.plot(V.index, V.values, lw=2 if i == 0 else 1.6, color=colors[i % len(colors)], label=short, solid_capstyle="round")
        ends.append([float(V.iloc[-1]), f"{short}  ${V.iloc[-1]:,.0f}", colors[i % len(colors)], V.index[-1]])
    # end labels, pushed apart so they don't overlap
    gap = (hi - lo) * 0.045
    ends.sort(key=lambda e: e[0])
    for k in range(1, len(ends)):
        if ends[k][0] - ends[k - 1][0] < gap:
            ends[k][0] = ends[k - 1][0] + gap
    for y, text, color, x in ends:
        ax.annotate(text, (x, y), xytext=(8, 0), textcoords="offset points", fontsize=8.5, color="#455366", va="center",
                    annotation_clip=False)
    ax.set_ylim(lo - (hi - lo) * 0.04, max(hi, ends[-1][0]) + (hi - lo) * 0.06)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6))
    ax.grid(True, axis="y", color="#e8ecf1", lw=0.8); ax.grid(False, axis="x")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#c9d1dc")
    ax.tick_params(colors="#687586", labelsize=9)
    ax.set_title("Value of $10,000 after trading costs, before tax", loc="left", fontsize=12, color="#0d1726")
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=len(names), labelcolor="#455366")
    fig.subplots_adjust(right=0.78, bottom=0.18)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    t0 = time.time()
    df, base, src = build_panel()
    log(f"panel: {len(df)} days, {df.index[0].date()} to {df.index[-1].date()}")
    W = rules(df, base)
    rng = np.random.default_rng(11)
    md = ["## Strategy backtest", "",
          f"Predictions: {src}. Prices: SPY with dividends reinvested; cash earns the 3-month T-bill rate. "
          f"Each rule sets its position at the close and earns the next day's return. Long only, no leverage. "
          f"Trading cost {COST * 1e4:.0f} bp per dollar traded (one way). Taxes: {TAX:.0%} on realized gains, settled "
          f"yearly with losses carried forward and on the final sale; buy and hold pays {LT_TAX:.0%} on its gain at the "
          f"end. 'Usual rate' = the model's base rate (5d {base['5d']:.1%}, 21d {base['21d']:.1%}), as on the website. "
          f"The 90% range is a moving-block bootstrap of the daily return gap; if it includes 0, the gap is within noise.",
          "", "This is a simulation for study, not investment advice. Past results don't predict future ones.", ""]
    md += ["## Full period", ""]
    full, res_full = run_period(df, W, "All days with predictions", rng)
    md += full + out_of_market_table(df, res_full)
    sub = df[df.index >= pd.Timestamp(SUB_START)]
    if len(sub) > 252:
        md += ["## Excluding the first year", "",
               "The 2020 crash dominates any timing rule's record, so here is the same test from "
               f"{SUB_START} on.", ""]
        sub_lines, _ = run_period(sub, W, "Later period", rng)
        md += sub_lines
    chart_names = [list(W)[0], "Model 5d: cash when High (≥2× usual rate)", "Model 5d: scaled (100% at usual rate → 0% at 2×)",
                   "VIX above 25 → cash", f"Realized-vol target {VOL_TARGET:.0%} (RV21)"]
    mlines, monthly = monthly_table(res_full, chart_names)
    md += mlines
    OUT.mkdir(exist_ok=True)
    draw(res_full, chart_names, OUT / "equity_curves.png")
    pd.DataFrame({n: r["V"] for n, r in res_full.items()}).to_csv(OUT / "equity_curves.csv", float_format="%.2f")
    pd.DataFrame({n: r["w"] for n, r in res_full.items()}).to_csv(OUT / "positions.csv", float_format="%.3f")
    (OUT / "monthly.json").write_text(json.dumps(monthly))
    md += ["The chart `equity_curves.png`, the daily values and the daily positions are in this run's artifact "
           "(`backtest-results`, under Artifacts on the run page).", ""]
    text = "\n".join(md)
    print(text)
    print("BACKTEST_MONTHLY_JSON=" + json.dumps(monthly, separators=(",", ":")))
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    log(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
