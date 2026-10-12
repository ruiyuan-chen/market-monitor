#!/usr/bin/env python3
"""
Drop-size experiment: do the risk models do better at predicting a smaller drop (for example 2%) than
the site's 3%?

Run it from the Actions tab ("Drop-size experiment" -> Run workflow). It downloads the same daily data
the site uses and runs the walk-forward test of scripts/risk_model.py for the site's three models
(turbulence model, VIX only, volatility only) with labels built from each drop size. Every model and
drop size is fit on the same days with the same monthly refits. It writes two tables per test period
to the run's summary page and does not change the website.

  1. Prediction quality, out of sample: AUC, Brier skill and the hit rate on the model's most alarming
     20% of days, next to the usual rate of each event.
  2. Trading: what $10,000 in SPY would have become when moving to cash
       a. at the close when the model's probability is at least 2x the usual rate (the site's "High"
          rule), and
       b. when today's reading is in the top 10% of the model's own earlier out-of-sample readings.
          This rule does not depend on the usual rate, so it still trades when 2x the usual rate is out
          of reach (a 2% drop within 21 days happens about half the time, and 2x that is over 100%).
     Same costs as the site's backtest (5 bp one way), before tax, with a 90% block-bootstrap range
     for the gap to buy and hold.

Each run is "history_start:test_start". The default compares the live setup (daily data since 2016,
tested from 2020) with a longer history (since 2003, tested from 2008, so the test includes 2008-09).

Local run on the notebook's CSV (VIX skipped; the S&P 500 index stands in for SPY):
  RISK_MODEL_PRICES_CSV=stock_details_5_years.csv RISK_MODEL_SPX_CSV=spx.csv THRESH_SPY_CSV=spx.csv \
  EXPERIMENT_RUNS=2018-01-01:2020-06-01 python scripts/threshold_experiment.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import backtest as bt  # noqa: E402
import model_experiment as me  # noqa: E402
import risk_model as rm  # noqa: E402

TURB = ["PC1", "PC2", "PC1_Change", "PC2_Change", "TurbScalar", "InternalTurbulence"]
MODELS = {
    "Turbulence model": TURB + ["RV21"],
    "VIX only": ["VIX"],
    "Volatility only": ["RV21"],
}
BENCH = "Volatility only"
DROPS = [float(x) for x in os.environ.get("EXPERIMENT_DROPS", "3,2").split(",") if x.strip()]
RUNS = os.environ.get("EXPERIMENT_RUNS", "2016-01-01:2020-01-01,2003-01-01:2008-01-01")
HORIZONS = (5, 21)
TOP_SHARE = 0.10          # rule b: cash on the 10% most alarming days
MIN_HISTORY = 126         # rule b needs this many earlier readings; invested before that


def log(msg: str) -> None:
    print(f"[drop-size {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def key(h: int, d: float) -> str:
    return f"{h}d, {d:g}% drop"


def money(x: float) -> str:
    return f"${x:,.0f}"


def market(start: str) -> pd.DataFrame:
    """SPY with dividends reinvested and the daily T-bill rate (cash)."""
    path = os.environ.get("THRESH_SPY_CSV")
    if path:
        s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        spy, irx = pd.to_numeric(s, errors="coerce").dropna(), None
    else:
        spy = bt.yahoo_close("SPY", start, True)
        try:
            irx = bt.yahoo_close("^IRX", start, False)
        except Exception as e:  # noqa: BLE001
            log(f"^IRX failed ({e}); cash earns 0")
            irx = None
    px = pd.DataFrame({"spy": spy}).dropna()
    px["rf"] = (irx.reindex(px.index).ffill().fillna(0) / 100 / 252) if irx is not None else 0.0
    px["ret"] = px["spy"].pct_change()
    return px


def quality_table(preds: dict, labels: dict, models: dict, start: str, rng) -> list[str]:
    from sklearn.metrics import roc_auc_score
    names = list(models)
    lines = ["| Horizon and drop | Usual rate | " + " | ".join(f"{m} AUC" for m in names)
             + " | Turbulence minus volatility-only AUC (90% range) | "
             + " | ".join(f"{m} Brier skill" for m in names)
             + " | Turbulence: drop rate on its top 20% days |",
             "|---|---|" + "---|" * (2 * len(names) + 2)]
    for h in HORIZONS:
        for d in DROPS:
            k = key(h, d)
            y = labels[k]
            common = y.notna() & (y.index >= pd.Timestamp(start))
            for m in names:
                common &= preds[(m, k)].notna()
            if common.sum() < 100:
                lines.append(f"| {k} | not enough days |" + " |" * (2 * len(names) + 2))
                continue
            yy = y[common].astype(int).values
            sc = {m: rm.scores(preds[(m, k)][common], y[common]) for m in names}
            if "Turbulence model" in sc and BENCH in sc:
                lo, hi = me.auc_diff_ci(preds[("Turbulence model", k)][common].values,
                                        preds[(BENCH, k)][common].values, yy, rng)
                diff = sc["Turbulence model"]["auc"] - roc_auc_score(yy, preds[(BENCH, k)][common].values)
                dtxt = f"{diff:+.3f} ({lo:+.3f} to {hi:+.3f})"
            else:
                dtxt = "–"
            top = sc.get("Turbulence model", {}).get("hitRateTop20")
            ttxt = f"{top:.1%}" if top is not None else "–"
            lines.append(f"| {k} | {yy.mean():.1%} | " + " | ".join(f"{sc[m]['auc']:.3f}" for m in names)
                         + f" | {dtxt} | " + " | ".join(f"{sc[m]['brierSkill']:+.3f}" for m in names)
                         + f" | {ttxt} |")
    lines.append("")
    return lines


def top_share_rule(p: pd.Series) -> pd.Series:
    """Cash (0) when today's reading is in the top TOP_SHARE of the model's readings so far, else 1."""
    q = p.expanding(min_periods=MIN_HISTORY).quantile(1 - TOP_SHARE)
    w = pd.Series(1.0, index=p.index)
    w[(p >= q) & q.notna()] = 0.0
    return bt.keep(w, p.notna())


def trading_table(preds: dict, labels: dict, models: dict, px: pd.DataFrame, start: str, rng) -> list[str]:
    starts = [preds[(m, key(h, d))].first_valid_index() for m in models for h in HORIZONS for d in DROPS]
    starts = [s for s in starts if s is not None]
    if not starts:
        return ["*No out-of-sample predictions.*", ""]
    first = max(min(starts), pd.Timestamp(start))
    df = px[px.index >= first].copy()
    if len(df) < 252:
        return ["*Not enough market data for the trading test.*", ""]
    P, rf = df["spy"], df["rf"]
    bh = bt.simulate(pd.Series(1.0, index=df.index), P, rf, bt.COST, 0.0, 0.0)
    mb = bt.metrics(bh["V"], rf)
    lines = [f"Trading days {df.index[0].date()} to {df.index[-1].date()} ({mb['years']:.1f} years). "
             f"Buy and hold SPY: $10,000 became **{money(mb['end'])}** ({mb['cagr']:.1%} a year, worst drawdown "
             f"{mb['maxdd']:.1%}).", "",
             "| Horizon and drop | Model | Rule | $10,000 became | vs. buy & hold / yr (90% range) | Worst drawdown | Days in cash | Round trips |",
             "|---|---|---|---|---|---|---|---|"]
    for h in HORIZONS:
        for d in DROPS:
            k = key(h, d)
            y = labels[k]
            base = float(y.dropna().mean())
            for m in models:
                p = preds[(m, k)].reindex(df.index).ffill(limit=5)
                if p.notna().sum() == 0:
                    continue
                for rule, w in ((f"cash at 2× usual rate (≥{2 * base:.0%})", bt.keep((p < 2 * base).astype(float), p.notna())),
                                (f"cash in its top {TOP_SHARE:.0%} of readings", top_share_rule(p))):
                    sim = bt.simulate(w, P, rf, bt.COST, 0.0, 0.0)
                    mm = bt.metrics(sim["V"], rf)
                    est, lo, hi = bt.boot_diff(sim["V"], bh["V"], rng)
                    cash = float((w.shift(1).reindex(df.index) == 0).mean())
                    trips = int(((w.shift(1) < 0.5) & (w >= 0.5)).sum())
                    gap = f"{est:+.1%} ({lo:+.1%} to {hi:+.1%})" if cash > 0 else "never in cash"
                    lines.append(f"| {k} | {m} | {rule} | {money(mm['end'])} | {gap} | "
                                 f"{mm['maxdd']:.1%} | {cash:.0%} | {trips} |")
    lines.append("")
    return lines


def main() -> None:
    runs = [tuple(r.split(":")) for r in RUNS.split(",") if r.strip()]
    first = min(r[0] for r in runs)
    csv = os.environ.get("RISK_MODEL_PRICES_CSV")
    if csv:
        close_all, vol_all = rm.prices_from_csv(csv)
    else:
        log(f"downloading daily prices since {first}")
        close_all, vol_all = rm.prices_from_yahoo(rm.read_universe(), first)
    spx_all = rm.spx_series(first)
    vix = me.vix_series(first)
    models = {k: v for k, v in MODELS.items() if vix is not None or "VIX" not in v}
    px_all = market(first)
    rng = np.random.default_rng(11)

    md = ["## Drop-size experiment", "",
          f"Drop sizes compared: {', '.join(f'{d:g}%' for d in DROPS)}. The event is the S&P 500 closing at least "
          "that much below the day's close within the next 5 or 21 trading days. Each model is a logistic "
          "regression refit monthly in the same walk-forward test as the site (clusters re-formed from the "
          f"trailing 2 years); every model and drop size uses the same days. Universe: {close_all.shape[1]} "
          "stocks from `model/universe.txt` (today's large companies, so earlier years are survivors only).", "",
          "**How to read it.** A smaller drop happens more often, so the usual rate rises. AUC (0.5 = coin "
          "flip, 1 = perfect ranking) and Brier skill (above 0 = better than always guessing the usual rate) "
          "can be compared across drop sizes. The trading rules hold SPY and move to cash at the close; "
          f"costs {bt.COST * 1e4:.0f} bp one way, before tax. A range that includes 0 means the gap is within noise.", ""]
    for hist_start, eval_start in runs:
        log(f"run: history from {hist_start}, test from {eval_start}")
        horizons = {key(h, d): (h, -d / 100) for h in HORIZONS for d in DROPS}
        cfg = dict(rm.CONFIG, history_start=hist_start, eval_start=eval_start, features=TURB, horizons=horizons)
        close = close_all.loc[hist_start:]
        close = close.loc[:, close.notna().sum() > 0]
        vol = vol_all.reindex(columns=close.columns).loc[close.index]
        spx = spx_all[(spx_all.index >= close.index[0]) & (spx_all.index <= close.index[-1])]
        prep = rm.prepare(close, vol, spx, cfg)
        F = rm.features_with(prep, rm.clusters_asof(prep, close.index[-1], cfg), cfg)
        index = F.index
        log(f"  {close.shape[1]} stocks; features {index[0].date()} to {index[-1].date()}")
        labels = {n: rm.make_labels(spx, h, thr).reindex(index) for n, (h, thr) in horizons.items()}
        preds = me.walk_forward_many(prep, vix, labels, models, cfg, index)
        md += [f"## Daily data since {hist_start}, tested from {eval_start}", "",
               "### Prediction quality (out of sample)", ""]
        md += quality_table(preds, labels, models, eval_start, rng)
        md += ["### Trading on the predictions", ""]
        md += trading_table(preds, labels, models, px_all, eval_start, rng)
    text = "\n".join(md)
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
