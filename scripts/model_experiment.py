#!/usr/bin/env python3
"""
Model experiment: compare feature sets and history lengths out of sample.

Run it from the Actions tab ("Model experiment" -> Run workflow). It downloads the same daily data the
site uses, runs the walk-forward test of scripts/risk_model.py for several variants at once (same
clusters, same days, same refit schedule) and writes a results table to the run's summary page. It
does not change the website.

Each run is "history_start:eval_start". The default compares
  - the live setup (daily data since 2016, tested from 2020), and
  - a longer history (daily data since 2003, tested from 2008, so the test includes 2008-09).
The longer run is also scored on 2020 onward, which isolates the effect of the extra training years.

Local run on the notebook's CSV (VIX is skipped unless RISK_MODEL_VIX_CSV is set):
  RISK_MODEL_PRICES_CSV=stock_details_5_years.csv RISK_MODEL_SPX_CSV=spx.csv \
  EXPERIMENT_RUNS=2018-01-01:2020-06-01 python scripts/model_experiment.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import risk_model as rm  # noqa: E402

TURB = ["PC1", "PC2", "PC1_Change", "PC2_Change", "TurbScalar", "InternalTurbulence"]
VARIANTS = {
    "Notebook features (6)": TURB,
    "Notebook + RV21 (live model)": TURB + ["RV21"],
    "Notebook + RV21 + VIX": TURB + ["RV21", "VIX"],
    "VIX only": ["VIX"],
    "RV21 only (baseline)": ["RV21"],
}
BASELINE = "RV21 only (baseline)"
RUNS = os.environ.get("EXPERIMENT_RUNS", "2016-01-01:2020-01-01,2003-01-01:2008-01-01")
BOOT_REPS, BOOT_BLOCK = 500, 63


def log(msg: str) -> None:
    print(f"[experiment {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def vix_series(start: str) -> pd.Series | None:
    path = os.environ.get("RISK_MODEL_VIX_CSV")
    if path:
        s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        return pd.to_numeric(s, errors="coerce").dropna().rename("VIX")
    if os.environ.get("RISK_MODEL_PRICES_CSV"):
        return None
    try:
        import yfinance as yf
        df = yf.download("^VIX", start=start, auto_adjust=False, progress=False, multi_level_index=False)
        s = df["Close"].dropna()
        idx = pd.to_datetime(s.index)
        s.index = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
        if len(s) > 500:
            return s.rename("VIX")
    except Exception as e:  # noqa: BLE001
        log(f"Yahoo ^VIX failed ({e}); trying FRED VIXCLS")
    try:
        from update_data import fred
        return fred("VIXCLS", start).rename("VIX")
    except Exception as e:  # noqa: BLE001
        log(f"FRED VIXCLS failed ({e}); VIX variants skipped")
        return None


def walk_forward_many(prep: dict, vix, labels: dict, variants: dict, cfg: dict, index) -> dict:
    """risk_model.walk_forward for several feature sets at once: the clusters and features are built
    once per monthly block and every variant is fit on exactly the same rows."""
    preds = {(v, n): pd.Series(np.nan, index=index) for v in variants for n in labels}
    i = index.searchsorted(pd.Timestamp(cfg["eval_start"]))
    blocks, t0 = 0, time.time()
    while i < len(index):
        d = index[i]
        F = rm.features_with(prep, rm.clusters_asof(prep, d - pd.Timedelta(days=1), cfg), cfg).reindex(index)
        if vix is not None:
            F["VIX"] = vix.reindex(index)
        j = min(len(index), i + cfg["refit_every"])
        for name, y in labels.items():
            h = cfg["horizons"][name][0]
            for v, cols in variants.items():
                Z = F[cols]
                m = rm.fit_rows(Z, y, max(0, i - h))
                block = Z.iloc[i:j]
                ok = block.notna().all(axis=1).values
                if m is not None and ok.any():
                    preds[(v, name)].iloc[np.arange(i, j)[ok]] = m.predict_proba(block[ok])[:, 1]
        i = j; blocks += 1
        if blocks % 24 == 0:
            log(f"  {blocks} blocks done ({d.date()}), {time.time() - t0:.0f}s")
    log(f"  walk-forward: {blocks} monthly blocks in {time.time() - t0:.0f}s")
    return preds


def auc_diff_ci(p1: np.ndarray, p0: np.ndarray, y: np.ndarray, rng) -> tuple[float, float]:
    """90% interval for AUC(p1) - AUC(p0) from a paired moving-block bootstrap. Blocks of about three
    months keep the overlap between neighbouring days' outcomes inside each resample."""
    from sklearn.metrics import roc_auc_score
    n = len(y); nb = max(1, n // BOOT_BLOCK)
    out = []
    for _ in range(BOOT_REPS):
        starts = rng.integers(0, max(1, n - BOOT_BLOCK + 1), nb)
        idx = (starts[:, None] + np.arange(BOOT_BLOCK)[None, :]).ravel()
        idx = idx[idx < n]
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        out.append(roc_auc_score(yy, p1[idx]) - roc_auc_score(yy, p0[idx]))
    if not out:
        return float("nan"), float("nan")
    return float(np.percentile(out, 5)), float(np.percentile(out, 95))


def score_window(preds: dict, labels: dict, variants: dict, start: str, title: str) -> list[str]:
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(7)
    lines = [f"### {title}", ""]
    for name, y in labels.items():
        common = y.notna() & (y.index >= pd.Timestamp(start))
        for v in variants:
            common &= preds[(v, name)].notna()
        if common.sum() < 100:
            lines += [f"*{name}: not enough days*", ""]
            continue
        yy = y[common].astype(int).values
        d0, d1 = y.index[common][0].date(), y.index[common][-1].date()
        lines += [f"**{name} horizon** ({int(common.sum())} days, {d0} to {d1}; drawdown followed on "
                  f"{yy.sum()} days, {yy.mean():.1%})", "",
                  "| Variant | AUC | AUC minus baseline (90% CI) | PR-AUC | Brier skill | Hit rate, top 20% |",
                  "|---|---|---|---|---|---|"]
        p_base = preds[(BASELINE, name)][common].values
        for v in variants:
            p = preds[(v, name)][common]
            s = rm.scores(p, y[common])
            if v == BASELINE:
                diff = "–"
            else:
                lo, hi = auc_diff_ci(p.values, p_base, yy, rng)
                diff = f"{s['auc'] - roc_auc_score(yy, p_base):+.3f} ({lo:+.3f} to {hi:+.3f})"
            lines.append(f"| {v} | {s['auc']:.3f} | {diff} | {s['prAuc']:.3f} | {s['brierSkill']:+.3f} | {s['hitRateTop20']:.1%} |")
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
    vix = vix_series(first)
    variants = {k: v for k, v in VARIANTS.items() if vix is not None or "VIX" not in v}

    md = ["## Model experiment", "",
          f"Universe: {close_all.shape[1]} stocks from `model/universe.txt` (today's large companies, so earlier "
          "years are survivors only). Labels: S&P 500 closes at least 3% below the day's close within 5 or 21 "
          "trading days. Every variant is a logistic regression refit monthly in the same walk-forward test "
          "(clusters re-formed from the trailing 2 years). The baseline uses only the past month's realized "
          "volatility (RV21). The interval is a 90% block-bootstrap range for the AUC difference; if it "
          "includes 0, the difference is within noise.", ""]
    for hist_start, eval_start in runs:
        log(f"run: history from {hist_start}, test from {eval_start}")
        cfg = dict(rm.CONFIG, history_start=hist_start, eval_start=eval_start, features=TURB)
        close = close_all.loc[hist_start:]; close = close.loc[:, close.notna().sum() > 0]
        vol = vol_all.reindex(columns=close.columns).loc[close.index]
        spx = spx_all[(spx_all.index >= close.index[0]) & (spx_all.index <= close.index[-1])]
        prep = rm.prepare(close, vol, spx, cfg)
        F = rm.features_with(prep, rm.clusters_asof(prep, close.index[-1], cfg), cfg)
        index = F.index
        log(f"  {close.shape[1]} stocks; features {index[0].date()} to {index[-1].date()}")
        labels = {n: rm.make_labels(spx, h, thr).reindex(index) for n, (h, thr) in cfg["horizons"].items()}
        preds = walk_forward_many(prep, vix, labels, variants, cfg, index)
        md += [f"## Daily data since {hist_start}, tested from {eval_start}", "",
               f"Features start {index[0].date()}; {close.shape[1]} stocks have prices in this period.", ""]
        md += score_window(preds, labels, variants, eval_start, f"Test period from {eval_start}")
        if eval_start < "2020-01-01" <= str(index[-1].date()):
            md += score_window(preds, labels, variants, "2020-01-01",
                               "Same days as the live setup's test (2020 on), trained on the longer history")
    text = "\n".join(md)
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
