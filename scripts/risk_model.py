#!/usr/bin/env python3
"""
Drawdown-risk model: Ryan Chen's market-turbulence model, run every trading day.

What it does
  1. Downloads daily prices for the stock universe in model/universe.txt and the S&P 500 index.
  2. Rebuilds the research notebook's features (verified to match it to machine precision):
       - clusters of co-moving stocks (PCA on the return-correlation matrix + k-means)
       - "financial turbulence": how decoupled each cluster is from the S&P 500 (rolling 21-day
         regression residual volatility x sqrt(1 - correlation)), summarized by an expanding PCA
       - "internal turbulence": the average z-score of six market-breadth measures
  3. Labels each day 1 if the S&P 500 closes at least 3% below that day's close at some point in the
     next 5 (or 21) trading days.
  4. Evaluates the model out of sample: walk forward one month at a time, re-forming the clusters and
     refitting the model using only data available on that date. Compares it with a volatility-only
     baseline.
  5. Re-forms the clusters as of today, fits the model on all labeled history and predicts today.

Output: data/model.json (read by the website) and data/model_log.json (append-only record of each
day's live prediction, scored once the outcome is known).

Differences from the notebook (README, "Risk model", explains each):
  - Clusters are formed from the trailing two years of returns as of each date, never from the full
    sample. Full-sample clusters use future information; this is the change that matters most.
  - Correlations use pairwise-complete data instead of treating missing returns as 0.
  - PCA signs are kept consistent between daily refits, and the PC "changes" are first differences
    (percent changes of a series that crosses zero blow up).
  - The last 5/21 days have no label yet (the notebook labeled them 0).
  - Logistic regression without resampling, so the probabilities are calibrated.

Local run on a CSV in the notebook's format (Date, Close, Volume, Company):
  RISK_MODEL_PRICES_CSV=stock_details_5_years.csv RISK_MODEL_SPX_CSV=spx.csv python scripts/risk_model.py
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# --------------------------------------------------------------------------------------
# Settings (edit here)
# --------------------------------------------------------------------------------------
CONFIG = {
    "history_start": "2016-01-01",   # first day of price history to download
    "eval_start": "2020-01-01",      # first day of the out-of-sample (walk-forward) test
    "cluster_years": 2,              # clusters are formed from this many trailing years of returns
    "n_clusters": 10,
    "cluster_pca_dims": 10,
    "turb_window": 21,               # days in the rolling regression
    "pca_components": 2,
    "pca_start_after": 30,
    "pc_change": "diff",             # "diff" (default) or "pct" (the notebook's percent change)
    "lookback_hl": 252, "ma_short": 50, "ma_long": 200, "z_win": 252, "min_names": 50,
    "horizons": {"5d": (5, -0.03), "21d": (21, -0.03)},
    "features": ["PC1", "PC2", "PC1_Change", "PC2_Change", "TurbScalar", "InternalTurbulence"],
    "refit_every": 21,               # walk-forward block length (about one month)
    "history_years_on_site": 6,      # out-of-sample history sent to the website
}

FEATURE_INFO = {
    "PC1": ("Turbulence factor 1", "First principal component of the cluster turbulence signals"),
    "PC2": ("Turbulence factor 2", "Second principal component of the cluster turbulence signals"),
    "PC1_Change": ("Change in factor 1", "Day-over-day change in turbulence factor 1"),
    "PC2_Change": ("Change in factor 2", "Day-over-day change in turbulence factor 2"),
    "TurbScalar": ("Financial turbulence", "Average decoupling of stock clusters from the S&P 500"),
    "InternalTurbulence": ("Internal turbulence", "Average z-score of six breadth measures: new lows, share below the 50- and 200-day averages, decliners, TRIN and dispersion"),
    "RV21": ("Realized volatility", "Annualized standard deviation of the last 21 daily S&P 500 returns"),
}


def log(msg: str) -> None:
    print(f"[risk-model {dt.datetime.now(dt.timezone.utc):%H:%M:%S}] {msg}", flush=True)


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------

def read_universe() -> list[str]:
    out = []
    for ln in (ROOT / "model" / "universe.txt").read_text().splitlines():
        ln = ln.split("#", 1)[0].strip()
        if ln:
            out.append(ln)
    return sorted(set(out))


def prices_from_csv(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """CSV in the notebook's long format: Date, Close, Volume, Company."""
    df = pd.read_csv(path, usecols=["Date", "Close", "Volume", "Company"])
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    close = df.pivot_table(index="Date", columns="Company", values="Close", aggfunc="last").sort_index()
    vol = df.pivot_table(index="Date", columns="Company", values="Volume", aggfunc="last").reindex(close.index)
    return close, vol


def prices_from_yahoo(tickers: list[str], start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    import yfinance as yf

    closes, vols = [], []
    for i in range(0, len(tickers), 100):
        chunk = tickers[i:i + 100]
        for attempt in range(3):
            try:
                df = yf.download(chunk, start=start, interval="1d", auto_adjust=True, actions=False,
                                 group_by="column", progress=False, threads=True, multi_level_index=True)
                if df is None or df.empty:
                    raise RuntimeError("empty download")
                closes.append(df["Close"]); vols.append(df["Volume"])
                break
            except Exception as e:  # noqa: BLE001
                log(f"  chunk {i // 100 + 1} attempt {attempt + 1} failed: {e}")
                time.sleep(10 * (attempt + 1))
        time.sleep(2)
    if not closes:
        raise RuntimeError("no price data downloaded")
    close = pd.concat(closes, axis=1); vol = pd.concat(vols, axis=1)
    idx = pd.to_datetime(close.index)
    idx = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
    close.index = idx; vol.index = idx
    close = close.loc[:, close.notna().sum() > 0]
    if close.shape[1] < 100:
        raise RuntimeError(f"only {close.shape[1]} tickers downloaded")
    log(f"  prices for {close.shape[1]} of {len(tickers)} tickers, {close.index[0].date()} to {close.index[-1].date()}")
    return close.sort_index(), vol.reindex(columns=close.columns).sort_index()


def spx_series(start: str) -> pd.Series:
    path = os.environ.get("RISK_MODEL_SPX_CSV")
    if path:
        s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        return pd.to_numeric(s, errors="coerce").dropna().rename("SPX")
    try:
        import yfinance as yf
        df = yf.download("^GSPC", start=start, auto_adjust=False, progress=False, multi_level_index=False)
        s = df["Close"].dropna()
        idx = pd.to_datetime(s.index)
        s.index = (idx.tz_localize(None) if idx.tz is not None else idx).normalize()
        if len(s) > 500:
            return s.rename("SPX")
    except Exception as e:  # noqa: BLE001
        log(f"  Yahoo ^GSPC failed ({e}); using FRED SP500")
    sys.path.insert(0, str(Path(__file__).parent))
    from update_data import fred
    return fred("SP500", start).rename("SPX")


# --------------------------------------------------------------------------------------
# Features (the notebook's method, vectorized)
# --------------------------------------------------------------------------------------

def form_clusters(rets: pd.DataFrame, start, end, cfg: dict) -> pd.Series:
    """PCA on the correlation matrix of daily returns in [start, end], then k-means."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    win = rets.loc[start:end]
    win = win.loc[:, win.notna().mean() >= 0.8]          # need most of the window
    corr = win.corr(min_periods=60).fillna(0.0)          # pairwise-complete correlations
    k = min(cfg["n_clusters"], max(2, corr.shape[0] // 5))
    dims = min(cfg["cluster_pca_dims"], corr.shape[0] - 1)
    X = PCA(n_components=dims, random_state=42).fit_transform(corr)
    labels = KMeans(n_clusters=k, random_state=42, n_init=50).fit_predict(X)
    return pd.Series(labels, index=corr.columns, name="Cluster")


def clusters_asof(prep: dict, date, cfg: dict) -> pd.Series:
    end = pd.Timestamp(date)
    return form_clusters(prep["rets"], end - pd.DateOffset(years=cfg["cluster_years"]), end, cfg)


def cluster_returns(rets: pd.DataFrame, close: pd.DataFrame, vol: pd.DataFrame, clusters: pd.Series) -> pd.DataFrame:
    """Each cluster's daily return, weighted by the previous day's dollar volume (equal weight if none)."""
    w_all = (close * vol).shift(1)
    out = {}
    for c in sorted(clusters.unique()):
        names = clusters.index[clusters == c]
        r = rets[names]; w = w_all[names].where(r.notna()).fillna(0.0)
        wsum = w.sum(axis=1)
        vw = (r.fillna(0.0) * w).sum(axis=1) / wsum.replace(0, np.nan)
        out[f"Cluster_{c}"] = vw.where(wsum > 0, r.mean(axis=1))
    return pd.DataFrame(out)


def turbulence_signals(panel: pd.DataFrame, bench: str, window: int) -> pd.DataFrame:
    """For each cluster: residual std of the S&P 500 regressed on the cluster (OLS over the previous
    `window` days) times sqrt(1 - correlation). Same numbers as the notebook's statsmodels loop."""
    y = panel[bench]
    sig = {}
    for col in panel.columns:
        if col == bench:
            continue
        pair = pd.concat([y, panel[col]], axis=1).dropna()
        c = pair.iloc[:, 0].rolling(window, min_periods=window - 1).corr(pair.iloc[:, 1])
        sy = pair.iloc[:, 0].rolling(window, min_periods=window - 1).std(ddof=0)
        rs = sy * np.sqrt((1 - c ** 2).clip(lower=0))     # population residual std of the OLS fit
        s = rs * np.sqrt((1 - c).clip(lower=0))
        sig[col] = s.reindex(panel.index).shift(1)         # day j uses days j-window .. j-1
    return pd.DataFrame(sig)


def expanding_pca(signal: pd.DataFrame, n_comp: int, start_after: int, change: str) -> pd.DataFrame:
    """Expanding-window PCA, scored on the last row each day (as in the notebook), computed from running
    sums so it is fast. Component signs are kept consistent from one day to the next."""
    S = signal.fillna(0.0).values
    seen = np.cumsum(signal.notna().values, axis=0) > 0    # columns that have appeared so far
    n, k = S.shape
    csum = np.cumsum(S, axis=0)
    cxx = np.cumsum(S[:, :, None] * S[:, None, :], axis=0)
    rows, idx, prev = [], [], None
    for i in range(start_after, n + 1):
        cols = seen[i - 1]
        if cols.sum() < n_comp:
            continue
        m = csum[i - 1][cols] / i
        cov = (cxx[i - 1][np.ix_(cols, cols)] - i * np.outer(m, m)) / (i - 1)
        w, v = np.linalg.eigh(cov)
        comps = v[:, ::-1][:, :n_comp].T                    # largest first
        # sklearn's sign convention (largest |loading| positive), then keep consistent over time
        signs = np.sign(comps[np.arange(n_comp), np.abs(comps).argmax(axis=1)]); comps = comps * signs[:, None]
        full = np.zeros((n_comp, k)); full[:, cols] = comps
        if prev is not None:
            for j in range(n_comp):
                if np.dot(full[j], prev[j]) < 0:
                    full[j] *= -1
        prev = full
        rows.append((S[i - 1] - np.where(cols, csum[i - 1] / i, 0.0)) @ full.T)
        idx.append(signal.index[i - 1])
    df = pd.DataFrame(rows, index=idx, columns=[f"PC{j + 1}" for j in range(n_comp)])
    for j in range(n_comp):
        pc = f"PC{j + 1}"
        df[f"{pc}_Change"] = df[pc].diff() if change == "diff" else df[pc].pct_change()
    return df


def breadth_measures(close: pd.DataFrame, vol: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rets = close.pct_change(fill_method=None)
    has = close.notna()
    N = has.sum(axis=1)
    ma50 = close.rolling(cfg["ma_short"], min_periods=cfg["ma_short"]).mean()
    ma200 = close.rolling(cfg["ma_long"], min_periods=cfg["ma_long"]).mean()
    hi = close.rolling(cfg["lookback_hl"], min_periods=cfg["lookback_hl"]).max()
    lo = close.rolling(cfg["lookback_hl"], min_periods=cfg["lookback_hl"]).min()
    cnt = lambda m: (m & has).sum(axis=1)   # as in the notebook, missing history counts as "no"
    adv_m, dec_m = rets > 0, rets < 0
    adv, dec = cnt(adv_m), cnt(dec_m)
    adv_v = vol.where(adv_m).sum(axis=1); dec_v = vol.where(dec_m).sum(axis=1)
    trin = (adv / dec.clip(lower=1e-12)) / (adv_v / dec_v.clip(lower=1e-12)).clip(lower=1e-12)
    b = pd.DataFrame({
        "N": N, "AD_Breadth": (adv - dec) / N,
        "Pct_NewHighs_252": cnt(close >= hi) / N, "Pct_NewLows_252": cnt(close <= lo) / N,
        "Pct_Above50": cnt(close > ma50) / N, "Pct_Above200": cnt(close > ma200) / N,
        "TRIN": trin, "XSec_Vol": rets.std(axis=1),
    })
    return b[b["N"] >= cfg["min_names"]]


def internal_turbulence(b: pd.DataFrame, zwin: int) -> pd.DataFrame:
    z = lambda s: (s - s.rolling(zwin).mean()) / s.rolling(zwin).std()
    comp = pd.DataFrame({
        "Z_NewLows": z(b["Pct_NewLows_252"]), "Z_Above50": z(1 - b["Pct_Above50"]),
        "Z_Above200": z(1 - b["Pct_Above200"]), "Z_AD": z(-b["AD_Breadth"]),
        "Z_TRIN": z(b["TRIN"]), "Z_XSecVol": z(b["XSec_Vol"]),
    })
    comp["InternalTurbulence"] = comp.mean(axis=1)
    return comp


def prepare(close: pd.DataFrame, vol: pd.DataFrame, spx: pd.Series, cfg: dict) -> dict:
    """Everything that does not depend on the clusters."""
    rets = close.pct_change(fill_method=None)
    br = breadth_measures(close, vol, cfg)
    it = internal_turbulence(br, cfg["z_win"])
    rv = (np.log(spx).diff().rolling(21).std() * math.sqrt(252)).rename("RV21")
    return {"close": close, "vol": vol, "spx": spx, "rets": rets, "breadth": br, "internal": it, "rv": rv,
            "spx_ret": spx.pct_change(fill_method=None).rename("SPX")}


def features_with(prep: dict, clusters: pd.Series, cfg: dict) -> pd.DataFrame:
    cret = cluster_returns(prep["rets"], prep["close"], prep["vol"], clusters)
    panel = cret.join(prep["spx_ret"], how="inner")
    signal = turbulence_signals(panel, "SPX", cfg["turb_window"])
    pcs = expanding_pca(signal, cfg["pca_components"], cfg["pca_start_after"], cfg["pc_change"])
    F = (pcs.join(signal.mean(axis=1).rename("TurbScalar"), how="left")
            .join(prep["internal"][["InternalTurbulence"]], how="inner").join(prep["rv"], how="left"))
    return F[F.index.isin(prep["spx"].index)].dropna(subset=cfg["features"])


# --------------------------------------------------------------------------------------
# Labels, model, evaluation
# --------------------------------------------------------------------------------------

def make_labels(spx: pd.Series, h: int, thr: float) -> pd.Series:
    """1 if the S&P 500 closes at least |thr| below today's close within the next h days; NaN if not yet known."""
    v = spx.values
    out = np.full(len(v), np.nan)
    for t in range(len(v) - h):
        out[t] = float(v[t + 1:t + h + 1].min() / v[t] - 1 <= thr)
    return pd.Series(out, index=spx.index)


def new_model():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=2000))])


def fit_rows(X: pd.DataFrame, y: pd.Series, end: int):
    """Fit on rows [0, end) that have a label; None if there is not enough data."""
    ytr = y.iloc[:end]
    m = ytr.notna().values & X.iloc[:end].notna().all(axis=1).values
    if m.sum() < 100 or ytr[m].nunique() < 2:
        return None
    return new_model().fit(X.iloc[:end][m], ytr[m].astype(int))


def walk_forward(prep: dict, labels: dict, cfg: dict, index: pd.DatetimeIndex) -> dict:
    """Out-of-sample probabilities. For each block of `refit_every` days starting at date d: form the
    clusters from returns up to the day before d, rebuild the features, train on rows whose labels were
    known before d (t <= i - h - 1), and predict the block."""
    preds = {name: pd.Series(np.nan, index=index) for name in labels}
    base = {name: pd.Series(np.nan, index=index) for name in labels}
    i = index.searchsorted(pd.Timestamp(cfg["eval_start"]))
    blocks = 0
    while i < len(index):
        d = index[i]
        F = features_with(prep, clusters_asof(prep, d - pd.Timedelta(days=1), cfg), cfg).reindex(index)
        X, R = F[cfg["features"]], F[["RV21"]]
        j = min(len(index), i + cfg["refit_every"])
        for name, y in labels.items():
            h = cfg["horizons"][name][0]
            for target, Z in ((preds, X), (base, R)):
                m = fit_rows(Z, y, max(0, i - h))
                block = Z.iloc[i:j]
                ok = block.notna().all(axis=1).values
                if m is not None and ok.any():
                    target[name].iloc[np.arange(i, j)[ok]] = m.predict_proba(block[ok])[:, 1]
        i = j; blocks += 1
    log(f"  walk-forward: {blocks} monthly blocks from {cfg['eval_start']}")
    return {"model": preds, "baseline": base}


def scores(p: pd.Series, y: pd.Series) -> dict:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    m = p.notna() & y.notna()
    p, y = p[m], y[m].astype(int)
    if len(p) < 50 or y.nunique() < 2:
        return {}
    base = float(y.mean())
    brier = float(brier_score_loss(y, p)); brier_c = float(brier_score_loss(y, np.full(len(y), base)))
    top = p >= p.quantile(0.8)
    return {
        "n": int(len(p)), "events": int(y.sum()), "baseRate": base,
        "auc": float(roc_auc_score(y, p)), "prAuc": float(average_precision_score(y, p)),
        "brier": brier, "brierClimatology": brier_c, "brierSkill": 1 - brier / brier_c,
        "hitRateTop20": float(y[top].mean()),
        "start": p.index[0].strftime("%Y-%m-%d"), "end": p.index[-1].strftime("%Y-%m-%d"),
    }


def calibration(p: pd.Series, y: pd.Series, edges=(0, .05, .10, .20, .30, .50, 1.0)) -> list:
    m = p.notna() & y.notna(); p, y = p[m], y[m]
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (p >= a) & ((p < b) if b < 1 else (p <= b))
        if sel.sum():
            out.append({"from": a, "to": b, "days": int(sel.sum()), "predicted": float(p[sel].mean()), "observed": float(y[sel].mean())})
    return out


def level_for(prob: float, base: float) -> str:
    r = prob / base if base > 0 else 1
    return "Low" if r < 0.75 else "Normal" if r < 1.25 else "Elevated" if r < 2 else "High"


def pairs(s: pd.Series, nd: int = 4) -> list:
    return [[i.strftime("%Y-%m-%d"), round(float(v), nd)] for i, v in s.dropna().items()]


# --------------------------------------------------------------------------------------

def build_model(cfg: dict = CONFIG) -> dict:
    t0 = time.time()
    csv = os.environ.get("RISK_MODEL_PRICES_CSV")
    if csv:
        log(f"prices from {csv}")
        close, vol = prices_from_csv(csv)
    else:
        log("downloading prices from Yahoo Finance")
        close, vol = prices_from_yahoo(read_universe(), cfg["history_start"])
    spx = spx_series(cfg["history_start"])
    spx = spx[(spx.index >= close.index[0]) & (spx.index <= close.index[-1])]
    log(f"  S&P 500: {len(spx)} days to {spx.index[-1].date()}")
    prep = prepare(close, vol, spx, cfg)

    # today's model: clusters formed from the trailing window up to the latest date
    live_clusters = clusters_asof(prep, close.index[-1], cfg)
    F = features_with(prep, live_clusters, cfg)
    X = F[cfg["features"]]
    asof = X.index[-1]
    log(f"  features: {X.shape[0]} days, {X.index[0].date()} to {asof.date()}; {live_clusters.size} stocks in {live_clusters.nunique()} clusters")

    labels = {name: make_labels(spx, h, thr).reindex(X.index) for name, (h, thr) in cfg["horizons"].items()}
    wf = walk_forward(prep, labels, cfg, X.index)

    cut = asof - pd.DateOffset(years=cfg["history_years_on_site"])
    keep = X.index >= max(pd.Timestamp(cfg["eval_start"]), cut)
    spx_f = spx.reindex(X.index)
    history = {"dates": [d.strftime("%Y-%m-%d") for d in X.index[keep]],
               "spx": [round(float(v), 2) for v in spx_f[keep]]}
    horizons = {}
    for name, (h, thr) in cfg["horizons"].items():
        y = labels[name]
        mask = y.notna()
        final = fit_rows(X, y, len(X))
        p_now = float(final.predict_proba(X.iloc[[-1]])[:, 1][0])
        base_rate = float(y[mask].mean())
        sc, lr = final.named_steps["scaler"], final.named_steps["model"]
        zx = (X.iloc[-1].values - sc.mean_) / sc.scale_
        drivers = sorted(({"feature": f, "label": FEATURE_INFO[f][0], "value": float(X.iloc[-1][f]), "z": float(z),
                           "coef": float(c), "contribution": float(c * z)} for f, z, c in zip(X.columns, zx, lr.coef_[0])),
                         key=lambda d: -abs(d["contribution"]))
        oos, bl = wf["model"][name], wf["baseline"][name]
        hist = oos.dropna()
        horizons[name] = {
            "days": h, "threshold": thr, "probability": p_now, "baseRate": base_rate,
            "ratio": p_now / base_rate if base_rate else None, "level": level_for(p_now, base_rate),
            "percentile": float((hist <= p_now).mean()) if len(hist) else None,
            "intercept": float(lr.intercept_[0]),
            "trainDays": int(mask.sum()), "trainEvents": int(y[mask].sum()),
            "evaluation": {"model": scores(oos, y), "volBaseline": scores(bl, y)},
            "calibration": calibration(oos, y),
            "drivers": drivers,
        }
        history[f"p_{name}"] = [None if pd.isna(v) else round(float(v), 4) for v in oos[keep]]
        history[f"y_{name}"] = [None if pd.isna(v) else int(v) for v in y[keep]]
        ev = horizons[name]["evaluation"]
        log(f"  {name}: today {p_now:.3f} (base {base_rate:.3f}) | out-of-sample AUC {ev['model'].get('auc', float('nan')):.3f}"
            f" vs volatility baseline {ev['volBaseline'].get('auc', float('nan')):.3f}")

    br, it = prep["breadth"].loc[:asof], prep["internal"].loc[:asof]
    pctile = lambda s, v: float((s.dropna() <= v).mean()) if s.notna().any() else None
    internals = []
    for key, label, fmt in [("Pct_Above200", "Stocks above their 200-day average", "pct"),
                            ("Pct_Above50", "Stocks above their 50-day average", "pct"),
                            ("Pct_NewHighs_252", "Stocks at a 52-week high", "pct"),
                            ("Pct_NewLows_252", "Stocks at a 52-week low", "pct"),
                            ("AD_Breadth", "Advancers minus decliners, share of stocks", "pct"),
                            ("TRIN", "Arms index (TRIN)", "num"),
                            ("XSec_Vol", "Dispersion of daily returns across stocks", "pct")]:
        v = float(br[key].iloc[-1])
        internals.append({"key": key, "label": label, "value": v, "format": fmt, "percentile": pctile(br[key], v)})
    for key, s, fmt in [("InternalTurbulence", it["InternalTurbulence"], "z"), ("TurbScalar", F["TurbScalar"], "num")]:
        v = float(s.dropna().iloc[-1])
        internals.append({"key": key, "label": FEATURE_INFO[key][0], "value": v, "format": fmt, "percentile": pctile(s, v)})

    start_hist = X.index[keep][0]
    out = {
        "asof": asof.strftime("%Y-%m-%d"),
        "spxClose": float(spx.loc[asof]),
        "horizons": horizons,
        "history": history,
        "internals": internals,
        "indicators": {"internalTurbulence": pairs(it["InternalTurbulence"].loc[start_hist:], 3),
                       "pctAbove200": pairs(br["Pct_Above200"].loc[start_hist:], 4)},
        "model": {
            "name": "Logistic regression on market-turbulence features",
            "features": [{"key": f, "label": FEATURE_INFO[f][0], "description": FEATURE_INFO[f][1]} for f in cfg["features"]],
            "universe": int(close.shape[1]), "clusters": int(live_clusters.nunique()), "clusterYears": cfg["cluster_years"],
            "evalStart": cfg["eval_start"], "refitEvery": cfg["refit_every"],
            "featureStart": X.index[0].strftime("%Y-%m-%d"),
        },
        "runtimeSeconds": round(time.time() - t0, 1),
    }
    update_log(out, spx)
    log(f"  done in {out['runtimeSeconds']}s")
    return out


def update_log(out: dict, spx: pd.Series) -> None:
    """Append today's live prediction to data/model_log.json and score earlier entries once their outcome is known."""
    path = DATA / "model_log.json"
    try:
        entries = json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        entries = []
    by_date = {e["date"]: e for e in entries}
    prev = by_date.get(out["asof"], {})
    by_date[out["asof"]] = {"date": out["asof"], "spx": round(out["spxClose"], 2),
                            **{f"p_{k}": round(v["probability"], 4) for k, v in out["horizons"].items()},
                            **{f"y_{k}": prev.get(f"y_{k}") for k in out["horizons"]}}
    idx = spx.index
    for e in by_date.values():
        d = pd.Timestamp(e["date"])
        if d not in idx:
            continue
        i = idx.get_loc(d)
        for k, v in out["horizons"].items():
            h, thr = v["days"], v["threshold"]
            if e.get(f"y_{k}") is None and i + h < len(idx):
                e[f"y_{k}"] = int(spx.iloc[i + 1:i + h + 1].min() / spx.iloc[i] - 1 <= thr)
    entries = sorted(by_date.values(), key=lambda e: e["date"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, separators=(",", ":")))
    out["liveLog"] = entries[-260:]


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    from update_data import now_iso, write_json
    data = build_model()
    data["updated"] = now_iso()
    write_json(DATA / "model.json", data)
    log("wrote data/model.json")
