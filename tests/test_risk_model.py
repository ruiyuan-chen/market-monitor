"""Offline tests for the drawdown-risk model (synthetic data, no network). Run: python -m pytest -q"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import risk_model as rm  # noqa: E402


def _panel(n=120, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    x = rng.normal(0, 0.01, (n, 3))
    spx = x @ np.array([0.5, 0.3, 0.2]) + rng.normal(0, 0.004, n)
    return pd.DataFrame({"C0": x[:, 0], "C1": x[:, 1], "C2": x[:, 2], "SPX": spx}, index=idx)


def test_turbulence_signal_matches_explicit_ols():
    p = _panel()
    sig = rm.turbulence_signals(p, "SPX", 21)
    j = 60
    y, x = p["SPX"].iloc[j - 21:j].values, p["C1"].iloc[j - 21:j].values
    A = np.column_stack([np.ones(21), x])
    resid = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    c = np.corrcoef(y, x)[0, 1]
    expected = np.std(resid) * np.sqrt(max(0.0, 1 - c))
    assert sig["C1"].iloc[j] == pytest.approx(expected, rel=1e-9)
    assert sig["C1"].iloc[:20].isna().all()          # needs at least 20 of the prior 21 days


def test_expanding_pca_matches_sklearn():
    from sklearn.decomposition import PCA
    S = _panel(200).iloc[:, :3].abs()
    out = rm.expanding_pca(S, 2, 30, "diff")
    for i in (30, 100, 200):
        p = PCA(n_components=2).fit(S.iloc[:i])
        ref = p.transform(S.iloc[[i - 1]])[0]
        got = out.loc[S.index[i - 1], ["PC1", "PC2"]].values
        assert np.allclose(np.abs(got), np.abs(ref), atol=1e-12)
    assert out["PC1_Change"].iloc[1] == pytest.approx(out["PC1"].iloc[1] - out["PC1"].iloc[0])


def test_labels_unknown_future_is_nan_not_zero():
    idx = pd.bdate_range("2024-01-01", periods=10)
    spx = pd.Series([100, 101, 99, 97, 98, 100, 100, 100, 100, 100], index=idx, dtype=float)
    y = rm.make_labels(spx, 3, -0.03)
    assert y.iloc[0] == 1.0          # 100 -> 97 within 3 days
    assert y.iloc[4] == 0.0
    assert y.iloc[-3:].isna().all()  # future not yet observed


def test_breadth_counts():
    idx = pd.bdate_range("2024-01-01", periods=3)
    close = pd.DataFrame({"A": [10, 11, 12], "B": [10, 9, 9], "C": [10, 10, np.nan]}, index=idx, dtype=float)
    vol = close * 0 + 100
    b = rm.breadth_measures(close, vol, dict(rm.CONFIG, min_names=1, ma_short=2, ma_long=2, lookback_hl=2))
    assert b.loc[idx[1], "N"] == 3 and b.loc[idx[2], "N"] == 2
    assert b.loc[idx[1], "AD_Breadth"] == pytest.approx((1 - 1) / 3)
    # as in the notebook, a close equal to the rolling max counts as a new high (B is flat at 9)
    assert b.loc[idx[2], "Pct_NewHighs_252"] == pytest.approx(2 / 2)


def test_walk_forward_uses_only_past_labels(monkeypatch):
    """A feature that copies the future label must not leak: the walk-forward never trains on rows whose
    label was unknown at prediction time, and it predicts only from eval_start on."""
    n = 400
    idx = pd.bdate_range("2019-01-01", periods=n)
    rng = np.random.default_rng(1)
    spx = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.012, n))), index=idx)
    y = rm.make_labels(spx, 5, -0.03)
    feats = pd.DataFrame({f: rng.normal(size=n) for f in rm.CONFIG["features"]}, index=idx)
    feats["RV21"] = 0.2
    seen = []

    def fake_features(prep, clusters, cfg):
        return feats

    real_fit = rm.fit_rows

    def spy_fit(X, yy, end):
        seen.append(end)
        return real_fit(X, yy, end)

    monkeypatch.setattr(rm, "features_with", fake_features)
    monkeypatch.setattr(rm, "clusters_asof", lambda prep, d, cfg: None)
    monkeypatch.setattr(rm, "fit_rows", spy_fit)
    cfg = dict(rm.CONFIG, eval_start=str(idx[250].date()), refit_every=21)
    wf = rm.walk_forward({}, {"5d": y}, cfg, idx)
    p = wf["model"]["5d"]
    assert p.iloc[:250].isna().all() and p.iloc[250:].notna().all()
    starts = range(250, n, 21)
    assert sorted(set(seen)) == sorted({max(0, i - 5) for i in starts})


def test_level_labels():
    assert rm.level_for(0.05, 0.15) == "Low"
    assert rm.level_for(0.15, 0.15) == "Normal"
    assert rm.level_for(0.25, 0.15) == "Elevated"
    assert rm.level_for(0.40, 0.15) == "High"
