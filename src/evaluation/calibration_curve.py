"""Kurva kalibrasi fleksibel (TRIPOD+AI 12e; Van Calster dkk. 2019): NumPy saja, keluaran agregat.

Metode: regresi logistik y ~ restricted cubic spline (RCS) dari logit(p), 4 knot pada persentil
5, 35, 65, 95 dari logit(p) (Harrell). Kurva = probabilitas teramati yang diprediksi model spline,
dievaluasi pada grid tetap p (50 titik, rata di skala logit, antara persentil 1 dan 99 dari p).
IK 95%: bootstrap per subject_id dengan knot dan grid TETAP dari sampel penuh; iterasi yang gagal
(mis. tanpa positif) menjadi NaN dan tidak dihitung.

Yang disimpan hanya grid (p prediksi, p teramati, IK); tidak ada nilai per kunjungan.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from src.evaluation.bootstrap import cluster_resample_indices
from src.evaluation.core import safe_logit

KNOT_PCTS = (5, 35, 65, 95)
GRID_PCTS = (1, 99)
N_GRID = 50


def rcs_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Basis restricted cubic spline (Harrell): kolom [x, s_1 .. s_{k-2}], linear di luar knot ujung."""
    x = np.asarray(x, dtype=np.float64)
    k = np.asarray(knots, dtype=np.float64)
    t_last, t_prev = k[-1], k[-2]
    scale = (t_last - k[0]) ** 2
    cols = [x]
    pos3 = lambda v: np.clip(v, 0, None) ** 3
    for j in range(len(k) - 2):
        s = (pos3(x - k[j]) - pos3(x - t_prev) * (t_last - k[j]) / (t_last - t_prev)
             + pos3(x - t_last) * (t_prev - k[j]) / (t_last - t_prev))
        cols.append(s / scale)
    return np.column_stack(cols)


def fit_logistic(X: np.ndarray, y: np.ndarray, max_iter: int = 50, ridge: float = 1e-8) -> np.ndarray | None:
    """IRLS/Newton untuk y ~ [1, X]. Kembalikan koefisien, atau None bila tidak konvergen/tidak hingga."""
    A = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(A.shape[1])
    if y.min() == y.max():
        return None
    for _ in range(max_iter):
        mu = expit(A @ w)
        W = mu * (1 - mu)
        H = A.T @ (A * W[:, None]) + ridge * np.eye(A.shape[1])
        g = A.T @ (y - mu)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return None
        w = w + step
        if not np.isfinite(w).all():
            return None
        if np.max(np.abs(step)) < 1e-9:
            return w
    return w if np.isfinite(w).all() else None


def setup(p: np.ndarray) -> dict:
    """Knot dan grid dari sampel penuh (dipakai juga di setiap iterasi bootstrap)."""
    x = safe_logit(p)
    knots = np.percentile(x, KNOT_PCTS)
    if not np.all(np.diff(knots) > 0):
        raise ValueError("Knot spline tidak unik (prediksi terlalu sedikit variasi).")
    lo, hi = np.percentile(p, GRID_PCTS)
    grid_p = expit(np.linspace(safe_logit([lo])[0], safe_logit([hi])[0], N_GRID))
    return {"knots": knots, "grid_p": grid_p, "grid_B": rcs_basis(safe_logit(grid_p), knots)}


def curve(y: np.ndarray, p: np.ndarray, s: dict) -> np.ndarray:
    """Probabilitas teramati (kurva) pada grid; NaN bila fit gagal."""
    w = fit_logistic(rcs_basis(safe_logit(p), s["knots"]), np.asarray(y, dtype=np.float64))
    if w is None:
        return np.full(len(s["grid_p"]), np.nan)
    return expit(w[0] + s["grid_B"] @ w[1:])


def calibration_curves(y, probs: dict[str, np.ndarray], clusters, n_boot: int = 2000, seed: int = 42,
                       alpha: float = 0.05) -> list[dict]:
    """Kurva untuk setiap model (probabilitas SETELAH kalibrasi terkunci) + IK bootstrap berpasangan.

    Kembalian: baris agregat {model, p_pred, p_obs, ci_low, ci_high, n_valid}.
    """
    y = np.asarray(y).astype(np.int64)
    setups = {m: setup(np.asarray(p, dtype=np.float64)) for m, p in probs.items()}
    point = {m: curve(y, probs[m], setups[m]) for m in probs}
    boots = {m: np.full((n_boot, N_GRID), np.nan) for m in probs}
    for i, idx in enumerate(cluster_resample_indices(clusters, n_boot, seed)):
        yb = y[idx]
        if not (0 < yb.sum() < len(yb)):
            continue
        for m in probs:
            boots[m][i] = curve(yb, probs[m][idx], setups[m])
    rows = []
    for m in probs:
        b = boots[m]
        ok = np.isfinite(b).all(axis=1)
        lo = np.quantile(b[ok], alpha / 2, axis=0) if ok.any() else np.full(N_GRID, np.nan)
        hi = np.quantile(b[ok], 1 - alpha / 2, axis=0) if ok.any() else np.full(N_GRID, np.nan)
        for j in range(N_GRID):
            rows.append({"model": m, "p_pred": float(setups[m]["grid_p"][j]), "p_obs": float(point[m][j]),
                         "ci_low": float(lo[j]), "ci_high": float(hi[j]), "n_valid": int(ok.sum())})
    return rows
