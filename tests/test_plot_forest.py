"""Tes forest plot Gambar 2 (src/evaluation/plot_forest.py). Data SINTETIS + laporan .md fold 18 yang ter-commit.

Jalankan: python -m pytest -q tests/test_plot_forest.py
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation import plot_forest as F
from src.evaluation.bootstrap import paired_bootstrap
from src.model.metrics import pr_auc, roc_auc

REPO = Path(__file__).resolve().parents[1]
MD18 = REPO / "reports" / "eval_fold18_uji_final.md"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def synthetic_metrics_csv(path: Path, models=F.MODELS, seed=0) -> None:
    """CSV dengan skema yang sama seperti keluaran run_fold19 (paired_bootstrap asli, bootstrap kecil)."""
    rng = np.random.default_rng(seed)
    n = 3000
    y = (rng.random(n) < 0.05).astype(int)
    subj = rng.integers(0, 2000, n)
    probs = {m: 1 / (1 + np.exp(-(rng.normal(0, 1, n) + (0.6 + 0.2 * i) * y * 2)))
             for i, m in enumerate(["M0", "M0S1"] + list(models))}
    probs["M0"] = (probs["M0"] > 0.9).astype(float)
    probs["M0S1"] = (probs["M0S1"] > 0.95).astype(float)
    fns = {"AUROC": lambda yy, p, m: np.nan if m in ("M0", "M0S1") else roc_auc(yy, p),
           "AUPRC": lambda yy, p, m: np.nan if m in ("M0", "M0S1") else pr_auc(yy, p)}
    b = paired_bootstrap(y, probs, subj, fns, n_boot=30, seed=42, diffs=[("M5", "M4")] if "M5" in models else [])
    metrics = b[~b["target"].str.contains(" - ", regex=False)]
    extra = pd.DataFrame([{"target": "M5", "metric": "Brier_mentah", "estimate": 0.02, "ci_low": np.nan,
                           "ci_high": np.nan, "n_valid": 0}])
    pd.concat([metrics, extra], ignore_index=True).to_csv(path, index=False)


@pytest.fixture
def reports18(tmp_path):
    shutil.copy(MD18, tmp_path / "eval_fold18_uji_final.md")
    synthetic_metrics_csv(tmp_path / "eval_fold18_uji_final_metrics.csv")
    return tmp_path


def test_header_laporan_fold18_terbaca():
    info = F.parse_header(MD18.read_text(encoding="utf-8"))
    assert info == {"n": 7520, "n_pos": "218", "prevalence_pct": 2.90}


def test_judul_harus_sesuai_fold():
    text = MD18.read_text(encoding="utf-8")
    assert F.check_fold(text, 18) is False
    with pytest.raises(SystemExit, match="tidak sesuai fold 19"):
        F.check_fold(text, 19)


def test_posthoc_terdeteksi():
    text = "# Evaluasi akhir fold 19\n\n> **POST-HOC.** Evaluasi fold 19 sudah pernah dijalankan\n"
    assert F.check_fold(text, 19) is True


def test_gambar_dibuat_tanpa_membaca_prediksi(reports18, monkeypatch):
    def larang(*a, **k):
        raise AssertionError("plot_forest tidak boleh membaca parquet")
    monkeypatch.setattr(pd, "read_parquet", larang)
    out = reports18 / "fig.png"
    F.main(["--fold", "18", "--reports-dir", str(reports18), "--out", str(out)])
    assert out.read_bytes()[:8] == PNG_MAGIC and out.stat().st_size > 10_000


def test_nama_keluaran_default(reports18):
    F.main(["--fold", "18", "--reports-dir", str(reports18)])
    assert (reports18 / "figures" / "gambar2_forest_fold18.png").exists()


def test_angka_sama_dengan_csv(reports18):
    pts = F.load_points(reports18 / "eval_fold18_uji_final_metrics.csv", F.MODELS)
    csv = pd.read_csv(reports18 / "eval_fold18_uji_final_metrics.csv")
    for _, r in pts.iterrows():
        c = csv[(csv["target"] == r["model"]) & (csv["metric"] == r["metric"])].iloc[0]
        assert (r["estimate"], r["ci_low"], r["ci_high"]) == (c["estimate"], c["ci_low"], c["ci_high"])
    assert len(pts) == 2 * len(F.MODELS)


def test_model_hilang_berhenti(tmp_path):
    shutil.copy(MD18, tmp_path / "eval_fold18_uji_final.md")
    synthetic_metrics_csv(tmp_path / "eval_fold18_uji_final_metrics.csv", models=["M1", "M2", "M3", "M4"])
    with pytest.raises(SystemExit, match="M5 AUROC: ditemukan 0 baris"):
        F.main(["--fold", "18", "--reports-dir", str(tmp_path), "--out", str(tmp_path / "x.png")])
    F.main(["--fold", "18", "--reports-dir", str(tmp_path), "--out", str(tmp_path / "x.png"),
            "--models", "M1", "M2", "M3", "M4"])
    assert (tmp_path / "x.png").exists()


def test_nilai_tidak_valid_berhenti(reports18):
    p = reports18 / "eval_fold18_uji_final_metrics.csv"
    df = pd.read_csv(p)
    df.loc[(df["target"] == "M3") & (df["metric"] == "AUPRC"), "ci_high"] = np.nan
    df.to_csv(p, index=False)
    with pytest.raises(SystemExit, match="M3 AUPRC: nilai tidak valid"):
        F.load_points(p, F.MODELS)


def test_csv_belum_ada_berhenti(tmp_path):
    shutil.copy(MD18, tmp_path / "eval_fold18_uji_final.md")
    with pytest.raises(SystemExit, match="tidak ada"):
        F.main(["--fold", "18", "--reports-dir", str(tmp_path)])


def test_desimal_koma():
    assert F.koma(0.8834) == "0,883"
    assert F.koma(2.9, 2) == "2,90"
