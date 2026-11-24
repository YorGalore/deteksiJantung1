"""Tes gambar kurva kalibrasi (src/evaluation/plot_calibration.py). Data SINTETIS + .md fold 18 yang ter-commit.

Jalankan: python -m pytest -q tests/test_plot_calibration.py
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from src.evaluation import plot_calibration as C
from src.evaluation.calibration_curve import calibration_curves
from src.evaluation.core import calibration_table

REPO = Path(__file__).resolve().parents[1]
MD18 = REPO / "reports" / "eval_fold18_uji_final.md"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def write_inputs(d: Path, models=C.MODELS) -> None:
    rng = np.random.default_rng(0)
    n = 4000
    probs = {m: expit(rng.normal(-3.3, 1.1, n)) for m in models}
    y = (rng.random(n) < probs[models[0]]).astype(int)
    pd.DataFrame(calibration_curves(y, probs, np.arange(n), n_boot=20, seed=42)).to_csv(
        d / "eval_fold18_uji_final_calibration_curve.csv", index=False)
    pd.DataFrame([{"model": m, "bin": r["bin"], "n": f"{r['n']:,}", "mean_pred": r["mean_pred"],
                   "observed": r["observed"]} for m in models for r in calibration_table(y, probs[m])]).to_csv(
        d / "eval_fold18_uji_final_calibration_bins.csv", index=False)
    shutil.copy(MD18, d / "eval_fold18_uji_final.md")


def test_gambar_kalibrasi_dibuat_tanpa_membaca_prediksi(tmp_path, monkeypatch):
    write_inputs(tmp_path)
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: (_ for _ in ()).throw(AssertionError("parquet")))
    C.main(["--fold", "18", "--reports-dir", str(tmp_path)])
    out = tmp_path / "figures" / "kalibrasi_fold18.png"
    assert out.read_bytes()[:8] == PNG_MAGIC and out.stat().st_size > 10_000


def test_pilihan_model_dan_model_hilang(tmp_path):
    write_inputs(tmp_path, models=["M4", "M5"])
    C.main(["--fold", "18", "--reports-dir", str(tmp_path), "--out", str(tmp_path / "a.png")])
    with pytest.raises(SystemExit, match="M1 tidak ada"):
        C.main(["--fold", "18", "--reports-dir", str(tmp_path), "--models", "M1", "M5"])


def test_csv_belum_ada_atau_fold_salah(tmp_path):
    shutil.copy(MD18, tmp_path / "eval_fold18_uji_final.md")
    with pytest.raises(SystemExit, match="tidak ada"):
        C.main(["--fold", "18", "--reports-dir", str(tmp_path)])
    write_inputs(tmp_path)
    (tmp_path / "eval_fold19.md").write_text((tmp_path / "eval_fold18_uji_final.md").read_text())
    for suf in ("calibration_curve", "calibration_bins"):
        shutil.copy(tmp_path / f"eval_fold18_uji_final_{suf}.csv", tmp_path / f"eval_fold19_{suf}.csv")
    with pytest.raises(SystemExit, match="tidak sesuai fold 19"):
        C.main(["--fold", "19", "--reports-dir", str(tmp_path)])


def test_nan_pada_estimasi_berhenti(tmp_path):
    write_inputs(tmp_path)
    p = tmp_path / "eval_fold18_uji_final_calibration_curve.csv"
    df = pd.read_csv(p)
    df.loc[(df.model == "M5") & (df.index % 7 == 0), "p_obs"] = np.nan
    df.to_csv(p, index=False)
    with pytest.raises(SystemExit, match="NaN"):
        C.main(["--fold", "18", "--reports-dir", str(tmp_path)])
