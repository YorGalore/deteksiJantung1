"""Pemeriksaan kewajaran XAI pada M1 asli, HANYA data validasi internal fold 0-16 (internal_val_0_16 == 1).

Jalankan dari root repo:
    python -u -m src.xai.run_xai_sanity

Keluaran: angka agregat saja (oklusi rata-rata per lead, profil Grad-CAM per detik, uji acak-bobot).
Tidak menulis berkas, tidak ada keluaran per pasien. Fold 17 tidak pernah dipakai.

Opsi fold (src/xai/selection.py), bawaan tetap validasi internal fold 0-16:
    python -u -m src.xai.run_xai_sanity --fold 18                       # uji kode sebelum rencana FINAL
    python -u -m src.xai.run_xai_sanity --fold 19 --include-fold-19     # setelah rencana analisis FINAL
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.dataset.run_signal_qc_pilot import CACHE, MASTER, PREFIX
from src.model.ecg_dataset import ECGDataset, VAL_COL, load_record_table
from src.model.ecg_preprocessing import LEAD_ORDER, LeadStandardizer
from src.model.resnet1d import resnet1d
from src.model.train_smoke import load_checkpoint
from src.xai.ecg_xai import grad_cam_1d, lead_occlusion
from src.xai.selection import CHOICES, pick_records, resolve

ROOT = Path(__file__).resolve().parents[2]
QC_CSV = ROOT / "data/interim/signal_qc_full.csv"
OUT_DIR = ROOT / "data/interim/m1"


def corr_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    den = np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1))
    return np.where(den > 1e-12, (a * b).sum(1) / np.maximum(den, 1e-12), 0.0)


def summarize_xai(model, model_random, x: torch.Tensor, y: np.ndarray) -> dict:
    """Agregat per kelas: oklusi per lead, profil Grad-CAM per bin 1 detik, dan korelasi dengan model acak."""
    occ = lead_occlusion(model, x)
    cam = grad_cam_1d(model, x)
    cam_rand = grad_cam_1d(model_random, x)
    out = {"n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum())}
    for name, m in (("pos", y == 1), ("neg", y == 0)):
        out[f"occ_{name}"] = occ[m].mean(0)
        out[f"cam_{name}"] = cam[m].reshape(int(m.sum()), 10, -1).mean((0, 2))
    out["corr_random_median"] = float(np.median(corr_rows(cam, cam_rand)))
    out["corr_random_p90"] = float(np.quantile(corr_rows(cam, cam_rand), 0.9))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fold", choices=CHOICES, default="internal",
                    help="internal = validasi internal fold 0-16 (bawaan, perilaku asli); 18 = uji kode; 19 = setelah FINAL")
    ap.add_argument("--include-fold-19", action="store_true")
    a = ap.parse_args(argv)

    folds, allow_locked, label = resolve(a.fold, a.include_fold_19)  # fold 17 ditolak; 19 hanya setelah FINAL
    table = load_record_table(QC_CSV, MASTER, folds, allow_locked=allow_locked)
    rng = np.random.default_rng(a.seed)
    pick = pick_records(table, a.fold, a.n_per_class, a.seed)
    stats = LeadStandardizer.load(OUT_DIR / "lead_stats.json")
    ds = ECGDataset(pick.reset_index(drop=True), CACHE, PREFIX, standardizer=stats)
    x = torch.stack([ds[i][0] for i in range(len(ds))])
    y = np.array([float(ds[i][1]) for i in range(len(ds))]).astype(int)

    model = resnet1d()
    load_checkpoint(OUT_DIR / "m1_best.pt", model)
    torch.manual_seed(a.seed)
    r = summarize_xai(model, resnet1d(), x, y)

    print(f"=== XAI sanity ({label}; {r['n_pos']} AMI, {r['n_neg']} non-AMI) ===")
    print("\nOklusi per lead (rata-rata penurunan logit saat lead dinolkan):")
    print(pd.DataFrame({"AMI": r["occ_pos"], "non-AMI": r["occ_neg"]}, index=LEAD_ORDER).round(3).to_string())
    print("\nGrad-CAM rata-rata per detik (0-1, tiap bin = 1 detik):")
    print(pd.DataFrame({"AMI": r["cam_pos"], "non-AMI": r["cam_neg"]}, index=range(1, 11)).round(3).to_string())
    print(f"\nUji acak-bobot: korelasi Grad-CAM model terlatih vs model acak: median {r['corr_random_median']:.3f}, "
          f"p90 {r['corr_random_p90']:.3f}")


if __name__ == "__main__":
    main()