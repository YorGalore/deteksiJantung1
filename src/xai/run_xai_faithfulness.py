"""Uji faithfulness XAI pada M1 asli, HANYA validasi internal fold 0-16 (internal_val_0_16 == 1).

Jalankan dari root repo:
    python -u -m src.xai.run_xai_faithfulness

Keluaran: angka agregat (rata-rata penurunan logit + selang 95% bootstrap). Tidak menulis berkas, tidak ada
keluaran per pasien. Fold 17 tidak pernah dipakai.

Opsi fold (src/xai/selection.py), bawaan tetap validasi internal fold 0-16:
    python -u -m src.xai.run_xai_faithfulness --fold 18                       # uji kode sebelum rencana FINAL
    python -u -m src.xai.run_xai_faithfulness --fold 19 --include-fold-19     # setelah rencana analisis FINAL
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.dataset.run_signal_qc_pilot import CACHE, MASTER, PREFIX
from src.model.ecg_dataset import ECGDataset, VAL_COL, load_record_table
from src.model.ecg_preprocessing import LeadStandardizer
from src.model.resnet1d import resnet1d
from src.model.train_smoke import load_checkpoint
from src.xai.ecg_xai import grad_cam_1d, lead_occlusion
from src.xai.selection import CHOICES, pick_records, resolve
from src.xai.faithfulness import lead_faithfulness, mean_ci, time_faithfulness

ROOT = Path(__file__).resolve().parents[2]
QC_CSV = ROOT / "data/interim/signal_qc_full.csv"
OUT_DIR = ROOT / "data/interim/m1"


def report(title: str, r: dict) -> str:
    rows = []
    for name in ("top", "random", "bottom"):
        m, lo, hi = mean_ci(r[name])
        rows.append({"wilayah": name, "penurunan_logit": round(m, 3), "CI95": f"[{lo:.3f}, {hi:.3f}]"})
    diff = r["top"] - r["random"]
    m, lo, hi = mean_ci(diff)
    rows.append({"wilayah": "top - random", "penurunan_logit": round(m, 3), "CI95": f"[{lo:.3f}, {hi:.3f}]"})
    frac = float((r["top"] > r["random"]).mean())
    return f"{title}\n{pd.DataFrame(rows).to_string(index=False)}\nrekaman dengan top > random: {frac:.2f}"


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
    pick = pick_records(table, a.fold, a.n_per_class, a.seed)
    ds = ECGDataset(pick.reset_index(drop=True), CACHE, PREFIX,
                    standardizer=LeadStandardizer.load(OUT_DIR / "lead_stats.json"))
    x = torch.stack([ds[i][0] for i in range(len(ds))])
    y = np.array([float(ds[i][1]) for i in range(len(ds))]).astype(int)
    model = resnet1d()
    load_checkpoint(OUT_DIR / "m1_best.pt", model)

    cam = grad_cam_1d(model, x)
    occ = lead_occlusion(model, x)
    print(f"=== Faithfulness XAI ({label}; penurunan logit = logit asli - logit setelah dinolkan) ===")
    for cls, name in ((1, "AMI"), (0, "non-AMI")):
        m = y == cls
        print(f"\n--- {name} (n={int(m.sum())}) ---")
        print(report("Waktu: 4 dari 20 blok 0,5 detik (Grad-CAM)", time_faithfulness(model, x[m], cam[m], seed=a.seed)))
        print(report("\nLead: 3 dari 12 lead (oklusi)", lead_faithfulness(model, x[m], occ[m], seed=a.seed)))


if __name__ == "__main__":
    main()