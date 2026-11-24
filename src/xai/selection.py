"""Pemilihan rekaman untuk uji XAI M1 (Grad-CAM acak-bobot, faithfulness, oklusi). Pandas saja, tanpa torch.

Pilihan --fold:
  internal  validasi internal fold 0-16 (internal_val_0_16 == 1); perilaku ASLI skrip XAI, tidak berubah.
  18        semua rekaman fold 18 (lolos QC); fold 18 tidak dipakai melatih M1. Dipakai untuk UJI KODE
            sebelum rencana analisis FINAL.
  19        semua rekaman fold 19 (lolos QC). Hanya dengan --include-fold-19, setelah rencana analisis
            berstatus FINAL dan di-commit (src/evaluation/lock.py).
Sampel: n_per_class AMI + n_per_class non-AMI, acak dengan seed (pandas .sample), sama seperti versi asli.
"""
from __future__ import annotations

import pandas as pd

from src.constants import FOLD_COL, FOLD_TEST, FOLD_THRESHOLD

VAL_COL = "internal_val_0_16"
CHOICES = ("internal", str(FOLD_THRESHOLD), str(FOLD_TEST))


def resolve(fold: str, include_fold_19: bool = False) -> tuple[list[int], bool, str]:
    """(fold yang dimuat, allow_locked, label laporan). Fold 19 diperiksa ke penjaga kunci."""
    if fold == "internal":
        return list(range(0, 17)), False, "validasi internal fold 0-16"
    if fold == str(FOLD_THRESHOLD):
        return [FOLD_THRESHOLD], True, "fold 18 (UJI KODE; fold 18 tidak dipakai melatih M1)"
    if fold == str(FOLD_TEST):
        if not include_fold_19:
            raise SystemExit("Fold 19 dikunci. Tambahkan --include-fold-19 hanya setelah rencana analisis FINAL.")
        from src.evaluation import lock
        info = lock.require_fold19_unlocked(lock.ROOT, need_m0_rule=False)
        return [FOLD_TEST], True, f"fold 19 (rencana analisis `{info['plan_commit']}`)"
    raise SystemExit(f"--fold harus salah satu dari {CHOICES}.")


def pick_records(table: pd.DataFrame, fold: str, n_per_class: int, seed: int) -> pd.DataFrame:
    """Ambil n_per_class AMI lalu n_per_class non-AMI (urutan sama dengan versi asli: kelas 1 dulu)."""
    if fold == "internal":
        pool = table[table[VAL_COL] == 1]
    else:
        pool = table[table[FOLD_COL] == int(fold)]
        if pool.empty or set(pool[FOLD_COL].unique()) != {int(fold)}:
            raise SystemExit(f"Tidak ada rekaman fold {fold} di tabel.")
    for c in (1, 0):
        if int((pool.label_ami == c).sum()) < n_per_class:
            raise SystemExit(f"Rekaman label {c} kurang dari {n_per_class}.")
    return pd.concat([pool[pool.label_ami == c].sample(n_per_class, random_state=seed) for c in (1, 0)])
