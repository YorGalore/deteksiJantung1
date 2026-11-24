"""Gambar kurva kalibrasi (setelah Platt terkunci) dari keluaran AGREGAT run_fold19.

Jalankan dari root repo:
    python -m src.evaluation.plot_calibration --fold 18     # uji dengan keluaran UJI KODE fold 18
    python -m src.evaluation.plot_calibration --fold 19     # setelah evaluasi fold 19 dijalankan sekali

Masukan (dibuat oleh src/evaluation/run_fold19.py):
  reports/{stem}_calibration_curve.csv  kolom model, p_pred, p_obs, ci_low, ci_high, n_valid (kurva fleksibel)
  reports/{stem}_calibration_bins.csv   kolom model, bin, n, mean_pred, observed (10 bin, jumlah kunjungan sama)
  reports/{stem}.md                     judul (fold) dan prevalensi
Keluaran: reports/figures/kalibrasi_fold{N}.png (300 dpi). Default panel: M4 (pembanding) dan M5 (utama).

Skrip ini TIDAK membaca prediksi atau data per kunjungan dan tidak menghitung metrik baru.
Catatan: di fold 18 kalibrasi Platt dipilih di fold yang sama, jadi gambar fold 18 bersifat in-sample (uji kode).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import FOLD_TEST, FOLD_THRESHOLD
from src.evaluation.plot_forest import INK, GRID, LABELS, MUTED, REPORTS, STEMS, check_fold, koma, parse_header

MODELS = ["M1", "M2", "M3", "M4", "M5", "M6"]
DEFAULT = ["M4", "M5"]
CURVE_COLS = ["model", "p_pred", "p_obs", "ci_low", "ci_high", "n_valid"]
BIN_COLS = ["model", "bin", "n", "mean_pred", "observed"]
BAND = "#c9d1d9"


def load_csv(path: Path, cols: list[str], models: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"{path.name} tidak ada. Jalankan dulu src.evaluation.run_fold19 (versi dengan kurva kalibrasi).")
    df = pd.read_csv(path, dtype={"n": str} if "n" in cols else None)
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise SystemExit(f"Kolom tidak ada di {path.name}: {miss}")
    for m in models:
        if not (df["model"] == m).any():
            raise SystemExit(f"{m} tidak ada di {path.name}.")
    return df[df["model"].isin(models)].copy()


def plot(curve: pd.DataFrame, bins: pd.DataFrame, models: list[str], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    fig, axes = plt.subplots(1, len(models), figsize=(3.4 * len(models), 3.5), squeeze=False)
    for ax, m in zip(axes[0], models):
        c = curve[curve["model"] == m].sort_values("p_pred")
        b = bins[bins["model"] == m]
        lim = 1.08 * float(max(c["p_pred"].max(), b["mean_pred"].max(), np.nanmax(c["p_obs"]), b["observed"].max()))
        ax.plot([0, lim], [0, lim], color=MUTED, linestyle=(0, (4, 3)), linewidth=1.0)
        ax.fill_between(c["p_pred"], c["ci_low"].clip(upper=lim), c["ci_high"].clip(upper=lim), color=BAND,
                        linewidth=0)
        ax.plot(c["p_pred"], c["p_obs"], color=INK, linewidth=2.0)
        ax.plot(b["mean_pred"], b["observed"], "o", markersize=5, markerfacecolor="white", markeredgecolor=INK,
                markeredgewidth=1.2, zorder=3)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        ax.set_title(LABELS.get(m, m), fontsize=9.5, color=INK, loc="left",
                     fontweight="bold" if m in ("M4", "M5") else "normal")
        ax.set_xlabel("Probabilitas prediksi (setelah Platt)", fontsize=8.5, color=INK)
        fmt = matplotlib.ticker.FuncFormatter(lambda v, _: koma(v, 2))
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
            axis.set_major_formatter(fmt)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8, colors=INK, length=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
    axes[0][0].set_ylabel("Proporsi IMA teramati", fontsize=8.5, color=INK)
    fig.tight_layout(w_pad=2.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Kurva kalibrasi dari CSV agregat run_fold19")
    ap.add_argument("--fold", type=int, required=True, choices=[FOLD_THRESHOLD, FOLD_TEST])
    ap.add_argument("--stem", help="nama dasar laporan (default: eval_fold18_uji_final / eval_fold19)")
    ap.add_argument("--models", nargs="+", default=DEFAULT, choices=MODELS)
    ap.add_argument("--reports-dir", type=Path, default=REPORTS)
    ap.add_argument("--out", type=Path, help="berkas PNG (default: reports/figures/kalibrasi_fold{N}.png)")
    a = ap.parse_args(argv)

    stem = a.stem or STEMS[a.fold]
    models = [m for m in MODELS if m in a.models]
    md_path = a.reports_dir / f"{stem}.md"
    if not md_path.exists():
        raise SystemExit(f"{md_path.name} tidak ada. Jalankan dulu src.evaluation.run_fold19 untuk fold ini.")
    md_text = md_path.read_text(encoding="utf-8")
    posthoc = check_fold(md_text, a.fold)
    info = parse_header(md_text)
    curve = load_csv(a.reports_dir / f"{stem}_calibration_curve.csv", CURVE_COLS, models)
    bins = load_csv(a.reports_dir / f"{stem}_calibration_bins.csv", BIN_COLS, models)
    if curve[["p_pred", "p_obs"]].isna().any().any():
        raise SystemExit("Kurva kalibrasi berisi NaN pada estimasi titik; periksa keluaran run_fold19.")
    out = a.out or a.reports_dir / "figures" / f"kalibrasi_fold{a.fold}{'_posthoc' if posthoc else ''}.png"
    plot(curve, bins, models, out)

    print(f"[fold {a.fold}] {stem}{' (POST-HOC)' if posthoc else ''}: kunjungan {info['n']:,} | positif "
          f"{info['n_pos']} | prevalensi {koma(info['prevalence_pct'], 2)}%")
    for m in models:
        c = curve[curve["model"] == m]
        dev = np.abs(c["p_obs"] - c["p_pred"])
        print(f"  {m}: grid {koma(c['p_pred'].min(), 3)}–{koma(c['p_pred'].max(), 3)} | selisih maks "
              f"|teramati - prediksi| pada grid {koma(dev.max(), 3)} | iterasi bootstrap sah {int(c['n_valid'].min())}")
    print(f"Tersimpan: {out}")


if __name__ == "__main__":
    main()
