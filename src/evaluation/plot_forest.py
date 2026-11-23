"""Gambar 2 naskah: forest plot AUROC dan AUPRC per model, dari keluaran AGREGAT run_fold19.

Jalankan dari root repo:
    python -m src.evaluation.plot_forest --fold 18     # uji dengan keluaran UJI KODE fold 18
    python -m src.evaluation.plot_forest --fold 19     # setelah evaluasi fold 19 dijalankan sekali

Masukan (dibuat oleh src/evaluation/run_fold19.py; hanya angka agregat):
  reports/{stem}_metrics.csv   kolom target, metric, estimate, ci_low, ci_high, n_valid
  reports/{stem}.md            baris "Kunjungan dinilai bersama: ... | positif: ... | prevalensi ...%"
  stem = eval_fold18_uji_final (fold 18) atau eval_fold19 (fold 19); --stem untuk laporan lain (mis. post-hoc).
Keluaran: reports/figures/gambar2_forest_fold{N}.png (300 dpi).

Skrip ini TIDAK membaca berkas prediksi, tabel master, atau data per kunjungan, dan tidak menghitung
metrik baru. Angka di gambar sama persis dengan angka di CSV (rencana analisis fold 19, Bagian 12).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import FOLD_TEST, FOLD_THRESHOLD

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
STEMS = {FOLD_THRESHOLD: "eval_fold18_uji_final", FOLD_TEST: "eval_fold19"}
MODELS = ["M1", "M2", "M3", "M4", "M5", "M6"]          # M0/M0S1 biner: tidak punya AUROC/AUPRC
LABELS = {
    "M1": "M1  Sinyal EKG",
    "M2": "M2  Rekam medis IGD",
    "M3": "M3  Pengukuran mesin EKG",
    "M4": "M4  Rekam medis + pengukuran",
    "M5": "M5  Fusi M1 + M2",
    "M6": "M6  Fusi M3 + M2",
}
EMPHASIS = {"M5", "M4"}                                 # model utama dan pembanding utama (huruf tebal)
METRICS = (("AUROC", "AUROC"), ("AUPRC", "AUPRC"))
REQUIRED_COLS = ["target", "metric", "estimate", "ci_low", "ci_high", "n_valid"]
INK, MUTED, GRID = "#1f2328", "#6e7781", "#e6e8eb"
HEADER = re.compile(
    r"Kunjungan dinilai bersama:\s*([\d.,]+)\s*\|\s*positif:\s*([^|]+?)\s*\|\s*prevalensi\s*([\d.]+)%")


def koma(x: float, nd: int = 3) -> str:
    """Desimal koma sesuai naskah (0,883)."""
    return f"{x:.{nd}f}".replace(".", ",")


def parse_header(md_text: str) -> dict:
    """Ambil n, positif, dan prevalensi (%) dari baris ringkasan laporan .md run_fold19."""
    m = HEADER.search(md_text)
    if not m:
        raise SystemExit("Baris 'Kunjungan dinilai bersama: ... | positif: ... | prevalensi ...%' tidak ditemukan "
                         "di laporan .md. Pastikan .md berasal dari run_fold19.")
    return {"n": int(re.sub(r"[.,]", "", m.group(1))), "n_pos": m.group(2).strip(), "prevalence_pct": float(m.group(3))}


def check_fold(md_text: str, fold: int) -> bool:
    """Judul .md harus sesuai fold. Kembalikan True bila laporan berlabel post-hoc."""
    first = md_text.lstrip().splitlines()[0] if md_text.strip() else ""
    want = "# Evaluasi akhir fold 19" if fold == FOLD_TEST else "# UJI KODE evaluasi akhir di fold 18"
    if first.strip() != want:
        raise SystemExit(f"Judul laporan .md tidak sesuai fold {fold}: '{first.strip()}' (diharapkan '{want}').")
    return "**POST-HOC.**" in md_text


def load_points(csv_path: Path, models: list[str]) -> pd.DataFrame:
    """Baris AUROC/AUPRC setiap model dari CSV metrik. Berhenti bila ada yang hilang atau tidak valid."""
    if not csv_path.exists():
        raise SystemExit(f"{csv_path.name} tidak ada. Jalankan dulu src.evaluation.run_fold19 untuk fold ini.")
    df = pd.read_csv(csv_path)
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        raise SystemExit(f"Kolom tidak ada di {csv_path.name}: {missing_cols}")
    keep = df[df["target"].isin(models) & df["metric"].isin([k for k, _ in METRICS])].copy()
    rows = []
    for m in models:
        for k, _ in METRICS:
            r = keep[(keep["target"] == m) & (keep["metric"] == k)]
            if len(r) != 1:
                raise SystemExit(f"{m} {k}: ditemukan {len(r)} baris di {csv_path.name} (harus tepat 1).")
            r = r.iloc[0]
            est, lo, hi = (float(r[c]) for c in ("estimate", "ci_low", "ci_high"))
            if not all(np.isfinite([est, lo, hi])) or not (0.0 <= lo <= hi <= 1.0) or int(r["n_valid"]) <= 0:
                raise SystemExit(f"{m} {k}: nilai tidak valid (estimasi {est}, IK [{lo}, {hi}], "
                                 f"iterasi sah {r['n_valid']}).")
            if not (lo <= est <= hi):
                print(f"  catatan: {m} {k} estimasi {est:.3f} di luar IK persentil [{lo:.3f}, {hi:.3f}] "
                      "(bisa terjadi pada bootstrap persentil; tetap digambar apa adanya).")
            rows.append({"model": m, "metric": k, "estimate": est, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def plot(points: pd.DataFrame, info: dict, models: list[str], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    ypos = {m: len(models) - 1 - i for i, m in enumerate(models)}           # model pertama di atas
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 0.36 * len(models) + 1.1), sharey=True)
    for ax, (key, title) in zip(axes, METRICS):
        d = points[points["metric"] == key]
        for _, r in d.iterrows():
            y = ypos[r["model"]]
            ax.hlines(y, r["ci_low"], r["ci_high"], color=INK, linewidth=1.6)
            ax.plot(r["estimate"], y, "o", color=INK, markersize=6.5, markeredgecolor="white", markeredgewidth=1.0,
                    zorder=3)
        lo, hi = d["ci_low"].min(), d["ci_high"].max()
        if key == "AUPRC":
            prev = info["prevalence_pct"] / 100
            ax.axvline(prev, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.0)
            ax.text(prev, len(models) - 0.45, f" prevalensi {koma(info['prevalence_pct'], 2)}%", color=MUTED,
                    fontsize=7.5, va="bottom", ha="left")
            ax.set_xlim(0, hi + 0.08 * hi)
        else:
            pad = 0.08 * (hi - lo)
            ax.set_xlim(lo - pad, hi + pad)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: koma(v, 2)))
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8, colors=INK, length=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
        ax.set_ylim(-0.6, len(models) - 0.2)
    axes[0].set_yticks([ypos[m] for m in models])
    axes[0].set_yticklabels([LABELS.get(m, m) for m in models], fontsize=8.5)
    for t, m in zip(axes[0].get_yticklabels(), models):
        t.set_fontweight("bold" if m in EMPHASIS else "normal")
    fig.tight_layout(w_pad=2.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Gambar 2: forest plot AUROC/AUPRC dari CSV agregat run_fold19")
    ap.add_argument("--fold", type=int, required=True, choices=[FOLD_THRESHOLD, FOLD_TEST])
    ap.add_argument("--stem", help="nama dasar laporan (default: eval_fold18_uji_final / eval_fold19)")
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--reports-dir", type=Path, default=REPORTS)
    ap.add_argument("--out", type=Path, help="berkas PNG keluaran (default: reports/figures/gambar2_forest_fold{N}.png)")
    a = ap.parse_args(argv)

    stem = a.stem or STEMS[a.fold]
    models = [m for m in MODELS if m in a.models]
    md_path = a.reports_dir / f"{stem}.md"
    if not md_path.exists():
        raise SystemExit(f"{md_path.name} tidak ada. Jalankan dulu src.evaluation.run_fold19 untuk fold ini.")
    md_text = md_path.read_text(encoding="utf-8")
    posthoc = check_fold(md_text, a.fold)
    info = parse_header(md_text)
    points = load_points(a.reports_dir / f"{stem}_metrics.csv", models)
    out = a.out or a.reports_dir / "figures" / f"gambar2_forest_fold{a.fold}{'_posthoc' if posthoc else ''}.png"
    plot(points, info, models, out)

    print(f"[fold {a.fold}] {stem}{' (POST-HOC)' if posthoc else ''}: kunjungan {info['n']:,} | positif "
          f"{info['n_pos']} | prevalensi {koma(info['prevalence_pct'], 2)}%")
    for m in models:
        vals = []
        for k, _ in METRICS:
            r = points[(points["model"] == m) & (points["metric"] == k)].iloc[0]
            vals.append(f"{k} {koma(r['estimate'])} [{koma(r['ci_low'])}; {koma(r['ci_high'])}]")
        print(f"  {m}: " + " | ".join(vals))
    print(f"Tersimpan: {out}")


if __name__ == "__main__":
    main()
