"""Kunci threshold ketiga di FOLD 18: spesifisitas setara M0 (rencana analisis fold 19, Bagian 4 dan 6).

Jalankan SEKALI dari root repo, sebelum rencana analisis FINAL:
    python -m src.evaluation.lock_m0_threshold

Aturan:
  - Target = spesifisitas M0 (definisi primer, aturan M0-v2) pada berkas predictions/M0_fold18.parquet utuh.
  - Untuk setiap model M1-M6: threshold TERKECIL di fold 18 yang memberi spesifisitas >= target
    (aturan yang sama dengan rule-in, core.threshold_for_specificity), dihitung pada berkas fold 18 utuh
    model itu (sama seperti locked_fold18.json).
  - Positif bila y_prob mentah >= threshold. Threshold diterapkan ke fold 19 tanpa diubah.
Keluaran: reports/locked_fold18_m0.json (hanya angka agregat). reports/locked_fold18.json TIDAK disentuh.
Fold 19 tidak dibaca. Berkas yang sudah ada tidak ditimpa (penguncian sekali jalan).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.constants import FOLD_THRESHOLD
from src.evaluation import lock
from src.evaluation import predictions as P
from src.evaluation.core import binary_metrics, threshold_for_specificity

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "locked_fold18_m0.json"
SCORE_MODELS = ["M1", "M2", "M3", "M4", "M5", "M6"]
RULE = ("threshold terkecil di fold 18 dengan spesifisitas >= spesifisitas M0 (primer) di fold 18; "
        "positif bila y_prob_raw >= threshold")


def build(models: list[str]) -> dict:
    m0 = P.load_predictions("M0", FOLD_THRESHOLD, pred_dir=P.PRED_DIR)
    y0, p0 = m0["y_true"].to_numpy(), m0["y_prob"].to_numpy()
    if not set(p0.tolist()) <= {0.0, 1.0}:
        raise SystemExit("M0_fold18 harus biner (y_prob 0/1).")
    b0 = binary_metrics(y0, p0, 0.5)
    target = b0["specificity"]
    from src.model.m0_rule import RULE_VERSION, rule_sha256
    out = {
        "fold": FOLD_THRESHOLD, "rule": RULE, "target_specificity": target,
        "m0": {"rule_version": RULE_VERSION, "rule_sha256": rule_sha256(), "n": int(len(m0)),
               "n_neg": int((y0 == 0).sum()), "specificity": target, "sensitivity": b0["sensitivity"],
               "predictions_sha256": lock.sha256(P.pred_path("M0", FOLD_THRESHOLD, P.PRED_DIR))},
        "models": {},
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    for m in models:
        df = P.load_predictions(m, FOLD_THRESHOLD, pred_dir=P.PRED_DIR)
        y, p = df["y_true"].to_numpy(), df["y_prob"].to_numpy()
        thr = threshold_for_specificity(y, p, target)
        bm = binary_metrics(y, p, thr)
        if bm["specificity"] < target:
            raise SystemExit(f"{m}: spesifisitas {bm['specificity']:.4f} < target {target:.4f} (tidak seharusnya terjadi).")
        out["models"][m] = {"y_prob_raw": thr, "n": int(len(df)),
                            "predictions_sha256": lock.sha256(P.pred_path(m, FOLD_THRESHOLD, P.PRED_DIR)),
                            "specificity_fold18": bm["specificity"], "sensitivity_fold18": bm["sensitivity"]}
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Kunci threshold spesifisitas setara M0 di fold 18")
    ap.add_argument("--models", nargs="+", default=SCORE_MODELS, choices=SCORE_MODELS)
    a = ap.parse_args(argv)
    if OUT.exists():
        raise SystemExit(f"{OUT.name} sudah ada: threshold setara M0 hanya dikunci SATU kali. Jangan ditimpa; "
                         "bila memang harus diulang (sebelum rencana FINAL), hapus berkasnya secara sadar dan catat "
                         "di decision log.")
    res = build(a.models)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"[fold 18] target spesifisitas = spesifisitas M0 = {res['target_specificity']:.4f} "
          f"(sensitivitas M0 {res['m0']['sensitivity']:.3f})")
    for m, e in res["models"].items():
        print(f"  {m}: threshold y_prob >= {e['y_prob_raw']:.6g} | spes {e['specificity_fold18']:.4f} | "
              f"sens {e['sensitivity_fold18']:.3f} (in-sample)")
    print(f"Tersimpan: {OUT.relative_to(ROOT) if OUT.is_relative_to(ROOT) else OUT}")


if __name__ == "__main__":
    main()
