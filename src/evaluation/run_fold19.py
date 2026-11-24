"""Evaluasi akhir sekali jalan (fold 19), diuji dulu di fold 18 (rencana analisis fold 19, Bagian 4-10).

Jalankan dari root repo:
    python -m src.evaluation.run_fold19 --fold 18 --n-boot 200        # uji cepat kode (BUKAN hasil)
    python -m src.evaluation.run_fold19 --fold 18                     # uji lengkap, 2.000 bootstrap
    python -m src.evaluation.run_fold19 --checksums                   # sha256 prediksi fold 19 (decision log)
    python -m src.evaluation.run_fold19 --fold 19 --include-fold-19   # SATU KALI, setelah semua syarat terpenuhi

Yang dilakukan (sama untuk fold 18 dan 19; parameter TIDAK dipilih ulang):
  1. Baca predictions/{M0..M6}_fold{N}.parquet dan gabungkan pada kunjungan yang sama.
  2. Pakai kalibrasi Platt + threshold rule-out/rule-in dari reports/locked_fold18.json apa adanya,
     dan threshold spesifisitas setara M0 dari reports/locked_fold18_m0.json (src.evaluation.lock_m0_threshold).
     M0 biner: positif bila y_prob = 1 (tanpa kalibrasi/threshold).
  3. Metrik per model, perbandingan utama M5 vs M4, perbandingan eksploratif, perbandingan dengan M0
     (termasuk pada spesifisitas setara M0), subgrup, dan analisis sensitivitas, semuanya dengan bootstrap
     berpasangan per subject_id.
  4. Kurva kalibrasi fleksibel setelah Platt (src/evaluation/calibration_curve.py) dengan IK bootstrap.
Keluaran (hanya angka agregat; hitungan < 10 ditulis "<10"):
  fold 18: reports/eval_fold18_uji_final.md + .csv pendukung   (UJI KODE, bukan hasil)
  fold 19: reports/eval_fold19.md + .csv pendukung             (hasil akhir; menolak jalan kedua)
Fold 19 hanya bisa dibuka bila rencana analisis FINAL, locked_fold18.json, locked_fold18_m0.json, dan aturan M0
sudah di-commit (src/evaluation/lock.py).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import FOLD_TEST, FOLD_THRESHOLD
from src.evaluation import lock
from src.evaluation import predictions as P
from src.evaluation.bootstrap import paired_bootstrap
from src.evaluation.calibration_curve import calibration_curves
from src.evaluation.core import (
    PlattCalibrator, binary_metrics, brier, calibration_intercept_slope, calibration_table, summarize,
)
from src.model.metrics import pr_auc, roc_auc

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
LOCKED = REPORTS / "locked_fold18.json"
LOCKED_M0 = REPORTS / "locked_fold18_m0.json"          # threshold spesifisitas setara M0
LOCKED_M0_REL = Path("reports/locked_fold18_m0.json")
MODELS = ["M0", "M0S1", "M1", "M2", "M3", "M4", "M5", "M6"]
BINARY_MODELS = {"M0", "M0S1"}            # M0 primer; M0S1 = analisis sensitivitas S1 (m0_rule.py)
M0_SENS = ("M0S1", "M0")
PRIMARY = ("M5", "M4")
EXPLORATORY = [("M1", "M3"), ("M6", "M4"), ("M5", "M2"), ("M5", "M1")]
PRIMARY_REPEAT = {"rawat_inap", "tanpa_tipe2_murni", "tanpa_cc_eksplisit"}   # rencana Bagian 9
SUBGROUP_COLS = ["label_source", "ami_stemi", "ami_nstemi", "ami_type2_only", "cc_explicit_mi",
                 "is_male", "age", "medrecon_available"]
MIN_COUNT = 10
NAN = float("nan")


# ---------------------------------------------------------------------------
# Parameter terkunci
# ---------------------------------------------------------------------------
@dataclass
class Locked:
    cal: PlattCalibrator | None
    thr_out: float
    thr_in: float
    thr_m0: float | None = None          # spesifisitas setara M0 (fold 18); None bila tidak dievaluasi


def load_locked(models: list[str], fold: int) -> tuple[dict[str, Locked], dict]:
    if not LOCKED.exists():
        raise SystemExit("reports/locked_fold18.json tidak ada. Jalankan src.evaluation.run_fold18 dulu.")
    raw = json.loads(LOCKED.read_text(encoding="utf-8"))
    out, notes = {}, {}
    for m in models:
        if m in BINARY_MODELS:
            out[m] = Locked(None, 0.5, 0.5, 0.5)
            continue
        if m not in raw:
            raise SystemExit(f"{m} tidak ada di locked_fold18.json; kalibrasi/threshold {m} belum dikunci.")
        e = raw[m]
        out[m] = Locked(PlattCalibrator.from_dict(e["calibration"]), float(e["threshold_rule_out"]["y_prob_raw"]),
                        float(e["threshold_rule_in"]["y_prob_raw"]))
        f18 = P.pred_path(m, FOLD_THRESHOLD, P.PRED_DIR)
        same = f18.exists() and lock.sha256(f18) == e["predictions_sha256"]
        notes[m] = "cocok" if same else "BERBEDA/TIDAK ADA"
        if fold == FOLD_THRESHOLD and not same:
            raise SystemExit(f"predictions/{m}_fold18.parquet berbeda dari berkas saat threshold dikunci "
                             "(sha256 tidak cocok). Jalankan ulang run_fold18 untuk model ini dulu.")
    score = [m for m in models if m not in BINARY_MODELS]
    if "M0" in models and score:
        load_locked_m0(out, score, fold, notes)
    return out, notes


def load_locked_m0(out: dict[str, Locked], score: list[str], fold: int, notes: dict) -> dict:
    """Isi Locked.thr_m0 dari locked_fold18_m0.json dan cek sha256 berkas fold 18 (M0 dan model)."""
    if not LOCKED_M0.exists():
        raise SystemExit("reports/locked_fold18_m0.json tidak ada. Jalankan src.evaluation.lock_m0_threshold dulu "
                         "(sekali, di fold 18, sebelum rencana analisis FINAL).")
    raw = json.loads(LOCKED_M0.read_text(encoding="utf-8"))
    checks = {"M0": raw["m0"]["predictions_sha256"]}
    for m in score:
        if m not in raw["models"]:
            raise SystemExit(f"{m} tidak ada di locked_fold18_m0.json.")
        out[m].thr_m0 = float(raw["models"][m]["y_prob_raw"])
        checks[m] = raw["models"][m]["predictions_sha256"]
    for m, want in checks.items():
        f18 = P.pred_path(m, FOLD_THRESHOLD, P.PRED_DIR)
        same = f18.exists() and lock.sha256(f18) == want
        notes[f"{m}@spes-M0"] = "cocok" if same else "BERBEDA/TIDAK ADA"
        if fold == FOLD_THRESHOLD and not same:
            raise SystemExit(f"predictions/{m}_fold18.parquet berbeda dari berkas saat threshold setara M0 dikunci "
                             "(sha256 tidak cocok).")
    out["_m0_target"] = Locked(None, raw["target_specificity"], raw["target_specificity"])
    return raw


# ---------------------------------------------------------------------------
# Metrik untuk bootstrap (cepat; validasi input sudah dilakukan saat berkas dibaca)
# ---------------------------------------------------------------------------
def _sens(y, pred):
    pos = y == 1
    return float(pred[pos].mean()) if pos.any() else NAN


def _spec(y, pred):
    neg = y == 0
    return float((~pred[neg]).mean()) if neg.any() else NAN


def _ppv(y, pred):
    return float(y[pred].mean()) if pred.any() else NAN


def _npv(y, pred):
    return float(1.0 - y[~pred].mean()) if (~pred).any() else NAN


def metric_fns(L: dict[str, Locked]) -> dict:
    def thr(m, which):
        return L[m].thr_out if which == "out" else L[m].thr_in

    fns = {
        "AUROC": lambda y, p, m: NAN if m in BINARY_MODELS else roc_auc(y, p),
        "AUPRC": lambda y, p, m: NAN if m in BINARY_MODELS else pr_auc(y, p),
        "Brier_kal": lambda y, p, m: NAN if m in BINARY_MODELS else float(np.mean((L[m].cal.transform(p) - y) ** 2)),
    }
    for which in ("out", "in"):
        for name, f in (("Sens", _sens), ("Spes", _spec), ("PPV", _ppv), ("NPV", _npv)):
            fns[f"{name}@rule-{which}"] = (lambda f, w: lambda y, p, m: f(y, p >= thr(m, w)))(f, which)
    if any(isinstance(v, Locked) and v.thr_m0 is not None for k, v in L.items() if not k.startswith("_")):
        for name, f in (("Sens", _sens), ("Spes", _spec), ("PPV", _ppv), ("NPV", _npv)):
            fns[f"{name}@spes-M0"] = (lambda f: lambda y, p, m: NAN if L[m].thr_m0 is None
                                      else f(y, p >= L[m].thr_m0))(f)
    return fns


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_wide(models: list[str], fold: int) -> tuple[pd.DataFrame, dict, dict]:
    frames = {m: P.load_predictions(m, fold, pred_dir=P.PRED_DIR, allow_test=(fold == FOLD_TEST)) for m in models}
    wide, info = P.align(frames)
    from src.model import train_tabular as T          # impor lambat: tidak memuat xgboost
    master, minfo = T.load_master()
    missing_cols = [c for c in SUBGROUP_COLS if c not in master.columns]
    if missing_cols:
        raise SystemExit(f"Kolom subgrup tidak ada di tabel master: {missing_cols}")
    sub = master[["ed_stay_id", "label_ami"] + SUBGROUP_COLS]
    wide = wide.merge(sub, on="ed_stay_id", how="left", validate="one_to_one", indicator=True)
    if (wide["_merge"] != "both").any():
        raise SystemExit(f"{int((wide['_merge'] != 'both').sum())} kunjungan prediksi tidak ada di tabel master.")
    if not (wide["label_ami"].astype(int) == wide["y_true"].astype(int)).all():
        raise SystemExit("y_true di berkas prediksi tidak sama dengan label_ami di tabel master.")
    wide = wide.drop(columns=["_merge", "label_ami"])
    return wide, info, minfo


def subgroup_masks(w: pd.DataFrame):
    """(nama, label, mask). Subtipe: semua negatif + positif subtipe itu (rencana Bagian 9)."""
    y = w["y_true"].to_numpy()
    age = pd.to_numeric(w["age"], errors="coerce")
    yield "rawat_inap", "Hanya pasien rawat inap (label_source = hosp) [wajib]", (w["label_source"] == "hosp").to_numpy()
    yield "tanpa_tipe2_murni", "Tanpa IMA tipe 2 murni", (w["ami_type2_only"] != 1).to_numpy()
    yield "tanpa_cc_eksplisit", "Tanpa keluhan yang menyebut diagnosis", (w["cc_explicit_mi"] != 1).to_numpy()
    yield "tipe2_murni", "IMA tipe 2 murni (positif tipe 2 murni + semua negatif)", \
        ((y == 0) | ((y == 1) & (w["ami_type2_only"] == 1))).to_numpy()
    yield "stemi", "STEMI (positif STEMI + semua negatif)", ((y == 0) | ((y == 1) & (w["ami_stemi"] == 1))).to_numpy()
    yield "nstemi", "NSTEMI (positif NSTEMI + semua negatif)", ((y == 0) | ((y == 1) & (w["ami_nstemi"] == 1))).to_numpy()
    yield "laki_laki", "Laki-laki", (w["is_male"] == 1).to_numpy()
    yield "perempuan", "Perempuan", (w["is_male"] == 0).to_numpy()
    yield "usia_lt65", "Usia < 65 tahun", (age < 65).fillna(False).to_numpy()
    yield "usia_ge65", "Usia >= 65 tahun", (age >= 65).fillna(False).to_numpy()
    yield "obat_ada", "Catatan obat rutin tersedia", (w["medrecon_available"] == 1).to_numpy()
    yield "obat_tidak", "Catatan obat rutin tidak tersedia", (w["medrecon_available"] == 0).to_numpy()


# ---------------------------------------------------------------------------
# Evaluasi
# ---------------------------------------------------------------------------
def default_pairs(models: list[str]) -> list[tuple[str, str]]:
    return [p for p in [PRIMARY] + EXPLORATORY if p[0] in models and p[1] in models]


def evaluate(wide: pd.DataFrame, models: list[str], L: dict[str, Locked], pairs, n_boot: int, seed: int) -> dict:
    y = wide["y_true"].to_numpy().astype(np.int64)
    probs = {m: wide[f"p_{m}"].to_numpy() for m in models}
    subj = wide["subject_id"].to_numpy()
    fns = metric_fns(L)
    score_models = [m for m in models if m not in BINARY_MODELS]
    vs_m0 = [(m, "M0") for m in score_models] if "M0" in models else []
    if all(m in models for m in M0_SENS):
        vs_m0.append(M0_SENS)
    print(f"  bootstrap keseluruhan {n_boot}x ...", flush=True)
    boot = paired_bootstrap(y, probs, subj, fns, n_boot=n_boot, seed=seed, diffs=list(pairs) + vs_m0)

    point = {}
    for m in models:
        p = probs[m]
        if m in BINARY_MODELS:
            point[m] = {"n": len(y), "n_pos": int(y.sum()), "prevalence": float(y.mean()),
                        "m_out": binary_metrics(y, p, 0.5), "m_in": None, "m_m0": None}
            continue
        pc = L[m].cal.transform(p)
        raw = summarize(y, p)
        cal = calibration_intercept_slope(y, pc)
        point[m] = {**raw, "brier_kal": brier(y, pc), "cal_intercept_kal": cal["calibration_intercept"],
                    "cal_slope_kal": cal["calibration_slope"], "m_out": binary_metrics(y, p, L[m].thr_out),
                    "m_in": binary_metrics(y, p, L[m].thr_in), "cal_table": calibration_table(y, pc),
                    "m_m0": None if L[m].thr_m0 is None else binary_metrics(y, p, L[m].thr_m0)}

    subgroups = []
    sub_fns = {k: fns[k] for k in ("AUROC", "Sens@rule-out", "Spes@rule-out")}
    for name, label, mask in subgroup_masks(wide):
        ys = y[mask]
        n, n_pos = int(mask.sum()), int(ys.sum())
        row = {"analisis": name, "label": label, "n": n, "n_pos": n_pos, "boot": None}
        if n_pos >= MIN_COUNT and (n - n_pos) >= MIN_COUNT:
            diffs = [PRIMARY] if (name in PRIMARY_REPEAT and PRIMARY in pairs) else []
            print(f"  bootstrap subgrup {name} {n_boot}x ...", flush=True)
            row["boot"] = paired_bootstrap(ys, {m: probs[m][mask] for m in models}, subj[mask], sub_fns,
                                           n_boot=n_boot, seed=seed, diffs=diffs)
        subgroups.append(row)
    switch = None
    if all(m in models for m in M0_SENS):
        moved = (probs["M0"] >= 0.5) & (probs["M0S1"] < 0.5)          # S1 bagian dari primer: hanya 1 -> 0
        pos = y == 1
        switch = {"semua": int(moved.sum()), "positif": int((moved & pos).sum()), "negatif": int((moved & ~pos).sum()),
                  **{k: int((moved & pos & (wide[c] == 1).to_numpy()).sum())
                     for k, c in (("positif STEMI", "ami_stemi"), ("positif NSTEMI", "ami_nstemi"),
                                  ("positif tipe 2 murni", "ami_type2_only"))}}
    print(f"  kurva kalibrasi fleksibel {n_boot}x ...", flush=True)
    curves = calibration_curves(y, {m: L[m].cal.transform(probs[m]) for m in score_models}, subj,
                                n_boot=n_boot, seed=seed) if score_models else []
    return {"boot": boot, "point": point, "subgroups": subgroups, "n": len(y), "n_pos": int(y.sum()),
            "m0_switch": switch, "cal_curves": curves,
            "m0_target": L["_m0_target"].thr_out if "_m0_target" in L else None}


def sanity_fold18(L: dict[str, Locked], models: list[str]) -> list[str]:
    """Di fold 18, parameter terkunci harus memberi sens >= 0,90 dan spes >= 0,90 (in-sample) dan
    kalibrasi sempurna setelah Platt. Dihitung pada berkas fold 18 MASING-MASING model secara utuh
    (sama seperti saat dikunci di run_fold18), bukan pada irisan kunjungan bersama, dan dijalankan
    SEBELUM bootstrap. Bila gagal, berkas atau parameter tidak cocok."""
    msgs = []
    for m in models:
        if m in BINARY_MODELS or m.startswith("_"):
            continue
        df = P.load_predictions(m, FOLD_THRESHOLD, pred_dir=P.PRED_DIR)
        y, p = df["y_true"].to_numpy(), df["y_prob"].to_numpy()
        so = binary_metrics(y, p, L[m].thr_out)["sensitivity"]
        si = binary_metrics(y, p, L[m].thr_in)["specificity"]
        c = calibration_intercept_slope(y, L[m].cal.transform(p))
        ci, cs = c["calibration_intercept"], c["calibration_slope"]
        ok = so >= 0.90 and si >= 0.90 and abs(ci) < 0.01 and abs(cs - 1) < 0.01
        extra = ""
        if L[m].thr_m0 is not None:
            target = L["_m0_target"].thr_out
            sm = binary_metrics(y, p, L[m].thr_m0)["specificity"]
            ok = ok and sm >= target
            extra = f", spes@setara-M0 {sm:.4f} (target {target:.4f})"
        msgs.append(f"{m}: sens rule-out {so:.3f}, spes rule-in {si:.3f}, intercept/slope setelah Platt "
                    f"{ci:.4f}/{cs:.4f}{extra} -> {'OK' if ok else 'GAGAL'}")
        if not ok:
            raise SystemExit("Uji fold 18 GAGAL: parameter terkunci tidak cocok dengan prediksi fold 18.\n" + "\n".join(msgs))
    return msgs


# ---------------------------------------------------------------------------
# Laporan
# ---------------------------------------------------------------------------
def cnt(n) -> str:
    return f"{int(n):,}" if int(n) >= MIN_COUNT else f"<{MIN_COUNT}"


def f3(x) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def ci_str(b: pd.DataFrame, target: str, metric: str) -> str:
    r = b[(b["target"] == target) & (b["metric"] == metric)]
    if r.empty or pd.isna(r["estimate"].iloc[0]):
        return "–"
    r = r.iloc[0]
    return f"{f3(r['estimate'])} [{f3(r['ci_low'])}, {f3(r['ci_high'])}]"


def suppress_parts(parts: dict[str, int]) -> dict[str, str]:
    """Sel < 10 ditulis "<10". Bila tepat satu bagian < 10, bagian lain yang terkecil ikut disembunyikan agar
    sel kecil tidak bisa dihitung dari total yang dilaporkan (supresi komplementer)."""
    out = {k: cnt(v) for k, v in parts.items()}
    small = [k for k, v in parts.items() if v < MIN_COUNT]
    rest = sorted((v, k) for k, v in parts.items() if v >= MIN_COUNT)
    if len(small) == 1 and rest:
        out[rest[0][1]] = "disembunyikan"
    return out


def md_table(rows: list[dict]) -> str:
    if not rows:
        return "_(tidak ada)_"
    cols = list(rows[0])
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def comparison_rows(b: pd.DataFrame, pairs, metrics) -> list[dict]:
    rows = []
    for a, c in pairs:
        for k in metrics:
            r = b[(b["target"] == f"{a} - {c}") & (b["metric"] == k)]
            if r.empty or pd.isna(r["estimate"].iloc[0]):
                continue
            r = r.iloc[0]
            rows.append({"pasangan": f"{a} - {c}", "metrik": k, "selisih": f3(r["estimate"]),
                         "IK 95%": f"[{f3(r['ci_low'])}, {f3(r['ci_high'])}]",
                         "IK tidak mencakup 0": "ya" if bool(r["ci_excludes_0"]) else "tidak",
                         "iterasi sah": int(r["n_valid"])})
    return rows


def build_report(res: dict, models, pairs, fold: int, meta: dict) -> tuple[str, dict[str, pd.DataFrame]]:
    b, pt = res["boot"], res["point"]
    score = [m for m in models if m not in BINARY_MODELS]
    has_m0 = res.get("m0_target") is not None
    head = ["# Evaluasi akhir fold 19" if fold == FOLD_TEST else "# UJI KODE evaluasi akhir di fold 18", "",
            "Dihasilkan otomatis oleh `src/evaluation/run_fold19.py`. Hanya angka agregat; hitungan < 10 ditulis \"<10\".", ""]
    if fold == FOLD_THRESHOLD:
        head += ["> **UJI KODE, BUKAN HASIL.** Kalibrasi dan threshold dipilih di fold 18 yang sama, jadi semua metrik "
                 "bergantung threshold dan kalibrasi di sini bersifat in-sample. Tujuannya hanya memastikan skrip "
                 "berjalan benar sebelum fold 19 dibuka.", ""]
    if meta.get("posthoc"):
        head += ["> **POST-HOC.** Evaluasi fold 19 sudah pernah dijalankan; laporan ini analisis ulang dan dilaporkan "
                 "terpisah dari hasil utama.", ""]
    head += [f"- Dibuat: {meta['created']} | bootstrap {meta['n_boot']}x per subject_id, seed {meta['seed']}",
             f"- Tabel master: `{meta['master_file']}` | filter sinyal: {meta['signal_filter']}",
             f"- Kunjungan dinilai bersama: {res['n']:,} | positif: {cnt(res['n_pos'])} | prevalensi "
             f"{res['n_pos'] / res['n'] * 100:.2f}% | per model: {meta['n_per_model']} | dibuang: {meta['dropped']}"]
    if meta.get("lock"):
        head.append(f"- Penguncian: rencana analisis `{meta['lock']['plan_commit']}`, locked_fold18.json "
                    f"`{meta['lock']['locked_fold18_commit']}`, aturan M0 `{meta['lock'].get('m0_rule_commit', '-')}`, "
                    f"locked_fold18_m0.json `{meta['lock'].get('m0_threshold_commit', '-')}`")
    if meta.get("m0_rule"):
        head.append(f"- Aturan M0: {meta['m0_rule']}")
    head += ["- SHA-256 berkas prediksi:", *[f"  - {m}: `{s}`" for m, s in meta["sha"].items()], ""]

    per_model = []
    for m in models:
        p = pt[m]
        if m in BINARY_MODELS:
            per_model.append({"model": m, "AUROC": "–", "AUPRC": "–", "Brier mentah": "–", "Brier kalibrasi": "–",
                              "intercept/slope kalibrasi": "–",
                              "rule-out: sens / spes / PPV / NPV": " / ".join(
                                  ci_str(b, m, f"{k}@rule-out") for k in ("Sens", "Spes", "PPV", "NPV")),
                              "rule-in: sens / spes / PPV / NPV": "(biner; sama dengan rule-out)",
                              **({"spes setara M0: sens / spes / PPV / NPV": "(titik operasi M0)"} if has_m0 else {})})
            continue
        per_model.append({
            "model": m, "AUROC": ci_str(b, m, "AUROC"), "AUPRC": ci_str(b, m, "AUPRC"), "Brier mentah": f3(p["brier"]),
            "Brier kalibrasi": ci_str(b, m, "Brier_kal"),
            "intercept/slope kalibrasi": f"{f3(p['cal_intercept_kal'])} / {f3(p['cal_slope_kal'])}",
            "rule-out: sens / spes / PPV / NPV": " / ".join(ci_str(b, m, f"{k}@rule-out") for k in ("Sens", "Spes", "PPV", "NPV")),
            "rule-in: sens / spes / PPV / NPV": " / ".join(ci_str(b, m, f"{k}@rule-in") for k in ("Sens", "Spes", "PPV", "NPV")),
            **({"spes setara M0: sens / spes / PPV / NPV": " / ".join(
                ci_str(b, m, f"{k}@spes-M0") for k in ("Sens", "Spes", "PPV", "NPV"))} if has_m0 else {}),
        })
    counts = []
    for m in models:
        for name, key in (("rule-out", "m_out"), ("rule-in", "m_in"), ("spes setara M0", "m_m0")):
            mm = pt[m].get(key)
            if mm is None:
                continue
            pos_cells = suppress_parts({"TP": mm["tp"], "FN": mm["fn"]})            # TP + FN = jumlah positif
            neg_cells = suppress_parts({"FP": mm["fp"], "TN": mm["tn"]})            # FP + TN = jumlah negatif
            counts.append({"model": m, "threshold": name if m not in BINARY_MODELS else "biner",
                           "y_prob mentah >=": "1 (biner)" if m in BINARY_MODELS else f"{mm['threshold']:.6g}",
                           "TP": pos_cells["TP"], "FP": neg_cells["FP"], "FN": pos_cells["FN"], "TN": neg_cells["TN"],
                           "F1": f3(mm["f1"])})

    support = ["AUROC", "AUPRC", "Brier_kal"] + [f"{k}@rule-{w}" for w in ("out", "in") for k in ("Sens", "Spes", "PPV", "NPV")]
    primary = [p for p in pairs if p == PRIMARY]
    explor = [p for p in pairs if p != PRIMARY]
    vs_m0 = [(m, "M0") for m in score if "M0" in models]
    if res.get("m0_switch") is not None:
        sw = res["m0_switch"]
        m0_sens = [
            "**Definisi M0: primer (M0) vs analisis sensitivitas S1 (M0S1)**", "",
            "M0S1 tidak menghitung pernyataan injury yang juga menyebut iskemia/perikarditis/repolarisasi dan tidak "
            "menghitung acute + iskemia. Setiap kunjungan M0S1 = 1 juga M0 = 1, jadi perpindahan hanya dari 1 ke 0.", "",
            md_table([{"kunjungan yang pindah M0 = 1 -> M0S1 = 0": k, "jumlah": v} for k, v in
                      {"semua": cnt(sw["semua"]),
                       **suppress_parts({k: sw[k] for k in ("positif", "negatif")}),
                       **suppress_parts({k: sw[k] for k in ("positif STEMI", "positif NSTEMI", "positif tipe 2 murni")}),
                       }.items()]),
            "", md_table(comparison_rows(b, [M0_SENS], ["Sens@rule-out", "Spes@rule-out", "PPV@rule-out", "NPV@rule-out"])),
            "",
        ]
    else:
        m0_sens = []

    lines = head + [
        "## 1. Kinerja setiap model", "",
        "Estimasi [IK 95% bootstrap per subject_id]. Brier mentah dan intercept/slope kalibrasi tanpa IK. "
        "Threshold dan kalibrasi dari `reports/locked_fold18.json`; prediksi positif bila y_prob mentah >= threshold.", "",
        md_table(per_model), "", "**Jumlah sel pada threshold terkunci**", "", md_table(counts), "",
        "## 2. Perbandingan utama", "",
        "M5 - M4. Metrik utama: ΔAUROC; bermakna bila IK 95% selisih tidak mencakup 0. Metrik lain pendukung.", "",
        md_table(comparison_rows(b, primary, support)) if primary else "_M5 atau M4 tidak dievaluasi._", "",
        "## 3. Perbandingan eksploratif (tanpa koreksi multiplisitas)", "",
        md_table(comparison_rows(b, explor, ["AUROC", "AUPRC"])), "",
        "## 4. Dibandingkan dengan M0 (eksploratif)", "",
        "Selisih sensitivitas dan spesifisitas model pada threshold terkuncinya dikurangi M0 (biner).", "",
        md_table(comparison_rows(b, vs_m0, ["Sens@rule-out", "Spes@rule-out", "Sens@rule-in", "Spes@rule-in"])), "",
        *([f"**Pada spesifisitas setara M0** (threshold dikunci di fold 18: spesifisitas model >= spesifisitas M0 "
           f"di fold 18 = {res['m0_target']:.4f}; `reports/locked_fold18_m0.json`)", "",
           md_table(comparison_rows(b, vs_m0, ["Sens@spes-M0", "Spes@spes-M0", "PPV@spes-M0", "NPV@spes-M0"])), ""]
          if has_m0 else []),
        *m0_sens,
        "## 5. Subgrup dan analisis sensitivitas", "",
        "AUROC, sensitivitas, dan spesifisitas pada threshold rule-out terkunci. Subgrup dengan positif atau negatif "
        "< 10 hanya dilaporkan jumlahnya. M5 - M4 diulang hanya pada tiga analisis yang direncanakan.", "",
    ]
    sub_rows, sub_diff = [], []
    for s in res["subgroups"]:
        if s["boot"] is None:
            sub_rows.append({"analisis": s["label"], "kunjungan": cnt(s["n"]), "positif": cnt(s["n_pos"]),
                             "model": "–", "AUROC": "–", "sens rule-out": "–", "spes rule-out": "–"})
            continue
        for m in models:
            sub_rows.append({"analisis": s["label"], "kunjungan": cnt(s["n"]), "positif": cnt(s["n_pos"]), "model": m,
                             "AUROC": ci_str(s["boot"], m, "AUROC"), "sens rule-out": ci_str(s["boot"], m, "Sens@rule-out"),
                             "spes rule-out": ci_str(s["boot"], m, "Spes@rule-out")})
        for r in comparison_rows(s["boot"], [PRIMARY] if s["analisis"] in PRIMARY_REPEAT else [], ["AUROC"]):
            sub_diff.append({"analisis": s["label"], **r})
    lines += [md_table(sub_rows), "", "**M5 - M4 pada analisis yang direncanakan**", "", md_table(sub_diff), "",
              "## 6. Kalibrasi setelah Platt terkunci", "",
              "Kurva kalibrasi fleksibel (regresi logistik restricted cubic spline pada logit p, 4 knot; IK 95% "
              f"bootstrap per subject_id, {meta['n_boot']}x) disimpan di `{meta['stem']}_calibration_curve.csv` "
              "(grid 50 titik antara persentil 1 dan 99 prediksi). Tabel di bawah: 10 bin berisi jumlah kunjungan sama "
              f"(juga di `{meta['stem']}_calibration_bins.csv`).", ""]
    for m in score:
        ct = [{"bin": r["bin"], "n": cnt(r["n"]), "rata-rata prediksi": f"{r['mean_pred']:.4f}",
               "proporsi teramati": f"{r['observed']:.4f}"} for r in pt[m]["cal_table"]]
        lines += [f"**{m}**", "", md_table(ct), ""]

    tables = {
        "metrics": b[~b["target"].str.contains(" - ", regex=False)].assign(
            estimate=lambda d: d["estimate"].astype(float)),
        "comparisons": b[b["target"].str.contains(" - ", regex=False)],
        "counts": pd.DataFrame(counts),
        "subgroups": pd.concat([s["boot"].assign(analisis=s["analisis"], n=cnt(s["n"]), n_pos=cnt(s["n_pos"]))
                                for s in res["subgroups"] if s["boot"] is not None], ignore_index=True)
        if any(s["boot"] is not None for s in res["subgroups"]) else pd.DataFrame(),
    }
    tables["calibration_curve"] = pd.DataFrame(res.get("cal_curves", []))
    tables["calibration_bins"] = pd.DataFrame(
        [{"model": m, "bin": r["bin"], "n": cnt(r["n"]), "mean_pred": r["mean_pred"], "observed": r["observed"]}
         for m in score for r in pt[m]["cal_table"]])
    extra = []
    for m in score:
        p = pt[m]
        for k, v in (("Brier_mentah", p["brier"]), ("intercept_mentah", p["calibration_intercept"]),
                     ("slope_mentah", p["calibration_slope"]), ("intercept_kalibrasi", p["cal_intercept_kal"]),
                     ("slope_kalibrasi", p["cal_slope_kal"])):
            extra.append({"target": m, "metric": k, "estimate": v, "ci_low": NAN, "ci_high": NAN, "n_valid": 0})
    tables["metrics"] = pd.concat([tables["metrics"], pd.DataFrame(extra)], ignore_index=True)
    return "\n".join(lines), tables


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def print_checksums(models) -> None:
    print("SHA-256 prediksi fold 19 (catat di decision log SEBELUM evaluasi):")
    for m, s in lock.checksums(P.PRED_DIR, FOLD_TEST, models).items():
        print(f"  {m}_fold19.parquet  {s or 'TIDAK ADA'}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Evaluasi akhir (fold 19), diuji dulu di fold 18")
    ap.add_argument("--fold", type=int, choices=[FOLD_THRESHOLD, FOLD_TEST])
    ap.add_argument("--include-fold-19", action="store_true")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--checksums", action="store_true", help="cetak sha256 prediksi fold 19 lalu berhenti")
    ap.add_argument("--overwrite-post-hoc", action="store_true",
                    help="jalankan ulang fold 19 sebagai analisis POST-HOC (laporan terpisah)")
    a = ap.parse_args(argv)
    models = [m for m in MODELS if m in a.models]
    if a.checksums:
        print_checksums(models)
        return
    if a.fold is None:
        ap.error("--fold wajib (18 untuk uji kode, 19 untuk evaluasi akhir)")

    meta = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "n_boot": a.n_boot, "seed": a.seed}
    stem = "eval_fold18_uji_final"
    if a.fold == FOLD_TEST:
        if not a.include_fold_19:
            raise SystemExit("Fold 19 dikunci (brief 11.5). Tambahkan --include-fold-19 hanya untuk evaluasi akhir.")
        meta["lock"] = lock.require_fold19_unlocked(lock.ROOT, need_m0_rule=bool(BINARY_MODELS & set(models)))
        if "M0" in models and any(m not in BINARY_MODELS for m in models):
            meta["lock"]["m0_threshold_commit"] = lock.committed_unchanged(LOCKED_M0_REL, lock.ROOT)
        stem = "eval_fold19"
        if (REPORTS / f"{stem}.md").exists():
            if not a.overwrite_post_hoc:
                raise SystemExit("reports/eval_fold19.md sudah ada: evaluasi fold 19 hanya dijalankan SATU kali. "
                                 "Analisis ulang hanya dengan --overwrite-post-hoc dan dicatat sebagai post-hoc.")
            meta["posthoc"] = True
            stem = f"eval_fold19_posthoc_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    meta["stem"] = stem
    L, lock_notes = load_locked(models, a.fold)
    print(f"[fold {a.fold}] model: {', '.join(models)} | parameter terkunci: {lock_notes}")
    wide, info, minfo = load_wide(models, a.fold)
    meta.update({"master_file": minfo["master_file"], "signal_filter": minfo["signal_filter"],
                 "n_per_model": info["n_per_model"], "dropped": info["dropped_per_model"],
                 "sha": lock.checksums(P.PRED_DIR, a.fold, models)})
    if BINARY_MODELS & set(models):
        from src.model.m0_rule import RULE_VERSION, rule_sha256
        meta["m0_rule"] = f"{RULE_VERSION}, sha256 `{rule_sha256()[:16]}...`"
    if any(info["dropped_per_model"].values()):
        print(f"  PERINGATAN: ada kunjungan yang tidak dimiliki semua model: {info['dropped_per_model']}")
    if a.fold == FOLD_THRESHOLD:
        print("\nCek parameter terkunci di fold 18 (harus OK semua):")
        for msg in sanity_fold18(L, models):
            print("  " + msg)
    pairs = default_pairs(models)
    print(f"\nBootstrap {a.n_boot}x (berpasangan per subject_id, seed {a.seed}) ...")
    res = evaluate(wide, models, L, pairs, a.n_boot, a.seed)

    text, tables = build_report(res, models, pairs, a.fold, meta)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / f"{stem}.md").write_text(text, encoding="utf-8")
    for name, df in tables.items():
        df.to_csv(REPORTS / f"{stem}_{name}.csv", index=False)

    b = res["boot"]
    print(f"\n=== RINGKASAN FOLD {a.fold}{' (UJI KODE, in-sample)' if a.fold == FOLD_THRESHOLD else ''} ===")
    print(f"kunjungan {res['n']:,} | positif {cnt(res['n_pos'])} | prevalensi {res['n_pos'] / res['n'] * 100:.2f}%")
    for m in models:
        if m in BINARY_MODELS:
            print(f"  {m}: sens {ci_str(b, m, 'Sens@rule-out')} | spes {ci_str(b, m, 'Spes@rule-out')}")
        else:
            print(f"  {m}: AUROC {ci_str(b, m, 'AUROC')} | AUPRC {ci_str(b, m, 'AUPRC')}"
                  + (f" | sens@spes setara M0 {ci_str(b, m, 'Sens@spes-M0')}" if res.get("m0_target") is not None else ""))
    if PRIMARY in pairs:
        print(f"  UTAMA M5 - M4: ΔAUROC {ci_str(b, 'M5 - M4', 'AUROC')}")
    print(f"\nTersimpan: reports/{stem}.md dan reports/{stem}_*.csv")


if __name__ == "__main__":
    main()
