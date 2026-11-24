"""Tes paket fold 19 Orang 3: aturan M0, penjaga penguncian, run_fold19, SHAP. Data SINTETIS saja.

Jalankan: python -m pytest -q tests/test_fold19.py
XGBoost (SHAP) hanya dijalankan di proses Python terpisah (macOS: torch + xgboost di satu proses bisa crash).
"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from src.constants import ANALYSIS_COLS, EMR_FEATURES, ID_COLS, LABEL_COLS, MM_FEATURES, SPLIT_COLS
from src.evaluation import lock, lock_m0_threshold, run_fold18, run_fold19
from src.evaluation import predictions as P
from src.evaluation.predictions import load_predictions, validate_predictions
from src.model import m0_rule
from src.model import train_tabular as T
from src.model.tabular import ARRIVAL_CATEGORIES

REPO = Path(__file__).resolve().parents[1]
HAS_XGB = importlib.util.find_spec("xgboost") is not None
HAS_SHAP = importlib.util.find_spec("shap") is not None


# ---------------------------------------------------------------------------
# Data sintetis
# ---------------------------------------------------------------------------
def synthetic_master(n=40000, seed=0) -> pd.DataFrame:
    """Skema sama dengan master_table_draft (65 kolom). Label bergantung pada beberapa fitur."""
    rng = np.random.default_rng(seed)
    subj = rng.integers(0, int(n * 0.6), n)
    fold = (subj * 7919) % 20
    m = pd.DataFrame({"ed_stay_id": np.arange(n, dtype="int64") + 30_000_000, "subject_id": subj.astype("int64"),
                      "study_id": np.arange(n, dtype="int64") + 40_000_000,
                      "file_name": pd.Series([f"files/p0/p{s}/s{i}/{i}" for i, s in enumerate(subj)], dtype="str"),
                      "fold": fold.astype("int64")})
    u16, u17 = np.unique(subj[fold <= 16]), np.unique(subj[fold <= 17])
    val16 = set(rng.choice(u16, size=int(0.1 * len(u16)), replace=False))
    val17 = set(rng.choice(u17, size=int(0.1 * len(u17)), replace=False))
    m["internal_val_0_16"] = (m.subject_id.isin(val16) & (m.fold <= 16)).astype("int8")
    m["internal_val_0_17"] = (m.subject_id.isin(val17) & (m.fold <= 17)).astype("int8")
    hr = rng.normal(85, 20, n)
    chest = (rng.random(n) < 0.18).astype("int8")
    qrs = rng.normal(95, 15, n)
    logit = -4.0 + 1.8 * chest + 0.025 * (hr - 85) + 0.03 * (qrs - 95) + rng.normal(0, 0.5, n)
    y = (rng.random(n) < expit(logit)).astype("int64")
    m["label_ami"] = y
    m["label_source"] = pd.Series(np.where((y == 1) | (rng.random(n) < 0.6), "hosp", "ed"), dtype="str")
    for c in LABEL_COLS[2:]:
        m[c] = np.zeros(n, dtype="int64")
    sub = rng.random(n)
    m["ami_stemi"] = ((y == 1) & (sub < 0.25)).astype("int64")
    m["ami_nstemi"] = ((y == 1) & (sub >= 0.25) & (sub < 0.9)).astype("int64")
    m["ami_type2"] = ((y == 1) & (sub >= 0.9)).astype("int64")
    m["ami_type2_only"] = m["ami_type2"]
    m["cc_explicit_mi"] = ((rng.random(n) < 0.02) | ((y == 1) & (rng.random(n) < 0.05))).astype("int8")
    m["cc_missing"] = (rng.random(n) < 0.01).astype("int8")
    m["medrecon_available"] = (rng.random(n) < 0.8).astype("int8")
    m["mm_missing_count"] = rng.integers(0, 3, n).astype("int16")
    m["first_ecg_matches_no0"] = 1
    m["signal_ok"] = pd.array([pd.NA] * n, dtype="Int8")
    m["age"] = rng.normal(62, 16, n)
    m["is_male"] = (rng.random(n) < 0.5).astype("int8")
    arr = rng.choice(ARRIVAL_CATEGORIES + [None], n, p=[0.35, 0.002, 0.05, 0.05, 0.5, 0.048])
    m["arrival_transport"] = pd.Series(arr, dtype="string")
    for c in ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain", "acuity"]:
        m[c] = rng.normal(50, 10, n)
    m["heartrate"] = np.where(rng.random(n) < 0.05, np.nan, hr)
    for c in [c for c in EMR_FEATURES if c.startswith("cc_")]:
        m[c] = (rng.random(n) < 0.05).astype("int8")
    m["cc_chest_pain"] = chest
    no_med = m["medrecon_available"].to_numpy() == 0
    for c in [c for c in EMR_FEATURES if c.startswith("med_")] + ["n_meds"]:
        v = (rng.random(n) < 0.2).astype(float) if c != "n_meds" else rng.integers(0, 15, n).astype(float)
        m[c] = np.where(no_med, np.nan, v)
    for c in MM_FEATURES:
        m[c] = rng.normal(100, 20, n)
    m["qrs_duration"] = qrs
    m["p_wave_absent"] = (rng.random(n) < 0.05).astype(float)
    return m[ID_COLS + SPLIT_COLS + LABEL_COLS + ANALYSIS_COLS + EMR_FEATURES + MM_FEATURES]


def synthetic_reports(master: pd.DataFrame, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(master)
    y = master["label_ami"].to_numpy()
    r = pd.DataFrame({"ed_stay_id": master["ed_stay_id"].to_numpy()})
    for i in range(18):
        r[f"report_{i}"] = pd.array([pd.NA] * n, dtype="string")
    r["report_0"] = pd.array(np.where(rng.random(n) < 0.97, "Sinus rhythm", None), dtype="string")
    acute = ((y == 1) & (rng.random(n) < 0.35)) | ((y == 0) & (rng.random(n) < 0.02))
    old = rng.random(n) < 0.08
    r["report_1"] = pd.array(np.where(acute, "*** ACUTE MI / STEMI ***",
                                      np.where(old, "Inferior infarct, age undetermined", None)), dtype="string")
    r["report_2"] = pd.array(np.where(rng.random(n) < 0.05, "Old MI, no acute changes", None), dtype="string")
    inj = ((y == 1) & (rng.random(n) < 0.15)) | ((y == 0) & (rng.random(n) < 0.02))
    r["report_3"] = pd.array(np.where(inj, "Inferior ST-T changes suggest myocardial injury/ischemia", None), dtype="string")
    return r


def write_preds(pred_dir: Path, master: pd.DataFrame, fold: int, seed=0):
    """Prediksi sintetis M0-M6 untuk satu fold; M5 sengaja paling baik."""
    rng = np.random.default_rng(seed + fold)
    d = master[master["fold"] == fold].reset_index(drop=True)
    y = d["label_ami"].to_numpy()
    s = np.where(y == 1, 1.0, 0.0)
    base = {"ed_stay_id": d["ed_stay_id"], "subject_id": d["subject_id"], "study_id": d["study_id"],
            "fold": fold, "y_true": y.astype("int8")}
    noise = {"M1": 1.0, "M2": 1.1, "M3": 1.6, "M4": 0.95, "M5": 0.7, "M6": 1.0}
    pred_dir.mkdir(parents=True, exist_ok=True)
    for m, sd in noise.items():
        p = expit(2.0 * s + rng.normal(0, sd, len(y)) - 1.0 + (2.5 if m in ("M1", "M2", "M3", "M4") else -2.0))
        pd.DataFrame({**base, "y_prob": p}).to_parquet(pred_dir / f"{m}_fold{fold}.parquet", index=False)
    m0 = ((y == 1) & (rng.random(len(y)) < 0.4)) | ((y == 0) & (rng.random(len(y)) < 0.03))
    pd.DataFrame({**base, "y_prob": m0.astype(float)}).to_parquet(pred_dir / f"M0_fold{fold}.parquet", index=False)
    m0s1 = m0 & (rng.random(len(y)) < 0.7)                                   # S1 bagian dari primer
    pd.DataFrame({**base, "y_prob": m0s1.astype(float)}).to_parquet(pred_dir / f"M0S1_fold{fold}.parquet", index=False)


def git(root: Path, *args):
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t"}
    r = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    return r.stdout


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """Repo tiruan: tabel master + mm_reports sintetis; semua jalur modul diarahkan ke tmp_path."""
    master = synthetic_master()
    (tmp_path / "data/processed").mkdir(parents=True)
    (tmp_path / "data/interim").mkdir(parents=True)
    master.to_parquet(tmp_path / "data/processed/master_table_draft.parquet", index=False)
    synthetic_reports(master).to_parquet(tmp_path / "data/interim/mm_reports.parquet", index=False)
    for name, val in {"ROOT": tmp_path, "MASTER_V1": tmp_path / "data/processed/master_table_v1.parquet",
                      "MASTER_DRAFT": tmp_path / "data/processed/master_table_draft.parquet",
                      "SIGNAL_QC": tmp_path / "data/interim/signal_qc.csv", "OUT_DIR": tmp_path / "data/interim/tabular",
                      "PRED_DIR": tmp_path / "predictions", "REPORTS": tmp_path / "reports"}.items():
        monkeypatch.setattr(T, name, val)
    monkeypatch.setattr(P, "PRED_DIR", tmp_path / "predictions")
    monkeypatch.setattr(P, "ROOT", tmp_path)
    monkeypatch.setattr(lock, "ROOT", tmp_path)
    monkeypatch.setattr(run_fold19, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(run_fold19, "LOCKED", tmp_path / "reports/locked_fold18.json")
    monkeypatch.setattr(run_fold19, "LOCKED_M0", tmp_path / "reports/locked_fold18_m0.json")
    monkeypatch.setattr(lock_m0_threshold, "ROOT", tmp_path)
    monkeypatch.setattr(lock_m0_threshold, "OUT", tmp_path / "reports/locked_fold18_m0.json")
    monkeypatch.setattr(run_fold18, "ROOT", tmp_path)
    monkeypatch.setattr(run_fold18, "REPORT", tmp_path / "reports/eval_fold18.md")
    monkeypatch.setattr(run_fold18, "LOCKED", tmp_path / "reports/locked_fold18.json")
    monkeypatch.setattr(run_fold18, "load_predictions",
                        lambda m, f: load_predictions(m, f, pred_dir=tmp_path / "predictions"))
    monkeypatch.setattr(run_fold18, "pred_path", lambda m, f: tmp_path / "predictions" / f"{m}_fold{f}.parquet")
    return tmp_path, master


def lock_fold18(root: Path, master: pd.DataFrame, m0: bool = True):
    write_preds(root / "predictions", master, 18)
    run_fold18.main(["--models", "M1", "M2", "M3", "M4", "M5", "M6", "--n-boot", "20"])
    if m0:
        lock_m0_threshold.main([])


# ---------------------------------------------------------------------------
# Aturan M0
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("*** ACUTE MI / STEMI ***", 1), ("Acute MI", 1), ("Anterior infarct, possibly acute", 1),
    ("Probable acute inferior infarct", 1), ("ST elevation consider anterior injury or acute infarct", 1),
    ("Inferior injury pattern", 1), ("Cannot rule out acute MI", 1), ("can not rule out acute infarct", 1),
    ("Inferior infarct, age undetermined", 0), ("Old anterior infarct", 0), ("Sinus rhythm", 0),
    ("Old MI, no acute changes", 0), ("No acute ST changes", 0), ("Acute pericarditis", 0),
    ("Mild ST depression", 0), ("Cannot rule out inferior infarct", 0), ("without acute injury", 0),
])
def test_m0_statement_rule(text, expected):
    f = m0_rule.statement_flags(pd.DataFrame({"ed_stay_id": [1], "text": [text.lower()]}))
    assert int(f["positive"].iloc[0]) == expected


@pytest.mark.parametrize("text,primer,s1", [
    ("Anterior infarct, recent", 1, 1), ("Anterior infarct, probably recent", 1, 1), ("Evolving inferior MI", 1, 1),
    ("Recent anteroseptal infarct", 1, 1), ("Consider acute ischemia", 1, 0), ("ST depression, ischemia, possibly acute", 1, 0),
    ("Inferior ST-T changes suggest myocardial injury/ischemia", 1, 0), ("Anterior injury pattern", 1, 1),
    ("ST elevation consider anterior injury or acute infarct", 1, 1),
    ("ST elevation, consistent with injury, pericarditis, or early repolarization", 1, 0),
    ("Marked ST depression, consider subendocardial injury", 1, 1), ("*** ACUTE MI ***", 1, 1),
    ("No acute ischemia", 0, 0), ("Lateral ST depression, consider ischemia", 0, 0), ("Acute pericarditis or ischemia", 0, 0),
    ("Old infarct, unchanged since recent tracing", 0, 0), ("Inferior infarct, recent, compared with previous", 0, 0),
    ("Septal infarct, age undetermined, compared with recent ecg of", 0, 0), ("Inferior infarct, age undetermined", 0, 0),
])
def test_m0_variants_primary_and_s1(text, primer, s1):
    for dtype in ("string", object):                                    # hasil tidak bergantung backend string pandas
        df = pd.DataFrame({"ed_stay_id": [1], "text": pd.Series([text.lower()]).astype(dtype)})
        f = m0_rule.statement_flags(df)
        assert (int(f["pos_M0"].iloc[0]), int(f["pos_M0S1"].iloc[0])) == (primer, s1)


def test_m0_per_visit_any_statement_and_empty_visits():
    r = pd.DataFrame({"ed_stay_id": [1, 2, 3, 4]})
    for i in range(18):
        r[f"report_{i}"] = pd.array([pd.NA] * 4, dtype="string")
    r["report_0"] = pd.array(["Sinus rhythm", "Sinus rhythm", pd.NA, "  "], dtype="string")
    r["report_5"] = pd.array(["Acute MI", "Old infarct", pd.NA, pd.NA], dtype="string")
    v = m0_rule.m0_per_visit(r).set_index("ed_stay_id")
    assert v["m0"].to_dict() == {1: 1, 2: 0, 3: 0, 4: 0}
    assert v["m0s1"].to_dict() == {1: 1, 2: 0, 3: 0, 4: 0}
    assert v["n_statements"].to_dict() == {1: 2, 2: 2, 3: 0, 4: 0}


def test_m0_audit_counts_categories():
    rows = {
        1: ["Sinus rhythm", "Inferior ST-T changes suggest myocardial injury/ischemia"],
        2: ["*** ACUTE MI ***"],
        3: ["Probable inferior infarct, age indeterminate"],
        4: ["Consider inferior infarct", "acute"],                       # terpotong ke baris berikutnya
        5: ["ST elevation consider early repolarization, pericarditis, or injury"],
        6: ["No acute infarct"],
        7: ["Old anterior infarct", None, "acute pericarditis"],         # tidak berurutan
        8: ["Inferior infarct, possibly acute", "Anterior injury pattern"],
    }
    recs = []
    for vid, texts in rows.items():
        r = {"ed_stay_id": vid}
        r.update({f"report_{i}": t for i, t in enumerate(texts)})
        recs.append(r)
    c, _ = m0_rule.audit_counts(pd.DataFrame(recs))
    assert c["kunjungan_m0"] == 4                                        # 1, 2, 5, 8
    assert c["kunjungan_m0s1"] == 2 and c["kunjungan_pindah"] == 2       # 2, 8 | 1, 5
    assert c["a_inj_saja"] == 3 and c["a_inj_iskemia"] == 1 and c["a_inj_peri_repol"] == 1
    assert c["a_inj_alternatif"] == 2                                    # "/" dan "or"
    assert c["a_kunj_hanya_inj"] == 2 and c["a_kunj_hanya_inj_iskemia"] == 1 and c["a_kunj_hanya_inj_peri_repol"] == 1
    assert c["dinegasi"] == 1 and c["b_acute_dan_infark"] == 1
    assert c["b_infark_usia"] == 2 and c["b_infark_lain"] == 1          # 3, 7 | 4
    assert c["b_acute_saja"] == 2 and c["b_stemi"] == 0                 # "acute", "acute pericarditis"
    assert c["c_terpotong_kunj_m0_0"] == 1 and c["c_terpotong_kunj_m0_1"] == 0
    assert c["b4_perikarditis"] == 1 and c["b4_lain"] == 1 and c["b4_singkatan_infark"] == 0
    assert c["c1_pasangan"] == 1 and c["c1_acute_pendek"] == 1 and c["c1_baris_pertama_menggantung"] == 0


def test_m0_audit_split_and_abbreviation_details():
    recs = [
        {"ed_stay_id": 1, "report_0": "Inferior infarct,", "report_1": "possibly acute"},
        {"ed_stay_id": 2, "report_0": "Consider acute", "report_1": "anterior infarct"},
        {"ed_stay_id": 3, "report_0": "Old anterior infarct", "report_1": "acute pericarditis"},
        {"ed_stay_id": 4, "report_0": "acute ASMI"},
        {"ed_stay_id": 5, "report_0": "Anterior infarct, recent"},
        {"ed_stay_id": 6, "report_0": "Acute ST elevation"},
    ]
    c, _ = m0_rule.audit_counts(pd.DataFrame(recs))
    assert c["kunjungan_m0"] == 1 and c["kunjungan_m0s1"] == 1            # "anterior infarct, recent"
    assert c["d_recent_saja"] == 1 and c["b3_recent_evolving"] == 0
    assert c["c_terpotong_kunj_m0_0"] == 3 and c["c1_pasangan"] == 3
    assert c["c1_acute_pendek"] == 2                       # "possibly acute", "consider acute"
    assert c["c1_baris_pertama_menggantung"] == 1          # "inferior infarct,"
    assert c["c1_acute_perikarditis"] == 1
    assert c["b4_singkatan_infark"] == 1 and c["b4_st_elevasi"] == 1


def test_m0_rule_fingerprint_is_stable():
    assert m0_rule.rule_sha256() == m0_rule.rule_sha256() and len(m0_rule.rule_sha256()) == 64


def test_m0_writes_contract_file_without_metrics(repo):
    root, master = repo
    m0_rule.main(["--folds", "18"])
    d = validate_predictions(pd.read_parquet(root / "predictions/M0_fold18.parquet"), expected_fold=18)
    s1 = validate_predictions(pd.read_parquet(root / "predictions/M0S1_fold18.parquet"), expected_fold=18)
    assert len(d) == int((master["fold"] == 18).sum()) and set(d["y_prob"].unique()) <= {0.0, 1.0}
    assert (s1["ed_stay_id"].to_numpy() == d["ed_stay_id"].to_numpy()).all()
    assert (s1["y_prob"] <= d["y_prob"]).all() and s1["y_prob"].sum() < d["y_prob"].sum()   # S1 bagian dari primer
    rep = (root / "reports/m0_rule.md").read_text()
    assert m0_rule.rule_sha256() in rep and "AUROC" not in rep                 # tanpa metrik kinerja
    assert all(w not in rep.lower() for w in ("spesifisitas", "ppv", "npv", "label_ami", "y_true"))
    assert "30000" not in rep                                   # tidak ada ed_stay_id di laporan


def test_m0_fold19_needs_flag_and_lock(repo):
    with pytest.raises(SystemExit, match="dikunci"):
        m0_rule.main(["--folds", "19"])
    with pytest.raises(SystemExit, match="Fold 19 dikunci"):
        m0_rule.main(["--folds", "19", "--include-fold-19"])     # bukan repo git / rencana tidak ada
    with pytest.raises(SystemExit, match="fold evaluasi"):
        m0_rule.main(["--folds", "17"])


# ---------------------------------------------------------------------------
# Penjaga penguncian
# ---------------------------------------------------------------------------
def _make_lock_files(root: Path, status="DRAF v1"):
    (root / "reports").mkdir(exist_ok=True)
    (root / "src/model").mkdir(parents=True, exist_ok=True)
    (root / "reports/rencana_analisis_fold19.md").write_text(f"# Rencana\n\n**Status: {status}**\n", encoding="utf-8")
    (root / "reports/locked_fold18.json").write_text("{}", encoding="utf-8")
    shutil.copy(REPO / "src/model/m0_rule.py", root / "src/model/m0_rule.py")


def test_lock_requires_committed_final_plan(tmp_path):
    _make_lock_files(tmp_path)
    git(tmp_path, "init", "-q")
    with pytest.raises(SystemExit, match="belum di-commit"):
        lock.require_fold19_unlocked(tmp_path)
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "draf")
    with pytest.raises(SystemExit, match="FINAL"):
        lock.require_fold19_unlocked(tmp_path)
    (tmp_path / "reports/rencana_analisis_fold19.md").write_text("**Status: FINAL (2026-11-25)**\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="berubah"):
        lock.require_fold19_unlocked(tmp_path)                 # sudah FINAL tapi belum di-commit
    git(tmp_path, "commit", "-qam", "final")
    info = lock.require_fold19_unlocked(tmp_path)
    assert set(info) == {"plan_commit", "locked_fold18_commit", "m0_rule_commit"}
    (tmp_path / "src/model/m0_rule.py").write_text("# diubah\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="m0_rule.py berubah"):
        lock.require_fold19_unlocked(tmp_path)


def test_lock_outside_git(tmp_path):
    _make_lock_files(tmp_path, "FINAL")
    with pytest.raises(SystemExit, match="bukan repositori git"):
        lock.require_fold19_unlocked(tmp_path)


# ---------------------------------------------------------------------------
# run_fold19
# ---------------------------------------------------------------------------
def test_run_fold19_on_fold18_reproduces_locked_parameters(repo, capsys):
    root, master = repo
    lock_fold18(root, master)
    run_fold19.main(["--fold", "18", "--n-boot", "40"])
    out = capsys.readouterr().out
    assert out.count("-> OK") == 6 and "GAGAL" not in out
    rep = (root / "reports/eval_fold18_uji_final.md").read_text()
    assert "UJI KODE, BUKAN HASIL" in rep and "M5 - M4" in rep and "Subgrup" in rep
    assert "30000" not in rep and "40000" not in rep                         # tidak ada ID
    for name in ("metrics", "comparisons", "counts", "subgroups"):
        assert (root / f"reports/eval_fold18_uji_final_{name}.csv").exists()
    counts = pd.read_csv(root / "reports/eval_fold18_uji_final_counts.csv", dtype=str)
    for c in ("TP", "FP", "FN", "TN"):                                       # sel kecil disembunyikan
        assert all(v in ("<10", "disembunyikan") or int(v.replace(",", "")) >= 10 for v in counts[c])
    comp = pd.read_csv(root / "reports/eval_fold18_uji_final_comparisons.csv")
    assert {"M5 - M4", "M1 - M3", "M6 - M4", "M5 - M2", "M5 - M1", "M1 - M0", "M0S1 - M0"} <= set(comp["target"])
    assert "IMA tipe 2 murni" in rep and "primer (M0) vs analisis sensitivitas S1" in rep
    assert not (root / "reports/eval_fold19.md").exists()


def test_run_fold19_detects_changed_fold18_predictions(repo):
    root, master = repo
    lock_fold18(root, master)
    d = pd.read_parquet(root / "predictions/M2_fold18.parquet")
    d["y_prob"] = np.clip(d["y_prob"] * 0.99, 0, 1)
    d.to_parquet(root / "predictions/M2_fold18.parquet", index=False)
    with pytest.raises(SystemExit, match="sha256 tidak cocok"):
        run_fold19.main(["--fold", "18", "--n-boot", "10"])


def test_run_fold19_sanity_uses_each_model_full_file(repo, capsys):
    """Bila satu model tidak punya semua kunjungan fold 18, cek parameter tetap dihitung pada berkas
    utuh tiap model (seperti saat dikunci), dan irisan kunjungan bersama hanya diberi peringatan."""
    root, master = repo
    write_preds(root / "predictions", master, 18)
    d = pd.read_parquet(root / "predictions/M1_fold18.parquet")
    d.iloc[25:].to_parquet(root / "predictions/M1_fold18.parquet", index=False)      # 25 kunjungan hilang
    run_fold18.main(["--models", "M1", "M2", "M3", "M4", "M5", "M6", "--n-boot", "20"])
    lock_m0_threshold.main([])
    run_fold19.main(["--fold", "18", "--n-boot", "20"])
    out = capsys.readouterr().out
    assert out.count("-> OK") == 6 and "GAGAL" not in out
    assert "PERINGATAN" in out
    assert out.index("Cek parameter terkunci") < out.index("Bootstrap 20x")          # gagal cepat sebelum bootstrap


def test_run_fold19_sanity_fails_on_wrong_threshold(repo):
    root, master = repo
    lock_fold18(root, master)
    j = json.loads((root / "reports/locked_fold18.json").read_text())
    j["M3"]["threshold_rule_out"]["y_prob_raw"] = 0.999
    (root / "reports/locked_fold18.json").write_text(json.dumps(j))
    with pytest.raises(SystemExit, match="Uji fold 18 GAGAL"):
        run_fold19.main(["--fold", "18", "--n-boot", "10"])


def test_run_fold19_fold19_locked_without_flag_or_git(repo):
    root, master = repo
    lock_fold18(root, master)
    write_preds(root / "predictions", master, 19)
    with pytest.raises(SystemExit, match="dikunci"):
        run_fold19.main(["--fold", "19"])
    with pytest.raises(SystemExit, match="Fold 19 dikunci"):
        run_fold19.main(["--fold", "19", "--include-fold-19"])


def test_run_fold19_full_once_then_posthoc(repo, capsys):
    root, master = repo
    lock_fold18(root, master, m0=False)
    m0_rule.main(["--folds", "18"])
    lock_m0_threshold.main([])                                               # setelah M0_fold18 final
    (root / "src/model").mkdir(parents=True)
    shutil.copy(REPO / "src/model/m0_rule.py", root / "src/model/m0_rule.py")
    (root / "reports/rencana_analisis_fold19.md").write_text("**Status: FINAL (uji)**\n", encoding="utf-8")
    (root / ".gitignore").write_text("data/\npredictions/\n", encoding="utf-8")
    git(root, "init", "-q")
    git(root, "add", ".")
    git(root, "commit", "-qm", "kunci")
    write_preds(root / "predictions", master, 19)
    run_fold19.main(["--checksums"])
    assert "M1_fold19.parquet" in capsys.readouterr().out
    run_fold19.main(["--fold", "19", "--include-fold-19", "--n-boot", "30"])
    rep = (root / "reports/eval_fold19.md").read_text()
    assert "# Evaluasi akhir fold 19" in rep and "UJI KODE" not in rep and "Penguncian: rencana analisis" in rep
    assert "locked_fold18_m0.json `" in rep and "spesifisitas setara M0" in rep
    assert (root / "reports/eval_fold19_calibration_curve.csv").exists()
    with pytest.raises(SystemExit, match="SATU kali"):
        run_fold19.main(["--fold", "19", "--include-fold-19", "--n-boot", "30"])
    run_fold19.main(["--fold", "19", "--include-fold-19", "--n-boot", "30", "--overwrite-post-hoc"])
    posthoc = list((root / "reports").glob("eval_fold19_posthoc_*.md"))
    assert len(posthoc) == 1 and "POST-HOC" in posthoc[0].read_text()


def test_complementary_suppression():
    assert run_fold19.suppress_parts({"TP": 4, "FN": 200}) == {"TP": "<10", "FN": "disembunyikan"}
    assert run_fold19.suppress_parts({"TP": 40, "FN": 200}) == {"TP": "40", "FN": "200"}
    assert run_fold19.suppress_parts({"a": 3, "b": 5, "c": 90}) == {"a": "<10", "b": "<10", "c": "90"}
    assert run_fold19.suppress_parts({"a": 3, "b": 15, "c": 90}) == {"a": "<10", "b": "disembunyikan", "c": "90"}


def test_metric_functions_match_core():
    from src.evaluation.core import binary_metrics
    rng = np.random.default_rng(5)
    y = (rng.random(3000) < 0.1).astype(int)
    p = np.clip(rng.random(3000) * 0.5 + y * 0.4, 0, 1)
    L = {"M1": run_fold19.Locked(None, 0.45, 0.6)}
    fns = run_fold19.metric_fns(L)
    for w, t in (("out", 0.45), ("in", 0.6)):
        m = binary_metrics(y, p, t)
        assert fns[f"Sens@rule-{w}"](y, p, "M1") == pytest.approx(m["sensitivity"])
        assert fns[f"Spes@rule-{w}"](y, p, "M1") == pytest.approx(m["specificity"])
        assert fns[f"PPV@rule-{w}"](y, p, "M1") == pytest.approx(m["ppv"])
        assert fns[f"NPV@rule-{w}"](y, p, "M1") == pytest.approx(m["npv"])


# ---------------------------------------------------------------------------
# SHAP (XGBoost di proses terpisah)
# ---------------------------------------------------------------------------
def run_py(root: Path, body: list[str]) -> subprocess.CompletedProcess:
    code = "\n".join([
        "import sys", "from pathlib import Path",
        "from src.model import train_tabular as T", "from src.evaluation import predictions as P",
        f"r = Path({str(root)!r})",
        "T.ROOT = r", "T.MASTER_V1 = r / 'data/processed/master_table_v1.parquet'",
        "T.MASTER_DRAFT = r / 'data/processed/master_table_draft.parquet'",
        "T.SIGNAL_QC = r / 'data/interim/signal_qc.csv'", "T.OUT_DIR = r / 'data/interim/tabular'",
        "T.PRED_DIR = r / 'predictions'", "T.REPORTS = r / 'reports'", "P.PRED_DIR = r / 'predictions'",
        "T.MAX_TREES = 200", "T.EARLY_STOP = 20",
        *body,
        "assert 'torch' not in sys.modules, 'torch termuat bersama xgboost'",
    ])
    return subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=900)


@pytest.mark.skipif(not HAS_XGB, reason="xgboost belum terpasang")
def test_shap_m2_fold18_aggregate_and_consistent(tmp_path):
    (tmp_path / "data/processed").mkdir(parents=True)
    synthetic_master(12000, seed=3).to_parquet(tmp_path / "data/processed/master_table_draft.parquet", index=False)
    r = run_py(tmp_path, ["T.main(['--model', 'M2', '--n-trials', '2'])"])
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    args = ["--model", "M2", "--fold", "18"] + (["--cek-shap"] if HAS_SHAP else [])
    r = run_py(tmp_path, ["from src.model import shap_tabular as S", f"S.main({args!r})"])
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    md = (tmp_path / "reports/shap_M2_fold18.md").read_text()
    csv = pd.read_csv(tmp_path / "reports/shap_M2_fold18.csv")
    assert (tmp_path / "reports/figures/shap_M2_fold18.png").exists()
    assert len(csv) == 33 and csv["peringkat"].tolist() == list(range(1, 34))   # 33 fitur asal (one-hot digabung)
    assert csv["label"].iloc[0] in {"Keluhan nyeri dada", "Denyut nadi triase"}  # sinyal yang ditanam
    assert csv["porsi_%"].sum() == pytest.approx(100.0)
    assert "30000" not in md                                                     # tidak ada ID
    if HAS_SHAP:
        line = next(s for s in r.stdout.splitlines() if "cek shap" in s)
        assert float(line.rsplit("cek shap", 1)[1].strip()) < 1e-4


def test_shap_stability_from_aggregate_csv(tmp_path, monkeypatch):
    from src.model import shap_tabular as S
    monkeypatch.setattr(T, "REPORTS", tmp_path)
    feats = [f"f{i}" for i in range(12)]
    a = pd.DataFrame({"fitur": feats, "label": [f"Fitur {i}" for i in range(12)], "mean_abs_shap": np.arange(12, 0, -1.0)})
    b = a.copy()
    b.loc[[0, 11], "mean_abs_shap"] = b.loc[[11, 0], "mean_abs_shap"].to_numpy()       # tukar teratas dan terbawah
    a.to_csv(tmp_path / "shap_M2_fold18.csv", index=False)
    b.to_csv(tmp_path / "shap_M2_fold19.csv", index=False)
    r = S.stability("M2")
    assert r["overlap"] == 9 and 0 < r["spearman"] < 1
    assert "Spearman" in (tmp_path / "shap_M2_stabilitas.md").read_text()


def test_shap_complementary_suppression():
    from src.model import shap_tabular as S
    rng = np.random.default_rng(0)
    n = 200
    x = np.r_[np.ones(5), np.zeros(n - 5)]                       # "= 1" hanya 5 kunjungan
    X = pd.DataFrame({"cc_chest_pain": x})
    raw = X.copy()
    out = S.summarize_groups(X, rng.normal(size=(n, 1)), raw, ["cc_chest_pain"])
    r = out.iloc[0]
    assert r["n = 1"] == "<10" and r["n = 0"] == "disembunyikan" and pd.isna(r["SHAP = 1"])


@pytest.mark.skipif(not HAS_XGB, reason="xgboost belum terpasang")
def test_shap_fold19_locked(tmp_path):
    r = run_py(tmp_path, ["from src.model import shap_tabular as S", "S.main(['--model', 'M4', '--fold', '19'])"])
    assert r.returncode != 0 and "dikunci" in (r.stdout + r.stderr)


# ---------------------------------------------------------------------------
# v1.7: threshold spesifisitas setara M0 (dikunci di fold 18) dan kurva kalibrasi fleksibel
# ---------------------------------------------------------------------------
def test_lock_m0_threshold_once_and_meets_target(repo, capsys):
    root, master = repo
    lock_fold18(root, master)
    j = json.loads((root / "reports/locked_fold18_m0.json").read_text())
    m0 = pd.read_parquet(root / "predictions/M0_fold18.parquet")
    neg = m0["y_true"] == 0
    assert j["target_specificity"] == pytest.approx(float((m0.loc[neg, "y_prob"] == 0).mean()))
    for m in ("M1", "M2", "M3", "M4", "M5", "M6"):
        d = pd.read_parquet(root / f"predictions/{m}_fold18.parquet")
        thr = j["models"][m]["y_prob_raw"]
        spec = float((d.loc[d.y_true == 0, "y_prob"] < thr).mean())
        assert spec >= j["target_specificity"]
        # threshold terkecil: satu langkah di bawahnya spesifisitas tidak lagi memenuhi
        below = np.nextafter(d.loc[d.y_true == 0, "y_prob"][d.loc[d.y_true == 0, "y_prob"] < thr].max(), -np.inf)
        assert float((d.loc[d.y_true == 0, "y_prob"] < below).mean()) < j["target_specificity"]
    assert "30000" not in json.dumps(j) and not list(root.glob("predictions/*fold19*"))
    with pytest.raises(SystemExit, match="SATU kali"):
        lock_m0_threshold.main([])


def test_run_fold19_needs_m0_lock(repo):
    root, master = repo
    lock_fold18(root, master, m0=False)
    with pytest.raises(SystemExit, match="lock_m0_threshold"):
        run_fold19.main(["--fold", "18", "--n-boot", "10"])


def test_run_fold19_m0_match_and_calibration_outputs(repo, capsys):
    root, master = repo
    lock_fold18(root, master)
    run_fold19.main(["--fold", "18", "--n-boot", "40"])
    out = capsys.readouterr().out
    assert out.count("-> OK") == 6 and "spes@setara-M0" in out
    rep = (root / "reports/eval_fold18_uji_final.md").read_text()
    assert "Pada spesifisitas setara M0" in rep and "spes setara M0" in rep
    comp = pd.read_csv(root / "reports/eval_fold18_uji_final_comparisons.csv")
    for m in ("M1", "M5"):
        r = comp[(comp.target == f"{m} - M0") & (comp.metric == "Spes@spes-M0")]
        assert len(r) == 1
    met = pd.read_csv(root / "reports/eval_fold18_uji_final_metrics.csv")
    target = json.loads((root / "reports/locked_fold18_m0.json").read_text())["target_specificity"]
    spes = met[(met.metric == "Spes@spes-M0") & met.target.isin(["M1", "M2", "M3", "M4", "M5", "M6"])]["estimate"]
    assert len(spes) == 6 and (spes >= target - 1e-12).all()                 # in-sample di fold 18
    curve = pd.read_csv(root / "reports/eval_fold18_uji_final_calibration_curve.csv")
    assert set(curve.model) == {"M1", "M2", "M3", "M4", "M5", "M6"} and len(curve) == 6 * 50
    assert ((curve.ci_low <= curve.p_obs + 1e-9) & (curve.p_obs <= curve.ci_high + 1e-9)).mean() > 0.95
    assert curve.n_valid.min() > 30
    bins = pd.read_csv(root / "reports/eval_fold18_uji_final_calibration_bins.csv", dtype={"n": str})
    assert len(bins) == 60 and all(v == "<10" or int(v.replace(",", "")) >= 10 for v in bins.n)
    counts = pd.read_csv(root / "reports/eval_fold18_uji_final_counts.csv")
    assert (counts.threshold == "spes setara M0").sum() == 6


def test_run_fold19_m0_threshold_tampered_fails(repo):
    root, master = repo
    lock_fold18(root, master)
    p = root / "reports/locked_fold18_m0.json"
    j = json.loads(p.read_text())
    j["models"]["M4"]["y_prob_raw"] = 0.0
    p.write_text(json.dumps(j))
    with pytest.raises(SystemExit, match="Uji fold 18 GAGAL"):
        run_fold19.main(["--fold", "18", "--n-boot", "10"])


def test_flexible_calibration_curve_tracks_truth():
    from src.evaluation import calibration_curve as C
    rng = np.random.default_rng(3)
    n = 20000
    p = expit(rng.normal(-3.3, 1.2, n))
    y_ok = (rng.random(n) < p).astype(int)                                 # terkalibrasi sempurna
    y_over = (rng.random(n) < p * 0.5).astype(int)                          # model 2x terlalu tinggi
    subj = np.arange(n)
    rows = pd.DataFrame(C.calibration_curves(y_ok, {"ok": p, "over": p}, subj, n_boot=30, seed=1))
    ok = rows[rows.model == "ok"]
    over = rows[rows.model == "over"]
    rows2 = pd.DataFrame(C.calibration_curves(y_over, {"over": p}, subj, n_boot=30, seed=1))
    mid = (ok.p_pred > 0.02) & (ok.p_pred < 0.3)
    assert np.abs(ok.p_obs[mid] - ok.p_pred[mid]).max() < 0.03
    assert (rows2.p_obs.to_numpy()[mid.to_numpy()] < 0.75 * rows2.p_pred.to_numpy()[mid.to_numpy()]).all()
    assert len(over) == 50 and over.p_pred.is_monotonic_increasing


def test_rcs_basis_linear_outside_knots():
    from src.evaluation.calibration_curve import rcs_basis
    k = np.array([-3.0, -1.0, 0.0, 2.0])
    x = np.array([3.0, 4.0, 5.0])
    B = rcs_basis(x, k)
    assert B.shape == (3, 3)
    d2 = np.diff(B, n=2, axis=0)                                            # linear di kanan knot terakhir
    assert np.allclose(d2, 0, atol=1e-10)
    assert np.allclose(rcs_basis(np.array([-5.0, -4.0]), k)[:, 1:], 0)      # nol di kiri knot pertama
