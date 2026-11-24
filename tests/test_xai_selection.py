"""Tes pemilihan rekaman XAI M1 (src/xai/selection.py). Pandas saja (tanpa torch); data SINTETIS.

Jalankan: python -m pytest -q tests/test_xai_selection.py
"""
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation import lock
from src.xai import selection as S

REPO = Path(__file__).resolve().parents[1]


def table(n=20000, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, 20, n)
    return pd.DataFrame({"ed_stay_id": np.arange(n) + 30_000_000, "file_name": [f"f{i}" for i in range(n)],
                         "fold": fold, "label_ami": (rng.random(n) < 0.08).astype(int),
                         "internal_val_0_16": ((fold <= 16) & (rng.random(n) < 0.1)).astype(int)})


def test_internal_sama_persis_dengan_kode_asli():
    t = table()
    t = t[t.fold <= 16]
    val = t[t["internal_val_0_16"] == 1]
    lama = pd.concat([val[val.label_ami == c].sample(40, random_state=42) for c in (1, 0)])
    baru = S.pick_records(t, "internal", 40, 42)
    assert lama["ed_stay_id"].tolist() == baru["ed_stay_id"].tolist()
    assert S.resolve("internal") == (list(range(0, 17)), False, "validasi internal fold 0-16")


def test_fold18_hanya_fold18_dan_seimbang():
    t = table()
    folds, allow, label = S.resolve("18")
    assert folds == [18] and allow and "UJI KODE" in label
    p = S.pick_records(t[t.fold.isin(folds)], "18", 30, 42)
    assert set(p.fold) == {18} and len(p) == 60 and p.label_ami.sum() == 30


def test_kelas_kurang_berhenti():
    t = table(n=3000)
    with pytest.raises(SystemExit, match="kurang dari"):
        S.pick_records(t[t.fold == 18], "18", 150, 42)


def test_fold19_butuh_flag_dan_kunci(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="--include-fold-19"):
        S.resolve("19")
    monkeypatch.setattr(lock, "ROOT", tmp_path)
    with pytest.raises(SystemExit, match="Fold 19 dikunci"):
        S.resolve("19", include_fold_19=True)


def _git(root, *args):
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t"}
    r = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr


def test_fold19_terbuka_setelah_final(tmp_path, monkeypatch):
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports/rencana_analisis_fold19.md").write_text("**Status: DRAF**\n", encoding="utf-8")
    (tmp_path / "reports/locked_fold18.json").write_text("{}", encoding="utf-8")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "draf")
    monkeypatch.setattr(lock, "ROOT", tmp_path)
    with pytest.raises(SystemExit, match="FINAL"):
        S.resolve("19", include_fold_19=True)
    (tmp_path / "reports/rencana_analisis_fold19.md").write_text("**Status: FINAL (uji)**\n", encoding="utf-8")
    _git(tmp_path, "commit", "-qam", "final")
    folds, allow, label = S.resolve("19", include_fold_19=True)
    assert folds == [19] and allow and label.startswith("fold 19 (rencana analisis `")


def test_skrip_xai_bawaan_tetap_internal():
    for f in ("run_xai_sanity.py", "run_xai_faithfulness.py"):
        src = (REPO / "src/xai" / f).read_text(encoding="utf-8")
        assert 'default="internal"' in src and "range(0, 17)" not in src and "resolve(a.fold, a.include_fold_19)" in src
