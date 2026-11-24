# Rencana Analisis Final (Fold 19)

**Status: DRAF v4 (2026-11-24), disusun Orang 3. Belum berlaku.**

Dokumen ini disimpan di `reports/rencana_analisis_fold19.md`. Dokumen menjadi final setelah:
- disetujui tim dan dosen pembimbing atas teks ini (Bagian 16);
- baris status di atas diganti menjadi status final, berikut tanggalnya;
- di-commit dan di-push ke repo tim privat.

Penjaga `src/evaluation/lock.py` menolak semua pekerjaan fold 19 sebelum ketiga hal itu terpenuhi. Setelah final, prediksi fold 19 boleh dibuat dan evaluasi dijalankan **satu kali** (PROJECT_BRIEF 11.5; rencana proyek 8.6). Analisis apa pun yang tidak tercantum di sini dilaporkan sebagai **post-hoc** dan dicatat di `reports/decision_log.md`.

Perubahan dari DRAF v1 (2026-11-14):
- aturan M0-v2 dan analisis sensitivitas M0S1;
- subgrup IMA tipe 2 murni;
- supresi komplementer untuk sel kecil;
- SHAP M2/M4 (fold 19 utama, fold 18 sebagai cek stabilitas);
- status M1_fold19;
- prosedur disesuaikan dengan kode yang sudah diuji di fold 18.

Perubahan dari DRAF v2 (2026-11-24):
- entri decision log Orang 2 tentang `M1_fold19` ditambahkan, termasuk konfirmasi bahwa `signal_qc_full.csv` identik dengan `signal_qc.csv` (Bagian 14);
- skrip Gambar 2 `src/evaluation/plot_forest.py`, sudah diuji di fold 18 (Bagian 12);
- CSV pendukung di-commit secara eksplisit karena `.gitignore` mengabaikan `*.csv` (Bagian 13).

Perubahan dari DRAF v3 (2026-11-24), sesuai keputusan tim 24 November (decision log 2026-11-24, "Keputusan A2"):
- threshold ketiga, **spesifisitas setara M0**, dikunci di fold 18 (`reports/locked_fold18_m0.json`, commit `eef89dc`) dan menjadi jawaban utama untuk perbandingan dengan M0 (Bagian 4, 6, 7);
- **kurva kalibrasi fleksibel** (restricted cubic spline) setelah Platt, dengan IK bootstrap, menggantikan pertanyaan "kurva ROC dan kalibrasi". Kurva ROC tidak dilaporkan (Bagian 7, 12);
- XAI: SHAP global saja, tanpa nilai per pasien. Uji Grad-CAM/oklusi M1 diulang di fold 19, dan uji kodenya dijalankan di fold 18 (Bagian 11, 13);
- aturan sel kecil ditegaskan: "<10" berlaku untuk hitungan, sedangkan proporsi dan IK dilaporkan apa adanya (Bagian 12);
- hasil fold 19 hanya di-push ke repo tim privat (Bagian 13);
- uji kode XAI M1 di fold 18 oleh Orang 2 sudah lolos (`reports/xai_fold18.txt`; Bagian 11 dan 14);
- semua butir Bagian 14 diperbarui.

---

## 1. Tujuan

Mengunci seluruh analisis fold 19 sebelum ada yang melihat hasil model pada fold 19. Yang dikunci: populasi, model, parameter, metrik, perbandingan, subgrup, analisis sensitivitas, XAI, dan prosedur.

## 2. Populasi evaluasi

- Fold 19 tabel master: satu baris per kunjungan IGD dengan EKG pertama (`ecg_time` terawal), hanya `signal_ok == 1` (setara `master_table_v1`).
- Perkiraan: 7.227 kunjungan, 10 gagal QC, jadi **sekitar 7.217 kunjungan** (sama dengan jumlah baris `M1_fold19`), dengan sekitar 260 positif (PROJECT_BRIEF). Angka pastinya dicetak oleh skrip evaluasi.
- **Semua model dinilai pada kunjungan yang sama** (irisan `ed_stay_id`). Kalau ada model yang tidak punya prediksi untuk sebagian kunjungan, skrip mencetak peringatan, jumlahnya dilaporkan, dan analisis memakai irisan. Diharapkan tidak ada kunjungan yang hilang.
- Prevalensi fold 19 dilaporkan sebagai garis dasar AUPRC.

## 3. Model yang dievaluasi

| Kode | Model | Data latih | Catatan |
|---|---|---|---|
| M0 | Interpretasi mesin EKG (`report_*`), aturan kata kunci M0-v2, definisi primer | Tanpa training | Dikunci di commit `650d599` (Bagian 10) |
| M0S1 | Seperti M0, definisi sempit (analisis sensitivitas S1) | Tanpa training | Dikunci bersama M0 |
| M1 | ResNet-1D, sinyal 12 lead | Fold 0–16, subset 2a (±13%) | Keterbatasan tercatat di decision log 2026-11-14 |
| M2 | XGBoost, fitur EMR | Fold 0–16 | |
| M3 | XGBoost, machine measurements | Fold 0–16 | |
| M4 | XGBoost, EMR + MM (Jalur B) | Fold 0–17 | Pembanding utama |
| M5 | Late fusion M1 + M2 (Jalur A) | Meta-learner di fold 17 | Model utama |
| M6 | Late fusion M3 + M2 | Meta-learner di fold 17 | |

Tidak ada model yang dilatih ulang, dituning, atau diganti setelah dokumen ini disetujui.

## 4. Parameter yang dikunci

Semua parameter dipilih di fold 18 dan diterapkan ke fold 19 **tanpa diubah**. Prediksi positif bila `y_prob` mentah ≥ threshold.

| Parameter | Aturan di fold 18 | Berkas | Commit |
|---|---|---|---|
| Kalibrasi Platt M1–M6 | Regresi logistik pada logit `y_prob` | `reports/locked_fold18.json` | `61b8749` |
| **Threshold rule-out** (utama) | Threshold terbesar dengan sensitivitas ≥ 0,90 | `reports/locked_fold18.json` | `61b8749` |
| **Threshold rule-in** (sekunder) | Threshold terkecil dengan spesifisitas ≥ 0,90 | `reports/locked_fold18.json` | `61b8749` |
| **Threshold spesifisitas setara M0** | Threshold terkecil dengan spesifisitas ≥ spesifisitas M0 (primer) di fold 18 = **0,9782** (7.143 dari 7.302 negatif) | `reports/locked_fold18_m0.json` | `eef89dc` |

- M0 dan M0S1 berupa keputusan biner (positif bila `y_prob` = 1), tanpa kalibrasi dan threshold.
- Nilai threshold spesifisitas setara M0 (`y_prob` mentah):

  | M1 | M2 | M3 | M4 | M5 | M6 |
  |---|---|---|---|---|---|
  | 0,887541 | 0,845454 | 0,789982 | 0,832852 | 0,168243 | 0,169488 |

- Uji kode di fold 18 (laporan terbaru `reports/eval_fold18_uji_final.md`, commit `0740e58`) memastikan beberapa hal:
  - parameter memberi sensitivitas rule-out 0,904 dan spesifisitas rule-in 0,900;
  - spesifisitas tepat 0,9782 pada threshold setara M0;
  - kalibrasi in-sample sempurna untuk keenam model;
  - sha256 berkas prediksi fold 18 sama dengan saat parameter dikunci;
  - angka lama tidak berubah setelah fitur baru ditambahkan.

## 5. Perbandingan utama (satu)

**M5 vs M4**: apakah late fusion sinyal EKG + EMR lebih baik daripada baseline concat tabular?

- Metrik utama: **ΔAUROC = AUROC(M5) − AUROC(M4)**.
- Uji: bootstrap berpasangan per `subject_id` (Bagian 8). Perbedaan dianggap **bermakna** bila IK 95% ΔAUROC tidak mencakup 0.
- Hasil dilaporkan apa adanya, ke arah mana pun.
- Metrik pendukung untuk pasangan yang sama, masing-masing dengan IK selisih:
  - ΔAUPRC dan ΔBrier (setelah kalibrasi terkunci);
  - Δsensitivitas, Δspesifisitas, ΔPPV, dan ΔNPV pada threshold rule-out dan rule-in terkunci.

## 6. Perbandingan eksploratif

Semua perbandingan di bagian ini dilaporkan dengan IK selisih berpasangan dan **diberi label eksploratif**. Tidak ada koreksi multiplisitas, dan hal ini dinyatakan eksplisit di laporan.

| Pasangan | Pertanyaan | Metrik |
|---|---|---|
| M1 vs M3 | Apakah sinyal mentah lebih informatif daripada machine measurements? | ΔAUROC, ΔAUPRC |
| M6 vs M4 | Dengan informasi yang sama, apakah late fusion lebih baik daripada concat? | ΔAUROC, ΔAUPRC |
| M5 vs M2 | Apakah EKG menambah nilai di atas EMR? | ΔAUROC, ΔAUPRC |
| M5 vs M1 | Apakah EMR menambah nilai di atas EKG? | ΔAUROC, ΔAUPRC |
| Setiap model (M1–M6) vs M0, **pada spesifisitas setara M0** | Pertanyaan sekunder (d) proposal: pada spesifisitas yang sama dengan interpretasi mesin, apakah model menemukan lebih banyak kasus? **Jawaban utama untuk perbandingan dengan M0.** | Δsensitivitas, Δspesifisitas, ΔPPV, ΔNPV |
| Setiap model (M1–M6) vs M0, pada threshold rule-out dan rule-in | Gambaran trade-off pada titik operasi klinis model | Δsensitivitas dan Δspesifisitas |
| M0S1 vs M0 | Seberapa besar pengaruh definisi M0? (analisis sensitivitas, Bagian 10) | Δsensitivitas, Δspesifisitas, ΔPPV, ΔNPV |

- Perbandingan dengan baseline hanya memakai M0 primer. Sensitivitas, spesifisitas, PPV, dan NPV M0 dan M0S1 dilaporkan dengan IK bootstrap.
- Threshold setara M0 dikunci di fold 18, jadi spesifisitas model di fold 19 tidak dijamin persis sama dengan spesifisitas M0. Karena itu Δspesifisitas ikut dilaporkan dan dibaca bersama Δsensitivitas.

## 7. Metrik per model

- n, jumlah positif, prevalensi.
- Tanpa threshold:
  - **AUROC** dan **AUPRC** (AUPRC dibandingkan dengan prevalensi);
  - **Brier** mentah dan setelah kalibrasi terkunci;
  - **intercept dan slope kalibrasi**, mentah dan setelah kalibrasi;
  - **kurva kalibrasi fleksibel** setelah Platt untuk M1–M6:
    - regresi logistik hasil observasi terhadap restricted cubic spline dari logit p, dengan 4 knot pada persentil 5/35/65/95 logit p;
    - dievaluasi pada grid 50 titik yang rata di skala logit, antara persentil 1 dan 99 dari p;
    - IK 95% dari bootstrap per `subject_id` dengan knot dan grid tetap;
    - Gambar kalibrasi naskah menampilkan M4 dan M5;
  - tabel kalibrasi setelah Platt (10 bin dengan jumlah kunjungan sama).
- Pada threshold rule-out, rule-in, dan spesifisitas setara M0: **sensitivitas, spesifisitas, PPV, NPV, F1**, serta TP/FP/FN/TN.
- M0 dan M0S1: sensitivitas, spesifisitas, PPV, NPV, F1, dan TP/FP/FN/TN pada satu titik operasi. AUROC, AUPRC, Brier, dan kurva kalibrasi tidak dihitung untuk keduanya.
- **Kurva ROC tidak dilaporkan.** AUROC dilaporkan sebagai angka dengan IK.
- **Akurasi tidak dipakai** karena data sangat tidak seimbang.
- IK 95% (Bagian 8) dihitung untuk AUROC, AUPRC, Brier setelah kalibrasi, sensitivitas, spesifisitas, PPV, NPV, dan titik-titik kurva kalibrasi. Brier mentah serta intercept/slope dilaporkan tanpa IK.

## 8. Interval kepercayaan

- Bootstrap **per `subject_id`**: semua kunjungan seorang pasien ikut terambil bersama.
- **2.000 iterasi**, seed 42, IK persentil 95%.
- **Berpasangan**: di setiap iterasi, semua model dinilai pada sampel yang sama, dan selisih dihitung per iterasi.
- Iterasi tanpa kasus positif atau negatif tidak dihitung. Jumlah iterasi yang sah dilaporkan. Untuk kurva kalibrasi, iterasi yang gagal konvergen juga tidak dihitung.
- Implementasi: `src/evaluation/bootstrap.py` dan `src/evaluation/calibration_curve.py`, dipanggil oleh `src/evaluation/run_fold19.py`.

## 9. Subgrup dan analisis sensitivitas

Tidak ada threshold atau kalibrasi baru; semua memakai parameter terkunci. Untuk setiap subgrup dilaporkan AUROC, sensitivitas, dan spesifisitas pada threshold rule-out, dengan IK, untuk semua model (M0 dan M0S1 tanpa AUROC). Subgrup dengan positif **atau** negatif < 10 hanya dilaporkan jumlahnya, tanpa estimasi.

| Analisis | Definisi | Status |
|---|---|---|
| Hanya pasien rawat inap | `label_source == "hosp"` | **Wajib** (99,3% positif berasal dari rawat inap) |
| Tanpa IMA tipe 2 murni | Buang positif dengan `ami_type2_only == 1` | Analisis sensitivitas |
| Tanpa keluhan yang menyebut diagnosis | Buang `cc_explicit_mi == 1` | Analisis sensitivitas |
| IMA tipe 2 murni | Positif `ami_type2_only == 1` + semua negatif | Subgrup subtipe |
| STEMI | Positif `ami_stemi == 1` + semua negatif | Subgrup subtipe |
| NSTEMI | Positif `ami_nstemi == 1` + semua negatif | Subgrup subtipe |
| Jenis kelamin | `is_male` = 1 / 0 | Subgrup |
| Usia | < 65 vs ≥ 65 tahun (`age`) | Subgrup |
| Ketersediaan obat rutin | `medrecon_available` = 1 / 0 | Subgrup |

Perbandingan utama M5 − M4 (ΔAUROC) diulang **hanya** pada tiga analisis pertama, dan hasilnya diberi label eksploratif. Di tabel naskah, kolom selisih untuk subgrup lain ditulis "–".

## 10. Baseline M0: definisi primer dan analisis sensitivitas

Rincian lengkap ada di decision log 2026-11-24 dan `reports/m0_rule.md`.

- **Aturan M0-v2**:
  - sidik jari sha256 `f515ab4f4d4d8d85d2e249ecacbad94fe67fc4eaf93f5648b8e6d452df19589d`;
  - kode `src/model/m0_rule.py`, commit `650d599`; entri decision log di commit `b769b98`.
  - Aturan dikunci sebelum kinerja M0 dilihat pada fold mana pun. Penyusunannya hanya memakai audit pola di fold 0–16 tanpa label.
- **Satuan**: satu pernyataan `report_0` … `report_17` dari EKG pertama (huruf kecil). Kunjungan positif bila minimal satu pernyataannya positif. Kunjungan tanpa teks report dianggap 0.
- **M0 (primer)**, "mesin menandai kemungkinan infark, injury, atau iskemia akut". Pernyataan positif bila memuat salah satu dari:
  - STEMI;
  - acute + infark/MI;
  - infark + recent/evolving;
  - injury;
  - acute + iskemia.

  Pernyataan yang dinegasi ("no"/"without") dikecualikan.
- **M0S1 (analisis sensitivitas S1)**, "mesin menyatakan infark akut/STEMI". Sama dengan M0, kecuali:
  - injury hanya dihitung bila pernyataan yang sama tidak menyebut iskemia, perikarditis, atau repolarisasi;
  - acute + iskemia tidak dihitung.

  Akibatnya setiap kunjungan M0S1 = 1 juga M0 = 1.
- **Varian S2 tidak dianalisis**. Menurut audit, S2 hanya berbeda dari M0 pada < 10 pernyataan, dan melaporkannya bisa membuka sel kecil.
- "Injury" di sini adalah istilah EKG (current of injury), bukan "myocardial injury" biokimia (Fourth Universal Definition of MI, 2018).
- **Yang dilaporkan**:
  - kinerja M0 dan M0S1 (Bagian 7);
  - selisih setiap model terhadap M0 pada spesifisitas setara M0 dan pada threshold model (Bagian 6);
  - selisih M0S1 − M0 (Bagian 6);
  - jumlah kunjungan yang pindah dari M0 = 1 ke M0S1 = 0, per label dan per subtipe (STEMI, NSTEMI, tipe 2 murni);
  - kinerja kedua definisi di subgrup subtipe (Bagian 9).
- Definisi primer tidak diganti setelah hasil terlihat. Kalau M0S1 ternyata lebih baik, temuan itu dilaporkan sebagai hasil analisis sensitivitas.
- Threshold spesifisitas setara M0 dihitung dari M0 primer saja dan tidak dihitung ulang untuk M0S1.

## 11. XAI

**SHAP M2 dan M4 (global saja)**
- Metode: TreeSHAP bawaan XGBoost (`pred_contribs`, setara `shap.TreeExplainer`), pada iterasi terbaik yang sama dengan prediksi. Skala: log-odds model mentah (sebelum Platt).
- **Hasil utama: fold 19**, supaya konsisten dengan semua hasil lain.
- **Fold 18 sebagai cek stabilitas.** SHAP fold 18 sudah dihitung pada 2026-11-24 (commit `76d29fc`). Fold 18 tidak dipakai untuk melatih pohon M2 maupun M4.
- Ukuran stabilitas (deskriptif, tanpa ambang lulus/gagal), dihitung dengan `python -m src.model.shap_tabular --model {M2,M4} --stabilitas`:
  - korelasi Spearman peringkat rata-rata |SHAP| semua fitur asal antara fold 18 dan fold 19;
  - jumlah fitur yang sama di 10 teratas.
- Yang dilaporkan per model:
  - rata-rata |SHAP| dan porsinya per fitur asal (one-hot cara kedatangan dijumlahkan);
  - arah pengaruh per kelompok (biner 1 vs 0; kontinu kuartil bawah vs atas; kategori per kategori);
  - gambar 15 fitur teratas.
- **Tidak ada penjelasan per pasien.** Penjelasan per pasien akan menampilkan data individual MIMIC, sedangkan pertanyaan penelitian berada di tingkat populasi. Kalau naskah butuh contoh ilustratif, hanya boleh dipakai profil sintetis yang diberi label "bukan pasien nyata".
- Pemeriksaan wajib (skrip berhenti bila gagal):
  - aditivitas ≤ 1e-3;
  - sigmoid(margin) sama dengan `y_prob` berkas prediksi (≤ 1e-5).
- Interpretasi: nilai SHAP menggambarkan pola yang dipelajari model (asosiasi), **bukan hubungan sebab-akibat**. Fitur yang saling berkorelasi bisa berbagi atau saling menggantikan kontribusi.

**Grad-CAM dan oklusi per lead M1 (dikerjakan Orang 2)**
- Hasil validasi internal fold 0–16 (decision log 2026-11-05) **tetap dipakai**:
  - 150 IMA + 150 non-IMA, seed 42;
  - uji acak-bobot;
  - faithfulness waktu;
  - uji lead. Uji lead tetap **tidak** dipakai sebagai bukti faithfulness karena hampir melingkar (catatan 5 November).
- **Uji yang sama diulang di fold 19** dengan sampel dan seed yang sama (150 IMA + 150 non-IMA, seed 42), sebagai konfirmasi pada data independen:
  - `python -u -m src.xai.run_xai_sanity --fold 19 --include-fold-19`
  - `python -u -m src.xai.run_xai_faithfulness --fold 19 --include-fold-19`
- **Uji kode di fold 18** (`--fold 18`) sudah dijalankan Orang 2 pada 2026-11-24 dengan skrip `jalankan_xai_fold18.sh` (commit `dbec679`). Keluarannya, yang hanya berisi angka agregat, disimpan di `reports/xai_fold18.txt`. Hasilnya **lolos**:
  - 11 tes lolos;
  - perilaku bawaan (validasi internal) mengulang angka 5 November (korelasi acak-bobot median −0,042, p90 0,176);
  - kedua jalan fold 18 berlabel uji kode.

  Rinciannya ada di decision log 2026-11-24. Angka fold 18 itu hanya uji kode, bukan hasil naskah.
- Opsi `--fold` ditambahkan ke skrip XAI atas izin Orang 2 (commit `dbec679`). Tanpa opsi itu, skrip tetap memilih rekaman validasi internal yang sama seperti versi asli.

**XAI lain**
- Koefisien meta-learner M5/M6 sudah dilaporkan dari fold 17 (decision log 2026-11-14) dan tidak dihitung ulang.

## 12. Pelaporan dan sel kecil

- Semua keluaran hanya berisi angka agregat; tidak ada ID atau nilai per pasien.
- **Aturan "<10" berlaku untuk hitungan.** Setiap hitungan < 10 ditulis "<10" (rencana proyek 2.5). **Proporsi** (sensitivitas, spesifisitas, PPV, NPV) **dan IK-nya dilaporkan apa adanya**, termasuk di subgrup.
- **Supresi komplementer**: bila satu bagian dari suatu total < 10, bagian lain yang terkecil ikut ditulis "disembunyikan". Ini berlaku untuk:
  - TP/FN dan FP/TN;
  - perpindahan M0 → M0S1 per label dan per subtipe;
  - jumlah kunjungan per kelompok SHAP biner dan per kategori.
- Keluaran resmi:
  - evaluasi: `reports/eval_fold19.md` dan CSV pendukung (`eval_fold19_metrics.csv`, `_comparisons.csv`, `_counts.csv`, `_subgroups.csv`, `_calibration_curve.csv`, `_calibration_bins.csv`);
  - M0: `reports/m0_rule.md`;
  - SHAP: `reports/shap_{M2,M4}_fold19.md`, `reports/shap_{M2,M4}_fold19.csv`, `reports/figures/shap_{M2,M4}_fold19.png`, `reports/shap_{M2,M4}_stabilitas.md`;
  - XAI M1: `reports/xai_m1_sanity_fold19.txt` dan `reports/xai_m1_faithfulness_fold19.txt`.
- Gambar naskah dibuat dari CSV agregat keluaran evaluasi, tanpa membaca prediksi dan tanpa analisis tambahan. Kedua skrip sudah diuji di fold 18.
  - Gambar 2 (forest plot AUROC/AUPRC): `python -m src.evaluation.plot_forest --fold 19` → `reports/figures/gambar2_forest_fold19.png`.
  - Gambar kalibrasi (M4 dan M5): `python -m src.evaluation.plot_calibration --fold 19` → `reports/figures/kalibrasi_fold19.png`.

## 13. Prosedur

1. Dokumen ini disetujui (Bagian 16). Baris status di bagian atas diganti menjadi final berikut tanggalnya, lalu di-commit dan **di-push ke repo tim privat sebelum langkah 3**, supaya ada cap waktu di GitHub. Hash commit dicatat di decision log.
2. **Selesai.** Parameter yang sudah dikunci:
   - aturan M0-v2: commit `650d599`, decision log `b769b98`, diuji di fold 18 (`76d29fc`);
   - threshold dan kalibrasi: `61b8749`;
   - threshold spesifisitas setara M0: `eef89dc`;
   - kode evaluasi terbaru diuji di fold 18 (`0740e58`).
3. Prediksi fold 19 dibuat **tanpa menghitung metrik apa pun**:
   - M1: **sudah dibuat** oleh Orang 2 (2026-11-21).
     - 7.217 baris; sha256 `d75dfb9e295cb10e1d98a743bf9474792b9458e38b24a4c48fde8ab7e1ee9460`, cocok antara salinan Orang 2 dan Orang 3.
     - Berkas tidak dibuat ulang.
     - Berkas ini dibuat sebelum rencana di-commit, dan dicatat apa adanya di decision log (Bagian 14).
   - M0 dan M0S1: `python -m src.model.m0_rule --folds 19 --include-fold-19`
   - M2, M3, M4: `python -m src.model.train_tabular --model M2 --predict-only --folds 19 --include-fold-19` (ulangi untuk M3 dan M4).
   - M5, M6 (setelah M1, M2, dan M3 fold 19 tersedia): `python -m src.model.train_meta --model M5 --predict-only --folds 19 --include-fold-19` (ulangi untuk M6).
4. Checksum sha256 semua berkas `predictions/*_fold19.parquet` dicetak dengan `python -m src.evaluation.run_fold19 --checksums`. Hasilnya dicatat di decision log, lalu di-commit dan di-push **sebelum** evaluasi.
5. Evaluasi dijalankan **satu kali**: `python -m src.evaluation.run_fold19 --fold 19 --include-fold-19`. Skrip menolak jalan kedua. Jalan ulang hanya bisa dengan `--overwrite-post-hoc`, ditulis ke laporan terpisah, dan dilaporkan sebagai post-hoc. Skrip juga mewajibkan `locked_fold18.json`, `locked_fold18_m0.json`, `m0_rule.py`, dan dokumen ini sudah di-commit tanpa perubahan.
6. SHAP fold 19, lalu cek stabilitas:
   - `python -m src.model.shap_tabular --model M2 --fold 19 --include-fold-19` (ulangi untuk M4);
   - `python -m src.model.shap_tabular --model M2 --stabilitas` (ulangi untuk M4).
7. Gambar naskah dari keluaran agregat langkah 5:
   - `python -m src.evaluation.plot_forest --fold 19`
   - `python -m src.evaluation.plot_calibration --fold 19`
8. XAI M1 fold 19 oleh Orang 2, di laptopnya. Repo lokalnya harus sudah berisi dokumen ini dalam status final (`git pull --no-rebase origin evaluation`).
   - `python -u -m src.xai.run_xai_sanity --fold 19 --include-fold-19 | tee reports/xai_m1_sanity_fold19.txt`
   - `python -u -m src.xai.run_xai_faithfulness --fold 19 --include-fold-19 | tee reports/xai_m1_faithfulness_fold19.txt`

   Kedua berkas keluaran (agregat) di-commit apa adanya.
9. Semua laporan agregat di-commit apa adanya. Karena `.gitignore` mengabaikan `*.csv`, CSV pendukung ditambahkan secara eksplisit: `git add -f reports/eval_fold19_*.csv reports/shap_M2_fold19.csv reports/shap_M4_fold19.csv`. Push **hanya ke repo tim privat** (`melinda-ab/deteksi-serangan-jantung`).
10. Tidak ada tuning ulang setelah melihat fold 19. Kalau terpaksa, perubahan dicatat di decision log dan dilaporkan sebagai post-hoc, terpisah dari hasil utama.

Tidak ada anggota tim yang menghitung metrik fold 19 di luar langkah 5, 6, dan 8.

## 14. Hal yang perlu diputuskan atau diselesaikan sebelum dokumen ini final

Keputusan butir pertama sampai kesembilan diambil pada 2026-11-24 (decision log, "Keputusan A2").

- [x] Metrik utama M5 vs M4: **ΔAUROC** (sesuai proposal 2.8).
- [x] M5 vs M1 sebagai perbandingan eksploratif (sesuai proposal 2.8).
- [x] "Sensitivitas pada spesifisitas M0" **dipakai** sebagai jawaban utama pertanyaan sekunder (d). Threshold-nya **dikunci di fold 18** (`eef89dc`), bukan dipilih dari fold 19. Usulan DRAF v3 untuk tidak memakainya dibatalkan.
- [x] Kurva kalibrasi fleksibel dilaporkan (TRIPOD+AI butir 12e; Van Calster dkk. 2019). Kurva ROC tidak dilaporkan.
- [x] Aturan "<10" berlaku untuk hitungan; proporsi dan IK dilaporkan apa adanya (Bagian 12).
- [x] Grad-CAM/oklusi M1: hasil validasi internal fold 0–16 tetap, dan uji yang sama diulang di fold 19 (Bagian 11).
- [x] SHAP global saja, tanpa penjelasan per pasien (Bagian 11).
- [x] Hasil fold 19 hanya di-push ke repo tim privat (Bagian 13).
- [x] Pembuat `M1_fold19`: Orang 2. Berkasnya sudah ada dan checksum-nya cocok.
- [x] `signal_qc_full.csv` identik dengan `signal_qc.csv`. sha256 lengkap dicatat di entri decision log 2026-11-21.
- [x] Entri decision log Orang 2 tentang `M1_fold19` sudah ditambahkan (commit `5bfc459`).
- [x] Uji kode XAI M1 di fold 18 oleh Orang 2 lolos (`jalankan_xai_fold18.sh`, commit `dbec679`, 2026-11-24). Keluaran ada di `reports/xai_fold18.txt` dan dicatat di decision log.
- [ ] Ejaan nama Orang 1 dikonfirmasi (Ana/Anna).
- [ ] Persetujuan **atas teks dokumen ini** oleh Orang 1, Orang 2, Orang 3, dan dosen pembimbing (Bagian 16). Tanggal persetujuan harus sama dengan atau setelah tanggal teks ini. Pernyataan setuju Orang 1 dan Orang 2 pada 23 November berlaku untuk draf sebelumnya, dan dicatat sebagai riwayat di decision log.

## 15. Keterbatasan yang sudah diketahui (dilaporkan apa pun hasilnya)

- M1 dilatih pada sekitar 13% data fold 0–16, sehingga M1 dan M5 kemungkinan dirugikan dibanding model tabular yang dilatih dengan data penuh.
- Label berasal dari diagnosis saat pulang (kode ICD), bukan kondisi saat EKG direkam. 99,3% positif berasal dari pasien rawat inap.
- Label mencakup IMA tipe 2. Interpretasi mesin EKG tidak dirancang untuk mendeteksinya, sehingga sensitivitas M0 diperkirakan rendah.
- Tidak ada troponin, sehingga performa pada NSTEMI kemungkinan lebih rendah.
- EKG MIMIC-IV-ECG direkam dengan mesin dari beberapa produsen. Teks interpretasi mesin (M0) tidak seragam antar-produsen, dan aturan kata kunci hanya mendekati keputusan mesin.
- Prediksi `M1_fold19` dibuat sebelum rencana ini di-commit. Berkasnya dikunci dengan checksum, dan belum ada metrik yang dihitung darinya.
- Data berasal dari satu rumah sakit di AS; validasi eksternal diperlukan sebelum model dipakai di tempat lain.
- Metrik pada threshold dan kurva kalibrasi di fold 18 bersifat in-sample. Hanya fold 19 yang menjadi hasil akhir.
- Threshold setara M0 dikunci di fold 18, sehingga spesifisitas model di fold 19 bisa sedikit berbeda dari spesifisitas M0.
- SHAP menggambarkan asosiasi yang dipelajari model, bukan sebab-akibat.
- XAI M1 dinilai pada sampel 300 rekaman per fold, bukan seluruh fold. Profil Grad-CAM relatif datar sepanjang rekaman (catatan 5 November), sehingga lokalisasinya sedang.

## 16. Persetujuan

Persetujuan diberikan atas teks dokumen ini. Tanggal diisi pada hari persetujuan, yaitu sama dengan atau setelah tanggal teks ini.

| Peran | Nama | Tanggal | Setuju |
|---|---|---|---|
| Orang 1 | Ana | | |
| Orang 2 | Ayu | | |
| Orang 3 | Freta Yordinia Laura | | |
| Dosen pembimbing | Drs. Medi, M.Kom. | | |
