# Rencana Analisis Final (Fold 19)

**Status: DRAF v3 (2026-11-24), disusun Orang 3. Belum berlaku.**

Dokumen ini disimpan di `reports/rencana_analisis_fold19.md`. Dokumen menjadi final setelah:
- disetujui tim (Bagian 16);
- baris status di atas diganti menjadi status final, berikut tanggalnya;
- di-commit ke repo.

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

- Kalibrasi Platt dan threshold M1–M6 diambil dari `reports/locked_fold18.json`, commit `61b8749` ("threshold M1-M6 dikunci di fold 18").
- **Threshold rule-out** (utama): sensitivitas ≥ 0,90 di fold 18. **Threshold rule-in** (sekunder): spesifisitas ≥ 0,90 di fold 18.
- Kedua threshold diterapkan ke fold 19 **tanpa diubah**. Prediksi positif bila `y_prob` mentah ≥ threshold.
- M0 dan M0S1 berupa keputusan biner (positif bila `y_prob` = 1), tanpa kalibrasi dan threshold.
- Uji kode di fold 18 (`reports/eval_fold18_uji_final.md`, commit `76d29fc`) memastikan parameter yang sama menghasilkan sensitivitas rule-out 0,904, spesifisitas rule-in 0,900, dan kalibrasi in-sample sempurna untuk keenam model. Uji itu juga memastikan sha256 berkas prediksi fold 18 sama dengan saat parameter dikunci.

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
| Setiap model (M1–M6) vs M0 | Apakah model lebih baik daripada interpretasi mesin EKG? | Δsensitivitas dan Δspesifisitas pada threshold rule-out dan rule-in model |
| M0S1 vs M0 | Seberapa besar pengaruh definisi M0? (analisis sensitivitas, Bagian 10) | Δsensitivitas, Δspesifisitas, ΔPPV, ΔNPV |

Perbandingan dengan baseline hanya memakai M0 primer. Sensitivitas, spesifisitas, PPV, dan NPV M0 dan M0S1 dilaporkan dengan IK bootstrap.

## 7. Metrik per model

- n, jumlah positif, prevalensi.
- Tanpa threshold:
  - **AUROC** dan **AUPRC** (AUPRC dibandingkan dengan prevalensi);
  - **Brier** mentah dan setelah kalibrasi terkunci;
  - **intercept dan slope kalibrasi**, mentah dan setelah kalibrasi;
  - tabel kalibrasi setelah Platt (10 bin dengan jumlah kunjungan sama).
- Pada threshold rule-out dan rule-in terkunci: **sensitivitas, spesifisitas, PPV, NPV, F1**, serta TP/FP/FN/TN.
- M0 dan M0S1: sensitivitas, spesifisitas, PPV, NPV, F1, dan TP/FP/FN/TN pada satu titik operasi. AUROC, AUPRC, dan Brier tidak dihitung untuk keduanya.
- **Akurasi tidak dipakai** karena data sangat tidak seimbang.
- IK 95% (Bagian 8) dihitung untuk AUROC, AUPRC, Brier setelah kalibrasi, sensitivitas, spesifisitas, PPV, dan NPV. Brier mentah serta intercept/slope dilaporkan tanpa IK.

## 8. Interval kepercayaan

- Bootstrap **per `subject_id`**: semua kunjungan seorang pasien ikut terambil bersama.
- **2.000 iterasi**, seed 42, IK persentil 95%.
- **Berpasangan**: di setiap iterasi, semua model dinilai pada sampel yang sama, dan selisih dihitung per iterasi.
- Iterasi tanpa kasus positif atau negatif tidak dihitung. Jumlah iterasi yang sah dilaporkan.
- Implementasi: `src/evaluation/bootstrap.py`, dipanggil oleh `src/evaluation/run_fold19.py`.

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

Perbandingan utama M5 − M4 (ΔAUROC) diulang **hanya** pada tiga analisis pertama, dan hasilnya diberi label eksploratif.

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
  - selisih M0S1 − M0 (Bagian 6);
  - jumlah kunjungan yang pindah dari M0 = 1 ke M0S1 = 0, per label dan per subtipe (STEMI, NSTEMI, tipe 2 murni);
  - kinerja kedua definisi di subgrup subtipe (Bagian 9).
- Definisi primer tidak diganti setelah hasil terlihat. Kalau M0S1 ternyata lebih baik, temuan itu dilaporkan sebagai hasil analisis sensitivitas.

## 11. XAI

**SHAP M2 dan M4**
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
- Pemeriksaan wajib (skrip berhenti bila gagal):
  - aditivitas ≤ 1e-3;
  - sigmoid(margin) sama dengan `y_prob` berkas prediksi (≤ 1e-5).
- Interpretasi: nilai SHAP menggambarkan pola yang dipelajari model (asosiasi), **bukan hubungan sebab-akibat**. Fitur yang saling berkorelasi bisa berbagi atau saling menggantikan kontribusi.

**XAI lain**
- Koefisien meta-learner M5/M6 sudah dilaporkan dari fold 17 (decision log 2026-11-14) dan tidak dihitung ulang.
- Grad-CAM dan oklusi per lead untuk M1 dikerjakan Orang 2. Fold yang dipakai diputuskan di Bagian 14.

## 12. Pelaporan dan sel kecil

- Semua keluaran hanya berisi angka agregat; tidak ada ID atau nilai per pasien.
- Setiap hitungan < 10 ditulis "<10" (rencana proyek 2.5).
- **Supresi komplementer**: bila satu bagian dari suatu total < 10, bagian lain yang terkecil ikut ditulis "disembunyikan". Ini berlaku untuk:
  - TP/FN dan FP/TN;
  - perpindahan M0 → M0S1 per label dan per subtipe;
  - jumlah kunjungan per kelompok SHAP biner dan per kategori.
- Keluaran resmi:
  - evaluasi: `reports/eval_fold19.md` dan CSV pendukung (metrik, perbandingan, jumlah sel, subgrup);
  - M0: `reports/m0_rule.md`;
  - SHAP: `reports/shap_{M2,M4}_fold19.md`, `reports/figures/shap_{M2,M4}_fold19.png`, `reports/shap_{M2,M4}_stabilitas.md`.
- Gambar naskah (forest plot AUROC/AUPRC) dibuat dengan `python -m src.evaluation.plot_forest --fold 19` dari `reports/eval_fold19_metrics.csv` dan `reports/eval_fold19.md`, tanpa membaca prediksi dan tanpa analisis tambahan. Skrip ini sudah diuji di fold 18 (`reports/figures/gambar2_forest_fold18.png`). Kurva ROC dan kurva kalibrasi: lihat Bagian 14.

## 13. Prosedur

1. Dokumen ini disetujui (Bagian 16). Baris status di bagian atas diganti menjadi final berikut tanggalnya, lalu di-commit. Hash commit dicatat di decision log.
2. **Selesai.** Aturan M0-v2 dikunci di commit `650d599`, dicatat di decision log (`b769b98`), dan kodenya diuji di fold 18 (`76d29fc`).
3. Prediksi fold 19 dibuat **tanpa menghitung metrik apa pun**:
   - M1: **sudah dibuat** oleh Orang 2 (2026-11-21).
     - 7.217 baris; sha256 `d75dfb9e295cb10e1d98a743bf9474792b9458e38b24a4c48fde8ab7e1ee9460`, cocok antara salinan Orang 2 dan Orang 3.
     - Berkas tidak dibuat ulang.
     - Berkas ini dibuat sebelum rencana di-commit, dan dicatat apa adanya di decision log (Bagian 14).
   - M0 dan M0S1: `python -m src.model.m0_rule --folds 19 --include-fold-19`
   - M2, M3, M4: `python -m src.model.train_tabular --model M2 --predict-only --folds 19 --include-fold-19` (ulangi untuk M3 dan M4).
   - M5, M6 (setelah M1, M2, dan M3 fold 19 tersedia): `python -m src.model.train_meta --model M5 --predict-only --folds 19 --include-fold-19` (ulangi untuk M6).
4. Checksum sha256 semua berkas `predictions/*_fold19.parquet` dicetak dengan `python -m src.evaluation.run_fold19 --checksums`. Hasilnya dicatat dan di-commit di decision log **sebelum** evaluasi.
5. Evaluasi dijalankan **satu kali**: `python -m src.evaluation.run_fold19 --fold 19 --include-fold-19`. Skrip menolak jalan kedua. Jalan ulang hanya bisa dengan `--overwrite-post-hoc`, ditulis ke laporan terpisah, dan dilaporkan sebagai post-hoc.
6. SHAP fold 19, lalu cek stabilitas:
   - `python -m src.model.shap_tabular --model M2 --fold 19 --include-fold-19` (ulangi untuk M4);
   - `python -m src.model.shap_tabular --model M2 --stabilitas` (ulangi untuk M4).
7. Gambar naskah dibuat dari keluaran agregat langkah 5 dan 6 (`python -m src.evaluation.plot_forest --fold 19`).
8. Semua laporan agregat di-commit apa adanya. Karena `.gitignore` mengabaikan `*.csv`, CSV pendukung ditambahkan secara eksplisit: `git add -f reports/eval_fold19_*.csv reports/shap_M2_fold19.csv reports/shap_M4_fold19.csv`.
9. Tidak ada tuning ulang setelah melihat fold 19. Kalau terpaksa, perubahan dicatat di decision log dan dilaporkan sebagai post-hoc, terpisah dari hasil utama.

Tidak ada anggota tim yang menghitung metrik fold 19 di luar langkah 5 dan 6.

## 14. Hal yang perlu diputuskan atau diselesaikan sebelum dokumen ini final

- [ ] Metrik utama M5 vs M4: **AUROC**. Sudah diimplementasikan dan diuji di fold 18.
- [ ] M5 vs M1 sebagai perbandingan eksploratif. Sudah diimplementasikan.
- [ ] Usulan DRAF v1 "sensitivitas setiap model pada spesifisitas yang sama dengan M0" **tidak dipakai**. Alasannya, threshold itu harus dipilih dari data fold 19 sendiri, dan perbandingan dengan M0 sudah dilakukan pada threshold terkunci (Bagian 6).
- [x] Pembuat `M1_fold19`: Orang 2. Berkasnya sudah ada dan checksum-nya cocok.
- [x] Orang 2 mengonfirmasi bahwa `signal_qc_full.csv` (yang dipakai untuk `M1_fold19`) identik dengan `data/interim/signal_qc.csv`, yang sha256-nya diawali `68353a95` dan diakhiri `bb23b`. Bila berbeda, jumlah kunjungan fold 19 yang berbeda dilaporkan (agregat) dan ditangani sebelum dokumen ini final. **Selesai:** identik; sha256 lengkap dicatat di entri decision log 2026-11-21.
- [x] Entri decision log Orang 2 tentang `M1_fold19` ditambahkan ke `reports/decision_log.md`. Entri itu mencakup tanggal pembuatan, checksum, dan pernyataan bahwa belum ada metrik yang dihitung. **Selesai:** ditambahkan 2026-11-24.
- [ ] Perlu atau tidaknya kurva ROC (dengan titik operasi M0) dan kurva kalibrasi di naskah. Kalau perlu, Orang 3 menambahkan ekspor titik kurva agregat ke `run_fold19.py` dan mengujinya di fold 18 sebelum dokumen ini final.
- [ ] Fold untuk Grad-CAM dan oklusi M1 (Orang 2). Usulan: sama dengan SHAP, yaitu fold 19 utama.
- [ ] Persetujuan dosen pembimbing: definisi label, aturan M0-v2, dan rencana ini.

## 15. Keterbatasan yang sudah diketahui (dilaporkan apa pun hasilnya)

- M1 dilatih pada sekitar 13% data fold 0–16, sehingga M1 dan M5 kemungkinan dirugikan dibanding model tabular yang dilatih dengan data penuh.
- Label berasal dari diagnosis saat pulang (kode ICD), bukan kondisi saat EKG direkam. 99,3% positif berasal dari pasien rawat inap.
- Label mencakup IMA tipe 2. Interpretasi mesin EKG tidak dirancang untuk mendeteksinya, sehingga sensitivitas M0 diperkirakan rendah.
- Tidak ada troponin, sehingga performa pada NSTEMI kemungkinan lebih rendah.
- EKG MIMIC-IV-ECG direkam dengan mesin dari beberapa produsen. Teks interpretasi mesin (M0) tidak seragam antar-produsen, dan aturan kata kunci hanya mendekati keputusan mesin.
- Prediksi `M1_fold19` dibuat sebelum rencana ini di-commit. Berkasnya dikunci dengan checksum, dan belum ada metrik yang dihitung darinya.
- Data berasal dari satu rumah sakit di AS; validasi eksternal diperlukan sebelum model dipakai di tempat lain.
- Metrik pada threshold di fold 18 bersifat in-sample. Hanya fold 19 yang menjadi hasil akhir.
- SHAP menggambarkan asosiasi yang dipelajari model, bukan sebab-akibat.

## 16. Persetujuan

| Peran | Nama | Tanggal | Setuju |
|---|---|---|---|
| Orang 1 | | | |
| Orang 2 | | | |
| Orang 3 | Freta Yordinia Laura | | |
| Dosen pembimbing | | | |
