## 2026-10-23 — Usulan aturan QC sinyal (Orang 2) — BELUM DISETUJUI

**Konteks.** Aturan PROJECT_BRIEF 10.2 (gagal jika ada lead dengan > 20% NaN, atau >= 3 lead datar) diterapkan pada 300 rekaman sampel (fold 0-16, acak: 100 positif + 200 negatif) dan 20 rekaman pilot. Hasil: 299 dari 300 sampel lolos.

**Temuan.** 4 dari 319 rekaman (1,3%) memiliki satu lead dengan artefak besar (maksimum 9,8-22,5 mV; p99 5,5-18,7 mV) sementara 11 lead lain normal (maksimum <= 2,4 mV). Aturan brief meloloskan keempatnya; setelah clip +-3 mV, lead itu menjadi derau jenuh. Ambang amplitudo 20 mV hanya menangkap 2 dari 4. Dua rekaman bervoltase tinggi yang tampak sah (maksimum 3,8 dan 5,2 mV, beberapa lead tinggi sekaligus) tidak termasuk kasus rusak.

**Usulan (perlu persetujuan tim).** Tambahkan aturan: gagal jika ada lead dengan |x| > 8 mV. Ambang ini berada di celah antara 5,2 mV (tampak sah) dan 9,8 mV (rusak) pada data yang ada. Alternatif: hanya menandai (`amplitude_flag`), atau menolkan lead yang rusak dan mempertahankan rekamannya.

**Catatan.** Dasar ambang hanya 6 rekaman. Validasi ulang pada rekaman tambahan dari fold 0-16 sebelum dikunci; jangan menyetel ambang memakai fold 19.


## 2026-10-23 — Clipping ±3 mV dipertahankan sementara (DISETUJUI; efektivitas BELUM divalidasi)

**Keputusan:** clipping ±3 mV tetap dipakai sesuai PROJECT_BRIEF 10.3 demi konsistensi dengan brief. Efektivitasnya terhadap performa dan stabilitas training belum divalidasi.

**Ketentuan:**
1. Clipping dilakukan dalam satuan mV (satuan header MIMIC-IV-ECG), setelah resample 500→100 Hz dan pemotongan 10 detik, sebelum standardisasi per-lead. Urutan ini sesuai brief 10.3.
2. Standardisasi tetap memakai statistik global yang dihitung dari data training saja. Clipping sebelum standardisasi juga melindungi statistik tersebut dari lead yang rusak. Tidak diubah secara sepihak.
3. `amplitude_flag` (> 20 mV) tetap penanda informatif. Aturan hard fail 8 mV per lead BELUM diadopsi (lihat entri usulan sebelumnya). QC mengikuti brief 10.2: gagal jika ada lead dengan > 20% NaN atau ≥ 3 lead datar.
4. Clipping tidak dimatikan dan standardisasi global tidak diubah sampai ablasi selesai dan keputusan baru dicatat di berkas ini.

**Rencana ablasi (setelah pipeline dan dataset stabil):**
- Bandingkan konfigurasi clipping (baseline brief ±3 mV, ambang lain, tanpa clipping dengan statistik robust, dan alternatif layak lain yang disepakati tim saat itu).
- Pemilihan HANYA memakai fold 0–16 (`internal_val_0_16`). Fold 17–19 tidak dipakai untuk memilih metode; fold 19 tidak disentuh sama sekali sebelum evaluasi akhir.
- Hasil dan keputusan akhir dicatat di decision_log.

**Catatan jejak fold pilot:** 20 rekaman pilot (`run_signal_qc_pilot`, tanpa filter fold) mencakup 2 rekaman di fold 18 dan 0 di fold 19. Pilot hanya QC sinyal tanpa label dan tanpa penyetelan threshold. Rekaman V6 rusak yang menjadi bukti usulan 8 mV berasal dari fold 4. Runner pilot tidak dipakai lagi untuk pemrosesan data.


## 2026-10-23 — Jalur QC tunggal sesuai brief, kontrak status signal_qc, validasi mV, pilot hanya fold 0-16 (DISETUJUI)

**Keputusan:**
1. Semua runner QC (`run_signal_qc_pilot`, `run_signal_qc_sample`, `run_signal_qc_brief`) memakai satu jalur di `src/dataset/signal_qc_core.py`: `read_ecg_record` -> `check_ecg_quality_brief` -> `prepare_ecg`, identik dengan `ECGDataset`. Fungsi lama (`load_ecg_record`, `check_ecg_quality`, `preprocess_ecg`) dipertahankan untuk tes lama, tetapi tidak dipakai pipeline (dijaga tes `test_runners_have_no_legacy_path`).
2. Kontrak status `signal_qc`: `pass` -> signal_ok=1; `qc_fail`, `preprocess_fail`, `load_fail` -> 0; `download_failed` dan `not_processed` -> kosong. Rekaman yang belum diperiksa tidak pernah dihitung lolos (`build_master` hanya menyimpan signal_ok == 1). `signal_qc.csv` produksi harus berisi SEMUA ed_stay_id master (`build_contract`); rekaman tanpa hasil = `not_processed`.
3. `read_ecg_record` memvalidasi satuan mV dari baris sinyal `.hea` mentah. wfdb mengisi satuan kosong menjadi mV secara diam-diam, sehingga `record.units` tidak cukup. Satuan kosong, tidak konsisten, atau bukan mV -> ValueError `unit_invalid` -> status `load_fail`. Alasan: clipping +-3 mV dan amplitude_flag hanya bermakna dalam mV.
4. Sampel pilot dibatasi fold 0-16. Fold 17-19 tidak boleh dipakai untuk menetapkan atau menyetel aturan QC.
5. Aturan QC TIDAK berubah: gagal jika ada lead > 20% NaN atau >= 3 lead datar. `amplitude_flag` (> 20 mV) tetap informatif; hard fail 8 mV tidak diadopsi. Clipping +-3 mV dipertahankan (lihat entri clipping); efektivitasnya belum divalidasi dan ablasinya direncanakan hanya pada fold 0-16.

**Keterbatasan bukti:**
- Validasi mV: 319 header di cache lokal semuanya menuliskan `mV` eksplisit (3.828 baris lead). Ini sampel dari fold 0-16 ditambah pilot, bukan seluruh dataset. Rekaman dengan satuan lain akan muncul sebagai `load_fail` dan perlu ditinjau, bukan diasumsikan mV.
- Inf tidak bisa ada di `.dat` 16-bit; penanganannya (diperlakukan seperti NaN) hanya diuji pada tingkat array.
- Hasil QC lama (`signal_qc_sample.csv`, aturan lama) bukan sumber QC dan tidak dipakai pipeline.
- Pilot lama (tanpa filter fold) memuat 2 rekaman fold 18 dan 0 fold 19; hanya QC sinyal tanpa label dan tanpa penyetelan. Berkas mentahnya masih ada di cache lokal dan tidak dipakai untuk keputusan apa pun.
- Cakupan QC pada seluruh dataset belum diketahui; baru sampel kecil (319 rekaman) yang diperiksa.

**Rencana ablasi clipping:** sama seperti entri clipping: bandingkan konfigurasi hanya pada fold 0-16 (`internal_val_0_16`), tidak pernah memakai fold 17-19 untuk memilih metode atau threshold.


## 2026-10-23 — Skala data M1: tahap 2a (positif penuh + negatif 3x, seed 42); fold 17-19 disiapkan terpisah

**Keputusan (usulan Orang 2; menunggu konfirmasi Orang 1 dan Orang 3):**
1. Training awal M1 (tahap 2a) memakai SEMUA positif fold 0-16 (4.134) + 3x negatif (12.402) acak, seed 42, total 16.536 rekaman unik. Sampel tahap 1 (1.000 rekaman) termuat di dalamnya. Komposisi setelah pembagian brief 7.2: training 14.835 rekaman (3.683 positif, 24,8%) dan validasi `internal_val_0_16 == 1` 1.701 rekaman (451 positif, 26,5%).
2. Negatif tambahan hanya bertahap, berdasarkan kebutuhan dan hasil validasi internal (fold 0-16), tidak berdasarkan fold 17-19.
3. Unduhan dan QC fold 17-19 (22.269 rekaman, 688 positif) disiapkan terpisah dengan aturan QC yang tetap. Tidak ada penyetelan threshold, clipping, atau preprocessing dari fold evaluasi. Fold 19 tetap disegel sampai evaluasi akhir (brief 11.5).

**Alasan:** laju unduh terukur sekitar 13,3 rekaman/menit (80 rekaman dalam 360 detik, satu jendela pengukuran); seluruh fold 0-16 yang belum di cache (125.639 rekaman) kira-kira 6,5 hari nonstop, 2a kira-kira 0,8 hari, fold 17-19 kira-kira 1,2 hari. Ekstrapolasi kasar dengan laju konstan, bukan jaminan.

**Perbedaan dari PROJECT_BRIEF (dicatat, bukan disembunyikan):**
- Brief 10.1 mengasumsikan seluruh 149 ribu EKG diunduh; 2a hanya sebagian fold 0-16.
- Brief 10.4 menyebut `pos_weight` sekitar 30 (prevalensi 3,2%). Dengan negatif disubsampel 3:1, `pos_weight = n_neg/n_pos` pada data training sekitar 3. Tujuannya sama (menyeimbangkan kelas), angkanya berbeda.
- Validasi internal ikut terdampak subsampling (sekitar 26% positif), sehingga AUPRC validasi tidak sebanding dengan prevalensi asli. Dipakai hanya untuk early stopping. Threshold dan kalibrasi tetap di fold 18 (prevalensi asli), sesuai brief.
- Penyimpanan praproses ke memmap/HDF5 (brief 10.3 langkah 7) belum dibuat; ECGDataset memproses dari cache WFDB ke RAM per run.

**Verifikasi pembagian (agregat, tabel master draft):** validasi internal 12.822 dari 126.734 rekaman fold 0-16 (10,1%); 0 pada fold >= 17; 0 pasien muncul di training dan validasi sekaligus.

**Verifikasi pipeline tahap 1 (999 rekaman lolos QC dari 1.000; 1 download_failed karena HTTP 500):** loss training turun; checkpoint dimuat ulang menghasilkan prediksi identik. Kontrol label diacak (3 seed, n_val=99 dengan 29 positif): AUROC validasi 0,466 / 0,458 / 0,570 versus label asli 0,664 / 0,701 / 0,686. Ini hanya uji pipeline pada subset diperkaya positif dan validasi kecil, BUKAN klaim performa model.

**Keterbatasan bukti:** laju unduh dari satu jendela pengukuran (kemungkinan ada unduhan lain yang berjalan bersamaan); subset tahap 1 berproporsi 30% positif; 1 dari 80 rekaman pada pengukuran laju gagal QC (alasan tidak disimpan).


## 2026-10-23 — Klarifikasi tahap 2a: urutan sampling vs pembagian, pos_weight, dan fungsi signal_qc_stage2a.csv

Mengoreksi/melengkapi entri "Skala data M1: tahap 2a" di atas. Tidak ada keputusan baru; mengikuti PROJECT_BRIEF.

1. **Urutan sampling dan pembagian.** Sampel 2a (semua positif + 3x negatif, seed 42) diambil dari seluruh pool fold 0-16 SEBELUM pembagian train/validasi, lalu dibagi dengan flag `internal_val_0_16` dari tabel master (berbasis pasien). Sampling TIDAK dilakukan terpisah di dalam tiap partisi. Hasil: training 14.835 rekaman (3.683 positif, 11.152 negatif, 3,03:1) dan validasi 1.701 rekaman (451 positif, 1.250 negatif, 2,77:1). Seluruh positif ikut, jadi semua 451 positif validasi ada. Tidak ada kesepakatan tim untuk sampling per partisi.
2. **pos_weight.** `train_m1` memakai `pos_weight = n_neg/n_pos` dari data training setelah sampling (11.152/3.683 = 3,03). Angka sekitar 30 di brief 10.4 adalah rasio pada data penuh (110.229 negatif / 3.683 positif = 29,9). Negatif yang diambil 10,1% dari 110.229, sehingga tiap negatif sampel mewakili sekitar 9,9 negatif; 29,9 / 9,9 = 3,03. Jadi 3,03 pada sampel setara secara efektif dengan sekitar 30 pada data penuh (bobot relatif positif terhadap negatif sama; hanya skala total yang berbeda). Memakai 30 pada sampel akan memberi bobot positif sekitar 10x terlalu besar. Probabilitas keluaran tetap tidak terkalibrasi ke prevalensi asli; kalibrasi di fold 18 (Orang 3). Transformasi monoton tidak mengubah AUROC/AUPRC.
3. **Validasi diperkaya.** Validasi internal berisi sekitar 26,5% positif; dipakai hanya untuk early stopping, bukan untuk melaporkan AUPRC. Validasi berprevalensi asli opsional (sekitar 11.100 rekaman tambahan, sekitar 17-18 jam unduhan) dan hanya dilakukan jika tim menginginkannya.
4. **signal_qc_stage2a.csv hanya untuk subset 2a** (16.536 baris), dibaca oleh `train_m1`. Bukan `signal_qc.csv` dan tidak dipakai `build_master`. Kontrak QC seluruh master dibuat nanti dengan `build_contract` atas semua `ed_stay_id` master, setelah tim memutuskan perlakuan baris tanpa sinyal di fold 0-16 (rekaman yang tidak diproses berstatus `not_processed` dan tidak dihitung lolos). Jangan menyalin berkas stage menjadi `signal_qc.csv`.


## 2026-10-29 — Hasil M1 (tahap 2a), QC fold 17-18, dan serah terima prediksi ke Orang 3

**Keputusan / status (Orang 2):**
1. M1 (ResNet-1D, dari nol) dilatih pada tahap 2a sesuai entri 2026-10-23 dan PROJECT_BRIEF 7.2 dan 10.4: seed 42, CPU, `train = fold <= 16 & internal_val_0_16 == 0`, `val = internal_val_0_16 == 1`, `pos_weight = n_neg/n_pos` = 3,03, early stopping pada AUPRC validasi (patience 5), AdamW lr 1e-3. Training 14.816 rekaman (3.680 positif), validasi 1.700 rekaman (450 positif); turun dari rencana 14.835/1.701 karena 20 rekaman tidak lolos (lihat QC di bawah). Memuat dan praproses 788 detik.
2. Checkpoint terbaik epoch 12: AUROC validasi 0,855 dan AUPRC validasi 0,715. Early stopping berhenti di epoch 17. Setelah epoch 12 loss training terus turun sementara metrik validasi memburuk (overfitting); checkpoint yang dipakai tetap epoch 12. Validasi berisi sekitar 26,5% positif, jadi AUPRC ini BUKAN performa pada prevalensi asli (3,2%) dan tidak boleh dibandingkan langsung dengan hasil akhir.
3. Prediksi M1 untuk fold 17 dan 18 ditulis ke `predictions/M1_fold17.parquet` (7.512 baris, 210 positif) dan `predictions/M1_fold18.parquet` (7.520 baris, 218 positif) dengan kontrak PROJECT_BRIEF 12.1. `y_prob` belum terkalibrasi; kalibrasi dan threshold di fold 18 dikerjakan Orang 3. Orang 2 tidak menghitung metrik di fold 17/18. Fold 19 belum disentuh (tidak diunduh, tidak diprediksi) dan tetap dikunci sampai rencana analisis disepakati (brief 11.5).
4. Paket serah terima untuk Orang 3 (kode, checkpoint, statistik standardisasi, catatan) dan paket prediksi dibuat terpisah di luar repo dan dikirim lewat jalur privat tim, bukan lewat Git.

**QC sinyal (aturan brief 10.2, tanpa perubahan aturan):**
- Tahap 2a (16.536 rekaman, fold 0-16): `pass` 16.516, `qc_fail` 19 (11 karena >=3 lead datar, 8 karena lead dengan >20% NaN), `download_failed` 1 (TimeoutError, fold 13, kegagalan jaringan sesaat). Lolos per label: negatif 12.386 dari 12.402, positif 4.130 dari 4.134. Kegagalan tersebar 0-3 per fold. `amplitude_flag` (>20 mV, informatif) pada 35 rekaman; clipping +-3 mV dan tidak adanya hard fail 8 mV tidak diubah.
- Fold 17 dan 18 (15.042 rekaman): `pass` 15.032 (fold 17: 7.512, fold 18: 7.520), `qc_fail` 9 (7 lead NaN, 2 lead datar; fold 17: 5, fold 18: 4), `download_failed` 1 (HTTPError, fold 17). Positif 428 dari 428 lolos; negatif 14.604 dari 14.614. `amplitude_flag` pada 46 rekaman. Aturan QC tidak memakai label; hasil QC tidak dipakai untuk menyetel apa pun.

**Koreksi perkiraan:** entri dan pesan sebelumnya memperkirakan fold 17-18 sekitar 21 ribu rekaman. Angka sebenarnya 15.042. Dengan total fold 17-19 sebesar 22.269 rekaman dan 688 positif, fold 19 sekitar 7.227 rekaman dan 260 positif (turunan aritmetika, belum dihitung dari tabel master).

**Perbedaan dari PROJECT_BRIEF (dicatat, bukan disembunyikan):**
- M1 dilatih pada subset fold 0-16 (positif penuh + negatif 3x), bukan seluruh data (brief 10.4). `pos_weight` 3,03 setara efektif dengan sekitar 30 pada data penuh (lihat klarifikasi 2026-10-23).
- Validasi internal diperkaya positif (sekitar 26,5%); dipakai hanya untuk early stopping.
- Training di CPU (torch +cpu); GPU belum dipakai. Penyimpanan praproses ke memmap/HDF5 (brief 10.3 langkah 7) belum dibuat.
- `signal_qc.csv` penuh (149.003 `ed_stay_id`) dan `master_table_v1` BELUM ada. Berkas `signal_qc_stage2a.csv` dan `signal_qc_eval.csv` hanya subset dan tidak dipakai `build_master`.

**Terbuka / butuh keputusan tim (tidak diputuskan sepihak oleh Orang 2):**
- D1: jalur unduhan untuk sisa kohort. Laju terukur sekitar 5-10 rekaman/menit (batas PhysioNet sekitar 2,5 KB/detik per koneksi, maksimal 8 koneksi); server tidak mengirim `.dat` terkompresi (tanpa `Content-Encoding`). Bucket AWS publik `s3://physionet-open/mimic-iv-ecg/` ternyata kosong; halaman dataset tidak menyebut bucket GCS. Opsi yang masih ada: uji kecepatan ZIP resmi 33,8 GB, kemungkinan akses cloud lewat pengaturan PhysioNet, atau penyesuaian lingkup. Tidak ada anggota tim yang memiliki salinan sinyal.
- D2: perlakuan rekaman yang benar-benar tidak tersedia di server (saat ini 1 rekaman di fold 17) apakah dikeluarkan dari v1 sebagai "sinyal tidak tersedia".
- D3: toleransi pembuatan `master_table_v1` (hanya jika tidak ada ID tanpa hasil QC, atau batas yang disepakati).
- D4: pemilik dan lokasi kode M4 (XGBoost dilatih ulang pada v1), versi Python tim, dan penambahan `xgboost` ke requirements.
- D5: QC fold 19 dengan aturan yang sama (tanpa tuning) agar semua model dinilai pada kunjungan yang sama; prediksi tetap dikunci.

**Keterbatasan bukti:** AUROC/AUPRC di atas dari validasi internal yang kecil dan diperkaya positif, satu kali jalan (seed 42), tanpa interval kepercayaan. Perkiraan waktu unduhan adalah ekstrapolasi kasar dari laju yang berubah-ubah. Konfirmasi Orang 1 dan Orang 3 atas skala 2a (entri 2026-10-23) belum tercatat di log ini.

## 2026-11-05 — Unduhan ZIP resmi, QC sinyal penuh, dan kontrak signal_qc.csv

**Jalur unduhan (penyimpangan dari brief 10.1).** Brief 10.1 meminta hanya ECG kohort yang diunduh (daftar URL). Jalur per rekaman ≈2,5 KB/s per koneksi (maks. 8 koneksi), terlalu lambat untuk ±117 ribu rekaman sisa, dan tim tidak memiliki jalur lebih cepat (keputusan D1). Dipakai ZIP resmi PhysioNet `mimic-iv-ecg-1.0.zip` (36.295.288.053 byte, 1.601.077 anggota) lewat unduhan Range yang dapat dilanjutkan (`src/dataset/download_zip.py`, ukuran dan ETag diperiksa, struktur ZIP terbaca). Hanya rekaman kohort yang diekstrak ke `data/raw/ecg_cache/` (`src/dataset/extract_zip_records.py`); sisa isi ZIP tidak dipakai.

**Verifikasi.** Seluruh 149.003 file_name master ada di ZIP (.hea dan .dat). Ekstraksi folds 0–18: 141.776 rekaman, 109.924 diekstrak baru dan 31.852 rekaman unduhan lama dibandingkan byte per byte dengan ZIP: seluruhnya identik. CRC ZIP dan ukuran .dat dari header lolos untuk semua rekaman; tidak ada yang hilang, rusak, atau salah ukuran. Fold 19 diekstrak dengan flag eksplisit (hanya menyalin berkas, tanpa label/model).

**Rekaman fold 17 yang dulu HTTP 500.** Pada 26 dan 29 Oktober satu rekaman fold 17 gagal diunduh (HTTP 500 pada .hea dan .dat) dan diduga tidak tersedia. Ternyata rekaman itu ada dan utuh di ZIP resmi, sehingga keputusan D2 (mengeluarkannya dari master_table_v1 sebagai "sinyal tidak tersedia") tidak diperlukan; rekaman itu diperlakukan seperti lainnya sesuai hasil QC.

**Keputusan tim (2 November).** D1: tidak ada jalur unduhan yang lebih cepat dari tim. D2: rekaman HTTP 500 dikeluarkan bila memang tidak tersedia (kini tidak relevan, lihat di atas). D3: terserah; dipakai default kontrak hanya ditulis bila setiap ed_stay_id master punya status akhir (pass/qc_fail/preprocess_fail/load_fail), dijaga oleh `build_signal_qc.py`. D4: Python 3.14.10. D5: QC fold 19 boleh dengan aturan yang sama tanpa tuning; prediksi fold 19 tetap terkunci.

**QC sinyal penuh (aturan brief tidak diubah, tanpa tuning).** 149.003 rekaman: 148.861 lolos, 142 qc_fail (0,10%): 85 `three_or_more_flat_leads`, 57 `lead_nan_over_20pct`; tidak ada load_fail/preprocess_fail. Tingkat lolos 0,999 untuk label 0 (144.043/144.181) dan label 1 (4.818/4.822). Gagal per fold 1–14 (fold 19: 10 dari 7.227). amplitude_flag (>20 mV, informasional): 360 rekaman lolos. Hasil QC fold evaluasi hanya dilaporkan agregat dan tidak dipakai mengubah aturan.

**Kontrak.** `data/interim/signal_qc.csv` (ed_stay_id, signal_ok; 149.003 baris, 148.861 bernilai 1 dan 142 bernilai 0) dan `signal_qc_status.csv` (status lengkap) dibuat lewat `python -m src.dataset.build_signal_qc`. Berkas lokal, tidak di-commit. master_table_draft.parquet tidak diubah.

## 2026-11-05 — Fold 17 diperbarui dan pemeriksaan XAI M1 (eksploratif)

**Fold 17.** predictions/M1_fold17.parquet dibuat ulang dengan model dan statistik lead yang sama setelah satu rekaman yang dulu HTTP 500 ternyata tersedia di ZIP resmi: 7.513 baris (210 positif). Untuk 7.512 ID lama y_prob identik (selisih maks 0,0) dan y_true tidak berubah; hanya ditambah 1 baris negatif. Fold 18 tidak berubah (7.520 baris). Kedua fold kini sejajar dengan signal_qc.csv (signal_ok=1).

**XAI M1,** hanya validasi internal fold 0-16 (150 AMI + 150 non-AMI, seed 42; fold 17-19 tidak disentuh; src/xai/run_xai_sanity.py, run_xai_faithfulness.py). Uji acak-bobot: korelasi Grad-CAM model terlatih vs model acak median -0,04 (p90 0,18), jadi peta bergantung pada bobot terlatih. Faithfulness waktu (menolkan 4 dari 20 blok 0,5 detik): pada AMI penurunan logit wilayah Grad-CAM tertinggi 0,94, acak 0,70, terendah 0,37; selisih top-acak 0,24 (CI95 0,18-0,30; 81% rekaman). Pada non-AMI efeknya kecil (selisih 0,09; 56% rekaman). Profil Grad-CAM relatif datar sepanjang rekaman, sehingga lokalisasinya sedang dan bukan penunjuk segmen ST. Uji lead (menolkan 3 dari 12 lead) berselisih besar tetapi hampir melingkar karena skor oklusi dihitung dengan perturbasi yang sama; tidak dipakai sebagai bukti faithfulness.

**Keputusan:** hasil ini eksploratif. Klaim klinis tentang lead atau segmen tertentu tidak dibuat. Batasan: satu seed, n=150 per kelas, validasi diperkaya positif, perturbasi nol di luar distribusi. Tidak ada perubahan pada model, QC, atau preprocessing.
## 2026-11-14 — Skala data M1 tahap 2a disetujui sebagai M1 final; M4 dipegang Orang 3; lokasi kode Orang 3 (DISETUJUI)

**Keputusan:**
1. M1 tahap 2a (entri 2026-10-23 dan 2026-10-29: semua positif fold 0-16 + 3x negatif acak, seed 42; 16.516 rekaman lolos QC, sekitar 13% dari 126.734 rekaman fold 0-16) dipakai sebagai M1 final. M1 tidak dilatih ulang pada data penuh. Ini menutup butir "Konfirmasi Orang 1 dan Orang 3 atas skala 2a belum tercatat" pada entri 2026-10-29.
2. Persetujuan ini dicatat SEBELUM metrik M1 pada fold 17 atau fold 18 dihitung oleh siapa pun, sehingga keputusan tidak dipengaruhi hasil evaluasi.
3. M4 (XGBoost EMR + MM, training fold 0-17 dengan `internal_val_0_17 == 0`, early stopping pada `internal_val_0_17 == 1`) dikerjakan Orang 3 (menutup D4 bagian pemilik M4).
4. Kode Orang 3: model tabular di `src/model/`, evaluasi di `src/evaluation/`, dikerjakan di branch `evaluation`.

**Alasan:** keterbatasan waktu dan komputasi (training M1 di CPU; penyimpanan praproses memmap/HDF5 belum dibuat; memuat 2a saja butuh sekitar 788 detik).

**Konsekuensi untuk pelaporan (keterbatasan, wajib disebut di laporan):**
- M1, dan karena itu M5, belajar dari data EKG yang jauh lebih sedikit (sekitar 13% fold 0-16) dibanding M2/M3 (fold 0-16 penuh) dan M4 (fold 0-17 penuh). Perbandingan utama M5 vs M4 dan perbandingan M1 vs M3 harus menyertakan catatan ini; selisih yang tidak menguntungkan M5/M1 tidak bisa dipisahkan dari efek jumlah data latih.
- Probabilitas M1 tidak terkalibrasi terhadap prevalensi asli (negatif disubsampel, `pos_weight` 3,03). Kalibrasi Platt dan threshold ditentukan di fold 18 oleh Orang 3.

**Disetujui oleh:** Orang 1 (Anna), Orang 3 (Dini). Tanggal persetujuan: 14 Nov

## 2026-11-14 — Spesifikasi model tabular M2, M3, M4 (Orang 3) (DIKUNCI sebelum training)

**Keputusan:**
1. Fitur: M2 = `EMR_FEATURES` (33), M3 = `MM_FEATURES` (10), M4 = keduanya (43), semuanya dari `src/constants.py`. Kode menolak kolom terlarang, label, analisis, dan split (`src/model/tabular.py`, `check_feature_list`).
2. `arrival_transport`: one-hot dengan kategori tetap AMBULANCE, HELICOPTER, OTHER, UNKNOWN, WALK IN (dicek pada master_table_draft); nilai kosong = semua 0; kategori baru = error. Alasan: SHAP per kategori mudah dibaca dan kolom matriks sama di semua fold.
3. Pembagian: M2/M3 train fold 0-16 & `internal_val_0_16 == 0`, validasi `internal_val_0_16 == 1`; M4 train fold 0-17 & `internal_val_0_17 == 0`, validasi `internal_val_0_17 == 1`. Fold 17-19 tidak dipakai untuk early stopping atau tuning model dasar.
4. XGBoost (`tree_method=hist`), `scale_pos_weight = n_neg/n_pos` data training, maksimal 3000 pohon, early stopping 100 putaran pada `aucpr` validasi internal. NaN dibiarkan.
5. Tuning ringan: random search 20 kombinasi (seed 42; trial 1 = max_depth 4, learning_rate 0,05, min_child_weight 1, subsample 0,8, colsample_bytree 0,8), ruang: max_depth {3,4,5,6,8}, learning_rate {0,02; 0,03; 0,05; 0,1}, min_child_weight {1,3,5,10}, subsample {0,6..1,0}, colsample_bytree {0,5..1,0}. Model dipilih menurut average precision (AUPRC) pada validasi internal. Model terpilih tidak dilatih ulang pada train + validasi.
6. Prediksi: M2/M3 untuk fold 17 dan 18; M4 hanya fold 18 (fold 17 adalah data latih M4, jadi prediksinya tidak sah dan tidak ditulis). Fold 19 hanya dengan `--include-fold-19` setelah rencana analisis disepakati dan di-commit.
7. M2/M3/M4 dilatih pada master_table_draft yang difilter signal_ok == 1 dari signal_qc.csv (setara master_table_v1: 148.861 kunjungan), sehingga semua model memakai kunjungan yang sama dengan M1. Bila master_table_v1 resmi dari Orang 1 dipakai kemudian, M2/M3/M4 dilatih ulang dan threshold fold 18 dikunci ulang sebelum fold 19 disentuh.
8. Selama pengembangan, fold 18 hanya dipakai untuk kalibrasi dan threshold per model; perbandingan antar-model (termasuk M5 vs M4) tidak diputuskan dari fold 18, tetapi dari fold 19 sekali jalan.

9. Catatan teknis tim (macOS): torch dan xgboost masing-masing membawa pustaka OpenMP (`libomp`). Bila keduanya termuat di satu proses Python, training XGBoost bisa segmentation fault (terjadi 2026-11-14 saat `pytest` penuh). Aturan: model tabular dan SHAP dijalankan di proses terpisah dari kode M1/XAI EKG (`python -m ...` masing-masing), dan tes XGBoost dijalankan di subprocess. Mac butuh `brew install libomp` untuk XGBoost.

**Disetujui oleh:** Orang 3 (Dini). Tanggal: 14 Nov

## 2026-11-14 — Spesifikasi meta-learner M5 dan M6 (Orang 3) (DIKUNCI sebelum training)

**Keputusan:**
1. M5 (late fusion utama, Jalur A) = regresi logistik pada logit p_M1 dan logit p_M2. M6 (late fusion tabular) = regresi logistik pada logit p_M3 dan logit p_M2. Keduanya dilatih HANYA di fold 17, pada kunjungan yang sama di semua input (inner join `ed_stay_id`; `study_id` wajib sama antar-input).
2. Input = logit probabilitas mentah model dasar (dipotong ke [1e-6, 1-1e-6]), distandarkan dengan rata-rata/SD fold 17. Probabilitas terkalibrasi tidak dipakai sebagai input karena kalibrasi Platt adalah transformasi linear pada skala logit dan diserap oleh regresi logistik (diuji di `tests/test_meta.py`).
3. Regularisasi L2 ringan, C = 1 (formulasi scikit-learn: C * jumlah log-loss + 0,5 * ||w||^2, intercept tidak dipenalti). C tidak dituning: tidak ada data validasi terpisah di fold 17, dan fold 18 tidak boleh dipakai untuk memilih. Implementasi NumPy (`src/model/meta_learner.py`, metode Newton dengan Hessian eksak; berhenti bila Newton decrement <= 1e-14 x |f|), diverifikasi sama dengan scikit-learn (selisih maksimum 2,6e-7 di 27 skenario) dan lolos uji stres 20.000 fit bootstrap tanpa kegagalan; tanpa scikit-learn agar aman bersama torch di macOS. Versi awal (v3) memakai L-BFGS-B dengan toleransi terlalu ketat sehingga berstatus ABNORMAL di macOS; diganti sebelum meta-learner dilatih pada data asli.
4. Gerbang rencana proyek 3.4: fold 17 berisi 210 positif (>= 100), jadi cross-fitting tidak diperlukan. Kode berhenti bila positif fold 17 < 100.
5. XAI fusion: koefisien terstandar (per 1 SD logit), odds ratio per 1 SD, dan CI 95% bootstrap per `subject_id` (2000x, standardisasi dan fit diulang di setiap sampel). Porsi |koefisien| hanya gambaran bobot relatif, bukan dekomposisi kausal, karena input berkorelasi.
6. Prediksi: fold 18 saja (fold 17 adalah data latih meta-learner). Fold 19 hanya dengan `--include-fold-19` setelah prediksi fold 19 model dasar tersedia dan rencana analisis dikunci.
7. Kalibrasi Platt dan threshold M5/M6 ditentukan di fold 18 dengan aturan yang sama seperti model lain (`src/evaluation/run_fold18.py`).

**Disetujui oleh:** Orang 3 (Dini). Tanggal: 14 Nov

## 2026-11-24 — Aturan baseline M0 (M0-v2) dikunci sebelum M0 dinilai (Orang 3)

**Latar belakang:**
- Draf M0-v1 (sha256 aturan `e696c60b…f33e`) tidak pernah di-commit.
- Draf itu diaudit di fold 0-16 tanpa membaca label (`python -m src.model.m0_rule --audit-angka`). Hasil audit (angka agregat):
  - 126.611 kunjungan; 3.010 bernilai M0 = 1.
  - 1.395 kunjungan positif hanya karena kata "injury". Dari jumlah itu, 372 kunjungan positif hanya karena pernyataan injury yang juga menyebut iskemia.
  - Pernyataan injury yang disebut bersama perikarditis/repolarisasi: <10.
  - Pernyataan infark + recent/evolving tanpa "acute" (tidak terhitung di v1): 230.
  - Pernyataan acute + iskemia tanpa infark/injury (tidak terhitung di v1): 159.
  - Pernyataan acute + perikarditis/repolarisasi: 279. Ini benar tidak dihitung.
  - Kunjungan M0 = 0 dengan infark dan acute di dua baris berurutan: 48. Baris acute yang pendek, baris pertama yang menggantung, dan baris acute yang menyebut perikarditis masing-masing <10. Jadi sebagian besar pasangan ini tidak berpola kalimat terpotong, dan aturan tetap dinilai per baris.
- Belum ada kinerja M0 yang dilihat pada fold mana pun.

**Keputusan:**
1. M0 adalah baseline klinis tanpa training: aturan kata kunci pada pernyataan interpretasi otomatis mesin EKG (`report_0` ... `report_17`) dari EKG pertama setiap kunjungan.
   - Sumber teks: `data/interim/mm_reports.parquet`.
   - Kunjungan yang dinilai sama dengan model lain (`train_tabular.load_master`).
   - Satuan penilaian adalah satu pernyataan (huruf kecil). Kunjungan positif bila minimal satu pernyataannya positif. Kunjungan tanpa teks report = 0.
2. **Definisi primer `M0`** ("mesin menandai kemungkinan infark, injury, atau iskemia akut"). Pernyataan positif bila cocok salah satu pola berikut:
   - `stemi`;
   - acute + MI/infarct dalam satu pernyataan, dengan urutan mana pun;
   - infark + recent/evolving: kata pemisahnya hanya kata keraguan (possibly, probably, dll.), dan pernyataan tidak memuat old, age un/indetermined, atau kata pembanding (compared, previous, prior, since, tracing, unchanged);
   - kata `injury`;
   - acute + iskemia: maksimal 2 kata pemisah, dan pemisahnya bukan or/and/perikarditis/repolarisasi.
3. **Analisis sensitivitas `M0S1`** ("mesin menyatakan infark akut/STEMI"): sama dengan M0, dengan dua perbedaan:
   - injury hanya dihitung bila pernyataan yang sama tidak menyebut iskemia, perikarditis, atau repolarisasi;
   - acute + iskemia tidak dihitung.

   Akibatnya setiap kunjungan M0S1 = 1 juga M0 = 1.
4. **Varian S2 tidak dijalankan sebagai analisis terpisah.** S2 hanya mengeluarkan injury yang disebut bersama perikarditis/repolarisasi. Menurut audit, pernyataan seperti itu <10, jadi S2 praktis identik dengan M0. Kalau S2 dilaporkan, selisih kedua tabel bisa membuka sel < 10.
5. Infark lama atau infark dengan usia tidak pasti (old, age un/indetermined) tanpa acute/recent/evolving tidak dihitung di kedua definisi. Ini lebih ketat daripada definisi sistem komersial pada Al-Zaiti dkk. (Nature Medicine 2023), yang juga menghitung "age indeterminate".
6. Pola negasi: `\b(?:no|without)\b[^,;]*\b(?:acute|stemi|infarct\w*|injury|mi)\b`. Kata "not" sengaja tidak dipakai agar "cannot rule out acute MI" tetap positif.
7. Pencocokan memakai modul `re` Python, sehingga hasilnya tidak bergantung pada backend string pandas. Pola regex lengkap tercantum di `src/model/m0_rule.py` dan `reports/m0_rule.md`.
8. "Injury" di sini adalah istilah EKG (current of injury), bukan "myocardial injury" biokimia menurut Fourth Universal Definition of MI (2018).
9. Dasar pemilihan:
   - Studi validasi pernyataan otomatis menghitung lebih dari sekadar "acute MI". Faramand dkk. (J Electrocardiol 2021) menghitung Acute MI atau Acute Ischemia. Al-Zaiti dkk. (Nature Medicine 2023) menghitung Acute MI, termasuk "recent", dan Acute Ischemia.
   - De Bie dkk. (J Electrocardiol 2020) mencantumkan pernyataan injury dan "injury/ischemia" dalam daftar critical value beberapa program.
   - Mayoritas kasus di kohort adalah NSTEMI, sehingga definisi yang membuang semua pernyataan injury/iskemia akan membuat M0 nyaris hanya mendeteksi STEMI.
   - Definisi sempit tetap dilaporkan sebagai S1.
10. Pelaporan:
    - Kedua definisi dilaporkan dengan sensitivitas, spesifisitas, PPV, dan NPV, masing-masing dengan IK 95% bootstrap per `subject_id`.
    - Selisih M0S1 - M0 juga dilaporkan.
    - Jumlah kunjungan yang pindah dari M0 = 1 ke M0S1 = 0 dilaporkan per label dan per subtipe (STEMI, NSTEMI, tipe 2 murni).
    - Kinerja kedua definisi juga dilaporkan di subgrup STEMI, NSTEMI, dan IMA tipe 2 murni.
    - Perbandingan model lain dengan baseline hanya memakai M0 primer dan bersifat eksploratif.
11. Definisi primer tidak diganti setelah hasil terlihat. Bila M0S1 ternyata lebih baik, temuan itu dilaporkan apa adanya sebagai hasil analisis sensitivitas.
12. Sidik jari aturan (sha256 spesifikasi, dari `python -m src.model.m0_rule --spec`): `f515ab4f4d4d8d85d2e249ecacbad94fe67fc4eaf93f5648b8e6d452df19589d`.
13. Audit M0-v2 di fold 0-16 tanpa label (`--audit-angka`):
    - M0 = 1 pada 3.374 dari 126.611 kunjungan; M0S1 = 1 pada 2.858 kunjungan; 516 kunjungan pindah dari M0 = 1 ke M0S1 = 0.
    - Pola baru `infark_recent`: 230 pernyataan. Dari jumlah itu, yang juga menyebut old/age undetermined <10, dan yang memuat kata pembanding <10.
    - Pola baru `acute_iskemia`: 159 pernyataan. Yang juga menyebut perikarditis/repolarisasi <10.
    - Sisa infark + recent/evolving dan sisa acute + iskemia yang tidak tertangkap pola: masing-masing <10.
    - Pernyataan "acute" yang tetap tidak dihitung: 279, semuanya menyebut perikarditis/repolarisasi.
    - Kunjungan M0 = 0 dengan infark dan acute di dua baris berurutan: <10.
    - Infark tanpa acute/recent dan tanpa keterangan usia (mis. "consider ... infarct"): 1.681 pernyataan. Pernyataan ini sengaja tidak dihitung karena tidak menyatakan kejadian akut.
14. Fold 19: `M0_fold19` dan `M0S1_fold19` hanya dibuat dengan `--include-fold-19`, setelah rencana analisis berstatus FINAL dan `src/model/m0_rule.py` di-commit tanpa perubahan sesudahnya (dicek otomatis oleh `src/evaluation/lock.py`).

**Commit aturan M0:** 650d599

**Disetujui oleh:** Orang 3 (Freta Yordinia Laura). Tanggal: 2026-11-24

---

## 2026-11-24 — Skrip evaluasi akhir `run_fold19.py` dan penjaga kunci fold 19 (Orang 3)

**Keputusan:**
1. `src/evaluation/run_fold19.py` menjalankan evaluasi akhir dengan langkah yang sama untuk fold 18 (uji kode) dan fold 19 (hasil).
   - Tidak ada parameter yang dipilih ulang: kalibrasi Platt dan threshold rule-out/rule-in diambil apa adanya dari `reports/locked_fold18.json`.
   - Prediksi positif bila `y_prob` mentah >= threshold.
   - M0 dan M0S1 biner: positif bila `y_prob` = 1.
2. Semua model (M0, M0S1, M1-M6) dinilai pada kunjungan yang sama (irisan `ed_stay_id`). Bila ada kunjungan yang tidak dimiliki semua model, jumlahnya dicetak sebagai peringatan dan dicatat di laporan.
3. Uji di fold 18 bersifat in-sample, jadi BUKAN hasil. Sebelum bootstrap, skrip memeriksa dua hal:
   - Parameter terkunci, bila dipakai pada berkas fold 18 utuh setiap model, harus memberi sensitivitas rule-out >= 0,90 dan spesifisitas rule-in >= 0,90, serta intercept dan slope kalibrasi setelah Platt dalam 0,01 dari 0 dan 1.
   - sha256 berkas prediksi fold 18 harus sama dengan sha256 saat threshold dikunci.

   Bila salah satu gagal, skrip berhenti.
4. Metrik per model:
   - AUROC, AUPRC, Brier setelah kalibrasi, dan intercept/slope kalibrasi;
   - sensitivitas, spesifisitas, PPV, dan NPV pada threshold rule-out dan rule-in.

   IK 95% berasal dari bootstrap berpasangan per `subject_id`, 2.000 kali, seed 42, metode persentil.
5. Perbandingan:
   - Utama: M5 - M4, dengan metrik utama ΔAUROC (bermakna bila IK 95% tidak mencakup 0).
   - Eksploratif, tanpa koreksi multiplisitas: M1 - M3, M6 - M4, M5 - M2, M5 - M1, setiap model dikurangi M0, dan M0S1 - M0.
6. Subgrup dan analisis sensitivitas:
   - hanya rawat inap (`label_source = hosp`);
   - tanpa IMA tipe 2 murni;
   - tanpa keluhan yang menyebut diagnosis;
   - IMA tipe 2 murni, STEMI, dan NSTEMI (masing-masing positif subtipe itu + semua negatif);
   - laki-laki dan perempuan;
   - usia < 65 dan >= 65 tahun;
   - catatan obat rutin tersedia dan tidak tersedia.

   Untuk setiap subgrup dilaporkan AUROC, sensitivitas, dan spesifisitas pada threshold rule-out. Subgrup dengan positif atau negatif < 10 hanya dilaporkan jumlahnya. M5 - M4 diulang hanya pada tiga analisis pertama.
7. Hitungan < 10 ditulis "<10". Supresi komplementer juga diterapkan: bila satu bagian dari suatu total < 10, bagian lain yang terkecil ikut disembunyikan (TP/FN, FP/TN, perpindahan M0 -> M0S1 per label dan per subtipe). Laporan hanya berisi angka agregat.
8. Fold 19 hanya bisa dijalankan dengan `--include-fold-19` bila tiga berkas berikut sudah di-commit dan tidak berubah sesudahnya (dicek oleh `src/evaluation/lock.py`):
   - `reports/rencana_analisis_fold19.md`, yang berisi baris "Status: FINAL";
   - `reports/locked_fold18.json`;
   - `src/model/m0_rule.py`.

   Evaluasi fold 19 hanya dijalankan satu kali. Jalan ulang hanya bisa dengan `--overwrite-post-hoc`, ditulis ke laporan terpisah, dan dilaporkan sebagai analisis post-hoc.

**Disetujui oleh:** Orang 3 (Freta Yordinia Laura). Tanggal: 2026-11-24

---

## 2026-11-24 — Spesifikasi SHAP M2 dan M4 (Orang 3)

**Keputusan:**
1. Nilai SHAP M2 dan M4 dihitung dengan algoritma TreeSHAP bawaan XGBoost (`Booster.predict(pred_contribs=True)`), pada iterasi terbaik yang sama dengan prediksi (`iteration_range`). Ini definisi TreeSHAP yang sama dengan `shap.TreeExplainer` untuk model pohon. Opsi `--cek-shap` membandingkan keduanya pada 500 baris.
2. Skala nilai: log-odds model mentah (sebelum Platt). Platt hanya mengubah logit secara linear (b > 0), jadi urutan kepentingan fitur tidak berubah.
3. Ringkasan global per fitur asal:
   - rata-rata |SHAP|, porsi terhadap total, dan rata-rata SHAP;
   - kolom one-hot cara kedatangan dijumlahkan menjadi satu fitur;
   - arah pengaruh = rata-rata SHAP per kelompok (biner: 1 vs 0; kontinu: kuartil bawah vs kuartil atas di fold yang sama; kategori: per kategori).

   Kelompok < 10 kunjungan tidak ditampilkan. Tidak ada nilai per pasien di laporan.
4. Pemeriksaan wajib (skrip berhenti bila gagal):
   - jumlah kontribusi + bias = margin model (selisih <= 1e-3);
   - sigmoid(margin) = `y_prob` di berkas prediksi fold yang sama (selisih <= 1e-5).
5. M2 dan M4 tidak dilatih di fold 18 (M2: fold 0-16; M4: fold 0-17), jadi SHAP fold 18 dihitung pada data di luar training pohon.
   - Fold yang dipakai untuk SHAP di naskah: ditetapkan di rencana analisis fold 19, sebelum fold 19 dibuka.
   - SHAP fold 19 hanya dengan `--include-fold-19`, setelah rencana analisis FINAL di-commit.

**Disetujui oleh:** Orang 3 (Freta Yordinia Laura). Tanggal: 2026-11-24

---

## 2026-11-21 — Prediksi M1 untuk fold 19 (test terkunci) dibuat

**Pembukaan fold 19 untuk M1.** `predictions/M1_fold19.parquet` dibuat satu kali dengan perintah yang sama seperti fold 17 dan 18: `python -u -m src.model.predict_m1 --qc-csv data/interim/signal_qc_full.csv --folds 19 --include-fold-19`. Model, statistik lead, praproses, dan aturan QC tidak diubah (checkpoint `m1_best.pt` epoch 12, `lead_stats.json` dari data training fold 0-16). Hasil: 7.217 baris (7.227 rekaman fold 19 dikurangi 10 yang gagal QC sinyal), ID unik, `y_prob` 0-1 tanpa NaN; kolom sesuai kontrak (ed_stay_id, subject_id, study_id, fold, y_true, y_prob). SHA-256 berkas: `d75dfb9e295cb10e1d98a743bf9474792b9458e38b24a4c48fde8ab7e1ee9460`.

**Dasar pembukaan.** Rencana analisis fold 19: belum ada rencana yang di-commit saat berkas ini dibuat (2026-11-21); DRAF v1 (2026-11-14) belum di-commit, DRAF v2 di-commit 2026-11-24 (`16ca7d4`), versi final lihat entri persetujuan rencana analisis. Kode evaluasi Orang 3 sudah diuji di fold 18 sebelum fold 19 dibuka: saat berkas ini dibuat baru `run_fold18.py` (kalibrasi dan threshold M1-M6, commit `61b8749`, 2026-11-14) yang diuji; skrip evaluasi akhir `run_fold19.py` baru diuji di fold 18 pada 2026-11-24 (commit `76d29fc`). Keputusan skala M1: tetap M1 epoch 12 yang dilatih pada subset fold 0-16 (±13%), sesuai keterbatasan di decision log 2026-11-14; dikonfirmasi oleh Orang 1 (Anna) dan Orang 3 (Dini) pada 2026-11-14 (entri decision log "Skala data M1 tahap 2a disetujui sebagai M1 final"); tidak ada keputusan baru sejak itu.

**Batasan.** Orang 2 tidak menghitung atau melihat metrik fold 19; prediksi dikirim ke Orang 3 lewat jalur privat (berisi ID pasien, tidak di-commit). Tidak ada pelatihan ulang, tuning, atau perubahan threshold setelah fold 19 dibuat; jika terjadi, dicatat sebagai post-hoc. Pemeriksaan kode oleh Antigravity (hanya kode, tanpa data) tidak menemukan jalur yang melewati penguncian fold 19; angka tes yang dilaporkannya (41) tidak dipakai, dan suite lengkap lokal 144 tes lolos sebelum fold 19 dibuat.

**Tambahan 2026-11-24 (Orang 3).**
- SHA-256 salinan `M1_fold19.parquet` di laptop Orang 3 sama dengan nilai di atas.
- Berkas QC yang dipakai, `signal_qc_full.csv`, identik dengan `data/interim/signal_qc.csv` (dipakai model tabular dan M0). SHA-256 keduanya: `68353a9565f395b71cb7f0896bb0bfda84cb1eb7714eb08128e69312f99bb23b`. Akibatnya M1 dan model lain dinilai pada kunjungan fold 19 yang sama.
- Pembukaan fold 19 untuk M1 terjadi sebelum rencana analisis di-commit. Ini penyimpangan dari prosedur (PROJECT_BRIEF 11.5) dan dicatat apa adanya.
- Isian "dikonfirmasi oleh ... pada ..." di paragraf **Dasar pembukaan** ditulis Orang 3 atas izin Orang 2 (2026-11-24), merujuk entri decision log 2026-11-14.
- Sampai 2026-11-24, Orang 3 hanya memeriksa checksum, jumlah baris, dan nama kolom berkas ini; tidak ada metrik fold 19 yang dihitung. Metrik fold 19 hanya akan dihitung oleh `run_fold19.py` setelah rencana analisis final di-commit.

## 2026-11-24 — Keputusan A2 rencana analisis fold 19: threshold setara M0, kurva kalibrasi, XAI, dan pelaporan (Orang 3)

**Latar belakang:**
- Rencana analisis DRAF v3 (commit `5bfc459`) masih menyisakan beberapa keputusan di Bagian 14. Keputusan di bawah diambil pada 24 November 2026, sebelum fold 19 dibuka.
- Tidak ada metrik fold 19 yang dilihat. Semua angka yang disebut di sini berasal dari fold 18 (uji kode, in-sample) atau fold 0–16.

**Keputusan:**
1. **Metrik utama M5 vs M4 = ΔAUROC**, sesuai proposal 2.8.
2. **M5 vs M1** dilaporkan sebagai perbandingan eksploratif, sesuai proposal 2.8.
3. **Sensitivitas pada spesifisitas setara M0** dipakai sebagai jawaban utama pertanyaan sekunder (d) proposal. Usulan DRAF v3 untuk tidak memakainya dibatalkan.
   - Alasan: pada threshold rule-out dan rule-in, tidak ada model yang lebih unggul daripada M0 di sensitivitas **dan** spesifisitas sekaligus di fold 18. Perbandingan pada titik operasi itu hanya menggambarkan trade-off. Al-Zaiti dkk. (Nature Medicine 2023) bisa menyimpulkan modelnya lebih baik karena model itu unggul di kedua angka.
   - Supaya tidak memilih threshold dari data test, threshold **dikunci di fold 18** dengan `src/evaluation/lock_m0_threshold.py` → `reports/locked_fold18_m0.json` (commit `eef89dc`).
   - Aturan: threshold terkecil di fold 18 dengan spesifisitas ≥ spesifisitas M0 primer di fold 18 = 0,9782 (7.143 dari 7.302 negatif), dihitung pada berkas fold 18 utuh setiap model. Prediksi positif bila `y_prob` mentah ≥ threshold.
   - Nilai threshold (`y_prob` mentah): M1 0,887541; M2 0,845454; M3 0,789982; M4 0,832852; M5 0,168243; M6 0,169488. Semuanya memberi spesifisitas tepat 0,9782 di fold 18 (in-sample).
   - Skrip menolak menimpa berkas yang sudah ada. `locked_fold18.json` tidak disentuh. `run_fold19.py` memeriksa sha256 berkas prediksi fold 18 dan, untuk fold 19, mewajibkan `locked_fold18_m0.json` sudah di-commit tanpa perubahan.
4. **Kurva kalibrasi fleksibel** setelah Platt dilaporkan untuk M1–M6. Gambar naskah menampilkan M4 dan M5. Kurva ROC tidak dilaporkan.
   - Dasar: TRIPOD+AI 2024 butir 12e; Van Calster dkk. (BMC Medicine 2019) tentang kurva kalibrasi yang dihaluskan.
   - Metode (`src/evaluation/calibration_curve.py`):
     - regresi logistik hasil terhadap restricted cubic spline dari logit p, 4 knot pada persentil 5/35/65/95 (Harrell);
     - grid 50 titik yang rata di skala logit, antara persentil 1 dan 99 dari p;
     - IK 95% dari bootstrap per `subject_id`, 2.000×, dengan knot dan grid tetap;
     - iterasi yang gagal tidak dihitung.
   - Keluaran: `{stem}_calibration_curve.csv`, `{stem}_calibration_bins.csv`, dan `reports/figures/kalibrasi_fold{N}.png` (`src/evaluation/plot_calibration.py`).
   - Uji kode fold 18 (commit `0740e58`): selisih maksimum antara proporsi teramati dan prediksi pada grid 0,003 (M4) dan 0,012 (M5), in-sample. Semua angka lama di laporan fold 18 identik (118 dari 118 baris, selisih maksimum 0,0).
5. **Aturan "<10" berlaku untuk hitungan.** Proporsi (sensitivitas, spesifisitas, PPV, NPV) dan IK-nya dilaporkan apa adanya, termasuk di subgrup. Supresi komplementer untuk hitungan tetap berlaku.
6. **Grad-CAM dan oklusi M1:**
   - Hasil validasi internal fold 0–16 (entri 2026-11-05) tetap dipakai.
   - Uji yang sama diulang di fold 19 (150 IMA + 150 non-IMA, seed 42) sebagai konfirmasi pada data independen. Uji ini dijalankan Orang 2 setelah rencana FINAL.
   - Atas izin Orang 2, skrip `src/xai/run_xai_sanity.py` dan `run_xai_faithfulness.py` mendapat opsi `--fold` (`src/xai/selection.py`, commit `dbec679`):
     - `internal` (bawaan) memilih rekaman yang sama dengan versi asli;
     - `18` untuk uji kode;
     - `19` hanya bisa dipakai dengan `--include-fold-19` setelah rencana FINAL.
   - Uji kode fold 18 oleh Orang 2 (`jalankan_xai_fold18.sh`) dicatat di entri terpisah setelah hasilnya ada.
7. **SHAP global saja.** Tidak ada penjelasan per pasien, karena penjelasan per pasien akan menampilkan data individual MIMIC, sedangkan pertanyaan penelitian berada di tingkat populasi. Contoh ilustratif, kalau diperlukan, hanya memakai profil sintetis berlabel "bukan pasien nyata".
8. **Hasil fold 19 hanya di-push ke repo tim privat** (`melinda-ab/deteksi-serangan-jantung`).
9. **Persetujuan:**
   - Orang 1 (Ana) dan Orang 2 (Ayu) menyatakan setuju pada 23 November 2026 atas draf rencana yang ada saat itu. Pernyataan ini dicatat sebagai riwayat.
   - Karena teks rencana berubah sesudahnya (DRAF v3 dan v4, 24 November), persetujuan yang berlaku adalah persetujuan atas **teks final**. Persetujuan itu dicatat di Bagian 16 rencana, dengan tanggal yang sama dengan atau setelah tanggal teks final.

**Dokumen terkait:**
- rencana analisis DRAF v4 (`reports/rencana_analisis_fold19.md`);
- forest plot dari CSV agregat (`src/evaluation/plot_forest.py`, commit `0ce01fc`);
- kode threshold setara M0 dan kurva kalibrasi (`14a545d`).

**Diputuskan oleh:** Orang 3 (Freta Yordinia Laura), 2026-11-24. Persetujuan Orang 1, Orang 2, dan dosen pembimbing atas teks final dicatat saat rencana FINAL.

## 2026-11-24 — Uji kode XAI M1 di fold 18 (dijalankan Orang 2, dicatat Orang 3)

**Pelaksanaan:**
- Dijalankan Orang 2 (Ayu) di laptopnya dengan `jalankan_xai_fold18.sh`, pada commit `dbec679`, 2026-11-24 pukul 10.33.
- Skrip ini tidak membuka fold 19, tidak menulis ke repo, dan tidak melakukan commit.
- Keluaran lengkapnya, yang hanya berisi angka agregat, disimpan di `reports/xai_fold18.txt`.
- Sampel: 150 IMA + 150 non-IMA, seed 42, sama dengan validasi internal 5 November.

**Hasil (memenuhi ketiga syarat uji kode):**
1. Tes: 11 lolos (`tests/test_xai_selection.py` dan `tests/test_xai.py`).
2. Perilaku bawaan tidak berubah. Jalan validasi internal fold 0-16 mengulang angka 5 November: korelasi Grad-CAM model terlatih vs model acak median −0,042 dan p90 0,176 (5 November: −0,04 dan 0,18).
3. Kedua jalan `--fold 18` (sanity dan faithfulness) berlabel "fold 18 (UJI KODE; fold 18 tidak dipakai melatih M1)" dan hanya berisi angka agregat.

**Angka fold 18** (uji kode, bukan hasil naskah; dicatat sebagai pembanding hasil fold 19):
- Uji acak-bobot: median −0,060, p90 0,247. Peta Grad-CAM tetap bergantung pada bobot terlatih.
- Faithfulness waktu (menolkan 4 dari 20 blok 0,5 detik):
  - IMA: penurunan logit pada wilayah Grad-CAM tertinggi 0,982, acak 0,661, terendah 0,361. Selisih tertinggi − acak 0,322 (IK 95% 0,261–0,381); 85% rekaman.
  - Non-IMA: selisih tertinggi − acak 0,173 (IK 95% 0,122–0,223); 72% rekaman.
- Profil Grad-CAM tetap relatif datar sepanjang detik 1-9, sama dengan catatan 5 November.
- Uji lead (menolkan 3 dari 12 lead) tetap **tidak** dipakai sebagai bukti faithfulness, karena hampir melingkar (catatan 5 November).

**Kesimpulan:** kode XAI M1 siap dipakai di fold 19 sesuai rencana analisis (Bagian 11 dan 13). Butir Bagian 14 "uji kode XAI M1 di fold 18" dicentang di DRAF v4.

**Dicatat oleh:** Orang 3 (Freta Yordinia Laura), 2026-11-24, berdasarkan keluaran dari Orang 2.
