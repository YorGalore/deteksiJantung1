# UJI KODE evaluasi akhir di fold 18

Dihasilkan otomatis oleh `src/evaluation/run_fold19.py`. Hanya angka agregat; hitungan < 10 ditulis "<10".

> **UJI KODE, BUKAN HASIL.** Kalibrasi dan threshold dipilih di fold 18 yang sama, jadi semua metrik bergantung threshold dan kalibrasi di sini bersifat in-sample. Tujuannya hanya memastikan skrip berjalan benar sebelum fold 19 dibuka.

- Dibuat: 2026-11-24T00:23:25+00:00 | bootstrap 2000x per subject_id, seed 42
- Tabel master: `master_table_draft.parquet` | filter sinyal: signal_ok == 1 dari signal_qc.csv (148,861 baris tersisa)
- Kunjungan dinilai bersama: 7,520 | positif: 218 | prevalensi 2.90% | per model: {'M0': 7520, 'M0S1': 7520, 'M1': 7520, 'M2': 7520, 'M3': 7520, 'M4': 7520, 'M5': 7520, 'M6': 7520} | dibuang: {'M0': 0, 'M0S1': 0, 'M1': 0, 'M2': 0, 'M3': 0, 'M4': 0, 'M5': 0, 'M6': 0}
- Aturan M0: M0-v2, sha256 `f515ab4f4d4d8d85...`
- SHA-256 berkas prediksi:
  - M0: `6254f1f1be0916ab311c6170f2ece11e9e7ca97ce563a05cb42a2cb4f740e323`
  - M0S1: `f5824f8a58f7a923d64f5201fff8cfac2986a794f5f87e16a65b2b3532a59b74`
  - M1: `0295de7df6778740e6bcd078176b8f63365f81b7164ccafd3cf0e211d93bb7dc`
  - M2: `3ba8afce32496addfa8377795923b5aa83a0cbaa4f8b1e99c48e424017cd2ee4`
  - M3: `77350be62a64439cb8e2676dba089e7cd1c1ee53947bffae12495c4dd49abd07`
  - M4: `825de25602f933a1067384d667b80d010ff540c6867310d28403bef172e3a533`
  - M5: `4f5811e7ae6e58e0668cd6853173ee76431b0a61a184472a59a98ec4db281602`
  - M6: `e6caa6e7ce549849221494f5802337bab68032de4ddca6a31831fb21e007bae7`

## 1. Kinerja setiap model

Estimasi [IK 95% bootstrap per subject_id]. Brier mentah dan intercept/slope kalibrasi tanpa IK. Threshold dan kalibrasi dari `reports/locked_fold18.json`; prediksi positif bila y_prob mentah >= threshold.

| model | AUROC | AUPRC | Brier mentah | Brier kalibrasi | intercept/slope kalibrasi | rule-out: sens / spes / PPV / NPV | rule-in: sens / spes / PPV / NPV | spes setara M0: sens / spes / PPV / NPV |
|---|---|---|---|---|---|---|---|---|
| M0 | – | – | – | – | – | 0.124 [0.085, 0.166] / 0.978 [0.974, 0.982] / 0.145 [0.098, 0.201] / 0.974 [0.970, 0.978] | (biner; sama dengan rule-out) | (titik operasi M0) |
| M0S1 | – | – | – | – | – | 0.115 [0.075, 0.156] / 0.981 [0.977, 0.985] / 0.154 [0.101, 0.215] / 0.974 [0.970, 0.978] | (biner; sama dengan rule-out) | (titik operasi M0) |
| M1 | 0.846 [0.822, 0.870] | 0.212 [0.160, 0.274] | 0.154 | 0.025 [0.022, 0.029] | 0.000 / 1.000 | 0.904 [0.863, 0.940] / 0.613 [0.594, 0.633] / 0.065 [0.055, 0.076] / 0.995 [0.993, 0.997] | 0.528 [0.458, 0.599] / 0.900 [0.888, 0.911] / 0.136 [0.111, 0.164] / 0.985 [0.981, 0.988] | 0.225 [0.167, 0.289] / 0.978 [0.973, 0.983] / 0.236 [0.176, 0.300] / 0.977 [0.973, 0.981] |
| M2 | 0.831 [0.806, 0.857] | 0.144 [0.112, 0.192] | 0.165 | 0.026 [0.023, 0.030] | 0.000 / 1.000 | 0.904 [0.864, 0.942] / 0.559 [0.542, 0.577] / 0.058 [0.049, 0.067] / 0.995 [0.993, 0.997] | 0.482 [0.414, 0.551] / 0.900 [0.892, 0.908] / 0.126 [0.101, 0.152] / 0.983 [0.980, 0.986] | 0.193 [0.140, 0.245] / 0.978 [0.975, 0.982] / 0.209 [0.151, 0.267] / 0.976 [0.972, 0.980] |
| M3 | 0.761 [0.730, 0.791] | 0.085 [0.067, 0.109] | 0.183 | 0.027 [0.023, 0.032] | 0.000 / 1.000 | 0.904 [0.862, 0.941] / 0.454 [0.435, 0.474] / 0.047 [0.040, 0.055] / 0.994 [0.991, 0.996] | 0.353 [0.289, 0.423] / 0.900 [0.888, 0.911] / 0.095 [0.073, 0.120] / 0.979 [0.975, 0.983] | 0.119 [0.076, 0.167] / 0.978 [0.974, 0.982] / 0.141 [0.090, 0.190] / 0.974 [0.969, 0.978] |
| M4 | 0.855 [0.831, 0.879] | 0.180 [0.139, 0.237] | 0.153 | 0.026 [0.022, 0.030] | 0.000 / 1.000 | 0.904 [0.859, 0.943] / 0.581 [0.564, 0.600] / 0.061 [0.051, 0.071] / 0.995 [0.993, 0.997] | 0.550 [0.481, 0.619] / 0.900 [0.891, 0.909] / 0.141 [0.115, 0.169] / 0.985 [0.982, 0.988] | 0.229 [0.175, 0.288] / 0.978 [0.974, 0.982] / 0.239 [0.181, 0.303] / 0.977 [0.973, 0.981] |
| M5 | 0.883 [0.862, 0.903] | 0.276 [0.217, 0.344] | 0.024 | 0.024 [0.021, 0.028] | 0.000 / 1.000 | 0.904 [0.863, 0.942] / 0.675 [0.658, 0.692] / 0.077 [0.065, 0.089] / 0.996 [0.994, 0.998] | 0.619 [0.556, 0.685] / 0.900 [0.890, 0.910] / 0.156 [0.129, 0.184] / 0.988 [0.985, 0.990] | 0.307 [0.242, 0.377] / 0.978 [0.973, 0.983] / 0.296 [0.230, 0.363] / 0.979 [0.976, 0.983] |
| M6 | 0.853 [0.829, 0.877] | 0.168 [0.130, 0.220] | 0.026 | 0.026 [0.022, 0.030] | 0.000 / 1.000 | 0.904 [0.862, 0.942] / 0.616 [0.598, 0.634] / 0.066 [0.056, 0.076] / 0.995 [0.993, 0.997] | 0.541 [0.472, 0.612] / 0.900 [0.891, 0.909] / 0.139 [0.115, 0.166] / 0.985 [0.982, 0.988] | 0.225 [0.164, 0.292] / 0.978 [0.975, 0.982] / 0.236 [0.175, 0.304] / 0.977 [0.973, 0.981] |

**Jumlah sel pada threshold terkunci**

| model | threshold | y_prob mentah >= | TP | FP | FN | TN | F1 |
|---|---|---|---|---|---|---|---|
| M0 | biner | 1 (biner) | 27 | 159 | 191 | 7,143 | 0.134 |
| M0S1 | biner | 1 (biner) | 25 | 137 | 193 | 7,165 | 0.132 |
| M1 | rule-out | 0.302827 | 197 | 2,827 | 21 | 4,475 | 0.122 |
| M1 | rule-in | 0.711488 | 115 | 730 | 103 | 6,572 | 0.216 |
| M1 | spes setara M0 | 0.887541 | 49 | 159 | 169 | 7,143 | 0.230 |
| M2 | rule-out | 0.316234 | 197 | 3,223 | 21 | 4,079 | 0.108 |
| M2 | rule-in | 0.701304 | 105 | 730 | 113 | 6,572 | 0.199 |
| M2 | spes setara M0 | 0.845454 | 42 | 159 | 176 | 7,143 | 0.200 |
| M3 | rule-out | 0.311988 | 197 | 3,985 | 21 | 3,317 | 0.090 |
| M3 | rule-in | 0.696361 | 77 | 730 | 141 | 6,572 | 0.150 |
| M3 | spes setara M0 | 0.789982 | 26 | 159 | 192 | 7,143 | 0.129 |
| M4 | rule-out | 0.314095 | 197 | 3,056 | 21 | 4,246 | 0.114 |
| M4 | rule-in | 0.681518 | 120 | 730 | 98 | 6,572 | 0.225 |
| M4 | spes setara M0 | 0.832852 | 50 | 159 | 168 | 7,143 | 0.234 |
| M5 | rule-out | 0.0172195 | 197 | 2,376 | 21 | 4,926 | 0.141 |
| M5 | rule-in | 0.0661233 | 135 | 730 | 83 | 6,572 | 0.249 |
| M5 | spes setara M0 | 0.168243 | 67 | 159 | 151 | 7,143 | 0.302 |
| M6 | rule-out | 0.0146082 | 197 | 2,807 | 21 | 4,495 | 0.122 |
| M6 | rule-in | 0.0700142 | 118 | 730 | 100 | 6,572 | 0.221 |
| M6 | spes setara M0 | 0.169488 | 49 | 159 | 169 | 7,143 | 0.230 |

## 2. Perbandingan utama

M5 - M4. Metrik utama: ΔAUROC; bermakna bila IK 95% selisih tidak mencakup 0. Metrik lain pendukung.

| pasangan | metrik | selisih | IK 95% | IK tidak mencakup 0 | iterasi sah |
|---|---|---|---|---|---|
| M5 - M4 | AUROC | 0.028 | [0.013, 0.043] | ya | 2000 |
| M5 - M4 | AUPRC | 0.096 | [0.050, 0.139] | ya | 2000 |
| M5 - M4 | Brier_kal | -0.002 | [-0.002, -0.001] | ya | 2000 |
| M5 - M4 | Sens@rule-out | 0.000 | [-0.040, 0.040] | tidak | 2000 |
| M5 - M4 | Spes@rule-out | 0.093 | [0.083, 0.104] | ya | 2000 |
| M5 - M4 | PPV@rule-out | 0.016 | [0.012, 0.020] | ya | 2000 |
| M5 - M4 | NPV@rule-out | 0.001 | [-0.001, 0.003] | tidak | 2000 |
| M5 - M4 | Sens@rule-in | 0.069 | [0.011, 0.131] | ya | 2000 |
| M5 - M4 | Spes@rule-in | 0.000 | [-0.007, 0.008] | tidak | 2000 |
| M5 - M4 | PPV@rule-in | 0.015 | [0.000, 0.030] | ya | 2000 |
| M5 - M4 | NPV@rule-in | 0.002 | [0.000, 0.004] | ya | 2000 |

## 3. Perbandingan eksploratif (tanpa koreksi multiplisitas)

| pasangan | metrik | selisih | IK 95% | IK tidak mencakup 0 | iterasi sah |
|---|---|---|---|---|---|
| M1 - M3 | AUROC | 0.085 | [0.057, 0.113] | ya | 2000 |
| M1 - M3 | AUPRC | 0.127 | [0.076, 0.184] | ya | 2000 |
| M6 - M4 | AUROC | -0.002 | [-0.010, 0.006] | tidak | 2000 |
| M6 - M4 | AUPRC | -0.012 | [-0.034, 0.007] | tidak | 2000 |
| M5 - M2 | AUROC | 0.051 | [0.037, 0.067] | ya | 2000 |
| M5 - M2 | AUPRC | 0.132 | [0.085, 0.179] | ya | 2000 |
| M5 - M1 | AUROC | 0.037 | [0.020, 0.054] | ya | 2000 |
| M5 - M1 | AUPRC | 0.064 | [0.032, 0.100] | ya | 2000 |

## 4. Dibandingkan dengan M0 (eksploratif)

Selisih sensitivitas dan spesifisitas model pada threshold terkuncinya dikurangi M0 (biner).

| pasangan | metrik | selisih | IK 95% | IK tidak mencakup 0 | iterasi sah |
|---|---|---|---|---|---|
| M1 - M0 | Sens@rule-out | 0.780 | [0.726, 0.833] | ya | 2000 |
| M1 - M0 | Spes@rule-out | -0.365 | [-0.385, -0.345] | ya | 2000 |
| M1 - M0 | Sens@rule-in | 0.404 | [0.332, 0.477] | ya | 2000 |
| M1 - M0 | Spes@rule-in | -0.078 | [-0.090, -0.066] | ya | 2000 |
| M2 - M0 | Sens@rule-out | 0.780 | [0.723, 0.835] | ya | 2000 |
| M2 - M0 | Spes@rule-out | -0.420 | [-0.437, -0.401] | ya | 2000 |
| M2 - M0 | Sens@rule-in | 0.358 | [0.277, 0.433] | ya | 2000 |
| M2 - M0 | Spes@rule-in | -0.078 | [-0.087, -0.069] | ya | 2000 |
| M3 - M0 | Sens@rule-out | 0.780 | [0.726, 0.831] | ya | 2000 |
| M3 - M0 | Spes@rule-out | -0.524 | [-0.543, -0.504] | ya | 2000 |
| M3 - M0 | Sens@rule-in | 0.229 | [0.156, 0.304] | ya | 2000 |
| M3 - M0 | Spes@rule-in | -0.078 | [-0.090, -0.066] | ya | 2000 |
| M4 - M0 | Sens@rule-out | 0.780 | [0.721, 0.837] | ya | 2000 |
| M4 - M0 | Spes@rule-out | -0.397 | [-0.416, -0.378] | ya | 2000 |
| M4 - M0 | Sens@rule-in | 0.427 | [0.350, 0.503] | ya | 2000 |
| M4 - M0 | Spes@rule-in | -0.078 | [-0.088, -0.069] | ya | 2000 |
| M5 - M0 | Sens@rule-out | 0.780 | [0.722, 0.833] | ya | 2000 |
| M5 - M0 | Spes@rule-out | -0.304 | [-0.321, -0.286] | ya | 2000 |
| M5 - M0 | Sens@rule-in | 0.495 | [0.424, 0.568] | ya | 2000 |
| M5 - M0 | Spes@rule-in | -0.078 | [-0.088, -0.068] | ya | 2000 |
| M6 - M0 | Sens@rule-out | 0.780 | [0.722, 0.837] | ya | 2000 |
| M6 - M0 | Spes@rule-out | -0.363 | [-0.381, -0.344] | ya | 2000 |
| M6 - M0 | Sens@rule-in | 0.417 | [0.342, 0.493] | ya | 2000 |
| M6 - M0 | Spes@rule-in | -0.078 | [-0.087, -0.068] | ya | 2000 |

**Pada spesifisitas setara M0** (threshold dikunci di fold 18: spesifisitas model >= spesifisitas M0 di fold 18 = 0.9782; `reports/locked_fold18_m0.json`)

| pasangan | metrik | selisih | IK 95% | IK tidak mencakup 0 | iterasi sah |
|---|---|---|---|---|---|
| M1 - M0 | Sens@spes-M0 | 0.101 | [0.041, 0.165] | ya | 2000 |
| M1 - M0 | Spes@spes-M0 | 0.000 | [-0.006, 0.006] | tidak | 2000 |
| M1 - M0 | PPV@spes-M0 | 0.090 | [0.025, 0.156] | ya | 2000 |
| M1 - M0 | NPV@spes-M0 | 0.003 | [0.001, 0.005] | ya | 2000 |
| M2 - M0 | Sens@spes-M0 | 0.069 | [0.000, 0.136] | tidak | 2000 |
| M2 - M0 | Spes@spes-M0 | 0.000 | [-0.005, 0.005] | tidak | 2000 |
| M2 - M0 | PPV@spes-M0 | 0.064 | [-0.004, 0.133] | tidak | 2000 |
| M2 - M0 | NPV@spes-M0 | 0.002 | [0.000, 0.004] | ya | 2000 |
| M3 - M0 | Sens@spes-M0 | -0.005 | [-0.070, 0.064] | tidak | 2000 |
| M3 - M0 | Spes@spes-M0 | 0.000 | [-0.006, 0.006] | tidak | 2000 |
| M3 - M0 | PPV@spes-M0 | -0.005 | [-0.076, 0.062] | tidak | 2000 |
| M3 - M0 | NPV@spes-M0 | -0.000 | [-0.002, 0.002] | tidak | 2000 |
| M4 - M0 | Sens@spes-M0 | 0.106 | [0.039, 0.173] | ya | 2000 |
| M4 - M0 | Spes@spes-M0 | 0.000 | [-0.005, 0.005] | tidak | 2000 |
| M4 - M0 | PPV@spes-M0 | 0.094 | [0.028, 0.160] | ya | 2000 |
| M4 - M0 | NPV@spes-M0 | 0.003 | [0.001, 0.005] | ya | 2000 |
| M5 - M0 | Sens@spes-M0 | 0.183 | [0.113, 0.254] | ya | 2000 |
| M5 - M0 | Spes@spes-M0 | 0.000 | [-0.006, 0.006] | tidak | 2000 |
| M5 - M0 | PPV@spes-M0 | 0.151 | [0.082, 0.222] | ya | 2000 |
| M5 - M0 | NPV@spes-M0 | 0.005 | [0.003, 0.008] | ya | 2000 |
| M6 - M0 | Sens@spes-M0 | 0.101 | [0.026, 0.177] | ya | 2000 |
| M6 - M0 | Spes@spes-M0 | 0.000 | [-0.005, 0.005] | tidak | 2000 |
| M6 - M0 | PPV@spes-M0 | 0.090 | [0.020, 0.164] | ya | 2000 |
| M6 - M0 | NPV@spes-M0 | 0.003 | [0.001, 0.005] | ya | 2000 |

**Definisi M0: primer (M0) vs analisis sensitivitas S1 (M0S1)**

M0S1 tidak menghitung pernyataan injury yang juga menyebut iskemia/perikarditis/repolarisasi dan tidak menghitung acute + iskemia. Setiap kunjungan M0S1 = 1 juga M0 = 1, jadi perpindahan hanya dari 1 ke 0.

| kunjungan yang pindah M0 = 1 -> M0S1 = 0 | jumlah |
|---|---|
| semua | 24 |
| positif | <10 |
| negatif | disembunyikan |
| positif STEMI | <10 |
| positif NSTEMI | <10 |
| positif tipe 2 murni | <10 |

| pasangan | metrik | selisih | IK 95% | IK tidak mencakup 0 | iterasi sah |
|---|---|---|---|---|---|
| M0S1 - M0 | Sens@rule-out | -0.009 | [-0.023, 0.000] | tidak | 2000 |
| M0S1 - M0 | Spes@rule-out | 0.003 | [0.002, 0.004] | ya | 2000 |
| M0S1 - M0 | PPV@rule-out | 0.009 | [-0.007, 0.025] | tidak | 2000 |
| M0S1 - M0 | NPV@rule-out | -0.000 | [-0.001, 0.000] | tidak | 2000 |

## 5. Subgrup dan analisis sensitivitas

AUROC, sensitivitas, dan spesifisitas pada threshold rule-out terkunci. Subgrup dengan positif atau negatif < 10 hanya dilaporkan jumlahnya. M5 - M4 diulang hanya pada tiga analisis yang direncanakan.

| analisis | kunjungan | positif | model | AUROC | sens rule-out | spes rule-out |
|---|---|---|---|---|---|---|
| Hanya pasien rawat inap (label_source = hosp) [wajib] | 4,922 | 217 | M0 | – | 0.120 [0.079, 0.165] | 0.978 [0.974, 0.983] |
| Hanya pasien rawat inap (label_source = hosp) [wajib] | 4,922 | 217 | M0S1 | – | 0.111 [0.071, 0.156] | 0.982 [0.977, 0.986] |
| Hanya pasien rawat inap (label_source = hosp) [wajib] | 4,922 | 217 | M1 | 0.814 [0.786, 0.841] | 0.903 [0.862, 0.940] | 0.534 [0.512, 0.557] |
| Hanya pasien rawat inap (label_source = hosp) [wajib] | 4,922 | 217 | M2 | 0.788 [0.759, 0.816] | 0.903 [0.863, 0.942] | 0.454 [0.434, 0.475] |
| Hanya pasien rawat inap (label_source = hosp) [wajib] | 4,922 | 217 | M3 | 0.720 [0.684, 0.753] | 0.903 [0.860, 0.942] | 0.374 [0.353, 0.395] |
| Hanya pasien rawat inap (label_source = hosp) [wajib] | 4,922 | 217 | M4 | 0.816 [0.786, 0.844] | 0.903 [0.860, 0.943] | 0.479 [0.459, 0.499] |
| Hanya pasien rawat inap (label_source = hosp) [wajib] | 4,922 | 217 | M5 | 0.849 [0.825, 0.872] | 0.903 [0.862, 0.941] | 0.582 [0.562, 0.602] |
| Hanya pasien rawat inap (label_source = hosp) [wajib] | 4,922 | 217 | M6 | 0.812 [0.785, 0.840] | 0.903 [0.862, 0.942] | 0.517 [0.497, 0.537] |
| Tanpa IMA tipe 2 murni | 7,498 | 196 | M0 | – | 0.128 [0.082, 0.177] | 0.978 [0.974, 0.982] |
| Tanpa IMA tipe 2 murni | 7,498 | 196 | M0S1 | – | 0.117 [0.074, 0.164] | 0.981 [0.977, 0.985] |
| Tanpa IMA tipe 2 murni | 7,498 | 196 | M1 | 0.850 [0.824, 0.875] | 0.908 [0.863, 0.946] | 0.613 [0.594, 0.633] |
| Tanpa IMA tipe 2 murni | 7,498 | 196 | M2 | 0.834 [0.808, 0.858] | 0.898 [0.853, 0.939] | 0.559 [0.541, 0.577] |
| Tanpa IMA tipe 2 murni | 7,498 | 196 | M3 | 0.762 [0.731, 0.795] | 0.913 [0.868, 0.951] | 0.454 [0.435, 0.474] |
| Tanpa IMA tipe 2 murni | 7,498 | 196 | M4 | 0.857 [0.831, 0.882] | 0.908 [0.865, 0.948] | 0.581 [0.564, 0.600] |
| Tanpa IMA tipe 2 murni | 7,498 | 196 | M5 | 0.886 [0.865, 0.908] | 0.908 [0.865, 0.948] | 0.675 [0.657, 0.693] |
| Tanpa IMA tipe 2 murni | 7,498 | 196 | M6 | 0.855 [0.831, 0.879] | 0.903 [0.860, 0.945] | 0.616 [0.598, 0.634] |
| Tanpa keluhan yang menyebut diagnosis | 7,490 | 195 | M0 | – | 0.128 [0.082, 0.176] | 0.978 [0.974, 0.982] |
| Tanpa keluhan yang menyebut diagnosis | 7,490 | 195 | M0S1 | – | 0.118 [0.074, 0.165] | 0.981 [0.978, 0.985] |
| Tanpa keluhan yang menyebut diagnosis | 7,490 | 195 | M1 | 0.849 [0.822, 0.874] | 0.903 [0.857, 0.944] | 0.613 [0.592, 0.633] |
| Tanpa keluhan yang menyebut diagnosis | 7,490 | 195 | M2 | 0.828 [0.802, 0.854] | 0.897 [0.855, 0.938] | 0.559 [0.542, 0.577] |
| Tanpa keluhan yang menyebut diagnosis | 7,490 | 195 | M3 | 0.758 [0.724, 0.790] | 0.903 [0.860, 0.942] | 0.454 [0.434, 0.474] |
| Tanpa keluhan yang menyebut diagnosis | 7,490 | 195 | M4 | 0.854 [0.828, 0.879] | 0.897 [0.854, 0.939] | 0.582 [0.564, 0.600] |
| Tanpa keluhan yang menyebut diagnosis | 7,490 | 195 | M5 | 0.883 [0.862, 0.903] | 0.908 [0.863, 0.946] | 0.675 [0.656, 0.693] |
| Tanpa keluhan yang menyebut diagnosis | 7,490 | 195 | M6 | 0.851 [0.826, 0.875] | 0.903 [0.860, 0.944] | 0.616 [0.598, 0.634] |
| IMA tipe 2 murni (positif tipe 2 murni + semua negatif) | 7,324 | 22 | M0 | – | 0.091 [0.000, 0.238] | 0.978 [0.974, 0.982] |
| IMA tipe 2 murni (positif tipe 2 murni + semua negatif) | 7,324 | 22 | M0S1 | – | 0.091 [0.000, 0.238] | 0.981 [0.977, 0.985] |
| IMA tipe 2 murni (positif tipe 2 murni + semua negatif) | 7,324 | 22 | M1 | 0.808 [0.731, 0.881] | 0.864 [0.714, 1.000] | 0.613 [0.593, 0.632] |
| IMA tipe 2 murni (positif tipe 2 murni + semua negatif) | 7,324 | 22 | M2 | 0.806 [0.739, 0.865] | 0.955 [0.842, 1.000] | 0.559 [0.542, 0.575] |
| IMA tipe 2 murni (positif tipe 2 murni + semua negatif) | 7,324 | 22 | M3 | 0.753 [0.616, 0.869] | 0.818 [0.636, 0.960] | 0.454 [0.435, 0.473] |
| IMA tipe 2 murni (positif tipe 2 murni + semua negatif) | 7,324 | 22 | M4 | 0.841 [0.759, 0.908] | 0.864 [0.700, 1.000] | 0.581 [0.564, 0.599] |
| IMA tipe 2 murni (positif tipe 2 murni + semua negatif) | 7,324 | 22 | M5 | 0.851 [0.792, 0.903] | 0.864 [0.714, 1.000] | 0.675 [0.656, 0.691] |
| IMA tipe 2 murni (positif tipe 2 murni + semua negatif) | 7,324 | 22 | M6 | 0.831 [0.758, 0.893] | 0.909 [0.765, 1.000] | 0.616 [0.598, 0.633] |
| STEMI (positif STEMI + semua negatif) | 7,341 | 39 | M0 | – | 0.333 [0.182, 0.487] | 0.978 [0.974, 0.982] |
| STEMI (positif STEMI + semua negatif) | 7,341 | 39 | M0S1 | – | 0.333 [0.182, 0.487] | 0.981 [0.977, 0.985] |
| STEMI (positif STEMI + semua negatif) | 7,341 | 39 | M1 | 0.924 [0.875, 0.962] | 0.949 [0.875, 1.000] | 0.613 [0.593, 0.631] |
| STEMI (positif STEMI + semua negatif) | 7,341 | 39 | M2 | 0.879 [0.828, 0.923] | 0.949 [0.860, 1.000] | 0.559 [0.542, 0.576] |
| STEMI (positif STEMI + semua negatif) | 7,341 | 39 | M3 | 0.769 [0.698, 0.836] | 0.897 [0.793, 1.000] | 0.454 [0.435, 0.473] |
| STEMI (positif STEMI + semua negatif) | 7,341 | 39 | M4 | 0.894 [0.853, 0.930] | 0.974 [0.912, 1.000] | 0.581 [0.564, 0.600] |
| STEMI (positif STEMI + semua negatif) | 7,341 | 39 | M5 | 0.945 [0.909, 0.974] | 0.949 [0.860, 1.000] | 0.675 [0.657, 0.693] |
| STEMI (positif STEMI + semua negatif) | 7,341 | 39 | M6 | 0.894 [0.852, 0.930] | 0.949 [0.860, 1.000] | 0.616 [0.598, 0.634] |
| NSTEMI (positif NSTEMI + semua negatif) | 7,459 | 157 | M0 | – | 0.076 [0.037, 0.118] | 0.978 [0.974, 0.982] |
| NSTEMI (positif NSTEMI + semua negatif) | 7,459 | 157 | M0S1 | – | 0.064 [0.028, 0.103] | 0.981 [0.978, 0.985] |
| NSTEMI (positif NSTEMI + semua negatif) | 7,459 | 157 | M1 | 0.833 [0.803, 0.860] | 0.898 [0.844, 0.943] | 0.613 [0.593, 0.632] |
| NSTEMI (positif NSTEMI + semua negatif) | 7,459 | 157 | M2 | 0.824 [0.793, 0.854] | 0.885 [0.832, 0.935] | 0.559 [0.541, 0.578] |
| NSTEMI (positif NSTEMI + semua negatif) | 7,459 | 157 | M3 | 0.757 [0.721, 0.792] | 0.917 [0.871, 0.957] | 0.454 [0.435, 0.475] |
| NSTEMI (positif NSTEMI + semua negatif) | 7,459 | 157 | M4 | 0.848 [0.817, 0.876] | 0.892 [0.837, 0.941] | 0.581 [0.564, 0.601] |
| NSTEMI (positif NSTEMI + semua negatif) | 7,459 | 157 | M5 | 0.873 [0.848, 0.895] | 0.898 [0.847, 0.944] | 0.675 [0.656, 0.693] |
| NSTEMI (positif NSTEMI + semua negatif) | 7,459 | 157 | M6 | 0.846 [0.818, 0.873] | 0.892 [0.842, 0.941] | 0.616 [0.598, 0.635] |
| Laki-laki | 3,493 | 115 | M0 | – | 0.130 [0.070, 0.196] | 0.968 [0.960, 0.975] |
| Laki-laki | 3,493 | 115 | M0S1 | – | 0.113 [0.055, 0.179] | 0.972 [0.964, 0.978] |
| Laki-laki | 3,493 | 115 | M1 | 0.808 [0.774, 0.841] | 0.913 [0.862, 0.959] | 0.532 [0.506, 0.557] |
| Laki-laki | 3,493 | 115 | M2 | 0.803 [0.766, 0.838] | 0.922 [0.873, 0.966] | 0.474 [0.451, 0.498] |
| Laki-laki | 3,493 | 115 | M3 | 0.736 [0.690, 0.782] | 0.913 [0.859, 0.961] | 0.383 [0.359, 0.408] |
| Laki-laki | 3,493 | 115 | M4 | 0.833 [0.798, 0.867] | 0.913 [0.860, 0.961] | 0.496 [0.473, 0.519] |
| Laki-laki | 3,493 | 115 | M5 | 0.853 [0.822, 0.882] | 0.913 [0.855, 0.961] | 0.581 [0.557, 0.606] |
| Laki-laki | 3,493 | 115 | M6 | 0.830 [0.797, 0.862] | 0.922 [0.864, 0.966] | 0.532 [0.507, 0.556] |
| Perempuan | 4,027 | 103 | M0 | – | 0.117 [0.061, 0.176] | 0.987 [0.983, 0.990] |
| Perempuan | 4,027 | 103 | M0S1 | – | 0.117 [0.061, 0.176] | 0.990 [0.986, 0.993] |
| Perempuan | 4,027 | 103 | M1 | 0.875 [0.839, 0.907] | 0.893 [0.824, 0.948] | 0.683 [0.655, 0.710] |
| Perempuan | 4,027 | 103 | M2 | 0.855 [0.817, 0.889] | 0.883 [0.815, 0.942] | 0.631 [0.607, 0.655] |
| Perempuan | 4,027 | 103 | M3 | 0.780 [0.736, 0.824] | 0.893 [0.828, 0.951] | 0.515 [0.486, 0.544] |
| Perempuan | 4,027 | 103 | M4 | 0.875 [0.839, 0.907] | 0.893 [0.828, 0.953] | 0.655 [0.629, 0.680] |
| Perempuan | 4,027 | 103 | M5 | 0.907 [0.878, 0.932] | 0.893 [0.827, 0.948] | 0.755 [0.729, 0.778] |
| Perempuan | 4,027 | 103 | M6 | 0.873 [0.838, 0.905] | 0.883 [0.819, 0.946] | 0.688 [0.663, 0.711] |
| Usia < 65 tahun | 4,118 | 74 | M0 | – | 0.108 [0.039, 0.189] | 0.979 [0.974, 0.984] |
| Usia < 65 tahun | 4,118 | 74 | M0S1 | – | 0.108 [0.039, 0.189] | 0.982 [0.976, 0.986] |
| Usia < 65 tahun | 4,118 | 74 | M1 | 0.874 [0.832, 0.912] | 0.851 [0.764, 0.926] | 0.764 [0.740, 0.788] |
| Usia < 65 tahun | 4,118 | 74 | M2 | 0.872 [0.832, 0.907] | 0.824 [0.731, 0.909] | 0.745 [0.725, 0.766] |
| Usia < 65 tahun | 4,118 | 74 | M3 | 0.795 [0.743, 0.844] | 0.878 [0.800, 0.947] | 0.591 [0.564, 0.618] |
| Usia < 65 tahun | 4,118 | 74 | M4 | 0.893 [0.855, 0.925] | 0.851 [0.761, 0.929] | 0.741 [0.718, 0.763] |
| Usia < 65 tahun | 4,118 | 74 | M5 | 0.914 [0.878, 0.943] | 0.838 [0.747, 0.919] | 0.845 [0.826, 0.865] |
| Usia < 65 tahun | 4,118 | 74 | M6 | 0.892 [0.854, 0.923] | 0.838 [0.744, 0.919] | 0.795 [0.776, 0.815] |
| Usia >= 65 tahun | 3,402 | 144 | M0 | – | 0.132 [0.077, 0.189] | 0.977 [0.971, 0.983] |
| Usia >= 65 tahun | 3,402 | 144 | M0S1 | – | 0.118 [0.066, 0.173] | 0.981 [0.975, 0.986] |
| Usia >= 65 tahun | 3,402 | 144 | M1 | 0.795 [0.756, 0.830] | 0.931 [0.887, 0.969] | 0.425 [0.399, 0.452] |
| Usia >= 65 tahun | 3,402 | 144 | M2 | 0.769 [0.727, 0.810] | 0.944 [0.903, 0.981] | 0.327 [0.306, 0.347] |
| Usia >= 65 tahun | 3,402 | 144 | M3 | 0.700 [0.654, 0.743] | 0.917 [0.866, 0.959] | 0.284 [0.261, 0.306] |
| Usia >= 65 tahun | 3,402 | 144 | M4 | 0.799 [0.757, 0.838] | 0.931 [0.883, 0.972] | 0.384 [0.360, 0.405] |
| Usia >= 65 tahun | 3,402 | 144 | M5 | 0.838 [0.806, 0.870] | 0.938 [0.896, 0.973] | 0.463 [0.437, 0.487] |
| Usia >= 65 tahun | 3,402 | 144 | M6 | 0.797 [0.758, 0.835] | 0.938 [0.892, 0.978] | 0.393 [0.369, 0.415] |
| Catatan obat rutin tersedia | 6,186 | 166 | M0 | – | 0.078 [0.042, 0.118] | 0.980 [0.976, 0.984] |
| Catatan obat rutin tersedia | 6,186 | 166 | M0S1 | – | 0.066 [0.033, 0.103] | 0.983 [0.979, 0.987] |
| Catatan obat rutin tersedia | 6,186 | 166 | M1 | 0.825 [0.796, 0.853] | 0.898 [0.847, 0.942] | 0.588 [0.567, 0.610] |
| Catatan obat rutin tersedia | 6,186 | 166 | M2 | 0.825 [0.796, 0.853] | 0.898 [0.851, 0.941] | 0.552 [0.532, 0.572] |
| Catatan obat rutin tersedia | 6,186 | 166 | M3 | 0.741 [0.701, 0.779] | 0.886 [0.831, 0.933] | 0.432 [0.410, 0.454] |
| Catatan obat rutin tersedia | 6,186 | 166 | M4 | 0.847 [0.817, 0.874] | 0.898 [0.845, 0.940] | 0.574 [0.555, 0.595] |
| Catatan obat rutin tersedia | 6,186 | 166 | M5 | 0.875 [0.850, 0.899] | 0.904 [0.854, 0.947] | 0.665 [0.646, 0.685] |
| Catatan obat rutin tersedia | 6,186 | 166 | M6 | 0.846 [0.818, 0.873] | 0.898 [0.849, 0.941] | 0.608 [0.589, 0.629] |
| Catatan obat rutin tidak tersedia | 1,334 | 52 | M0 | – | 0.269 [0.151, 0.396] | 0.969 [0.958, 0.978] |
| Catatan obat rutin tidak tersedia | 1,334 | 52 | M0S1 | – | 0.269 [0.151, 0.396] | 0.971 [0.961, 0.980] |
| Catatan obat rutin tidak tersedia | 1,334 | 52 | M1 | 0.915 [0.876, 0.949] | 0.923 [0.841, 0.983] | 0.731 [0.703, 0.757] |
| Catatan obat rutin tidak tersedia | 1,334 | 52 | M2 | 0.849 [0.800, 0.893] | 0.923 [0.842, 0.984] | 0.591 [0.562, 0.620] |
| Catatan obat rutin tidak tersedia | 1,334 | 52 | M3 | 0.837 [0.786, 0.880] | 0.962 [0.902, 1.000] | 0.560 [0.529, 0.589] |
| Catatan obat rutin tidak tersedia | 1,334 | 52 | M4 | 0.876 [0.831, 0.916] | 0.923 [0.837, 0.983] | 0.615 [0.584, 0.645] |
| Catatan obat rutin tidak tersedia | 1,334 | 52 | M5 | 0.911 [0.871, 0.945] | 0.904 [0.814, 0.979] | 0.721 [0.693, 0.747] |
| Catatan obat rutin tidak tersedia | 1,334 | 52 | M6 | 0.872 [0.827, 0.912] | 0.923 [0.837, 0.983] | 0.650 [0.619, 0.678] |

**M5 - M4 pada analisis yang direncanakan**

| analisis | pasangan | metrik | selisih | IK 95% | IK tidak mencakup 0 | iterasi sah |
|---|---|---|---|---|---|---|
| Hanya pasien rawat inap (label_source = hosp) [wajib] | M5 - M4 | AUROC | 0.034 | [0.015, 0.053] | ya | 2000 |
| Tanpa IMA tipe 2 murni | M5 - M4 | AUROC | 0.030 | [0.014, 0.046] | ya | 2000 |
| Tanpa keluhan yang menyebut diagnosis | M5 - M4 | AUROC | 0.029 | [0.013, 0.046] | ya | 2000 |

## 6. Kalibrasi setelah Platt terkunci

Kurva kalibrasi fleksibel (regresi logistik restricted cubic spline pada logit p, 4 knot; IK 95% bootstrap per subject_id, 2000x) disimpan di `eval_fold18_uji_final_calibration_curve.csv` (grid 50 titik antara persentil 1 dan 99 prediksi). Tabel di bawah: 10 bin berisi jumlah kunjungan sama (juga di `eval_fold18_uji_final_calibration_bins.csv`).

**M1**

| bin | n | rata-rata prediksi | proporsi teramati |
|---|---|---|---|
| 1 | 752 | 0.0017 | 0.0013 |
| 2 | 752 | 0.0033 | 0.0013 |
| 3 | 752 | 0.0051 | 0.0027 |
| 4 | 752 | 0.0075 | 0.0053 |
| 5 | 752 | 0.0108 | 0.0040 |
| 6 | 752 | 0.0153 | 0.0146 |
| 7 | 752 | 0.0220 | 0.0226 |
| 8 | 752 | 0.0324 | 0.0399 |
| 9 | 752 | 0.0508 | 0.0505 |
| 10 | 752 | 0.1409 | 0.1476 |

**M2**

| bin | n | rata-rata prediksi | proporsi teramati |
|---|---|---|---|
| 1 | 752 | 0.0004 | 0.0013 |
| 2 | 752 | 0.0014 | 0.0000 |
| 3 | 752 | 0.0034 | 0.0040 |
| 4 | 752 | 0.0064 | 0.0066 |
| 5 | 752 | 0.0103 | 0.0066 |
| 6 | 752 | 0.0158 | 0.0133 |
| 7 | 752 | 0.0241 | 0.0306 |
| 8 | 752 | 0.0368 | 0.0332 |
| 9 | 752 | 0.0591 | 0.0638 |
| 10 | 752 | 0.1323 | 0.1303 |

**M3**

| bin | n | rata-rata prediksi | proporsi teramati |
|---|---|---|---|
| 1 | 752 | 0.0056 | 0.0053 |
| 2 | 752 | 0.0076 | 0.0013 |
| 3 | 752 | 0.0096 | 0.0080 |
| 4 | 752 | 0.0120 | 0.0120 |
| 5 | 752 | 0.0150 | 0.0146 |
| 6 | 752 | 0.0189 | 0.0213 |
| 7 | 752 | 0.0249 | 0.0399 |
| 8 | 752 | 0.0360 | 0.0293 |
| 9 | 752 | 0.0580 | 0.0598 |
| 10 | 752 | 0.1022 | 0.0984 |

**M4**

| bin | n | rata-rata prediksi | proporsi teramati |
|---|---|---|---|
| 1 | 752 | 0.0004 | 0.0013 |
| 2 | 752 | 0.0012 | 0.0000 |
| 3 | 752 | 0.0026 | 0.0013 |
| 4 | 752 | 0.0046 | 0.0066 |
| 5 | 752 | 0.0075 | 0.0053 |
| 6 | 752 | 0.0124 | 0.0186 |
| 7 | 752 | 0.0201 | 0.0146 |
| 8 | 752 | 0.0329 | 0.0359 |
| 9 | 752 | 0.0581 | 0.0505 |
| 10 | 752 | 0.1502 | 0.1556 |

**M5**

| bin | n | rata-rata prediksi | proporsi teramati |
|---|---|---|---|
| 1 | 752 | 0.0001 | 0.0013 |
| 2 | 752 | 0.0006 | 0.0000 |
| 3 | 752 | 0.0016 | 0.0027 |
| 4 | 752 | 0.0032 | 0.0013 |
| 5 | 752 | 0.0058 | 0.0000 |
| 6 | 752 | 0.0102 | 0.0080 |
| 7 | 752 | 0.0174 | 0.0239 |
| 8 | 752 | 0.0295 | 0.0346 |
| 9 | 752 | 0.0534 | 0.0412 |
| 10 | 752 | 0.1681 | 0.1769 |

**M6**

| bin | n | rata-rata prediksi | proporsi teramati |
|---|---|---|---|
| 1 | 752 | 0.0002 | 0.0013 |
| 2 | 752 | 0.0009 | 0.0000 |
| 3 | 752 | 0.0024 | 0.0013 |
| 4 | 752 | 0.0047 | 0.0066 |
| 5 | 752 | 0.0081 | 0.0080 |
| 6 | 752 | 0.0130 | 0.0106 |
| 7 | 752 | 0.0209 | 0.0239 |
| 8 | 752 | 0.0343 | 0.0279 |
| 9 | 752 | 0.0595 | 0.0638 |
| 10 | 752 | 0.1458 | 0.1463 |
