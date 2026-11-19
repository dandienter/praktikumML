# Bab 3 - Data Preparation Lanjutan: Missing Values dan Imbalanced Data
<p>
  <a href="https://colab.research.google.com/github/dandienter/praktikumML/blob/main/bab-03-data-preparation/praktikum-bab-03.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
  </a>
  <a href="Modul_Bab3_Implementasi_Python.pdf">📄 Modul PDF</a>
</p>

## Tujuan
Mengerjakan **Latihan Praktikum Modul Bab 3 bagian 3.10**: membandingkan strategi imputasi missing value (mean vs KNN Imputer) dan mengukur efektivitas penanganan data tidak seimbang (baseline, SMOTE, undersampling, class_weight) dengan metrik F1-score.

## Isi
- `praktikum-bab-03.ipynb` - notebook praktikum (sudah dieksekusi, lengkap dengan output & visualisasi)
- `[Modul_Bab3_Implementasi_Python.pdf](Modul_Bab3_Implementasi_Python.pdf)` - modul PDF format buku (cover + penjelasan lengkap tiap kode dan hasil + interpretasi), isi sinkron dengan notebook di folder ini

## Latihan yang Dikerjakan
- **Latihan 1 (modul 3.10):** mean imputation vs KNN Imputer (k=5) pada data sintetis, masing-masing dilatih Logistic Regression dan dibandingkan skor F1. Hasil: KNN Imputer F1 0,308 sedikit mengungguli mean imputation F1 0,296.
- **Latihan 2 (modul 3.10 / Praktikum 3.7):** perbandingan empat pendekatan imbalanced data memakai Stratified split dan Logistic Regression. Hasil F1: SMOTE 0,367, class_weight 0,346, undersampling 0,336, baseline 0,308. SMOTE paling efektif untuk kasus ini.

## Implementasi Python
Menjalankan data preparation secara end-to-end (Modul 3.9): dataset sintetis 2000 sampel (imbalance 95/5, missing value 5%). Missing value diimputasi KNN Imputer dan 40 outlier terdeteksi Isolation Forest. Perbandingan baseline vs SMOTE: akurasi 0,948 vs 0,777 dan F1 0,323 vs 0,288, tetapi recall kelas minoritas melonjak dari 0,217 ke 0,783, sebuah tradeoff precision-recall yang dibahas apa adanya di interpretasi notebook.

## Percobaan Mandiri
- **PM1 - Pengaruh contamination IsolationForest (0.01/0.02/0.05):** outlier dibuang 19/38/95, F1 turun 0.519 ke 0.200. Jangan buang outlier terlalu agresif.
- **PM2 - SMOTE k_neighbors 5 vs 3:** F1 0.367 vs 0.340, tetangga lebih banyak lebih halus.
- **PM3 - Tanpa vs dengan buang outlier:** F1 naik dari 0.308 ke 0.417 setelah 40 outlier dibuang.

## Sumber Data
Data sintetis dari `sklearn.datasets.make_classification` (n=2000, 8 fitur, rasio kelas 0.95/0.05) dengan simulasi missing value 5% acak, dibuat langsung di dalam notebook. Tidak ada file CSV.

## Cara Menjalankan
1. Buka `praktikum-bab-03.ipynb` di Google Colab atau Jupyter Notebook lokal.
2. Jalankan cell berurutan dari atas ke bawah (Run All).
3. Tidak ada dataset yang perlu diunduh; cell Sel 0 memasang `imbalanced-learn` otomatis kalau belum ada (khusus Colab).

---
<sub>← <a href="../README.md">Kembali ke daftar bab</a></sub>
