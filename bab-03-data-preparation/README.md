# Bab 3 - Data Preparation Lanjutan: Missing Values dan Imbalanced Data

## Tujuan
Mengerjakan **Latihan Praktikum Modul Bab 3 bagian 3.10**: membandingkan strategi imputasi missing value (mean vs KNN Imputer) dan mengukur efektivitas penanganan data tidak seimbang (baseline, SMOTE, undersampling, class_weight) dengan metrik F1-score.

## Isi
- `praktikum-bab-03.ipynb` - notebook Latihan Praktikum (sudah dieksekusi, lengkap dengan output & visualisasi)
- `praktikum-bab-*.pdf` - versi PDF bergaya modul

## Latihan yang Dikerjakan
- **Latihan 1 (modul 3.10):** mean imputation vs KNN Imputer (k=5) pada data sintetis, masing-masing dilatih Logistic Regression dan dibandingkan skor F1. Hasil: KNN Imputer F1 0,308 sedikit mengungguli mean imputation F1 0,296.
- **Latihan 2 (modul 3.10 / Praktikum 3.7):** perbandingan empat pendekatan imbalanced data memakai Stratified split dan Logistic Regression. Hasil F1: SMOTE 0,367, class_weight 0,346, undersampling 0,336, baseline 0,308. SMOTE paling efektif untuk kasus ini.

## Sumber Data
Data sintetis dari `sklearn.datasets.make_classification` (n=2000, 8 fitur, rasio kelas 0.95/0.05) dengan simulasi missing value 5% acak, dibuat langsung di dalam notebook. Tidak ada file CSV.

## Cara Menjalankan
1. Buka `praktikum-bab-03.ipynb` di Google Colab atau Jupyter Notebook lokal.
2. Jalankan cell berurutan dari atas ke bawah (Run All).
3. Tidak ada dataset yang perlu diunduh; cell Sel 0 memasang `imbalanced-learn` otomatis kalau belum ada (khusus Colab).
