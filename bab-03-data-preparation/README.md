# Bab 3 — Data Preparation Lanjutan: Missing Values, Outlier, dan Imbalanced Data

## Tujuan
Mempraktikkan imputasi missing value (KNN Imputer), deteksi outlier (Isolation Forest), dan penanganan data tidak seimbang (SMOTE, undersampling, class_weight) persis mengikuti Modul Bab 3 bagian 3.9.

## Isi
- `praktikum-bab-03.ipynb` — notebook praktikum (sudah dieksekusi, lengkap dengan output & visualisasi)
- `praktikum-bab-*.pdf` — versi PDF bergaya modul

## Sumber Data
Data sintetis dari `sklearn.datasets.make_classification` (n=2000, 8 fitur, rasio kelas 0.95/0.05) dengan simulasi missing value 5% acak, persis mengikuti modul. Tidak ada file CSV.

## Cara Menjalankan
1. Buka `praktikum-bab-03.ipynb` di Google Colab atau Jupyter Notebook lokal.
2. Jalankan cell berurutan dari atas ke bawah (Run All).
3. Tidak ada dataset yang perlu diunduh; cell Sel 0 memasang `imbalanced-learn` otomatis kalau belum ada (khusus Colab).
