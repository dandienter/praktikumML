# Bab 9: Ensemble Methods

## Tujuan
Memahami ide ensemble (bagging vs boosting) lewat Latihan Praktikum Modul Bab 9 bagian 9.12 (pengaruh `n_estimators` terhadap OOB score Random Forest dan perbandingan empat model dengan Stratified 5-Fold CV) plus tiga Percobaan Mandiri: ketahanan ensemble terhadap label noise, pengaruh learning rate di Gradient Boosting, dan hard voting vs soft voting.

## Isi
- `praktikum-bab-09.ipynb` - notebook praktikum (sudah dieksekusi, lengkap dengan output & visualisasi)
- `praktikum-bab-*.pdf` - versi PDF bergaya modul
- `grafik-n-estimators.png`, `grafik-banding-model.png` - grafik hasil praktikum

## Sumber Data
`sklearn.datasets.load_breast_cancer` (dataset publik Breast Cancer: 569 baris, 30 fitur). Tidak ada file yang perlu diunduh.

## Cara Menjalankan
1. Buka `praktikum-bab-09.ipynb` di Google Colab atau Jupyter Notebook lokal.
2. Jalankan cell berurutan dari atas ke bawah (Run All).
3. Dataset dimuat langsung dari `sklearn.datasets.load_breast_cancer` - tidak perlu mengunduh apa pun.
