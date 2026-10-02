# Bab 4 — Feature Engineering: Konstruksi, Seleksi Fitur, dan Pipeline Scikit-Learn

## Tujuan
Membangun feature construction, seleksi fitur (filter/wrapper/embedded), dan Pipeline + ColumnTransformer yang bebas data leakage (California Housing + Titanic).

## Isi
- `praktikum-bab-04.ipynb` — notebook praktikum (sudah dieksekusi, lengkap dengan output & visualisasi)
- `praktikum-bab-*.pdf` — versi PDF bergaya modul
- `data/` — dataset yang dipakai notebook (format CSV, bisa dipakai offline)

## Sumber Data
california_housing.csv — diekspor dari `sklearn.datasets.fetch_california_housing` (20640 rumah); titanic.csv — diekspor dari `seaborn.load_dataset('titanic')` (891 penumpang); breast_cancer.csv — untuk latihan perbandingan feature selection.

## Cara Menjalankan
1. Buka `praktikum-bab-04.ipynb` di Google Colab (unggah file + folder `data/`) atau Jupyter Notebook lokal.
2. Jalankan cell berurutan dari atas ke bawah (Run All).
3. Dataset dibaca dari folder `data/` — tidak perlu mengunduh apa pun.
