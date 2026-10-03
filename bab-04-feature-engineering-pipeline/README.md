# Bab 4: Feature Engineering: Konstruksi, Seleksi Fitur, dan Pipeline Scikit-Learn
<p>
  <a href="https://colab.research.google.com/github/dandienter/praktikumML/blob/main/bab-04-feature-engineering-pipeline/praktikum-bab-04.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
  </a>
  <a href="praktikum-bab-04.pdf">📄 Modul PDF</a>
</p>

## Tujuan
Latihan Praktikum Bab 4: membandingkan tiga pendekatan feature selection (filter, wrapper, embedded) dan membangun pipeline lengkap dengan ColumnTransformer (dataset Breast Cancer + Titanic).

## Isi
- `praktikum-bab-04.ipynb` - notebook Latihan Praktikum (sudah dieksekusi, lengkap dengan output & visualisasi)
- `[praktikum-bab-04.pdf](praktikum-bab-04.pdf)` - modul PDF bergaya modul (cover + daftar isi + kode + output + visualisasi)
- `data/` - dataset yang dipakai notebook (format CSV, bisa dipakai offline)

## Latihan Praktikum
- **Praktikum 1 - Membandingkan tiga pendekatan feature selection (dataset Breast Cancer):** filter (`SelectKBest` chi-square), wrapper (`RFE` + LogisticRegression), embedded (L1 LogisticRegression via `SelectFromModel`). Masing-masing memilih 10 fitur terbaik; hasilnya dibandingkan dalam satu tabel plus grafik 10 fitur dengan skor chi-square tertinggi.
- **Praktikum 2 - Membangun pipeline lengkap (dataset Titanic):** `ColumnTransformer` (numerik: median + StandardScaler; kategorikal: most_frequent + OneHotEncoder) + `LogisticRegression`, dievaluasi dengan akurasi di data uji plus confusion matrix.

## Percobaan Mandiri
Tiga percobaan mandiri di dataset sintetis properti (500 baris, setup datanya ditulis ulang tiap cell): (A) `pipeline.predict` vs preprocessing manual yang benar vs manual yang salah (lupa fitur konstruksi), (B) dengan vs tanpa feature construction `rasio_kamar_per_luas` (R2 sama-sama 0,987), (C) trade-off `n_estimators` 50 vs 200 (R2 sama, 200 pohon 6x lebih lambat).

## Sumber Data
- `breast_cancer.csv` - diekspor dari `sklearn.datasets.load_breast_cancer`; dipakai di latihan perbandingan feature selection (modul: "dataset klasifikasi pilihan Anda").
- `titanic.csv` - diekspor dari `seaborn.load_dataset('titanic')` (891 penumpang); dipakai di latihan pipeline.

## Cara Menjalankan
1. Buka `praktikum-bab-04.ipynb` di Google Colab (klik badge di README utama) atau Jupyter Notebook lokal.
2. Jalankan cell berurutan dari atas ke bawah (Run All). **Sel 0** otomatis mengunduh file `data/` yang dibutuhkan dari GitHub saat dibuka di Colab.

---
<sub>← <a href="../README.md">Kembali ke daftar bab</a></sub>
