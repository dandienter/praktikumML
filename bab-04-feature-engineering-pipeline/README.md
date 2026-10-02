# Bab 4 — Feature Engineering: Konstruksi, Seleksi Fitur, dan Pipeline Scikit-Learn

## Tujuan
Membangun feature construction, seleksi fitur (filter/wrapper/embedded), dan Pipeline + ColumnTransformer yang bebas data leakage (dataset sintetis properti + breast cancer + Titanic).

## Isi
- `praktikum-bab-04.ipynb` — notebook praktikum (sudah dieksekusi, lengkap dengan output & visualisasi)
- `praktikum-bab-*.pdf` — versi PDF bergaya modul
- `data/` — dataset yang dipakai notebook (format CSV, bisa dipakai offline)

## Sumber Data
- **Dataset sintetis properti** (praktikum 4.1–4.2, 4.11–4.15 + percobaan mandiri): dibangkitkan di dalam notebook sesuai Modul Bab 4 bagian 4.10 (`np.random.default_rng(42)`, n=500; kolom `luas_bangunan`, `jumlah_kamar`, `tipe_properti`, `kondisi_bangunan`, target `harga`). Tidak perlu file CSV.
- `titanic.csv` — diekspor dari `seaborn.load_dataset('titanic')` (891 penumpang); dipakai di latihan pipeline.
- `breast_cancer.csv` — diekspor dari `sklearn.datasets.load_breast_cancer`; dipakai di latihan perbandingan feature selection (modul: "dataset klasifikasi pilihan Anda").

## Cara Menjalankan
1. Buka `praktikum-bab-04.ipynb` di Google Colab (klik badge di README utama) atau Jupyter Notebook lokal.
2. Jalankan cell berurutan dari atas ke bawah (Run All). **Sel 0** otomatis mengunduh file `data/` yang dibutuhkan dari GitHub saat dibuka di Colab.
3. Dataset sintetis properti dibangkitkan langsung di notebook, tidak perlu mengunduh apa pun.
