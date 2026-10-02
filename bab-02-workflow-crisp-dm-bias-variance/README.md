# Bab 2 — Workflow Machine Learning, CRISP-DM, dan Diagnosis Bias-Variance

## Tujuan
Memahami 6 fase CRISP-DM, pembagian data train/validation/test, stratified sampling, dan cara mendiagnosis bias-variance lewat learning curve (data sintetis + Decision Tree).

## Isi
- `praktikum-bab-02.ipynb` — notebook praktikum (sudah dieksekusi, lengkap dengan output & visualisasi)
- `praktikum-bab-*.pdf` — versi PDF bergaya modul
- `data/` — dataset yang dipakai notebook (format CSV, bisa dipakai offline)

## Sumber Data
Data sintetis `make_classification` (1000 sampel, 10 fitur, 6 informatif, imbalance kelas 0.8/0.2, `random_state=42`) untuk bagian 2.12 (modul); telco-customer-churn.csv — dipakai untuk cuplikan data pada latihan pemetaan CRISP-DM (2.13).

## Cara Menjalankan
1. Buka `praktikum-bab-02.ipynb` di Google Colab (unggah file + folder `data/`) atau Jupyter Notebook lokal.
2. Jalankan cell berurutan dari atas ke bawah (Run All).
3. Dataset dibaca dari folder `data/` — tidak perlu mengunduh apa pun.
