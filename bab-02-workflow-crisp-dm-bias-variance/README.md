# Bab 2: Workflow Machine Learning, CRISP-DM, dan Diagnosis Bias-Variance
<p>
  <a href="https://colab.research.google.com/github/dandienter/praktikumML/blob/main/bab-02-workflow-crisp-dm-bias-variance/praktikum-bab-02.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
  </a>
  <a href="praktikum-bab-02.pdf">📄 Modul PDF</a>
</p>

## Tujuan
Memahami 6 fase CRISP-DM lewat studi kasus nyata dan cara mendiagnosis bias-variance lewat learning curve (data sintetis + Decision Tree).

## Isi
- `praktikum-bab-02.ipynb` - notebook Latihan Praktikum (sudah dieksekusi, lengkap dengan output & visualisasi)
- `[praktikum-bab-02.pdf](praktikum-bab-02.pdf)` - modul PDF bergaya modul (cover + daftar isi + kode + output + visualisasi)
- `data/` - dataset yang dipakai notebook (format CSV, bisa dipakai offline)

## Latihan Praktikum
- **Latihan 1 - Memetakan proyek nyata ke CRISP-DM:** studi kasus prediksi churn pelanggan telekomunikasi (dataset Telco Customer Churn, 7.043 pelanggan). Eksplorasi data menemukan sinyal bisnis terkuat di `Contract` (month-to-month churn 42,7% vs kontrak 2 tahun 2,8%), lalu tiap temuan dipetakan ke 6 fase CRISP-DM dengan aktivitas konkret.
- **Latihan 2 - Diagnosis bias-variance:** membandingkan learning curve Decision Tree untuk `max_depth` 1, 4, dan None pada data sintetis. Hasil: `max_depth=1` underfitting (train 0.849, validasi 0.800), `max_depth=4` paling seimbang (train 0.952, validasi 0.866), `max_depth=None` overfitting (train 1.000, validasi 0.858). Dilengkapi 3 grafik learning curve.

## Percobaan Mandiri
- **PM1 - Split tanpa vs dengan stratify:** proporsi kelas 1 bergeser (train 21,1% vs test 17,0%) tanpa stratify, terjaga (20,3% vs 20,5%) dengan stratify.
- **PM2 - Decision Tree vs Logistic Regression:** akurasi test 0.865 vs 0.84, pohon menang karena pola datanya non-linear.
- **PM3 - class_weight="balanced":** F1 kelas minoritas naik dari 0.543 ke 0.569.

## Sumber Data
Data sintetis `make_classification` (1000 sampel, 10 fitur, 6 informatif, imbalance kelas 0.8/0.2, `random_state=42`) dibuat langsung di cell Latihan 2; `data/telco-customer-churn.csv` dipakai untuk Latihan 1.

## Cara Menjalankan
1. Buka `praktikum-bab-02.ipynb` di Google Colab (unggah file + folder `data/`) atau Jupyter Notebook lokal.
2. Jalankan cell berurutan dari atas ke bawah (Run All).
3. Dataset dibaca dari folder `data/` - tidak perlu mengunduh apa pun.

---
<sub>← <a href="../README.md">Kembali ke daftar bab</a></sub>
