# Bab 9: Ensemble Methods
<p>
  <a href="https://colab.research.google.com/github/dandienter/praktikumML/blob/main/bab-09-ensemble-methods/praktikum-bab-09.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
  </a>
  <a href="Modul_Bab9_Implementasi_Python.pdf">📄 Modul PDF</a>
</p>

## Tujuan
Memahami ide ensemble (bagging vs boosting) lewat Latihan Praktikum Modul Bab 9 bagian 9.12 (pengaruh `n_estimators` terhadap OOB score Random Forest dan perbandingan empat model dengan Stratified 5-Fold CV) plus tiga Percobaan Mandiri: ketahanan ensemble terhadap label noise, pengaruh learning rate di Gradient Boosting, dan hard voting vs soft voting.

## Isi
- `praktikum-bab-09.ipynb` - notebook praktikum (sudah dieksekusi, lengkap dengan output & visualisasi)
- `[Modul_Bab9_Implementasi_Python.pdf](Modul_Bab9_Implementasi_Python.pdf)` - modul PDF format buku (cover + penjelasan lengkap tiap kode dan hasil + interpretasi), isi sinkron dengan notebook di folder ini
- `grafik-n-estimators.png`, `grafik-banding-model.png` - grafik hasil praktikum

## Implementasi Python
Menjalankan ensemble learning secara end-to-end (Modul 9.11) pada dataset Wine: Decision Tree tunggal 0,9556, Random Forest 1,0000, AdaBoost 0,9778, Gradient Boosting 0,9778. OOB score Random Forest 0,9699, dilengkapi grafik feature importance.

## Sumber Data
`sklearn.datasets.load_breast_cancer` (dataset publik Breast Cancer: 569 baris, 30 fitur). Tidak ada file yang perlu diunduh.

## Cara Menjalankan
1. Buka `praktikum-bab-09.ipynb` di Google Colab atau Jupyter Notebook lokal.
2. Jalankan cell berurutan dari atas ke bawah (Run All).
3. Dataset dimuat langsung dari `sklearn.datasets.load_breast_cancer` - tidak perlu mengunduh apa pun.

---
<sub>← <a href="../README.md">Kembali ke daftar bab</a></sub>
