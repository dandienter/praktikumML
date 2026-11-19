# Bab 7: Evaluasi Model Klasifikasi
<p>
  <a href="https://colab.research.google.com/github/dandienter/praktikumML/blob/main/bab-07-evaluasi-model/praktikum-bab-07.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
  </a>
  <a href="praktikum-bab-07.pdf">📄 Modul PDF</a>
</p>

## Tujuan
Menyusun laporan evaluasi model klasifikasi yang komprehensif (confusion matrix, precision/recall/F1, ROC curve, PR curve) serta membandingkan beberapa algoritma secara andal memakai Stratified k-Fold cross-validation. Dilengkapi tiga percobaan mandiri: single split vs cross-validation, pengaruh `random_state` ke hasil split, dan perbandingan akurasi vs F1 di data imbalance.

## Isi
- `praktikum-bab-07.ipynb` - notebook praktikum (sudah dieksekusi, lengkap dengan output & visualisasi)
- `[praktikum-bab-07.pdf](praktikum-bab-07.pdf)` - modul PDF bergaya modul (cover + daftar isi + kode + output + visualisasi)
- `[Modul_Bab7_Implementasi_Python.pdf](Modul_Bab7_Implementasi_Python.pdf)` - modul PDF format buku (cover + penjelasan lengkap tiap kode dan hasil + interpretasi), isi sinkron dengan notebook di folder ini

## Implementasi Python
Menjalankan evaluasi model secara end-to-end (Modul 7.12): Logistic Regression pada Breast Cancer. Confusion matrix TP=71, FP=1, FN=1, TN=41; akurasi 0,9825; F1 0,9861; Stratified 5-Fold CV menghasilkan F1 rata-rata 0,9794 +/- 0,0127; dilengkapi kurva ROC dari probabilitas prediksi.

## Sumber Data
`sklearn.datasets.load_breast_cancer` (dataset publik Breast Cancer Wisconsin: 569 baris, 30 fitur, 2 kelas). Tidak ada file yang perlu diunduh.

## Cara Menjalankan
1. Buka `praktikum-bab-07.ipynb` di Google Colab atau Jupyter Notebook lokal.
2. Jalankan cell berurutan dari atas ke bawah (Run All).
3. Dataset dimuat langsung dari `sklearn.datasets.load_breast_cancer` - tidak perlu mengunduh apa pun.

---
<sub>← <a href="../README.md">Kembali ke daftar bab</a></sub>
