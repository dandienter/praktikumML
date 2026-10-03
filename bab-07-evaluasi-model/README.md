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

## Sumber Data
`sklearn.datasets.load_breast_cancer` (dataset publik Breast Cancer Wisconsin: 569 baris, 30 fitur, 2 kelas). Tidak ada file yang perlu diunduh.

## Cara Menjalankan
1. Buka `praktikum-bab-07.ipynb` di Google Colab atau Jupyter Notebook lokal.
2. Jalankan cell berurutan dari atas ke bawah (Run All).
3. Dataset dimuat langsung dari `sklearn.datasets.load_breast_cancer` - tidak perlu mengunduh apa pun.

---
<sub>← <a href="../README.md">Kembali ke daftar bab</a></sub>
