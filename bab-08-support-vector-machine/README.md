# Bab 8: Support Vector Machine
<p>
  <a href="https://colab.research.google.com/github/dandienter/praktikumML/blob/main/bab-08-support-vector-machine/praktikum-bab-08.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
  </a>
  <a href="Modul_Bab8_Implementasi_Python.pdf">📄 Modul PDF</a>
</p>

## Tujuan
Memahami cara kerja SVM (margin maksimum, kernel trick, parameter C dan gamma) lewat dua Latihan Praktikum (perbandingan kernel linear/polynomial/RBF dan visualisasi decision boundary) plus tiga Percobaan Mandiri: pengaruh parameter C (C kecil vs C besar), kernel linear vs RBF di data melingkar, dan pentingnya scaling fitur.

## Isi
- `praktikum-bab-08.ipynb` - notebook praktikum (sudah dieksekusi, lengkap dengan output & visualisasi)
- `[Modul_Bab8_Implementasi_Python.pdf](Modul_Bab8_Implementasi_Python.pdf)` - modul PDF format buku (cover + penjelasan lengkap tiap kode dan hasil + interpretasi), isi sinkron dengan notebook di folder ini

## Implementasi Python
Menjalankan SVM secara end-to-end (Modul 8.10): Pipeline StandardScaler + SVC kernel RBF dengan GridSearchCV atas C {0.1, 1, 10, 100} dan gamma {0.001, 0.01, 0.1, 1}. Parameter terbaik C=10 dan gamma=0.01 (skor CV 0,9802), akurasi data uji 0,9825, dilengkapi heatmap peta akurasi grid.

## Sumber Data
`sklearn.datasets.load_breast_cancer` (dataset publik Wisconsin Breast Cancer: 569 baris, 30 fitur). Tidak ada file yang perlu diunduh.

## Cara Menjalankan
1. Buka `praktikum-bab-08.ipynb` di Google Colab atau Jupyter Notebook lokal.
2. Jalankan cell berurutan dari atas ke bawah (Run All).
3. Dataset dimuat langsung dari `sklearn.datasets.load_breast_cancer` - tidak perlu mengunduh apa pun.

---
<sub>← <a href="../README.md">Kembali ke daftar bab</a></sub>
