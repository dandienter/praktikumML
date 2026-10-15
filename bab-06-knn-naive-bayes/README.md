# Bab 6: K-Nearest Neighbors dan Naive Bayes
<p>
  <a href="https://colab.research.google.com/github/dandienter/praktikumML/blob/main/bab-06-knn-naive-bayes/praktikum-bab-06.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
  </a>
  <a href="praktikum-bab-06.pdf">📄 Modul PDF</a>
</p>

## Tujuan
Mengikuti modul Bab 6 bagian 6.16 (Praktikum 6.1-6.7): membandingkan KNN dan Naive Bayes pada dataset Wine (akurasi, pengaruh nilai k, klasifikasi teks spam), lalu mengerjakan Latihan Praktikum 6.17: perbandingan KNN vs Naive Bayes pada dataset Breast Cancer (akurasi, waktu pelatihan, waktu prediksi) dan klasifikasi teks MultinomialNB dengan evaluasi train-test split.

## Isi
- `praktikum-bab-06.ipynb` - notebook praktikum (sudah dieksekusi, lengkap dengan output & visualisasi)
- `[praktikum-bab-06.pdf](praktikum-bab-06.pdf)` - modul PDF bergaya modul (cover + daftar isi + kode + output + visualisasi)

## Sumber Data
- `sklearn.datasets.load_wine` (dataset publik Wine Recognition: 178 baris, 13 fitur).
- `sklearn.datasets.load_breast_cancer` (dataset publik Breast Cancer Wisconsin: 569 baris, 30 fitur). Tidak ada file yang perlu diunduh.
- Data teks kecil buatan sendiri (6 kalimat latih untuk contoh 6.7, 16 kalimat untuk Latihan 2) untuk klasifikasi spam Bahasa Indonesia, ditulis langsung di notebook.

## Cara Menjalankan
1. Buka `praktikum-bab-06.ipynb` di Google Colab atau Jupyter Notebook lokal.
2. Jalankan cell berurutan dari atas ke bawah (Run All).
3. Dataset dimuat langsung dari `sklearn.datasets` - tidak perlu mengunduh apa pun.

---
<sub>← <a href="../README.md">Kembali ke daftar bab</a></sub>
