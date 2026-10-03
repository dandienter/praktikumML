# Bab 6: K-Nearest Neighbors dan Naive Bayes
<p>
  <a href="https://colab.research.google.com/github/dandienter/praktikumML/blob/main/bab-06-knn-naive-bayes/praktikum-bab-06.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
  </a>
  <a href="praktikum-bab-06.pdf">📄 Modul PDF</a>
</p>

## Tujuan
Membandingkan performa KNN dan Naive Bayes pada dataset yang sama (akurasi, waktu pelatihan, waktu prediksi), serta menerapkan Multinomial Naive Bayes untuk klasifikasi teks sederhana. Dilengkapi tiga percobaan mandiri: pengaruh nilai k di KNN (k = 1, 5, 15), perbandingan `weights="uniform"` vs `"distance"`, dan perbandingan GaussianNB vs MultinomialNB di data count sintetis.

## Isi
- `praktikum-bab-06.ipynb` - notebook praktikum (sudah dieksekusi, lengkap dengan output & visualisasi)
- `[praktikum-bab-06.pdf](praktikum-bab-06.pdf)` - modul PDF bergaya modul (cover + daftar isi + kode + output + visualisasi)

## Sumber Data
- `sklearn.datasets.load_breast_cancer` (dataset publik Breast Cancer Wisconsin: 569 baris, 30 fitur). Tidak ada file yang perlu diunduh.
- Data teks kecil buatan sendiri (6 kalimat latih + teks uji) untuk Praktikum 2, ditulis langsung di notebook.

## Cara Menjalankan
1. Buka `praktikum-bab-06.ipynb` di Google Colab atau Jupyter Notebook lokal.
2. Jalankan cell berurutan dari atas ke bawah (Run All).
3. Dataset dimuat langsung dari `sklearn.datasets` - tidak perlu mengunduh apa pun.

---
<sub>← <a href="../README.md">Kembali ke daftar bab</a></sub>
