# Bab 10: Clustering
<p>
  <a href="https://colab.research.google.com/github/dandienter/praktikumML/blob/main/bab-10-clustering/praktikum-bab-10.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
  </a>
  <a href="Modul_Bab10_Implementasi_Python.pdf">📄 Modul PDF</a>
</p>

## Tujuan
Memahami clustering (unsupervised learning) dan menentukan jumlah cluster optimal lewat Elbow Method serta Silhouette Score, lalu membandingkan K-Means, Agglomerative Clustering, dan DBSCAN pada dataset yang sama. Dilengkapi tiga Percobaan Mandiri: perbandingan inisialisasi KMeans (random vs k-means++), pengaruh scaling fitur terhadap clustering, dan sensitivitas DBSCAN terhadap parameter eps.

## Isi
- `praktikum-bab-10.ipynb` - notebook praktikum (sudah dieksekusi, lengkap dengan output & visualisasi)
- `[Modul_Bab10_Implementasi_Python.pdf](Modul_Bab10_Implementasi_Python.pdf)` - modul PDF format buku (cover + penjelasan lengkap tiap kode dan hasil + interpretasi), isi sinkron dengan notebook di folder ini

## Implementasi Python
Menjalankan clustering secara end-to-end (Modul 10.11) pada 300 pelanggan sintetis dari tiga segmen: K-Means dan Hierarchical sama-sama meraih Silhouette Score 0,7185, sedangkan DBSCAN membentuk 2 cluster padat plus 1 titik noise dengan Silhouette 0,7298. Dilengkapi visualisasi scatter ketiga algoritma berdampingan.

## Sumber Data
Data sintetis pelanggan dari `sklearn.datasets.make_blobs` (1000 baris, 2 fitur: "Annual Income" dan "Spending Score", 5 pusat cluster; pengganti dataset Mall Customer Segmentation dari Kaggle sesuai izin modul). Tidak ada file yang perlu diunduh.

## Cara Menjalankan
1. Buka `praktikum-bab-10.ipynb` di Google Colab atau Jupyter Notebook lokal.
2. Jalankan cell berurutan dari atas ke bawah (Run All).
3. Dataset dibuat langsung lewat `sklearn.datasets.make_blobs` - tidak perlu mengunduh apa pun.

---
<sub>← <a href="../README.md">Kembali ke daftar bab</a></sub>
