# Bab 11: PCA dan Reduksi Dimensi

## Tujuan
Mengerjakan Latihan Praktikum Modul Bab 11 bagian 11.12: menentukan jumlah komponen PCA optimal lewat variance threshold dan membandingkan visualisasi PCA, t-SNE, dan LDA pada dataset yang sama. Dilengkapi tiga Percobaan Mandiri: analisis varians PCA lewat scree plot, pengaruh PCA terhadap akurasi dan waktu prediksi KNN, serta perbandingan visualisasi 2D antara fitur mentah dan komponen PCA.

## Isi
- `praktikum-bab-11.ipynb` - notebook praktikum (sudah dieksekusi, lengkap dengan output & visualisasi)
- `praktikum-bab-*.pdf` - versi PDF bergaya modul

## Sumber Data
`sklearn.datasets.load_breast_cancer` (569 baris, 30 fitur) untuk Praktikum 1 dan `sklearn.datasets.load_wine` (178 baris, 13 fitur, 3 kelas) untuk Praktikum 2. Tidak ada file yang perlu diunduh.

## Cara Menjalankan
1. Buka `praktikum-bab-11.ipynb` di Google Colab atau Jupyter Notebook lokal.
2. Jalankan cell berurutan dari atas ke bawah (Run All).
3. Dataset dimuat langsung dari `sklearn.datasets` - tidak perlu mengunduh apa pun.
