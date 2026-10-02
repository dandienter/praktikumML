# Bab 5: Decision Tree dan Teknik Pruning

## Tujuan
Memahami teknik pruning untuk mengatasi overfitting pada decision tree lewat Latihan Praktikum: pre-pruning (grid manual `max_depth` x `min_samples_leaf`) dan post-pruning (cost complexity pruning path).

## Isi
- `praktikum-bab-05.ipynb` - notebook Latihan Praktikum (sudah dieksekusi, lengkap dengan output & visualisasi)
- `praktikum-bab-*.pdf` - versi PDF bergaya modul

## Latihan Praktikum
1. **Latihan 1 - Grid manual pre-pruning.** Grid search manual atas 12 kombinasi `max_depth` {2, 4, 6, None} x `min_samples_leaf` {1, 5, 10} pada Breast Cancer. Hasil terbaik: akurasi test 0,9474 (`max_depth=4`, `min_samples_leaf=10`). Dilengkapi heatmap akurasi per kombinasi.
2. **Latihan 2 - Cost complexity pruning path.** Untuk tiap `alpha` di `ccp_alphas`, latih decision tree dan catat akurasi test serta jumlah daun. Alpha optimal 0,002866 memangkas pohon dari 19 jadi 12 daun sambil menaikkan akurasi test dari 0,9123 ke 0,9386. Dilengkapi plot `ccp_alpha` vs akurasi test.

## Sumber Data
`sklearn.datasets.load_breast_cancer` (dataset publik Breast Cancer Wisconsin: 569 baris, 30 fitur). Tidak ada file yang perlu diunduh.

## Cara Menjalankan
1. Buka `praktikum-bab-05.ipynb` di Google Colab atau Jupyter Notebook lokal.
2. Jalankan cell berurutan dari atas ke bawah (Run All).
3. Dataset dimuat langsung dari `sklearn.datasets` - tidak perlu mengunduh apa pun.
