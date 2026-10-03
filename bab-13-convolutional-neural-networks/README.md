# Bab 13: Convolutional Neural Networks
<p>
  <a href="https://colab.research.google.com/github/dandienter/praktikumML/blob/main/bab-13-convolutional-neural-networks/praktikum-bab-13.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
  </a>
  <a href="praktikum-bab-13.pdf">📄 Modul PDF</a>
</p>

## Tujuan
Membangun CNN sederhana dari nol pada subset CIFAR-10 lalu membandingkannya dengan transfer learning memakai MobileNetV2 (bobot ImageNet), sesuai Latihan Praktikum Modul Bab 13 bagian 13.12. Dilengkapi tiga Percobaan Mandiri pada CNN mini (subset 2.000 train / 500 val): data augmentation vs tanpa, jumlah filter 16 vs 32, dan dropout 0 vs 0.5.

## Isi
- `praktikum-bab-13.ipynb` - notebook praktikum (sudah dieksekusi, lengkap dengan output & visualisasi)
- `[praktikum-bab-13.pdf](praktikum-bab-13.pdf)` - modul PDF bergaya modul (cover + daftar isi + kode + output + visualisasi)

## Sumber Data
`keras.datasets.cifar10` (otomatis diunduh saat notebook dijalankan). Karena training di CPU, dipakai subset stratified: 500 citra per kelas untuk train (5.000 total) dan 100 citra per kelas untuk test (1.000 total). Tidak ada file yang perlu diunduh manual.

## Cara Menjalankan
1. Buka `praktikum-bab-13.ipynb` di Google Colab atau Jupyter Notebook lokal.
2. Jalankan cell berurutan dari atas ke bawah (Run All).
3. Di Colab, jalankan Sel 0 dulu untuk memastikan `tensorflow` terinstall. Bobot MobileNetV2 (~14 MB) ikut terunduh otomatis saat cell transfer learning dijalankan.

---
<sub>← <a href="../README.md">Kembali ke daftar bab</a></sub>
