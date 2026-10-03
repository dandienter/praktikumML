# Bab 12: Neural Networks
<p>
  <a href="https://colab.research.google.com/github/dandienter/praktikumML/blob/main/bab-12-neural-networks/praktikum-bab-12.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
  </a>
  <a href="praktikum-bab-12.pdf">📄 Modul PDF</a>
</p>

## Tujuan
Memahami cara membangun dan melatih MLP dengan Keras melalui dua latihan modul (eksperimen arsitektur hidden layer dan early stopping untuk mencegah overfitting) plus tiga Percobaan Mandiri: banding jumlah hidden layer, banding fungsi aktivasi (relu/tanh/sigmoid), dan banding learning rate/optimizer (dataset Digits).

## Isi
- `praktikum-bab-12.ipynb` - notebook praktikum (sudah dieksekusi, lengkap dengan output & visualisasi)
- `[praktikum-bab-12.pdf](praktikum-bab-12.pdf)` - modul PDF bergaya modul (cover + daftar isi + kode + output + visualisasi)

## Sumber Data
`sklearn.datasets.load_digits` (dataset publik Digits: 1797 baris, 64 fitur piksel, 10 kelas angka tulisan tangan). Tidak ada file yang perlu diunduh.

## Cara Menjalankan
1. Buka `praktikum-bab-12.ipynb` di Google Colab atau Jupyter Notebook lokal.
2. Jalankan cell berurutan dari atas ke bawah (Run All).
3. Dataset dimuat langsung dari `sklearn.datasets.load_digits` - tidak perlu mengunduh apa pun. Cell Sel 0 otomatis mengecek/menginstall `tensorflow` bila belum ada.

---
<sub>← <a href="../README.md">Kembali ke daftar bab</a></sub>
