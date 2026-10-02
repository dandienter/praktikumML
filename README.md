# Praktikum Machine Learning

Kumpulan praktikum Machine Learning - 14 bab, dari konsep dasar hingga mini project end-to-end.
Setiap bab berisi notebook Python (.ipynb) yang sudah dieksekusi lengkap dengan output dan
penjelasan. Bab 1 berisi praktikum dasar, sedangkan Bab 2-13 berfokus pada Latihan Praktikum
dari modul. Dataset disimpan dalam format CSV di tiap folder bab (atau dimuat langsung dari
scikit-learn) sehingga notebook bisa dijalankan secara offline maupun di Google Colab.

## Identitas

| | |
|---|---|
| **Nama** | Ahmad Dandi Subhani |
| **NPM** | 202343500126 |
| **Kelas** | R7B |
| **Mata Kuliah** | Machine Learning |
| **Dosen** | Nurfidah Dwitiyanti, M.Si. |

## Struktur Repository

```
praktikumML/
├── README.md
├── GLOSARIUM.md               # glosarium istilah & tools semua bab
├── GLOSARIUM.pdf              # versi PDF glosarium
├── .gitignore
├── tools/
│   └── build_pdf.py              # pembangun PDF bergaya modul (HTML + WeasyPrint)
├── bab-01-konsep-dasar-machine-learning/
│   ├── praktikum-bab-01.ipynb
│   ├── praktikum-bab-01.pdf
│   └── README.md
├── bab-02-workflow-crisp-dm-bias-variance/
│   ├── praktikum-bab-02.ipynb
│   ├── praktikum-bab-02.pdf
│   ├── README.md
│   └── data/telco-customer-churn.csv
├── bab-03-data-preparation/
│   ├── praktikum-bab-03.ipynb
│   ├── praktikum-bab-03.pdf
│   └── README.md
├── bab-04-feature-engineering-pipeline/
│   ├── praktikum-bab-04.ipynb
│   ├── praktikum-bab-04.pdf
│   ├── README.md
│   └── data/{titanic.csv, breast_cancer.csv}
├── bab-05-decision-tree-pruning/
│   ├── praktikum-bab-05.ipynb
│   ├── praktikum-bab-05.pdf
│   ├── README.md
│   └── data/breast_cancer.csv
├── bab-06-knn-naive-bayes/
│   ├── praktikum-bab-06.ipynb
│   ├── praktikum-bab-06.pdf
│   └── README.md
├── bab-07-evaluasi-model/
│   ├── praktikum-bab-07.ipynb
│   ├── praktikum-bab-07.pdf
│   └── README.md
├── bab-08-support-vector-machine/
│   ├── praktikum-bab-08.ipynb
│   ├── praktikum-bab-08.pdf
│   └── README.md
├── bab-09-ensemble-methods/
│   ├── praktikum-bab-09.ipynb
│   ├── praktikum-bab-09.pdf
│   └── README.md
├── bab-10-clustering/
│   ├── praktikum-bab-10.ipynb
│   ├── praktikum-bab-10.pdf
│   └── README.md
├── bab-11-pca-reduksi-dimensi/
│   ├── praktikum-bab-11.ipynb
│   ├── praktikum-bab-11.pdf
│   └── README.md
├── bab-12-neural-networks/
│   ├── praktikum-bab-12.ipynb
│   ├── praktikum-bab-12.pdf
│   └── README.md
├── bab-13-convolutional-neural-networks/
│   ├── praktikum-bab-13.ipynb
│   ├── praktikum-bab-13.pdf
│   └── README.md
└── bab-14-mini-project-end-to-end/   # ⏳ segera hadir
```

## Daftar Bab

| Bab | Judul | Status |
|-----|-------|--------|
| 1 | Konsep Dasar Machine Learning | ✅ Lengkap |
| 2 | Workflow Machine Learning, CRISP-DM, dan Diagnosis Bias-Variance | ✅ Lengkap |
| 3 | Data Preparation Lanjutan: Missing Values, Outlier, dan Imbalanced Data | ✅ Lengkap |
| 4 | Feature Engineering: Konstruksi, Seleksi Fitur, dan Pipeline Scikit-Learn | ✅ Lengkap |
| 5 | Decision Tree: Algoritma CART/ID3 dan Teknik Pruning | ✅ Lengkap |
| 6 | K-Nearest Neighbors (KNN) dan Naive Bayes Classifier | ✅ Lengkap (Latihan Praktikum) |
| 7 | Evaluasi Model: Confusion Matrix, Precision-Recall, ROC-AUC, dan Stratified K-Fold CV | ✅ Lengkap (Latihan Praktikum) |
| 8 | Support Vector Machine: Hyperplane, Margin, dan Kernel Trick | ✅ Lengkap (Latihan Praktikum) |
| 9 | Ensemble Learning: Random Forest, Bagging, dan Boosting | ✅ Lengkap (Latihan Praktikum) |
| 10 | Clustering: K-Means, Hierarchical Clustering, dan DBSCAN | ✅ Lengkap (Latihan Praktikum) |
| 11 | Dimensionality Reduction: PCA, t-SNE, dan LDA | ✅ Lengkap (Latihan Praktikum) |
| 12 | Neural Network: Multi-Layer Perceptron (MLP) | ✅ Lengkap (Latihan Praktikum) |
| 13 | Convolutional Neural Network (CNN) dan Transfer Learning | ✅ Lengkap (Latihan Praktikum) |
| 14 | Mini Project End-to-End: Komparasi Model, Tuning, dan Deployment | ⏳ Segera hadir |

## Glosarium

Istilah-istilah dan tools dari seluruh bab (Bab 1-13) dirangkum ringkas di satu dokumen:

- [GLOSARIUM.md](GLOSARIUM.md) - versi markdown
- [GLOSARIUM.pdf](GLOSARIUM.pdf) - versi PDF siap cetak/baca

## Tools yang Digunakan

| Perangkat | Versi |
|---|---|
| Python | 3.12.3 |
| numpy | 2.5.3 |
| pandas | 3.0.6 |
| matplotlib | 3.11.2 |
| seaborn | 0.13.2 |
| scikit-learn | 1.9.1 |
| imbalanced-learn | 0.14.2 |
| WeasyPrint (pembuat PDF) | 70.0 |

## Sumber Data

| Dataset | Sumber | Dipakai di |
|---|---|---|
| Iris (150 sampel, 4 fitur, 3 spesies) | `sklearn.datasets.load_iris` (data real Fisher), dimuat langsung di notebook | Bab 1 |
| Dataset sintetis klasifikasi (1000 sampel, 10 fitur, imbalance 0.8/0.2) | `sklearn.datasets.make_classification`, dibuat di cell Latihan 2 | Bab 2 (Latihan 2: diagnosis bias-variance) |
| Telco Customer Churn (7043 pelanggan) | Kaggle - Telco Customer Churn | Bab 2 (Latihan 1) |
| Dataset sintetis klasifikasi (2000 sampel, 8 fitur, imbalance 0.95/0.05, missing 5% simulasi) | `sklearn.datasets.make_classification`, dibuat di cell setup Latihan | Bab 3 (Latihan 1 & 2) |
| Breast Cancer Wisconsin (569 sampel, 30 fitur) | `sklearn.datasets.load_breast_cancer`, diekspor ke CSV | Bab 4 (latihan), 5, 6, 7, 8, 9, 11 |
| Titanic (891 penumpang) | `seaborn.load_dataset('titanic')`, diekspor ke CSV | Bab 4 (Latihan 2) |
| Dataset teks spam kecil (buatan sendiri) | Didefinisikan langsung di notebook | Bab 6 (Latihan: MultinomialNB) |
| Dataset sintetis clustering (5 blob) | `sklearn.datasets.make_blobs`, sebagai pengganti Mall Customer Segmentation | Bab 10 |
| Wine (178 sampel, 13 fitur) | `sklearn.datasets.load_wine`, dimuat langsung di notebook | Bab 11 |
| Digits (1797 citra 8x8, 10 kelas angka) | `sklearn.datasets.load_digits`, dimuat langsung di notebook | Bab 12 |
| CIFAR-10 subset (500 citra/kelas train, 100 citra/kelas test) | `keras.datasets.cifar10`, diunduh otomatis saat dijalankan | Bab 13 |

## Cara Menjalankan

### Google Colab

Klik badge di bawah untuk membuka notebook langsung di Colab (atau unggah file `.ipynb` + folder `data/` secara manual):

| Bab | Buka di Colab |
|-----|---------------|
| Bab 1 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dandienter/praktikumML/blob/main/bab-01-konsep-dasar-machine-learning/praktikum-bab-01.ipynb) |
| Bab 2 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dandienter/praktikumML/blob/main/bab-02-workflow-crisp-dm-bias-variance/praktikum-bab-02.ipynb) |
| Bab 3 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dandienter/praktikumML/blob/main/bab-03-data-preparation/praktikum-bab-03.ipynb) |
| Bab 4 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dandienter/praktikumML/blob/main/bab-04-feature-engineering-pipeline/praktikum-bab-04.ipynb) |
| Bab 5 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dandienter/praktikumML/blob/main/bab-05-decision-tree-pruning/praktikum-bab-05.ipynb) |
| Bab 6 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dandienter/praktikumML/blob/main/bab-06-knn-naive-bayes/praktikum-bab-06.ipynb) |
| Bab 7 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dandienter/praktikumML/blob/main/bab-07-evaluasi-model/praktikum-bab-07.ipynb) |
| Bab 8 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dandienter/praktikumML/blob/main/bab-08-support-vector-machine/praktikum-bab-08.ipynb) |
| Bab 9 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dandienter/praktikumML/blob/main/bab-09-ensemble-methods/praktikum-bab-09.ipynb) |
| Bab 10 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dandienter/praktikumML/blob/main/bab-10-clustering/praktikum-bab-10.ipynb) |
| Bab 11 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dandienter/praktikumML/blob/main/bab-11-pca-reduksi-dimensi/praktikum-bab-11.ipynb) |
| Bab 12 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dandienter/praktikumML/blob/main/bab-12-neural-networks/praktikum-bab-12.ipynb) |
| Bab 13 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dandienter/praktikumML/blob/main/bab-13-convolutional-neural-networks/praktikum-bab-13.ipynb) |

> Catatan: setiap notebook diawali **Sel 0 - Persiapan Awal** yang otomatis mengunduh file-file `data/` dari repo ini saat dibuka di Colab, jadi tinggal **Runtime > Run all** dan semuanya langsung jalan (termasuk Latihan Praktikum).

### Jupyter Lokal

```bash
git clone https://github.com/dandienter/praktikumML.git
cd praktikumML
python -m venv .venv && source .venv/bin/activate
pip install jupyter pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
jupyter notebook
# buka bab-0N-.../praktikum-bab-0N.ipynb, jalankan semua cell berurutan
```

### Membangun Ulang PDF

```bash
.venv/bin/python tools/build_pdf.py <folder>/praktikum-bab-0N.ipynb \
  --bab N --judul "<Judul Bab>" --out <folder>/praktikum-bab-0N.pdf
```
