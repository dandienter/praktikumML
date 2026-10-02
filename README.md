# Praktikum Machine Learning

Kumpulan praktikum Machine Learning — 14 bab, dari konsep dasar hingga mini project end-to-end.
Setiap bab berisi notebook Python (.ipynb) yang sudah dieksekusi lengkap dengan output, penjelasan,
percobaan mandiri, studi kasus, dan latihan praktikum yang dikerjakan. Dataset disimpan dalam
format CSV di tiap folder bab sehingga notebook bisa dijalankan secara offline.

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
├── .gitignore
├── tools/
│   └── build_pdf.py              # pembangun PDF bergaya modul (HTML + WeasyPrint)
├── bab-01-konsep-dasar-machine-learning/
│   ├── praktikum-bab-01.ipynb
│   ├── praktikum-bab-01.pdf
│   ├── README.md
│   └── data/iris.csv
├── bab-02-workflow-crisp-dm-bias-variance/
│   ├── praktikum-bab-02.ipynb
│   ├── praktikum-bab-02.pdf
│   ├── README.md
│   └── data/{breast_cancer.csv, telco-customer-churn.csv}
├── bab-03-data-preparation/
│   ├── praktikum-bab-03.ipynb
│   ├── praktikum-bab-03.pdf
│   ├── README.md
│   └── data/telco-customer-churn.csv
├── bab-04-feature-engineering-pipeline/
│   ├── praktikum-bab-04.ipynb
│   ├── praktikum-bab-04.pdf
│   ├── README.md
│   └── data/{california_housing.csv, titanic.csv, breast_cancer.csv}
├── bab-05-decision-tree-pruning/
│   ├── praktikum-bab-05.ipynb
│   ├── praktikum-bab-05.pdf
│   ├── README.md
│   └── data/breast_cancer.csv
├── bab-06-knn-naive-bayes/           # ⏳ segera hadir
├── bab-07-evaluasi-model/            # ⏳ segera hadir
├── bab-08-support-vector-machine/    # ⏳ segera hadir
├── bab-09-ensemble-learning/        # ⏳ segera hadir
├── bab-10-clustering/                # ⏳ segera hadir
├── bab-11-dimensionality-reduction/  # ⏳ segera hadir
├── bab-12-neural-network-mlp/        # ⏳ segera hadir
├── bab-13-cnn-transfer-learning/     # ⏳ segera hadir
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
| 6 | K-Nearest Neighbors (KNN) dan Naive Bayes Classifier | ⏳ Segera hadir |
| 7 | Evaluasi Model: Confusion Matrix, Precision-Recall, ROC-AUC, dan Stratified K-Fold CV | ⏳ Segera hadir |
| 8 | Support Vector Machine: Hyperplane, Margin, dan Kernel Trick | ⏳ Segera hadir |
| 9 | Ensemble Learning: Random Forest, Bagging, dan Boosting | ⏳ Segera hadir |
| 10 | Clustering: K-Means, Hierarchical Clustering, dan DBSCAN | ⏳ Segera hadir |
| 11 | Dimensionality Reduction: PCA, t-SNE, dan LDA | ⏳ Segera hadir |
| 12 | Neural Network: Multi-Layer Perceptron (MLP) | ⏳ Segera hadir |
| 13 | Convolutional Neural Network (CNN) dan Transfer Learning | ⏳ Segera hadir |
| 14 | Mini Project End-to-End: Komparasi Model, Tuning, dan Deployment | ⏳ Segera hadir |

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
| Iris (150 sampel, 4 fitur, 3 spesies) | `sklearn.datasets.load_iris` (data real Fisher), diekspor ke CSV | Bab 1 |
| Breast Cancer Wisconsin (569 sampel, 30 fitur) | `sklearn.datasets.load_breast_cancer`, diekspor ke CSV | Bab 2, 4, 5 |
| Telco Customer Churn (7043 pelanggan) | Kaggle — Telco Customer Churn | Bab 2 (latihan), Bab 3 |
| California Housing (20640 rumah) | `sklearn.datasets.fetch_california_housing`, diekspor ke CSV | Bab 4 |
| Titanic (891 penumpang) | `seaborn.load_dataset('titanic')`, diekspor ke CSV | Bab 4 |

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
