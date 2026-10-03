# Bab 14 — Mini Project End-to-End: Prediksi Penyakit Jantung

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dandienter/praktikumML/blob/main/bab-14-mini-project-end-to-end/praktikum-bab-14.ipynb)
[![PDF](https://img.shields.io/badge/PDF-Modul-blue)](praktikum-bab-14.pdf)

Proyek utama praktikum (Modul Bab 14, bagian 14.16): menyelesaikan proyek Machine Learning
end-to-end mengikuti seluruh tahapan **CRISP-DM** — dari pemilihan dataset sampai deployment.

> 🚀 **Aplikasi deployment** (web Streamlit + bot Telegram) ada di repo terpisah:
> **[dandienter/heart-disease-predictor](https://github.com/dandienter/heart-disease-predictor)**
>
> 🌐 **Web live:** https://heart-disease-predictor-dandie-46126b49.koyeb.app

## 📁 Isi Bab

| File | Keterangan |
|---|---|
| [`praktikum-bab-14.ipynb`](praktikum-bab-14.ipynb) | Notebook lengkap (40 cell, sudah dieksekusi) |
| [`praktikum-bab-14.pdf`](praktikum-bab-14.pdf) | Modul PDF (19 halaman) |
| [`data/heart.csv`](data/heart.csv) | Dataset Heart Disease UCI Cleveland (297 baris) |
| `model_heart.pkl` | Model terbaik tersimpan (dipakai aplikasi deployment) |
| `fitur_heart.pkl` | Daftar 15 nama fitur |

## 🧪 Eksperimen yang Dilakukan

1. **Business Understanding** — tujuan: deteksi dini penyakit jantung; kriteria sukses: recall tinggi
2. **EDA** — distribusi umur per target, sebaran kelas (164 sehat vs 139 sakit), heatmap korelasi
3. **Data Preparation** — split 80/20 stratified, scaling via Pipeline (anti data leakage)
4. **Feature Engineering** — `age_group` (kelompok umur) + `bp_chol_ratio` (rasio TD/kolesterol)
5. **Modeling** — 3 algoritma + `GridSearchCV` 5-fold: Logistic Regression, Random Forest, SVM (RBF)
6. **Evaluation** — akurasi, precision, recall, F1, ROC-AUC + confusion matrix & kurva ROC
7. **Pemilihan model** — SVM-RBF (akurasi 86,7% · recall 78,6% · F1 84,6% · ROC-AUC 95,7%),
   dipilih berdasarkan **recall tertinggi** karena false negative berbahaya pada kasus medis
8. **Deployment** — model disimpan `.pkl`, dipakai web Streamlit & bot Telegram
9. **Percobaan mandiri** — (1) pentingnya scaling per algoritma, (2) model ke-4 Gradient Boosting,
   (3) pengaruh feature engineering

## 📊 Hasil Utama

| Model | Akurasi | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 83,3% | 84,6% | 78,6% | 81,5% | 94,4% |
| Random Forest | 81,7% | 87,0% | 71,4% | 78,4% | 94,6% |
| **SVM (RBF) ✅** | **86,7%** | **91,7%** | **78,6%** | **84,6%** | **95,7%** |

## 🌐 Deployment

| Antarmuka | Repo | Cara pakai |
|---|---|---|
| Web (Streamlit) | [live di Koyeb](https://heart-disease-predictor-dandie-46126b49.koyeb.app) · [`heart-disease-predictor`](https://github.com/dandienter/heart-disease-predictor) | isi 13 fitur → klik Prediksi |
| Bot Telegram | [`heart-disease-predictor`](https://github.com/dandienter/heart-disease-predictor) | `/start` → jawab 13 pertanyaan → terima hasil |

## 🔙 Kembali

← [README Utama](../README.md)
