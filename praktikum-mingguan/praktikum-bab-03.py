# ============================================================
# Praktikum Bab 03: Data Preparation
# Diekstrak dari bab-03-data-preparation/praktikum-bab-03.ipynb
# ============================================================

# Praktikum Machine Learning: Bab 3
# Data Preparation Lanjutan: Missing Values, Outlier, dan Imbalanced Data
#
# | | |
# |---|---|
# | **Nama** | Ahmad Dandi Subhani |
# | **NPM** | 202343500126 |
# | **Kelas** | R7B |
# | **Mata Kuliah** | Machine Learning |
# | **Dosen** | Nurfidah Dwitiyanti, M.Si. |
#
# **Catatan:** notebook ini saya jalankan di Google Colab / Jupyter Notebook. Notebook ini hanya berisi **Latihan Praktikum Modul Bab 3 bagian 3.10**, memakai **data sintetis** dari `sklearn.datasets.make_classification` (n=2000, 8 fitur, rasio kelas 0.95/0.05) dengan **simulasi missing value 5%**. Tidak perlu file CSV apa pun.

# Ringkasan Konsep
#
# - **Missing value.** Data yang hilang terjadi karena berbagai sebab. Penanganannya ada tiga jalur: hapus baris yang bermasalah, imputasi statistik sederhana (mean/median), atau imputasi berbasis model seperti **KNN Imputer** yang menebak nilai kosong dari k tetangga terdekat sehingga pola antar-fitur tetap terjaga.
# - **Outlier.** Data yang menyimpang jauh dari pola umum dan bisa merusak model. **Isolation Forest** mendeteksinya dengan cara "mengisolasi" titik-titik aneh lewat pohon acak; parameter `contamination` adalah tebakan kita soal proporsi outlier di data.
# - **Imbalanced data.** Kelas minoritas jauh lebih sedikit dari kelas mayoritas. Di sini **akurasi menipu**, karena model bisa dapat akurasi tinggi hanya dengan selalu menebak kelas mayoritas. Penanganannya: **SMOTE** (oversampling: membuat sampel sintetis kelas minoritas), undersampling acak, atau `class_weight="balanced"`.

# Sel 0 - Persiapan Awal (Khusus Google Colab)
#
# Kalau notebook ini dibuka di Google Colab, jalankan cell ini dulu supaya `imbalanced-learn` (paket SMOTE) tersedia. Kalau dibuka di Jupyter lokal yang paketnya sudah terpasang, cell ini tidak mengubah apa-apa.

# Latihan Praktikum (Modul Bab 3, bagian 3.10)
#
# Latihan 1 - Perbandingan Strategi Imputasi
#
# **Soal (modul 3.10):** bandingkan mean imputation vs KNN Imputer pada data sintetis di atas; latih LogisticRegression untuk tiap strategi; bandingkan skor F1.

# Setup Latihan: library, data sintetis, imputasi, dan model dasar.
# Cell ini membuat semua variabel yang dipakai Latihan 1 dan Latihan 2.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from imblearn.over_sampling import SMOTE

# Dataset sintetis: 2000 baris, 8 fitur, kelas tidak seimbang (0.95 / 0.05)
X, y = make_classification(n_samples=2000, n_features=8,
    weights=[0.95, 0.05], random_state=42)
df = pd.DataFrame(X, columns=[f"fitur_{i}" for i in range(8)])
df["target"] = y
fitur_cols = [c for c in df.columns if c.startswith("fitur")]

# Simulasi missing value 5% di seluruh kolom
rng = np.random.default_rng(42)
mask = rng.random(df.shape) < 0.05
df_missing = df.mask(pd.DataFrame(mask, columns=df.columns))

# Imputasi KNN untuk data latih/uji (baris yang targetnya hilang dibuang)
imputer = KNNImputer(n_neighbors=5)
df_imputed = df_missing.copy()
df_imputed[fitur_cols] = imputer.fit_transform(df_missing[fitur_cols])
df_imputed = df_imputed.dropna(subset=["target"])
X = df_imputed[fitur_cols]
y = df_imputed["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size=0.2, stratify=y, random_state=42)

# Model dasar: baseline tanpa penanganan imbalance + SMOTE (hanya di train set)
model_base = LogisticRegression(max_iter=1000).fit(X_train, y_train)
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
model_smote = LogisticRegression(max_iter=1000).fit(X_train_sm, y_train_sm)
print("Setup latihan selesai. Data latih:", X_train.shape, "| Data uji:", X_test.shape)

from sklearn.impute import SimpleImputer
print(f"{'strategi imputasi':>18} | {'akurasi':>8} | {'F1':>7}")
print("-" * 41)
for nama, imp in [("Mean Imputation", SimpleImputer(strategy="mean")),
                  ("KNN Imputer (k=5)", KNNImputer(n_neighbors=5))]:
    df_lat = df_missing.copy()
    df_lat[fitur_cols] = imp.fit_transform(df_missing[fitur_cols])
    df_lat = df_lat.dropna(subset=["target"])
    Xl, yl = df_lat[fitur_cols], df_lat["target"].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(Xl, yl, test_size=0.2,
        stratify=yl, random_state=42)
    m_lat = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    yp = m_lat.predict(Xte)
    print(f"{nama:>18} | {accuracy_score(yte, yp):>8.3f} | {f1_score(yte, yp):>7.3f}")

# Grafik Latihan 1: perbandingan F1 mean imputation vs KNN Imputer
hasil_imp = {}
for nama, imp in [("Mean Imputation", SimpleImputer(strategy="mean")),
                  ("KNN Imputer (k=5)", KNNImputer(n_neighbors=5))]:
    df_lat = df_missing.copy()
    df_lat[fitur_cols] = imp.fit_transform(df_missing[fitur_cols])
    df_lat = df_lat.dropna(subset=["target"])
    Xl, yl = df_lat[fitur_cols], df_lat["target"].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(Xl, yl, test_size=0.2,
        stratify=yl, random_state=42)
    m_lat = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    hasil_imp[nama] = f1_score(yte, m_lat.predict(Xte))
plt.figure(figsize=(5.5, 3.2))
plt.bar(hasil_imp.keys(), hasil_imp.values())
plt.ylabel("F1-score")
plt.title("Latihan 1: Mean Imputation vs KNN Imputer")
for i, v in enumerate(hasil_imp.values()):
    plt.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
plt.tight_layout()
plt.show()

# **Jawaban Latihan 1:** KNN Imputer (F1 0,308) sedikit mengungguli mean imputation (F1 0,296). Selisihnya kecil, tapi imputasi berbasis tetangga tetap lebih baik karena memanfaatkan pola antar-fitur, bukan sekadar rata-rata kolom.

# Latihan 2 (Praktikum 3.7) - Efektivitas Penanganan Imbalanced Data
#
# **Soal (modul 3.10):** terapkan `RandomUnderSampler` dan `LogisticRegression(class_weight="balanced")`; bandingkan F1-Score keempat pendekatan (baseline, SMOTE, undersampling, class_weight); tulis kesimpulan pendekatan mana yang paling efektif.

from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42)
X_train_rus,y_train_rus = rus.fit_resample(X_train,y_train)
model_rus = LogisticRegression(max_iter=1000).fit(X_train_rus, y_train_rus)
model_cw = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X_train,y_train)
# Bandingkan F1-Score keempat pendekatan: baseline, SMOTE, undersampling, class_weight
print(f"{'pendekatan':>14} | {'akurasi':>8} | {'F1':>7}")
print("-" * 37)
for nama, model in [("Baseline", model_base), ("SMOTE", model_smote),
                    ("Undersampling", model_rus), ("class_weight", model_cw)]:
    yp = model.predict(X_test)
    print(f"{nama:>14} | {accuracy_score(y_test, yp):>8.3f} | {f1_score(y_test, yp):>7.3f}")

# Grafik Latihan 2: perbandingan F1 keempat pendekatan imbalance
hasil_imb = {}
for nama, model in [("Baseline", model_base), ("SMOTE", model_smote),
                    ("Undersampling", model_rus), ("class_weight", model_cw)]:
    hasil_imb[nama] = f1_score(y_test, model.predict(X_test))
plt.figure(figsize=(6.5, 3.2))
bars = plt.bar(hasil_imb.keys(), hasil_imb.values())
plt.ylabel("F1-score")
plt.title("Latihan 2: efektivitas penanganan imbalanced data")
for b, v in zip(bars, hasil_imb.values()):
    plt.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.3f}",
             ha="center", fontsize=9)
plt.tight_layout()
plt.show()

# **Kesimpulan Latihan 2:** SMOTE paling efektif untuk kasus ini dengan F1 tertinggi 0,367, disusul class_weight (0,346), undersampling (0,336), dan baseline terakhir (0,308). SMOTE menang karena menambah informasi kelas minoritas lewat sampel sintetis tanpa membuang data mayoritas, sedangkan undersampling kehilangan banyak data latih sehingga performanya turun.

# Percobaan Mandiri
#
# Bagian ini berisi percobaan saya sendiri di luar modul, semuanya memakai data sintetis yang sama supaya hasilnya bisa dibandingkan langsung dengan praktikum di atas.
#
# Percobaan Mandiri 1 - Pengaruh `contamination` pada IsolationForest
#
# Parameter `contamination` adalah tebakan proporsi outlier. Saya coba tiga nilai, buang outlier yang terdeteksi, lalu latih ulang model baseline untuk melihat efeknya ke F1.

# Setup data Percobaan Mandiri (mandiri, tidak bergantung cell lain)
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE

X, y = make_classification(n_samples=2000, n_features=8,
    weights=[0.95, 0.05], random_state=42)
df = pd.DataFrame(X, columns=[f"fitur_{i}" for i in range(8)])
df["target"] = y
rng = np.random.default_rng(42)
mask = rng.random(df.shape) < 0.05
df_missing = df.mask(pd.DataFrame(mask, columns=df.columns))
fitur_cols = [c for c in df.columns if c.startswith("fitur")]
df_imputed = df_missing.copy()
df_imputed[fitur_cols] = KNNImputer(n_neighbors=5).fit_transform(df_missing[fitur_cols])
df_imputed = df_imputed.dropna(subset=["target"]).copy()
df_imputed["target"] = df_imputed["target"].astype(int)

print(f"{'contamination':>13} | {'outlier dibuang':>14} | {'F1 (setelah buang)':>18}")
print("-" * 51)
for c in [0.01, 0.02, 0.05]:
    iso_pm = IsolationForest(contamination=c, random_state=42)
    label_pm = iso_pm.fit_predict(df_imputed[fitur_cols])
    n_out = int((label_pm == -1).sum())
    df_pm = df_imputed[label_pm != -1]
    Xtr, Xte, ytr, yte = train_test_split(df_pm[fitur_cols], df_pm["target"],
        test_size=0.2, stratify=df_pm["target"], random_state=42)
    m_pm = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    print(f"{c:>13} | {n_out:>14} | {f1_score(yte, m_pm.predict(Xte)):>18.3f}")

# **Kesimpulan:** menaikkan contamination membuat semakin banyak baris dibuang (19, 38, 95) dan F1 justru turun dari 0,519 ke 0,200, jadi outlier tidak boleh dibuang terlalu agresif karena ikut membuang informasi penting.
#
# Percobaan Mandiri 2 - SMOTE dengan `k_neighbors` 5 vs 3
#
# SMOTE membuat sampel sintetis dari interpolasi `k_neighbors` tetangga terdekat. Saya bandingkan nilai default (5) dengan 3.

# Setup data Percobaan Mandiri (mandiri, tidak bergantung cell lain)
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE

X, y = make_classification(n_samples=2000, n_features=8,
    weights=[0.95, 0.05], random_state=42)
df = pd.DataFrame(X, columns=[f"fitur_{i}" for i in range(8)])
df["target"] = y
rng = np.random.default_rng(42)
mask = rng.random(df.shape) < 0.05
df_missing = df.mask(pd.DataFrame(mask, columns=df.columns))
fitur_cols = [c for c in df.columns if c.startswith("fitur")]
df_imputed = df_missing.copy()
df_imputed[fitur_cols] = KNNImputer(n_neighbors=5).fit_transform(df_missing[fitur_cols])
df_imputed = df_imputed.dropna(subset=["target"]).copy()
df_imputed["target"] = df_imputed["target"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    df_imputed[fitur_cols], df_imputed["target"],
    test_size=0.2, stratify=df_imputed["target"], random_state=42)
print(f"{'k_neighbors':>11} | {'F1 (SMOTE)':>11}")
print("-" * 27)
for k in [5, 3]:
    sm_pm = SMOTE(random_state=42, k_neighbors=k)
    Xsm_pm, ysm_pm = sm_pm.fit_resample(X_train, y_train)
    m_pm = LogisticRegression(max_iter=1000).fit(Xsm_pm, ysm_pm)
    print(f"{k:>11} | {f1_score(y_test, m_pm.predict(X_test)):>11.3f}")

# **Kesimpulan:** k_neighbors=5 memberi F1 sedikit lebih tinggi (0,367 vs 0,340), jadi sampel sintetis dari tetangga yang lebih banyak menghasilkan sebaran kelas minoritas yang lebih halus.
#
# Percobaan Mandiri 3 - Tanpa vs Dengan Buang Outlier
#
# Saya bandingkan model baseline (outlier hanya ditandai, tidak dibuang) dengan model yang dilatih setelah baris outlier dibuang.

# Setup data Percobaan Mandiri 3 (mandiri, tidak bergantung cell lain)
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

X, y = make_classification(n_samples=2000, n_features=8,
    weights=[0.95, 0.05], random_state=42)
df = pd.DataFrame(X, columns=[f"fitur_{i}" for i in range(8)])
df["target"] = y
rng = np.random.default_rng(42)
mask = rng.random(df.shape) < 0.05
df_missing = df.mask(pd.DataFrame(mask, columns=df.columns))
fitur_cols = [c for c in df.columns if c.startswith("fitur")]
df_imp = df_missing.copy()
df_imp[fitur_cols] = KNNImputer(n_neighbors=5).fit_transform(df_missing[fitur_cols])
# deteksi outlier DULU di 2000 baris (seperti Praktikum 3.3), baru buang target hilang
df_imp["is_outlier"] = IsolationForest(contamination=0.02, random_state=42).fit_predict(df_imp[fitur_cols])

# tanpa buang outlier: drop target hilang saja, latih baseline
df_base = df_imp.dropna(subset=["target"]).copy()
df_base["target"] = df_base["target"].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    df_base[fitur_cols], df_base["target"],
    test_size=0.2, stratify=df_base["target"], random_state=42)
model_base = LogisticRegression(max_iter=1000).fit(X_train, y_train)
f1_tanpa = f1_score(y_test, model_base.predict(X_test))
# dengan buang outlier: drop outlier lalu drop target hilang, latih ulang
df_no_out = df_imp[df_imp["is_outlier"] != -1].dropna(subset=["target"]).copy()
df_no_out["target"] = df_no_out["target"].astype(int)
Xtr, Xte, ytr, yte = train_test_split(df_no_out[fitur_cols], df_no_out["target"],
    test_size=0.2, stratify=df_no_out["target"], random_state=42)
m_no_out = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
f1_dengan = f1_score(yte, m_no_out.predict(Xte))
print(f"F1 tanpa buang outlier : {f1_tanpa:.3f}")
print(f"F1 dengan buang outlier: {f1_dengan:.3f}")

# **Kesimpulan:** membuang outlier sebelum training menaikkan F1 dari 0,308 ke 0,417, jadi data yang lebih bersih membantu model mengenali kelas minoritas, walau skornya tetap di bawah model SMOTE.

# Kesimpulan Bab 3
#
# 1. **Imputasi berbasis tetangga sedikit lebih baik dari rata-rata kolom.** Pada Latihan 1, KNN Imputer menghasilkan F1 0,308 sedangkan mean imputation 0,296, karena KNN memanfaatkan pola antar-fitur, bukan sekadar rata-rata.
# 2. **Akurasi menipu pada data tidak seimbang.** Model baseline Latihan 2 mencapai akurasi 0,953 tapi F1 hanya 0,308, artinya model hampir selalu menebak kelas mayoritas dan gagal mengenali kelas minoritas.
# 3. **SMOTE paling efektif menangani imbalance untuk kasus ini.** Urutan F1: SMOTE 0,367, class_weight 0,346, undersampling 0,336, baseline 0,308. SMOTE menang karena menambah sampel sintetis kelas minoritas tanpa membuang data mayoritas.
# 4. **Penanganan data kotor harus dievaluasi dengan metrik yang tepat.** Baik imputasi maupun penyeimbangan kelas, kesimpulannya sama: pada data tidak seimbang, F1-score jauh lebih jujur dibanding akurasi untuk menilai kualitas model.
