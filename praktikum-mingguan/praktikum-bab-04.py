# ============================================================
# Praktikum Bab 04: Feature Engineering Pipeline (mulai dari Implementasi Python)
# Diekstrak dari bab-04-feature-engineering-pipeline/praktikum-bab-04.ipynb
# ============================================================

# Praktikum 1 (4.3) - Membandingkan Tiga Pendekatan Feature Selection (Modul Bab 4)
#
# Dataset **Breast Cancer** (569 sampel, 30 fitur numerik. Semuanya non-negatif jadi skor chi-square valid). Tiga metode milih 10 fitur terbaik (embedded pake L1-logistic yang menyeleksi secara otomatis):
#
# - **Filter:** `SelectKBest(chi2, k=10)`: skor statistik murni, tanpa model.
# - **Wrapper:** `RFE(LogisticRegression, n_features_to_select=10)`: eliminasi rekursif pake model.
# - **Embedded:** `SelectFromModel(LogisticRegression(l1_ratio=1, C=0.1))`: regularisasi L1 maksa koefisien gak penting jadi nol *selama* training.

# Import tambahan untuk latihan feature selection
from sklearn.feature_selection import SelectKBest, chi2, RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression

# Dataset Breast Cancer dibaca dari file CSV lokal (data/breast_cancer.csv)
df_bc = pd.read_csv('data/breast_cancer.csv')
fitur_bc = [c for c in df_bc.columns if c not in ('target', 'target_name')]
X_bc, y_bc = df_bc[fitur_bc].values, df_bc['target'].values
nama_fitur = np.array(fitur_bc)

# 1) Filter
skb = SelectKBest(chi2, k=10).fit(X_bc, y_bc)
f_filter = set(np.argsort(skb.scores_)[-10:])

# 2) Wrapper
rfe = RFE(LogisticRegression(max_iter=5000), n_features_to_select=10).fit(X_bc, y_bc)
f_wrapper = set(rfe.get_support(indices=True))

# 3) Embedded (L1)
l1 = SelectFromModel(
    LogisticRegression(l1_ratio=1, solver='saga', C=0.1, max_iter=10000,
                       random_state=42)
).fit(X_bc, y_bc)
f_embedded = set(l1.get_support(indices=True))

semua = sorted(f_filter | f_wrapper | f_embedded)
tabel = pd.DataFrame({
    'fitur': [nama_fitur[i] for i in semua],
    'filter (chi2)': ['ya' if i in f_filter else '-' for i in semua],
    'wrapper (RFE)': ['ya' if i in f_wrapper else '-' for i in semua],
    'embedded (L1)': ['ya' if i in f_embedded else '-' for i in semua],
})
print(tabel.to_string(index=False))
print(f"\nJumlah fitur terpilih: filter={len(f_filter)}, "
      f"wrapper={len(f_wrapper)}, embedded={len(f_embedded)}")
print(f"Irisan ketiga metode (dipilih semua): "
      f"{[nama_fitur[i] for i in sorted(f_filter & f_wrapper & f_embedded)]}")

# Grafik Praktikum 1: skor chi2 untuk 10 fitur teratas (metode filter)
skor = skb.scores_
top10 = np.argsort(skor)[-10:][::-1]
plt.figure(figsize=(7, 3.6))
plt.barh([nama_fitur[i] for i in top10], skor[top10])
plt.xlabel("Skor chi2")
plt.title("Praktikum 1: 10 fitur dengan skor chi2 tertinggi (filter)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# **Penjelasan grafiknya:** metode filter menilai tiap fitur secara individual lewat skor chi2. Fitur-fitur `worst ...` (nilai terburuk tiap pengukuran sel) mendominasi peringkat teratas, masuk akal karena sel kanker ganas cenderung punya nilai ekstrem. Kelemahan filter yang terlihat di sini: fitur terpilih saling berkorelasi (banyak varian pengukuran yang sama), sedangkan wrapper/embedded memilih kombinasi yang saling melengkapi di dalam model.

# **Penjelasan outputnya:**
#
# 1. Ketiga metode **gak milih himpunan yang identik**. Dari 18 fitur yang dipilih minimal satu metode, cuma **2** yang disepakati ketiganya: `mean radius` dan `worst radius`. Ini wajar: filter menilai tiap fitur *sendiri-sendiri* jadi cenderung milih fitur-fitur yang saling berkorelasi (banyak varian "worst ..." dan "mean ..."), sedangkan wrapper dan embedded mempertimbangkan **kontribusi fitur dalam konteks model**. RFE malah banyak milih fitur "*... error*" dan "*worst ...*" yang saling melengkapi di dalam model.
# 2. **Filter (chi2)** tercepat: cuma ngitung statistik sekali. **Wrapper (RFE)** paling lambat: ngelatih LogisticRegression berulang kali sambil buang fitur terlemah. **Embedded (L1)** cuma butuh satu kali training dan otomatis nentuin jumlah fitur (di sini 9 fitur, bukan genap 10).
# 3. **Kesimpulan:** gak ada metode yang "selalu bener". Filter cocok buat penyaringan awal yang cepet di data berdimensi besar; wrapper buat akurasi maksimal kalo komputasi bukan masalah; embedded sebagai jalan tengah yang elegan: seleksi terjadi sebagai efek samping training. Di praktiknya ketiganya sering **dikombinasikan** (filter dulu buat buang yang jelas-jelas noise, terus wrapper/embedded).

# Praktikum 2 - Membangun Pipeline Lengkap (Dataset Titanic) (Modul Bab 4)
#
# Buat latihan pipeline saya pake juga dataset Titanic dari seaborn (891 penumpang), filenya `data/titanic.csv`. Ini contoh klasik data campuran: numerik (`age`, `fare`, `sibsp`, `parch`) + kategorikal (`sex`, `embarked`, `class`) dengan target klasifikasi `survived`. Pipeline: `ColumnTransformer` (numerik jadi median + scaler; kategorikal jadi most_frequent + one-hot) + `LogisticRegression`, dievaluasi pake **akurasi** di data uji.

# Import tambahan untuk latihan pipeline Titanic
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Dataset Titanic dibaca dari file CSV lokal (data/titanic.csv) - bisa dipakai offline
df_t = pd.read_csv('data/titanic.csv')
print("Dataset Titanic berhasil dimuat:", df_t.shape)
fitur_num_t = ['age', 'fare', 'sibsp', 'parch']
fitur_cat_t = ['sex', 'embarked', 'class']
X_t = df_t[fitur_num_t + fitur_cat_t]
y_t = df_t['survived']

print("\nMissing value per kolom:", X_t.isna().sum().to_dict())

Xtr, Xte, ytr, yte = train_test_split(X_t, y_t, test_size=0.2, random_state=42)

pipe_titanic = Pipeline(steps=[
    ('preprocessing', ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')),
                          ('scaler', StandardScaler())]), fitur_num_t),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                          ('onehot', OneHotEncoder(
                              handle_unknown='ignore'))]), fitur_cat_t),
    ])),
    ('model', LogisticRegression(max_iter=1000)),
])
pipe_titanic.fit(Xtr, ytr)
akurasi = accuracy_score(yte, pipe_titanic.predict(Xte))
print(f"\nAkurasi pada data uji: {akurasi:.4f}")
print("Prediksi 1 penumpang baru:", pipe_titanic.predict(Xte.iloc[[0]])[0],
      "(aktual:", yte.iloc[0], ")")

# Grafik Praktikum 2: confusion matrix pipeline Titanic di data uji
from sklearn.metrics import ConfusionMatrixDisplay
plt.figure(figsize=(4.2, 3.6))
ConfusionMatrixDisplay.from_estimator(pipe_titanic, Xte, yte,
                                      display_labels=["Tidak selamat", "Selamat"])
plt.title("Praktikum 2: Confusion matrix (pipeline Titanic)")
plt.tight_layout()
plt.show()

# **Penjelasan grafiknya:** confusion matrix menunjukkan di mana model paling sering salah. Pola umumnya: model cukup baik mengenali penumpang yang tidak selamat, tapi sebagian penumpang yang selamat diprediksi tidak selamat. Ini wajar untuk model linear sederhana tanpa feature engineering lanjutan, dan alasannya akurasi 0,80 bukan 1,00.

# **Penjelasan outputnya:**
#
# 1. Dataset Titanic berhasil dimuat (891 baris, 15 kolom) dari file lokal `data/titanic.csv`.
# 2. Missing value kedeteksi di `age` (sekitar 177) dan `embarked` (2). Keduanya ditangani otomatis di dalam pipeline (median buat numerik, modus buat kategorikal), tanpa satu baris pun kode pembersihan manual di luar pipeline.
# 3. Akurasi **0,7989** di data uji: wajar buat model linear sederhana tanpa *feature engineering* lanjutan, dan konsisten sama benchmark umum dataset Titanic.
# 4. Prediksi satu baris data mentah langsung jalan tanpa preprocessing manual. Sekali lagi nunjukin keunggulan pipeline: data baru cukup dilempar ke `pipe.predict()`, preprocessing yang tepat (termasuk penanganan NaN dan kategori gak dikenal via `handle_unknown='ignore'`) udah "terkunci" di dalamnya. Catatan jujur: prediksi contoh baris itu 0 padahal aktualnya 1. Pengingat kalo akurasi 80% artinya sekitar 1 dari 5 prediksi meleset.

# Percobaan Mandiri
#
# Tiga percobaan mandiri untuk menguji pemahaman di dataset sintetis properti: (A) apa risiko preprocessing manual di luar pipeline, (B) apakah feature construction `rasio_kamar_per_luas` benar-benar membantu, dan (C) bagaimana trade-off jumlah pohon vs waktu latih.

# Percobaan Mandiri A - Pipeline vs Preprocessing Manual
#
# Skenario deployment: model sudah dilatih, lalu datang **1 baris data properti baru** (fitur mentah). Saya bandingkan tiga cara memprediksinya:
#
# 1. `pipeline.predict(data_baru)`: cara pipeline (satu baris kode),
# 2. preprocessing **manual** yang direplikasi dengan benar (hasilnya harus identik),
# 3. preprocessing **manual** yang salah: lupa membuat fitur `rasio_kamar_per_luas`.

# Percobaan Mandiri A: pipeline vs preprocessing manual (1 data properti baru)
# Semua setup dibuat ulang di cell ini supaya mandiri.
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

rng = np.random.default_rng(42)
n = 500
df = pd.DataFrame({
    "luas_bangunan": rng.normal(120, 40, n).clip(30, None),
    "jumlah_kamar": rng.integers(1, 6, n),
    "tipe_properti": rng.choice(["Rumah", "Apartemen", "Ruko"], n),
    "kondisi_bangunan": rng.choice(["Buruk", "Sedang", "Baik", "Sangat Baik"], n),
})
df["harga"] = (df["luas_bangunan"] * 15 + df["jumlah_kamar"] * 20 + rng.normal(0, 50, n)) * 1_000_000
df["rasio_kamar_per_luas"] = df["jumlah_kamar"] / df["luas_bangunan"]

fitur_numerik = ["luas_bangunan", "jumlah_kamar", "rasio_kamar_per_luas"]
urutan_kondisi = [["Buruk", "Sedang", "Baik", "Sangat Baik"]]
preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]), fitur_numerik),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), ["tipe_properti"]),
    ("ord", OrdinalEncoder(categories=urutan_kondisi), ["kondisi_bangunan"]),
])

X = df.drop(columns=["harga"])
y = df["harga"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", RandomForestRegressor(n_estimators=200, random_state=42)),
])
pipeline.fit(X_train, y_train)

baru = pd.DataFrame({
    "luas_bangunan": [150],
    "jumlah_kamar": [4],
    "tipe_properti": ["Rumah"],
    "kondisi_bangunan": ["Baik"],
})
baru["rasio_kamar_per_luas"] = baru["jumlah_kamar"] / baru["luas_bangunan"]

# Cara 1: langsung lewat pipeline
p1 = pipeline.predict(baru)[0]
print(f"1) pipeline.predict             : Rp{p1:,.0f}")

# Cara 2: preprocessing manual yang direplikasi dengan benar
prep = pipeline.named_steps["preprocessing"]
model_rf = pipeline.named_steps["model"]
p2 = model_rf.predict(prep.transform(baru))[0]
print(f"2) manual (replikasi benar)      : Rp{p2:,.0f}")

# Cara 3: preprocessing manual yang salah (lupa fitur konstruksi)
baru_salah = baru.drop(columns=["rasio_kamar_per_luas"])
try:
    pipeline.predict(baru_salah)
except Exception as e:
    print(f"3) manual (lupa rasio_kamar_per_luas): {type(e).__name__}")

# **Penjelasan outputnya:**
#
# 1. Cara 1 dan cara 2 hasilnya **identik** (Rp2.340.104.768), membuktikan pipeline hanya membungkus langkah yang sama dalam satu objek.
# 2. Cara 3 langsung **error** (`ValueError`) karena preprocessor mengharapkan kolom `rasio_kamar_per_luas`. Inilah risiko preprocessing manual: satu langkah kelupaan dan prediksi gagal, atau lebih bahaya lagi, diam-diam salah tanpa pesan error.
# 3. Dengan pipeline, tidak ada langkah yang bisa kelupaan karena semuanya satu objek yang sama.

# Percobaan Mandiri B - Dengan vs Tanpa Feature Construction
#
# Apakah fitur hasil konstruksi (`rasio_kamar_per_luas`) benar-benar membantu? Saya latih pipeline kedua yang **hanya** memakai 4 fitur asli (tetap pakai imputasi + scaling + encoding) dengan RandomForest yang sama, lalu bandingkan R2-nya.

# Percobaan Mandiri B: dengan vs tanpa feature construction
# Dataset dan kedua pipeline dibuat ulang di cell ini supaya mandiri.
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

rng = np.random.default_rng(42)
n = 500
df = pd.DataFrame({
    "luas_bangunan": rng.normal(120, 40, n).clip(30, None),
    "jumlah_kamar": rng.integers(1, 6, n),
    "tipe_properti": rng.choice(["Rumah", "Apartemen", "Ruko"], n),
    "kondisi_bangunan": rng.choice(["Buruk", "Sedang", "Baik", "Sangat Baik"], n),
})
df["harga"] = (df["luas_bangunan"] * 15 + df["jumlah_kamar"] * 20 + rng.normal(0, 50, n)) * 1_000_000
df["rasio_kamar_per_luas"] = df["jumlah_kamar"] / df["luas_bangunan"]
urutan_kondisi = [["Buruk", "Sedang", "Baik", "Sangat Baik"]]
y = df["harga"]

# Pipeline dengan feature construction
pre_dengan = ColumnTransformer(transformers=[
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]), ["luas_bangunan", "jumlah_kamar", "rasio_kamar_per_luas"]),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), ["tipe_properti"]),
    ("ord", OrdinalEncoder(categories=urutan_kondisi), ["kondisi_bangunan"]),
])
pipe_dengan = Pipeline([("preprocessing", pre_dengan),
                        ("model", RandomForestRegressor(n_estimators=200, random_state=42))])

# Pipeline tanpa feature construction (hanya 4 fitur asli)
pre_tanpa = ColumnTransformer(transformers=[
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]), ["luas_bangunan", "jumlah_kamar"]),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), ["tipe_properti"]),
    ("ord", OrdinalEncoder(categories=urutan_kondisi), ["kondisi_bangunan"]),
])
pipe_tanpa = Pipeline([("preprocessing", pre_tanpa),
                       ("model", RandomForestRegressor(n_estimators=200, random_state=42))])

X_dengan = df.drop(columns=["harga"])
X_tanpa = df.drop(columns=["harga", "rasio_kamar_per_luas"])
Xtr, Xte, ytr, yte = train_test_split(X_dengan, y, test_size=0.2, random_state=42)
_, Xte_t, _, yte_t = train_test_split(X_tanpa, y, test_size=0.2, random_state=42)

pipe_dengan.fit(Xtr, ytr)
pipe_tanpa.fit(Xtr.drop(columns=["rasio_kamar_per_luas"]), ytr)
print(f"R2 tanpa feature construction : {r2_score(yte_t, pipe_tanpa.predict(Xte_t)):.3f}")
print(f"R2 dengan feature construction: {r2_score(yte, pipe_dengan.predict(Xte)):.3f}")

# **Penjelasan outputnya:**
#
# 1. R2 **sama-sama 0,987** dengan maupun tanpa `rasio_kamar_per_luas`. Fitur konstruksi ternyata tidak menambah apa-apa di sini.
# 2. Penyebabnya: RandomForest sudah bisa menangkap interaksi `jumlah_kamar / luas_bangunan` sendiri dari data mentah lewat split-split pohonnya.
# 3. Pelajarannya: feature construction tidak selalu membantu. Selalu verifikasi pakai metrik, jangan asal tambah fitur.

# Percobaan Mandiri C - n_estimators = 50 vs 200
#
# Berapa harga yang dibayar untuk 200 pohon dibanding 50 pohon? Saya latih pipeline lengkap dengan kedua nilai itu lalu bandingkan **R2** dan **waktu latih**.

# Percobaan Mandiri C: n_estimators 50 vs 200 (trade-off waktu vs performa)
# Dataset dan preprocessor dibuat ulang di cell ini supaya mandiri.
import time
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

rng = np.random.default_rng(42)
n = 500
df = pd.DataFrame({
    "luas_bangunan": rng.normal(120, 40, n).clip(30, None),
    "jumlah_kamar": rng.integers(1, 6, n),
    "tipe_properti": rng.choice(["Rumah", "Apartemen", "Ruko"], n),
    "kondisi_bangunan": rng.choice(["Buruk", "Sedang", "Baik", "Sangat Baik"], n),
})
df["harga"] = (df["luas_bangunan"] * 15 + df["jumlah_kamar"] * 20 + rng.normal(0, 50, n)) * 1_000_000
df["rasio_kamar_per_luas"] = df["jumlah_kamar"] / df["luas_bangunan"]

preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]), ["luas_bangunan", "jumlah_kamar", "rasio_kamar_per_luas"]),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), ["tipe_properti"]),
    ("ord", OrdinalEncoder(categories=[["Buruk", "Sedang", "Baik", "Sangat Baik"]]), ["kondisi_bangunan"]),
])

X = df.drop(columns=["harga"])
y = df["harga"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

for n_est in [50, 200]:
    p = Pipeline([("preprocessing", preprocessor),
                  ("model", RandomForestRegressor(n_estimators=n_est, random_state=42))])
    t0 = time.time()
    p.fit(X_train, y_train)
    dt = time.time() - t0
    skor = r2_score(y_test, p.predict(X_test))
    print(f"n_estimators={n_est:3d} -> R2 {skor:.3f}, waktu latih {dt:.1f} detik")

# **Penjelasan outputnya:**
#
# 1. 50 pohon: R2 **0,987** dalam **0,5 detik**. 200 pohon: R2 **tetap 0,987** tapi waktunya **3 detik**, enam kali lipat.
# 2. Di dataset kecil ini 200 pohon tidak lebih baik sama sekali, hanya jauh lebih lambat.
# 3. Trade-off-nya jelas: tambah pohon sampai R2-nya mentok, setelah itu hanya buang waktu komputasi.

# Kesimpulan Bab 4
#
# 1. **Feature selection (Praktikum 1):** tiga pendekatan milih himpunan fitur yang **beda-beda** di dataset Breast Cancer. Dari 18 fitur yang dipilih minimal satu metode, cuma 2 yang disepakati ketiganya (`mean radius` dan `worst radius`). Filter (chi2) tercepat karena cuma ngitung statistik; wrapper (RFE) paling mahal karena ngelatih model berulang kali; embedded (L1) jalan tengah yang elegan karena seleksi terjadi sebagai efek samping training. Gak ada metode yang selalu terbaik, ketiganya sering dikombinasikan.
# 2. **Pipeline (Praktikum 2):** `ColumnTransformer` + `Pipeline` nangani data campuran Titanic (numerik + kategorikal) dalam satu objek dan mencapai akurasi **0,7989** di data uji. Missing value di `age` dan `embarked` ditangani otomatis di dalam pipeline, tanpa kode pembersihan manual.
# 3. **Keuntungan terbesar pipeline** keliatan pas deployment: `pipe.predict(data_mentah)` selalu nerapin preprocessing yang identik sama training, termasuk kategori gak dikenal (`handle_unknown='ignore'`). Jadi gak ada risiko preprocessing pas training beda sama pas prediksi.
# 4. **Confusion matrix** nunjukin model paling sering salah di penumpang yang selamat tapi diprediksi tidak selamat. Wajar buat model linear sederhana tanpa feature engineering lanjutan, dan pengingat kalo akurasi 80% artinya sekitar 1 dari 5 prediksi meleset.
