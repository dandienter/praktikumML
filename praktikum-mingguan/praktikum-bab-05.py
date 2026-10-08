# ============================================================
# Praktikum Bab 05: Decision Tree Pruning (mulai dari Implementasi Python)
# Diekstrak dari bab-05-decision-tree-pruning/praktikum-bab-05.ipynb
# ============================================================

# Praktikum Machine Learning: Bab 5
# Decision Tree: Algoritma CART/ID3 dan Teknik Pruning
#
# | | |
# |---|---|
# | **Nama** | Ahmad Dandi Subhani |
# | **NPM** | 202343500126 |
# | **Kelas** | R7B |
# | **Mata Kuliah** | Machine Learning |
# | **Dosen** | Nurfidah Dwitiyanti, M.Si. |
#
# **Catatan:** notebook ini saya jalanin di **Google Colab / Jupyter Notebook**. Dataset yang saya pake: Breast Cancer Wisconsin dari sklearn.datasets (569 pasien, 30 fitur), filenya `data/breast_cancer.csv`. Jadi nggak perlu unduh file eksternal.

# Ringkasan Konsep
#
# **Decision Tree (pohon keputusan)** itu model klasifikasi yang kerjanya ngebagi data secara **rekursif (recursive binary splitting)**: mulai dari *root node*, tiap *internal node* milih satu fitur sama satu nilai ambang (*threshold*) buat misahin data jadi dua cabang, terus prosesnya diulang di tiap cabang sampe kebentuk *leaf node* (daun) yang isinya label kelas prediksi. Jalur dari akar ke daun bisa dibaca sebagai aturan `if-then`, makanya model ini gampang banget **diinterpretasi**.
#
# **Kriteria pemilihan split.** Buat nentuin fitur dan threshold terbaik di tiap node, dipake ukuran *ketidakmurnian* (impurity) node:
# - **Gini impurity**: `Gini = 1 - jumlah(p_k^2)`. Makin kecil makin murni. Cepet dihitung dan jadi default di scikit-learn.
# - **Entropy / Information Gain**: `Entropy = -jumlah(p_k * log2(p_k))`, terus *Information Gain* = Entropy(induk) - Entropy(tertimbang anak). Split dengan Information Gain terbesar yang dipilih.
#
# Keduanya biasanya ngasih hasil mirip sih. Gini dikit lebih cepet, Entropy dikit lebih peka ke distribusi yang timpang.
#
# **CART vs ID3.** *CART (Classification and Regression Trees)* ngasilin pohon **biner** (tiap split selalu dua cabang) pake kriteria Gini/Entropy, ini yang dipake di `sklearn`. *ID3* pake Information Gain dan bisa bikin cabang sebanyak kategori fitur (multi-way split), cocok buat fitur kategorikal, tapi gampang bias ke fitur yang kategorinya banyak.
#
# **Overfitting pada tree.** Pohon yang nggak dibatesin bakal tumbuh sampe tiap daun murni. Hasilnya akurasi latih 100%, tapi akurasi uji malah turun karena modelnya ngapalin *noise*. Solusinya **pruning**:
# - **Pre-pruning** (dipotong sebelum tumbuh): `max_depth` (batas kedalaman), `min_samples_leaf` (minimal sampel di daun), `min_samples_split`.
# - **Post-pruning** (dipotong abis tumbuh penuh): **Cost Complexity Pruning** dengan parameter `ccp_alpha`. Cabang yang perbaikin impurity-nya lebih kecil dari `alpha` dipangkas. `alpha` yang optimal dipilih dari *pruning path*, bukan dari sekali tebak.

# Sel 0 - Persiapan Awal (Khusus Google Colab)
#
# Kalau notebook ini dibuka di Google Colab lewat link GitHub, jalankan sel di bawah ini dulu (atau langsung **Runtime > Run all**). Sel ini mengunduh otomatis file-file `data/` yang dibutuhkan dari repo GitHub, karena Colab hanya memuat file `.ipynb`-nya saja tanpa folder `data/`. Kalau dijalankan di laptop dan file datanya sudah ada, sel ini langsung dilewati.

# 5.12 Latihan Praktikum (Modul Bab 5)
#
# Dua latihan di bawah ini **saya kerjain** langsung (soal + kode + jawaban).

# Persiapan Data
#
# Sebelum latihan, saya siapkan dulu dataset dan semua paket yang dipakai. Dataset Breast Cancer (569 pasien, 30 fitur) saya ambil langsung dari sklearn, terus saya bagi 80% latih dan 20% uji dengan `stratify` supaya proporsi kelasnya seimbang.

# Persiapan data untuk Latihan Praktikum Bab 5
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("Ukuran data latih:", X_train.shape)
print("Ukuran data uji  :", X_test.shape)

# Latihan 1 - Percobaan mandiri Parameter Pre-Pruning (Modul Bab 5, bagian 5.12)
#
# **Soal:** Lakuin grid search manual atas kombinasi `max_depth` di {2, 4, 6, None} x `min_samples_leaf` di {1, 5, 10}. Cetak akurasi test tiap kombinasi dalam tabel rapi, terus simpulin kombinasi terbaiknya.

print(f"{'max_depth':>10s} | {'min_samples_leaf':>16s} | {'Akurasi Test':>12s}")
print("-" * 46)

hasil_grid = []
for md in [2, 4, 6, None]:
    for msl in [1, 5, 10]:
        m = DecisionTreeClassifier(max_depth=md, min_samples_leaf=msl, random_state=42)
        m.fit(X_train, y_train)
        ak = accuracy_score(y_test, m.predict(X_test))
        hasil_grid.append((ak, md, msl))
        print(f"{str(md):>10s} | {msl:16d} | {ak:12.4f}")

terbaik = max(hasil_grid, key=lambda t: t[0])
print(f"\nKombinasi terbaik: max_depth={terbaik[1]}, min_samples_leaf={terbaik[2]} "
      f"-> akurasi test {terbaik[0]:.4f}")

# Grafik Latihan 1: heatmap akurasi test (max_depth x min_samples_leaf)
import pandas as pd
mat = pd.DataFrame(index=["2", "4", "6", "None"], columns=["1", "5", "10"], dtype=float)
for ak, md, msl in hasil_grid:
    mat.loc[str(md), str(msl)] = ak
plt.figure(figsize=(5.5, 3.8))
im = plt.imshow(mat.values, cmap="YlGn")
plt.xticks(range(3), ["1", "5", "10"])
plt.yticks(range(4), ["2", "4", "6", "None"])
plt.xlabel("min_samples_leaf")
plt.ylabel("max_depth")
plt.title("Latihan 1: akurasi test per kombinasi pre-pruning")
for r in range(4):
    for c_ in range(3):
        plt.text(c_, r, f"{mat.values[r, c_]:.3f}", ha="center", va="center", fontsize=9)
plt.colorbar(im, label="Akurasi test")
plt.tight_layout()
plt.show()

# **Penjelasan grafiknya:** heatmap memperjelas pola yang sama: kolom `min_samples_leaf=10` paling terang (akurasi tertinggi) di semua baris `max_depth`. Kombinasi terbaik 0,9474 muncul tiga kali, dan versi paling sederhana (`max_depth=4`, `min_samples_leaf=10`) jadi pilihan utama.

# **Penjelasan outputnya:**
# 1. Dari 12 kombinasi, akurasi test tertinggi itu **0.9474**, dicapai **tiga** kombinasi: `(max_depth=4, min_samples_leaf=10)`, `(6, 10)`, sama `(None, 10)`.
# 2. Polanya konsisten: `min_samples_leaf=10` **selalu** lebih bagus daripada 1 atau 5 di tiap nilai `max_depth`. Ngebatasin daun minimum itu pre-pruning yang efektif di dataset ini.
# 3. `max_depth=None` nggak selalu jelek **asal** `min_samples_leaf`-nya cukup gede (10): pohon dibiarin tumbuh tapi tiap daun harus didukung minimal 10 sampel, jadi split noise tetep dicegah.
# 4. **Kesimpulan:** kombinasi terbaik yang paling sederhana itu **`max_depth=4, min_samples_leaf=10`** (akurasi test 0.9474). Dipilih yang paling dangkal di antara yang seri, karena pohon lebih kecil = lebih gampang diinterpretasi (prinsip *Occam's razor*).

# Latihan 2 - Cost Complexity Pruning Path (Modul Bab 5, bagian 5.12)
#
# **Soal:** Buat tiap `alpha` di `path.ccp_alphas`, latih decision tree dan catat akurasi test-nya. Plot `ccp_alpha` vs akurasi test, tentuin alpha optimal, terus bandingin jumlah leaf node sebelum dan sesudah pruning.

# Hitung pruning path dulu dari data latih
path = DecisionTreeClassifier(random_state=42).cost_complexity_pruning_path(X_train, y_train)
alphas = path.ccp_alphas
ak_test_ccp, n_daun = [], []

for a in alphas:
    m = DecisionTreeClassifier(ccp_alpha=a, random_state=42)
    m.fit(X_train, y_train)
    ak_test_ccp.append(accuracy_score(y_test, m.predict(X_test)))
    n_daun.append(m.get_n_leaves())

print(f"{'ccp_alpha':>10s} | {'Akurasi Test':>12s} | {'Jumlah Daun':>11s}")
print("-" * 41)
for a, ak, nd in zip(alphas, ak_test_ccp, n_daun):
    print(f"{a:10.6f} | {ak:12.4f} | {nd:11d}")

idx_opt = int(np.argmax(ak_test_ccp))   # alpha dengan akurasi test tertinggi
alpha_opt = alphas[idx_opt]
print(f"\nAlpha optimal: {alpha_opt:.6f} -> akurasi test {ak_test_ccp[idx_opt]:.4f}, "
      f"jumlah daun {n_daun[idx_opt]}")
print(f"Sebelum pruning (alpha=0): {n_daun[0]} daun -> sesudah (alpha optimal): "
      f"{n_daun[idx_opt]} daun")

plt.figure(figsize=(8, 5))
plt.plot(alphas, ak_test_ccp, marker="o", label="Akurasi Test")
plt.axvline(alpha_opt, color="red", linestyle="--",
            label=f"Alpha optimal = {alpha_opt:.4f}")
plt.xlabel("ccp_alpha")
plt.ylabel("Akurasi Test")
plt.title("Cost Complexity Pruning: ccp_alpha vs Akurasi Test")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# **Penjelasan outputnya:**
# 1. Di `alpha=0` (tanpa pruning): 19 daun, akurasi test 0.9123. Begitu alpha dinaikin dikit, jumlah daun turun bertahap (19 ke 18 ke 15 ke 12, dan seterusnya). Cabang yang lemah dipangkas satu per satu.
# 2. **Alpha optimal = 0.002866** dengan akurasi test **0.9386** dan **12 daun**. Pruning nyusutin pohon sekitar 37% (19 jadi 12 daun) *sambil naikin* akurasi.
# 3. Kurvanya nunjukin pola naik dulu terus turun: akurasi naik dulu (noise kebuang), terus **anjlok ke 0.6316** di alpha maksimum (0.3266) pas pohon tinggal 1 daun. Pruning berlebihan = underfitting.
# 4. Perbandingannya: sebelum pruning 19 daun (akurasi 0.9123) vs sesudah pruning optimal 12 daun (akurasi 0.9386). Post-pruning terbukti ngasih model yang **lebih kecil DAN lebih akurat**.

# Percobaan Mandiri
#
# Bagian ini isinya percobaan mandiri: saya ubah-ubah parameter terus ngamatin dampaknya ke performa sama bentuk pohon.

# Percobaan Mandiri 1 - Pengaruh `max_depth` 1 sampai 10 (kurva validasi manual)
#
# Bagian ini saya ngerjain Percobaan mandiri 1 dari Modul Machine Learning Bab 5. Saya latih 10 pohon dengan `max_depth` beda-beda, terus saya plot akurasi train vs test. Tujuannya: nemuin **titik di mana model mulai overfitting** (akurasi train terus naik, akurasi test berhenti naik atau malah turun).

# Percobaan Mandiri 1: pengaruh max_depth 1-10 (kurva validasi manual)
# Setup data dibuat ulang di cell ini supaya mandiri.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data_bc = load_breast_cancer(as_frame=True)
X_bc, y_bc = data_bc.data, data_bc.target
X_train, X_test, y_train, y_test = train_test_split(
    X_bc, y_bc, test_size=0.2, stratify=y_bc, random_state=42
)

depths = list(range(1, 11))
ak_train, ak_test = [], []

for d in depths:
    m = DecisionTreeClassifier(max_depth=d, random_state=42)
    m.fit(X_train, y_train)
    ak_train.append(accuracy_score(y_train, m.predict(X_train)))
    ak_test.append(accuracy_score(y_test, m.predict(X_test)))
    print(f"max_depth={d:2d} | akurasi train = {ak_train[-1]:.4f} | "
          f"akurasi test = {ak_test[-1]:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(depths, ak_train, marker="o", label="Akurasi Train")
plt.plot(depths, ak_test, marker="s", label="Akurasi Test")
plt.xlabel("max_depth")
plt.ylabel("Akurasi")
plt.title("Kurva Validasi Manual: Pengaruh max_depth terhadap Akurasi")
plt.xticks(depths)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# **Penjelasan outputnya:**
# 1. Akurasi **train naik terus** dari 0.9231 (depth=1) sampai **1.0000** di depth 7 ke atas. Pohon makin dalam makin jago ngapalin data latih.
# 2. Akurasi **test mentok di 0.9386 pada depth 3-4**, terus **turun ke 0.9123** dan stagnan di depth 6 ke atas. Kurvanya naik dulu terus turun.
# 3. **Titik overfitting** kelihatan jelas di sekitar **depth 5-6**: setelah itu kurva train sama test **melebar** (gap makin gede). Model makin kompleks tapi kemampuan generalisasinya nggak nambah.
# 4. Hasil ini cocok sama grid di Latihan 1: kombinasi terbaik ada di `max_depth=4` (akurasi test 0,9474), pas di puncak kurva validasi.

# Percobaan Mandiri 2 - `min_samples_leaf=1` vs `20` pada pohon tanpa batas
#
# Bagian ini saya ngerjain Percobaan mandiri 2 dari Modul Machine Learning Bab 5. `min_samples_leaf` itu pre-pruning yang ngelarang daun isinya terlalu dikit sampel. Saya bandingin pohon *full* (default `min_samples_leaf=1`) lawan `min_samples_leaf=20`.

# Percobaan Mandiri 2: min_samples_leaf=1 vs 20 pada pohon tanpa batas
# Setup data dibuat ulang di cell ini supaya mandiri.
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data_bc = load_breast_cancer(as_frame=True)
X_bc, y_bc = data_bc.data, data_bc.target
X_train, X_test, y_train, y_test = train_test_split(
    X_bc, y_bc, test_size=0.2, stratify=y_bc, random_state=42
)

print(f"{'min_samples_leaf':>16s} | {'Akurasi Train':>13s} | "
      f"{'Akurasi Test':>12s} | {'Gap':>6s} | {'Daun':>4s} | {'Kedalaman':>9s}")
print("-" * 80)
for msl in [1, 20]:
    m = DecisionTreeClassifier(min_samples_leaf=msl, random_state=42)
    m.fit(X_train, y_train)
    ak_tr = accuracy_score(y_train, m.predict(X_train))
    ak_te = accuracy_score(y_test, m.predict(X_test))
    print(f"{msl:16d} | {ak_tr:13.4f} | {ak_te:12.4f} | {ak_tr-ak_te:6.4f} | "
          f"{m.get_n_leaves():4d} | {m.get_depth():9d}")

# **Penjelasan outputnya:**
# 1. `min_samples_leaf=1`: **19 daun**, kedalaman 7, akurasi train **1.0000** vs test **0.9123** (gap 0.0877). Pohon penuh yang overfitting.
# 2. `min_samples_leaf=20`: daunnya nyusut drastis jadi **7**, kedalaman 4, train **0.9473** vs test **0.9035** (gap cuma 0.0437).
# 3. Ngebesarin `min_samples_leaf` maksa tiap daun didukung minimal 20 sampel. Split yang cuma nangkep segelintir *outlier* dilarang, jadinya pohon lebih sederhana dan gap-nya nyempit.
# 4. Yang menarik, di dataset ini `min_samples_leaf=20` malah nurunin dikit akurasi test (0.9035 < 0.9123): pruning yang **terlalu agresif** malah bikin *underfitting*. Nilai tengah (misal 5-10) lebih seimbang, sesuai hasil grid di Latihan 1.

# Percobaan Mandiri 3 - Decision Tree vs Random Forest kecil (`n_estimators=50`)
#
# Bagian ini saya ngerjain Percobaan mandiri 3 dari Modul Machine Learning Bab 5. Random Forest itu *ensemble* dari banyak decision tree yang dilatih di *bootstrap sample* acak. Kira-kira ensemble selalu menang nggak ya lawan satu pohon terbaik di dataset ini?

# Percobaan Mandiri 3: Decision Tree (max_depth=4) vs Random Forest kecil (n_estimators=50)
# Setup data dan model dibuat ulang di cell ini supaya mandiri.
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data_bc = load_breast_cancer(as_frame=True)
X_bc, y_bc = data_bc.data, data_bc.target
X_train, X_test, y_train, y_test = train_test_split(
    X_bc, y_bc, test_size=0.2, stratify=y_bc, random_state=42
)

pohon = DecisionTreeClassifier(max_depth=4, random_state=42)
pohon.fit(X_train, y_train)

rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(X_train, y_train)

ak_tr_tree = accuracy_score(y_train, pohon.predict(X_train))
ak_te_tree = accuracy_score(y_test, pohon.predict(X_test))
ak_tr_rf = accuracy_score(y_train, rf.predict(X_train))
ak_te_rf = accuracy_score(y_test, rf.predict(X_test))

print(f"{'Model':35s} | {'Akurasi Train':>13s} | {'Akurasi Test':>12s}")
print("-" * 70)
print(f"{'Decision Tree (max_depth=4)':35s} | {ak_tr_tree:13.4f} | "
      f"{ak_te_tree:12.4f}")
print(f"{'Random Forest (50 pohon)':35s} | {ak_tr_rf:13.4f} | "
      f"{ak_te_rf:12.4f}")

# **Penjelasan outputnya:**
# 1. Random Forest (50 pohon): akurasi test **0.9561**, ngalahin decision tree tunggal (0.9386). Selisihnya **sekitar 1,8 poin**.
# 2. Akurasi train RF nyampe **1.0000** tapi gap-nya (0.0439) tetep kecil. *Voting* antar 50 pohon yang beragam ngeredam variansi tanpa ngorbanin generalisasi. Ini inti cara kerja ensemble.
# 3. Tapi ada harga yang dibayar: modelnya berubah dari **1 pohon yang bisa dibaca manusia** jadi **50 pohon black-box**. Interpretabilitasnya ilang.
# 4. Kesimpulannya: ensemble **cenderung menang di akurasi** di dataset ini, tapi decision tree tetep unggul kalo **aturan yang bisa dijelasin** (misal buat dokter) lebih penting daripada selisih beberapa persen akurasi.

# Kesimpulan Bab 5
#
# 1. **Pre-pruning lewat grid manual efektif.** Dari 12 kombinasi `max_depth` x `min_samples_leaf`, akurasi test tertinggi 0,9474 dicapai tiga kombinasi: (4, 10), (6, 10), dan (None, 10). Pola yang paling konsisten: `min_samples_leaf=10` selalu menang di tiap nilai `max_depth`, jadi membatasi ukuran daun minimum terbukti mencegah split noise. Versi paling sederhana, `max_depth=4` dan `min_samples_leaf=10`, saya pilih sebagai kombinasi terbaik.
# 2. **Post-pruning (cost complexity pruning) menyusutkan pohon sekaligus menaikkan akurasi.** Tanpa pruning (alpha=0) pohon punya 19 daun dengan akurasi test 0,9123. Alpha optimal 0,002866 memangkasnya jadi 12 daun (susut sekitar 37%) dengan akurasi test naik ke 0,9386. Kurva alpha vs akurasi menunjukkan pola naik dulu lalu turun, dan pruning berlebihan (alpha maksimum 0,3266, pohon tinggal 1 daun) menjatuhkan akurasi ke 0,6316 alias underfitting.
# 3. **Pemilihan parameter pruning jangan asal tebak.** Grid manual untuk pre-pruning dan `cost_complexity_pruning_path` untuk post-pruning memberi cara sistematis memilih `max_depth`, `min_samples_leaf`, dan `ccp_alpha` yang optimal berdasarkan akurasi test, bukan tebakan.
# 4. **Pelajaran praktisnya:** kalau pohon overfitting, dua tuas yang bisa saya pakai adalah membatasi pertumbuhan pohon dari awal (pre-pruning) atau memangkas cabang yang lemah setelah pohon jadi (post-pruning). Keduanya saya buktikan sendiri di latihan ini dengan angka yang terukur.
