# ============================================================
# Praktikum Bab 09: Ensemble Methods (mulai dari Implementasi Python)
# Diekstrak dari bab-09-ensemble-methods/praktikum-bab-09.ipynb
# ============================================================

# Praktikum 1 - Pengaruh n_estimators
#
# **Tujuan:** mengamati pengaruh jumlah pohon (`n_estimators`) terhadap performa dan OOB score Random Forest.
#
# **Soal (Praktikum 9.6 modul):** latih Random Forest dengan `n_estimators` = 10, 50, 100, 300, 500 di atas dataset Breast Cancer, aktifkan `oob_score=True`, lalu catat OOB score dan akurasi test untuk tiap nilai.

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

for n in [10, 50, 100, 300, 500]:
    model = RandomForestClassifier(n_estimators=n,
        oob_score=True,random_state=42)
    model.fit(X_train, y_train)
    print(f"n_estimators={n}: OOB={model.oob_score_:.3f}, Test Acc={accuracy_score(y_test, model.predict(X_test)):.3f}")

import matplotlib.pyplot as plt

n_list = [10, 50, 100, 300, 500]
oob_list, acc_list = [], []
for n in n_list:
    m = RandomForestClassifier(n_estimators=n, oob_score=True, random_state=42)
    m.fit(X_train, y_train)
    oob_list.append(m.oob_score_)
    acc_list.append(accuracy_score(y_test, m.predict(X_test)))

plt.figure(figsize=(7, 4))
plt.plot(n_list, oob_list, marker="o", label="OOB score")
plt.plot(n_list, acc_list, marker="s", label="Akurasi test")
plt.xscale("log")
plt.xlabel("n_estimators (skala log)")
plt.ylabel("Skor")
plt.title("Pengaruh n_estimators terhadap OOB score dan akurasi test")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

# **Penjelasan outputnya:**
# - `n_estimators=10`: OOB=0.941, Test Acc=0.956. Di nilai ini muncul warning karena 10 pohon terlalu sedikit untuk estimasi OOB yang reliabel, jadi skor OOB-nya kurang bisa dipercaya.
# - `n_estimators=50`: OOB=0.943, Test Acc=0.965. Akurasi test naik dan warning hilang.
# - `n_estimators=100`: OOB=0.956, Test Acc=0.965.
# - `n_estimators=300`: OOB=0.965, Test Acc=0.965. Ini titik terbaik: OOB tertinggi.
# - `n_estimators=500`: OOB=0.963, Test Acc=0.965. OOB malah sedikit turun, akurasi test mentok.
# - Kesimpulannya: tambah pohon memang menaikkan skor, tapi setelah sekitar 100-300 pohon hasilnya jenuh. Buat dataset ini `n_estimators=100` sudah cukup karena akurasi testnya sama dengan n=500 (0.965) tapi waktu latihnya jauh lebih singkat.

# Praktikum 2 - Perbandingan Empat Model
#
# **Tujuan:** membandingkan Decision Tree, Random Forest, AdaBoost, dan Gradient Boosting secara menyeluruh menggunakan Stratified k-Fold CV.
#
# **Instruksi (modul):** gunakan `StratifiedKFold(n_splits=5)` dan `cross_val_score` dengan `scoring='f1_macro'` untuk keempat model. Sajikan hasilnya dalam tabel dan tuliskan kesimpulan model mana yang terbaik untuk dataset ini.

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
hasil = {}
print(f"{'Model':<18} {'F1-macro rata-rata':<18} {'Std'}")
print("-" * 48)
for nama, model in models.items():
    skor = cross_val_score(model, X, y, cv=skf, scoring="f1_macro")
    hasil[nama] = (skor.mean(), skor.std())
    print(f"{nama:<18} {skor.mean():.4f}             +/- {skor.std():.4f}")

nama_model = list(hasil.keys())
mean_skor = [hasil[k][0] for k in nama_model]
std_skor = [hasil[k][1] for k in nama_model]

plt.figure(figsize=(8, 4))
bars = plt.bar(nama_model, mean_skor, yerr=std_skor, capsize=6, color=["#7aa7d9", "#5cbf6a", "#e8a13c", "#c96a6a"])
plt.ylabel("F1-macro rata-rata (5-fold CV)")
plt.title("Perbandingan empat model dengan Stratified 5-Fold CV")
plt.ylim(min(mean_skor) - 0.05, 1.0)
for b, v in zip(bars, mean_skor):
    plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.004, f"{v:.4f}", ha="center", fontsize=9)
plt.tight_layout()
plt.show()

# **Penjelasan outputnya:**
# - Decision Tree: F1-macro 0.9028 +/- 0.0317. Paling rendah dan paling tidak stabil (std terbesar), wajar karena satu pohon gampang overfitting ke tiap fold.
# - Random Forest: F1-macro 0.9529 +/- 0.0135. Tertinggi sekaligus paling stabil (std terkecil).
# - AdaBoost: F1-macro 0.9487 +/- 0.0241. Hampir menyamai Random Forest.
# - Gradient Boosting: F1-macro 0.9446 +/- 0.0266. Bagus, tapi sedikit di bawah dua ensemble lainnya di dataset ini.
# - Kesimpulan: model terbaik untuk dataset Breast Cancer ini adalah **Random Forest**, karena F1-macro rata-ratanya tertinggi (0.9529) dengan simpangan terkecil (0.0135). Sesuai teori, ensemble mengurangi variansi dibanding satu Decision Tree tunggal.

# Percobaan Mandiri
#
# Tiga percobaan tambahan di luar Latihan Praktikum: ketahanan ensemble terhadap label noise, pengaruh learning rate pada Gradient Boosting, dan perbandingan hard voting vs soft voting.

# Percobaan Mandiri 1 - Decision Tree vs Ensemble di Data Noisy

# [jupyter magic, dinonaktifkan] %matplotlib inline
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score

# 15 persen label sengaja dibalik untuk mensimulasikan data yang noisy
X, y = make_classification(n_samples=500, n_features=10, n_informative=5,
                           flip_y=0.15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Bagging": BaggingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
}

hasil_train, hasil_test = {}, {}
for nama, model in models.items():
    model.fit(X_train, y_train)
    acc_train = accuracy_score(y_train, model.predict(X_train))
    acc_test = accuracy_score(y_test, model.predict(X_test))
    hasil_train[nama] = acc_train
    hasil_test[nama] = acc_test
    print(f"{nama}: train={acc_train:.3f}, test={acc_test:.3f}")

nama = list(models.keys())
plt.figure(figsize=(8, 4.5))
plt.bar([i - 0.2 for i in range(len(nama))], [hasil_train[n] for n in nama],
        width=0.4, label="Train")
plt.bar([i + 0.2 for i in range(len(nama))], [hasil_test[n] for n in nama],
        width=0.4, label="Test")
plt.xticks(range(len(nama)), nama)
plt.ylabel("Akurasi")
plt.title("Decision Tree vs Ensemble pada Data Berlabel Noise")
plt.legend()
plt.ylim(0.6, 1.05)
plt.tight_layout()
plt.show()

# **Penjelasan outputnya:**
# - Decision Tree tunggal: train 1.000, test 0.713. Gap-nya besar sekali, dia hafal data training termasuk label yang salah (overfitting ke noise).
# - Bagging: train 0.983, test 0.773. AdaBoost: train 0.869, test 0.787. Keduanya punya akurasi test lebih tinggi dengan gap train-test yang lebih kecil.
# - Kesimpulan: model ensemble lebih tahan terhadap label noise dibanding satu decision tree, karena rata-rata suara banyak model meredam pengaruh noise.

# Percobaan Mandiri 2 - Pengaruh Learning Rate di Gradient Boosting

# [jupyter magic, dinonaktifkan] %matplotlib inline
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

hasil = []
for lr in [0.01, 0.1, 1.0]:
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=lr,
                                       random_state=42)
    model.fit(X_train, y_train)
    acc_train = accuracy_score(y_train, model.predict(X_train))
    acc_test = accuracy_score(y_test, model.predict(X_test))
    hasil.append((lr, acc_train, acc_test))
    print(f"learning_rate={lr}: train={acc_train:.3f}, test={acc_test:.3f}")

print()
print(f"{'learning_rate':<15} {'train':<8} {'test'}")
print("-" * 30)
for lr, acc_train, acc_test in hasil:
    print(f"{lr:<15} {acc_train:<8.3f} {acc_test:.3f}")

labels = [str(h[0]) for h in hasil]
plt.figure(figsize=(7, 4.5))
plt.plot(labels, [h[1] for h in hasil], marker="o", label="Train")
plt.plot(labels, [h[2] for h in hasil], marker="s", label="Test")
plt.xlabel("learning_rate")
plt.ylabel("Akurasi")
plt.title("Pengaruh learning_rate pada Gradient Boosting (n_estimators=100)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()

# **Penjelasan outputnya:**
# - learning_rate=0.01: train 0.987, test 0.956. Model belajar paling lambat, akurasi train-nya paling rendah: tanda underfit kalau jumlah pohonnya tidak ditambah.
# - learning_rate=0.1: train 1.000, test 0.956.
# - learning_rate=1.0: train 1.000, test 0.965. Tiap pohon melompat paling jauh, akurasi test-nya paling tinggi di sini tapi juga paling berisiko overfit kalau datanya lebih sulit.
# - Kesimpulan: di dataset ini perbedaannya tidak jauh, tapi polanya sesuai teori. learning rate terlalu kecil bikin model butuh lebih banyak pohon, terlalu besar bikin model tidak stabil. Nilai 0.1 adalah titik tengah yang aman.

# Percobaan Mandiri 3 - Hard Voting vs Soft Voting

# [jupyter magic, dinonaktifkan] %matplotlib inline
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

estimators = [
    ("lr", Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=2000))])),
    ("dt", DecisionTreeClassifier(random_state=42)),
    ("knn", Pipeline([("scaler", StandardScaler()),
                      ("clf", KNeighborsClassifier())])),
]

hasil = {}
for voting in ["hard", "soft"]:
    model = VotingClassifier(estimators, voting=voting)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    hasil[voting] = acc
    print(f"Voting {voting}: akurasi test = {acc:.3f}")

plt.figure(figsize=(6.5, 4.5))
plt.bar(hasil.keys(), hasil.values())
plt.title("Hard Voting vs Soft Voting (Logistic Regression + Decision Tree + KNN)")
plt.ylabel("Akurasi test")
plt.ylim(min(hasil.values()) - 0.05, 1.0)
for k, v in hasil.items():
    plt.text(k, v + 0.003, f"{v:.3f}", ha="center")
plt.tight_layout()
plt.show()

# **Penjelasan outputnya:**
# - Hard voting: akurasi test 0.974. Soft voting: akurasi test 0.974, hasilnya sama persis.
# - Artinya ketiga model dasar (Logistic Regression, Decision Tree, KNN) kebanyakan sudah sepakat dalam klasifikasinya, jadi cara menggabungkannya tidak mengubah hasil.
# - Kesimpulan: soft voting tidak selalu menang dari hard voting. Soft voting baru membantu saat model anggotanya beda pendapat dan nilai probabilitasnya bisa dipercaya.

# Kesimpulan Bab 9
#
# - Menambah `n_estimators` pada Random Forest menaikkan OOB score dan akurasi test sampai titik jenuh, setelah itu tambahannya tidak signifikan tapi waktu latih terus bertambah, jadi cukup pilih nilai yang sudah stabil.
# - OOB score terbukti jadi perkiraan performa yang praktis karena dihitung gratis dari data out-of-bag tanpa perlu split validasi tambahan.
# - Dari perbandingan 5-fold CV, model ensemble (Random Forest, AdaBoost, Gradient Boosting) mengungguli satu Decision Tree tunggal dalam F1-macro, sesuai teori bahwa gabungan banyak model mengurangi variansi.
# - Untuk dataset Breast Cancer ini, model terbaik adalah yang punya F1-macro rata-rata tertinggi dengan std terkecil, karena artinya performanya bagus sekaligus stabil di semua fold.
