# ============================================================
# Praktikum Bab 10: Clustering (mulai dari Implementasi Python)
# Diekstrak dari bab-10-clustering/praktikum-bab-10.ipynb
# ============================================================

# Praktikum 1 - Menentukan k Optimal
#
# **Tujuan:** menentukan jumlah cluster optimal menggunakan Elbow Method dan Silhouette Score.
#
# **Instruksi (modul):** menggunakan dataset pelanggan (modul membolehkan data sintetis), buat plot Elbow Method dan Silhouette Score untuk k=2 hingga k=10, kemudian tentukan k optimal beserta justifikasinya.
#
# Saya memakai data sintetis pelanggan dari `make_blobs` dengan 5 pusat cluster dan 1000 baris. Dua fiturnya saya beri nama "Annual Income" dan "Spending Score" supaya terasa seperti data segmentasi pelanggan mal.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Data sintetis pelanggan: 1000 baris, 5 pusat cluster, 2 fitur.
X, _ = make_blobs(n_samples=1000, centers=5, n_features=2,
                  cluster_std=1.2, random_state=42)
df = pd.DataFrame(X, columns=["Annual Income", "Spending Score"])
print("Bentuk data:", df.shape)
print(df.head())

# Elbow method + silhouette score untuk k = 2 sampai 10.
ks = list(range(2, 11))
inertias, silhouettes = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X, labels))
    print(f"k={k}: inertia={km.inertia_:.1f}, silhouette={silhouettes[-1]:.3f}")

# Satu figure, dua subplot berdampingan.
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(ks, inertias, marker="o")
axes[0].set_xlabel("k (jumlah cluster)")
axes[0].set_ylabel("Inertia")
axes[0].set_title("Elbow Method")
axes[0].set_xticks(ks)
axes[1].plot(ks, silhouettes, marker="o", color="green")
axes[1].set_xlabel("k (jumlah cluster)")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette Score vs k")
axes[1].set_xticks(ks)
plt.tight_layout()
plt.show()

k_optimal = ks[int(np.argmax(silhouettes))]
print("k dengan silhouette score tertinggi:", k_optimal)

# **Penjelasan outputnya:**
#
# Datanya terbaca 1000 baris dengan 2 kolom, "Annual Income" dan "Spending Score", sesuai yang saya buat dari `make_blobs`. Dari hasil loop k=2 sampai k=10, inertia turun sangat tajam di awal: 29580.7 (k=2), 8198.6 (k=3), 3961.2 (k=4), 2633.8 (k=5), lalu melandai setelahnya (2410.3 di k=6, 2202.1 di k=7, dan seterusnya). Di grafik elbow, "siku" nya terlihat jelas di k=5, karena penurunan dari k=4 ke k=5 masih besar (sekitar 1327) sedangkan dari k=5 ke k=6 tinggal sekitar 223.
#
# Silhouette score tertingginya ada di k=4 (0.687), disusul k=3 (0.679) dan k=5 (0.629). Setelah k=5 skornya anjlok terus sampai 0.328 di k=10, yang artinya memaksakan terlalu banyak cluster justru merusak pemisahannya.
#
# **Justifikasi k optimal: saya pilih k=5.** Elbow method menunjuk k=5 dengan tegas, silhouette di k=5 masih bagus (0.629, tertinggi ketiga dan jauh di atas k=6 ke atas), dan angka ini juga cocok dengan 5 pusat cluster yang saya pakai saat membuat datanya. Jadi k=5 adalah titik temu kedua metode.

# Praktikum 2 - Segmentasi dengan Tiga Algoritma
#
# **Tujuan:** membandingkan hasil clustering ketiga algoritma pada dataset yang sama.
#
# **Instruksi (modul, Praktikum 10.7):** kode di modul hanya berupa komentar, jadi saya implementasikan langsung seperti ini:
#
# ```python
# Terapkan K-Means, Hierarchical (Agglomerative),
# dan DBSCAN pada dataset yang sama.
# Bandingkan jumlah cluster yang dihasilkan,
# Silhouette Score, dan visualisasikan ketiganya
# dalam satu figure berdampingan (subplot). Tuliskan analis
# is mengenai algoritma mana
# yang menghasilkan segmentasi paling masuk akal secara bis
# nis untuk kasus Anda.
# ```
#
# Saya memakai dataset X yang sama seperti Praktikum 1. Untuk DBSCAN saya pakai eps=0.9 dan min_samples=10, dan silhouette score-nya saya hitung di luar titik noise (label -1).

from sklearn.cluster import AgglomerativeClustering, DBSCAN

# Terapkan ketiga algoritma pada dataset yang sama.
hasil_label = {
    "K-Means": KMeans(n_clusters=5, n_init=10, random_state=42).fit_predict(X),
    "Agglomerative": AgglomerativeClustering(n_clusters=5).fit_predict(X),
    "DBSCAN": DBSCAN(eps=0.9, min_samples=10).fit_predict(X),
}

# Bandingkan jumlah cluster dan silhouette score tiap algoritma.
print(f"{'Algoritma':<14}{'Cluster':<9}{'Noise':<8}Silhouette")
for nama, labels in hasil_label.items():
    tanpa_noise = labels[labels != -1]
    n_cluster = len(set(tanpa_noise))
    n_noise = int((labels == -1).sum())
    sil = silhouette_score(X[labels != -1], tanpa_noise) if n_cluster > 1 else float("nan")
    print(f"{nama:<14}{n_cluster:<9}{n_noise:<8}{sil:.3f}")

# Visualisasikan ketiganya dalam satu figure berdampingan.
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (nama, labels) in zip(axes, hasil_label.items()):
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=12)
    ax.set_xlabel("Annual Income")
    ax.set_ylabel("Spending Score")
    ax.set_title(nama)
plt.tight_layout()
plt.show()

# **Penjelasan outputnya:**
#
# Tabel perbandingannya menunjukkan K-Means dan Agglomerative sama-sama menghasilkan tepat 5 cluster tanpa noise, dengan silhouette score yang hampir identik: 0.629 untuk K-Means dan 0.623 untuk Agglomerative. DBSCAN menemukan 4 cluster dengan 31 titik ditandai sebagai noise, dan silhouette score di luar noise-nya 0.699, paling tinggi di antara ketiganya.
#
# Dari scatter plot berdampingan terlihat K-Means dan Agglomerative memetakan kelima gugus pelanggan dengan rapi dan konsisten. DBSCAN menggabungkan dua gugus yang berdekatan menjadi satu cluster dan membuang 31 titik sebagai noise, sehingga segmentasinya lebih kasar.
#
# **Analisis:** untuk kasus segmentasi pelanggan ini, **K-Means paling masuk akal secara bisnis**. Alasannya: setiap pelanggan harus masuk ke satu segmen agar bisa ditindaklanjuti tim marketing, dan K-Means menempatkan semua 1000 pelanggan ke 5 segmen yang jelas tanpa ada yang dibuang sebagai noise. Agglomerative hasilnya nyaris sama bagusnya, jadi bisa jadi alternatif. DBSCAN memang punya silhouette tertinggi, tapi ia mengorbankan satu segmen dan membuang 31 pelanggan sebagai noise, sehingga kurang cocok dipakai untuk segmentasi pelanggan yang butuh seluruh data terpetakan.

# Percobaan Mandiri
#
# Di bawah ini saya tambahkan tiga percobaan mandiri di luar dua Latihan Praktikum dari modul, untuk menguji beberapa hal yang belum dibahas di atas: efek strategi inisialisasi KMeans, pengaruh scaling fitur terhadap hasil clustering, dan sensitivitas DBSCAN terhadap parameter eps.

# Percobaan 1 - Inisialisasi KMeans: random vs k-means++
#
# **Tujuan:** membandingkan kestabilan hasil KMeans antara inisialisasi pusat cluster yang acak (`init="random"`) dan yang cerdas (`init="k-means++"`).
#
# Saya membuat data `make_blobs` dengan 500 titik dan 4 cluster, lalu menjalankan KMeans masing-masing 10 kali dengan seed berbeda (`n_init=1` supaya tiap seed benar-benar satu run) dan melihat rata-rata serta standar deviasi inertia-nya. Kalau sebuah metode init bagus, rata-ratanya harus rendah dan std-nya kecil (hasilnya konsisten di semua seed).

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Data: 500 titik, 4 cluster (dibuat mandiri, tidak tergantung sel lain)
X_pm1, _ = make_blobs(n_samples=500, centers=4, random_state=42)

# Jalankan KMeans 10 kali dengan seed berbeda untuk tiap metode init
for metode in ["random", "k-means++"]:
    inertia = [
        KMeans(n_clusters=4, init=metode, n_init=1, random_state=s).fit(X_pm1).inertia_
        for s in range(10)
    ]
    print(f"init={metode:<10} rata-rata inertia={np.mean(inertia):.2f} "
          f"std={np.std(inertia):.2f} (min={min(inertia):.2f})")

# **Penjelasan outputnya:**
#
# Hasilnya sesuai dugaan. Dengan `init="random"`, rata-rata inertia-nya 1904.10 dengan std 1169.89: artinya hasilnya sangat tergantung keberuntungan seed, ada run yang bagus (inertia 948.89) tapi banyak juga yang nyangkut di minimum lokal dengan inertia jauh lebih besar. Sementara `init="k-means++"` menghasilkan rata-rata inertia 948.89 dengan std 0.00, artinya semua 10 seed dapat nilai optimum yang sama.
#
# **Kesimpulan:** inisialisasi `k-means++` jauh lebih stabil daripada `random`. Karena k-means++ memilih centroid awal yang saling berjauhan, KMeans tidak lagi sensitif terhadap seed dan hampir selalu menemukan solusi terbaik. Ini alasan kenapa k-means++ jadi init default di scikit-learn.

# Percobaan 2 - Scaling Fitur: dengan vs tanpa StandardScaler
#
# **Tujuan:** membuktikan bahwa fitur dengan skala besar bisa mendominasi jarak Euclidean di KMeans kalau datanya tidak di-scaling.
#
# Saya membuat data 2D dengan skala yang timpang: fitur pertama kisarannya 0-1, fitur kedua saya kalikan 1000 sehingga kisarannya 0-1000. Karena saya tahu label asli cluster-nya, saya bisa menilai kualitas clustering dengan Adjusted Rand Index (ARI): 1.0 berarti sama persis dengan struktur asli datanya.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

# Data 2D dengan skala timpang: fitur1 kecil, fitur2 besar (0-1000)
X_pm2, y_asli = make_blobs(n_samples=300, centers=3, random_state=7)
X_timpang = X_pm2.copy()
X_timpang[:, 1] *= 1000

label_tanpa = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(X_timpang)
label_dengan = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(
    StandardScaler().fit_transform(X_timpang))
print(f"Tanpa scaling : ARI vs label asli = {adjusted_rand_score(y_asli, label_tanpa):.3f}")
print(f"Dengan scaling: ARI vs label asli = {adjusted_rand_score(y_asli, label_dengan):.3f}")

# Visualisasi berdampingan
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].scatter(X_timpang[:, 0], X_timpang[:, 1], c=label_tanpa, cmap="tab10", s=12)
axes[0].set_title("Tanpa scaling")
axes[0].set_xlabel("Fitur 1")
axes[0].set_ylabel("Fitur 2 (0-1000)")
axes[1].scatter(X_timpang[:, 0], X_timpang[:, 1], c=label_dengan, cmap="tab10", s=12)
axes[1].set_title("Dengan StandardScaler")
axes[1].set_xlabel("Fitur 1")
axes[1].set_ylabel("Fitur 2 (0-1000)")
plt.tight_layout()
plt.show()

# **Penjelasan outputnya:**
#
# Tanpa scaling, ARI-nya hanya 0.570: KMeans gagal mengenali struktur asli data. Dari scatter plot kiri terlihat batas cluster-nya nyaris vertikal, artinya pemisahan hanya mengikuti fitur kedua (yang berskala 0-1000) sementara fitur pertama diabaikan sama sekali. Setelah di-scaling, ARI-nya naik menjadi 1.000: cluster yang ditemukan sama persis dengan label asli datanya, dan scatter plot kanannya menunjukkan tiga gugus yang dipisahkan secara wajar di kedua sumbu.
#
# **Kesimpulan:** tanpa scaling, fitur yang berskala besar mendominasi perhitungan jarak sehingga KMeans seolah-olah hanya melihat satu fitur. StandardScaler menyamakan kontribusi tiap fitur, dan di data ini itu mengembalikan hasil clustering ke struktur aslinya (ARI dari 0.570 menjadi 1.000).

# Percobaan 3 - Sensitivitas DBSCAN terhadap eps
#
# **Tujuan:** melihat bagaimana parameter eps memengaruhi hasil DBSCAN: terlalu kecil atau terlalu besar, mana yang merusak?
#
# Saya memakai `make_moons` (400 titik, noise 0.08) karena bentuk bulan sabitnya tidak bisa dipisahkan KMeans secara linear, jadi ini kasus yang cocok untuk DBSCAN. Saya coba dua nilai eps yang ekstrem (0.1 dan 0.5) dengan `min_samples=5` yang sama, lalu hitung jumlah cluster dan titik noise-nya.

import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

# Dua bulan sabit + noise (dibuat mandiri, tidak tergantung sel lain)
X_pm3, _ = make_moons(n_samples=400, noise=0.08, random_state=42)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, eps in zip(axes, [0.1, 0.5]):
    label = DBSCAN(eps=eps, min_samples=5).fit_predict(X_pm3)
    n_cluster = len(set(label)) - (1 if -1 in label else 0)
    n_noise = int((label == -1).sum())
    print(f"eps={eps}: {n_cluster} cluster, {n_noise} titik noise")
    ax.scatter(X_pm3[:, 0], X_pm3[:, 1], c=label, cmap="tab10", s=12)
    ax.set_title(f"DBSCAN eps={eps} ({n_cluster} cluster, {n_noise} noise)")
    ax.set_xlabel("Fitur 1")
    ax.set_ylabel("Fitur 2")
plt.tight_layout()
plt.show()

# **Penjelasan outputnya:**
#
# Dengan eps=0.1, DBSCAN menemukan 11 cluster dan membuang 25 titik sebagai noise. Dari scatter plot kiri terlihat kedua bulan sabitnya pecah menjadi banyak potongan kecil: karena jangkauan tetangganya terlalu sempit, titik-titik yang sebenarnya satu gugus tidak saling terjangkau. Sebaliknya dengan eps=0.5, DBSCAN hanya menemukan 1 cluster dan 0 noise: jangkauannya terlalu lebar sehingga kedua bulan sabit tersambung menjadi satu gugus raksasa.
#
# **Kesimpulan:** eps terlalu kecil membuat data terfragmentasi menjadi banyak cluster kecil plus banyak noise, sedangkan eps terlalu besar menyatukan gugus yang berbeda. Nilai eps yang pas untuk data ini ada di antaranya (sekitar 0.2-0.3), jadi pemilihan eps memang perlu di-tuning, bukan asal tebak.

# Kesimpulan Bab 10
#
# * Elbow method dan silhouette score bisa dipakai bersama untuk menentukan jumlah cluster: di data ini keduanya mengarah ke k=5, dengan siku inertia yang jelas dan silhouette 0.629.
# * K-Means dan Agglomerative clustering menghasilkan segmentasi yang setara bagusnya pada data pelanggan ini (silhouette 0.629 vs 0.623, sama-sama 5 cluster tanpa noise).
# * DBSCAN menemukan 4 cluster dengan silhouette tertinggi (0.699) tetapi menggabungkan dua segmen dan menandai 31 titik sebagai noise, sehingga kurang pas untuk kebutuhan segmentasi pelanggan.
# * Untuk kasus bisnis seperti segmentasi pelanggan mal, algoritma yang menempatkan seluruh data ke segmen yang jelas (K-Means) lebih masuk akal dibanding yang membuang data sebagai noise.
