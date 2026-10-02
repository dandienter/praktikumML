# Glosarium Praktikum Machine Learning

Kumpulan istilah dan tools yang dipakai di seluruh bab (Bab 1-13). Ditulis ringkas supaya
gampang dicari saat baca notebook atau PDF.

## Tools yang Dipakai

- **Python 3.12.3**: bahasa pemrograman yang dipakai di semua notebook.
- **Jupyter Notebook / Google Colab**: tempat menulis dan menjalankan kode Python per cell secara interaktif.
- **numpy 2.5.3**: library angka untuk array dan operasi matriks.
- **pandas 3.0.6**: library untuk mengolah data tabular (DataFrame).
- **matplotlib 3.11.2**: library untuk membuat grafik.
- **seaborn 0.13.2**: library grafik statistik yang dibangun di atas matplotlib.
- **scikit-learn 1.9.1**: library machine learning utama (model, preprocessing, metrik evaluasi).
- **imbalanced-learn 0.14.2**: library khusus untuk menangani data yang tidak seimbang (misal SMOTE).
- **TensorFlow / Keras**: framework deep learning untuk membangun neural network dan CNN.
- **WeasyPrint 70.0**: tool untuk mengubah HTML menjadi PDF (dipakai membangun PDF tiap bab).

## Bab 1 - Konsep Dasar Machine Learning

- **Machine Learning**: cabang ilmu komputer di mana program belajar dari pengalaman (data) tanpa diprogram eksplisit untuk tiap aturan.
- **Supervised learning**: pembelajaran dengan data berlabel, contohnya klasifikasi (dataset Iris).
- **Unsupervised learning**: pembelajaran tanpa label, contohnya clustering.
- **Klasifikasi**: tugas menebak kategori/label dari suatu data.
- **Regresi**: tugas menebak nilai angka kontinu.
- **Clustering**: mengelompokkan data berdasarkan kemiripan tanpa label.
- **Train/test split**: membagi data menjadi data latih dan data uji supaya performa model diukur secara jujur.
- **Overfitting**: model terlalu menghafal data latih sehingga jelek di data baru.
- **Underfitting**: model terlalu sederhana sehingga gagal menangkap pola data.

## Bab 2 - Workflow CRISP-DM dan Bias-Variance

- **CRISP-DM**: kerangka kerja proyek data mining dengan 6 fase: Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, Deployment.
- **Bias-variance tradeoff**: tarik-menarik antara bias tinggi (model terlalu sederhana) dan variance tinggi (model terlalu sensitif terhadap noise data). Model ideal seimbang di tengah.
- **Learning curve**: grafik skor train vs validasi terhadap ukuran data atau kompleksitas model, dipakai untuk mendiagnosis underfitting/overfitting.
- **Stratified sampling**: pembagian data yang menjaga proporsi tiap kelas di setiap bagian, penting untuk dataset yang tidak seimbang.
- **Churn**: pelanggan yang berhenti memakai layanan (studi kasus dataset Telco).

## Bab 3 - Data Preparation

- **Missing value**: data yang hilang/kosong. Ditangani dengan hapus baris, imputasi statistik sederhana (mean/median), atau imputasi berbasis model (KNN Imputer).
- **Outlier**: data yang menyimpang jauh dari pola umum dan bisa merusak model. Dideteksi misalnya dengan Isolation Forest.
- **Imbalanced data**: kelas minoritas jauh lebih sedikit dari kelas mayoritas, bikin model bias menebak kelas mayoritas.
- **Imputasi**: teknik mengisi nilai yang hilang. **Mean imputation** mengisi dengan rata-rata kolom, **KNN Imputer** mengisi berdasarkan tetangga terdekat.
- **SMOTE**: oversampling yang membuat sampel sintetis kelas minoritas supaya jumlahnya seimbang.
- **Undersampling**: mengurangi sampel kelas mayoritas secara acak.
- **class_weight="balanced"**: opsi di scikit-learn yang memberi bobot lebih besar ke kelas minoritas tanpa mengubah data.

## Bab 4 - Feature Engineering dan Pipeline

- **Feature construction**: membuat fitur baru dari fitur mentah agar pola lebih mudah dilihat model (misal rasio, binning).
- **Feature selection**: memilih subset fitur yang paling relevan. Ada tiga pendekatan: **filter** (skor statistik tanpa model, misal chi2), **wrapper** (evaluasi dengan model, misal RFE), **embedded** (seleksi saat training, misal Lasso/L1).
- **Pipeline**: menggabungkan preprocessing dan model jadi satu objek supaya alurnya konsisten dari training sampai deployment.
- **ColumnTransformer**: menerapkan transformasi berbeda ke kelompok kolom berbeda dalam satu langkah.
- **Data leakage**: kebocoran info data uji ke proses training (misal imputasi sebelum split) yang bikin skor evaluasi terlihat bagus padahal palsu. Pipeline mencegah ini karena tiap transformasi hanya di-fit di data latih.

## Bab 5 - Decision Tree dan Pruning

- **Decision Tree**: model klasifikasi berbentuk pohon keputusan yang kerjanya membelah data secara rekursif.
- **CART / ID3**: algoritma decision tree. CART memakai kriteria Gini dan bikin pohon biner, ID3 memakai Information Gain.
- **Gini impurity**: ukuran ketidakmurnian node, rumusnya 1 - jumlah(p^2). Makin kecil makin murni.
- **Entropy / Information Gain**: ukuran ketidakpastian (entropy) dan seberapa besar suatu split mengurangi ketidakpastian (information gain).
- **Overfitting pada tree**: pohon yang nggak dibatasi bakal tumbuh sampai tiap daun murni (akurasi latih 100%) tapi jelek di data baru karena menghafal noise.
- **Pre-pruning**: memotong sebelum tumbuh, misal batasan max_depth dan min_samples_leaf.
- **Post-pruning (cost complexity pruning)**: memotong setelah pohon tumbuh penuh memakai parameter ccp_alpha. Cabang yang perbaikan impurity-nya lebih kecil dari alpha dipangkas.

## Bab 6 - KNN dan Naive Bayes

- **KNN (K-Nearest Neighbors)**: algoritma lazy learning, tidak ada fase pelatihan eksplisit. Saat prediksi, ia mencari k tetangga terdekat lalu mengambil suara terbanyak.
- **Lazy learning**: model yang "menunda" kerja sampai ada data baru yang perlu diprediksi (kebalikan dari eager learning).
- **Naive Bayes**: classifier probabilistik berdasarkan teorema Bayes dengan asumsi fitur saling independen.
- **GaussianNB**: varian Naive Bayes untuk fitur numerik kontinu (asumsi tiap fitur berdistribusi normal).
- **MultinomialNB**: varian Naive Bayes untuk data hitungan/frekuensi, cocok untuk klasifikasi teks.
- **Scaling**: standardisasi fitur (misal StandardScaler) supaya KNN tidak didominasi fitur bernilai besar, karena KNN bekerja dengan jarak.

## Bab 7 - Evaluasi Model Klasifikasi

- **Confusion matrix**: tabel 2x2 berisi True Positive (TP), True Negative (TN), False Positive (FP), False Negative (FN).
- **Precision**: TP/(TP+FP), seberapa tepat prediksi positif model.
- **Recall**: TP/(TP+FN), seberapa banyak kasus positif yang berhasil ditangkap.
- **F1**: rata-rata harmonik precision dan recall, enak dipakai sebagai satu angka ringkasan.
- **ROC curve dan AUC**: kurva True Positive Rate vs False Positive Rate di berbagai threshold. AUC 1 berarti sempurna.
- **Stratified k-fold CV**: validasi silang yang tiap lipatannya menjaga proporsi kelas seperti data aslinya, lebih adil dari sekali split.

## Bab 8 - Support Vector Machine

- **SVM**: classifier yang mencari hyperplane (garis pemisah) dengan margin selebar mungkin.
- **Support vector**: titik-titik data terdekat yang menentukan posisi garis pemisah.
- **Kernel trick**: trik memetakan data ke ruang berdimensi lebih tinggi supaya bisa dipisahkan secara linear. Kernel umum: linear, polynomial, RBF.
- **Parameter C**: mengatur tradeoff antara margin lebar dan kesalahan klasifikasi. C besar bikin model ketat (risiko overfitting).
- **Parameter gamma**: mengatur seberapa jauh pengaruh satu titik training. Gamma besar bikin boundary berliku mengikuti data.
- **Scaling**: SVM sensitif terhadap skala fitur, jadi fitur harus distandardisasi dulu sebelum training.

## Bab 9 - Ensemble Methods

- **Ensemble learning**: menggabungkan banyak model lemah menjadi satu model yang kuat dan stabil.
- **Bagging**: melatih banyak model di sampel bootstrap berbeda lalu dirata-rata/divoting. Contoh: Random Forest.
- **Boosting**: melatih model berurutan, tiap model baru fokus ke sampel yang salah diprediksi sebelumnya. Contoh: AdaBoost, Gradient Boosting.
- **Random Forest**: ensemble decision tree dengan bootstrap dan pemilihan fitur acak tiap split.
- **n_estimators**: jumlah pohon dalam ensemble, makin banyak biasanya makin stabil.
- **OOB score (Out-of-Bag)**: skor validasi gratis dari sampel yang tidak terpakai di tiap bootstrap, tanpa perlu data validasi terpisah.

## Bab 10 - Clustering

- **Clustering**: unsupervised learning untuk mencari kelompok-kelompok alami di dalam data.
- **K-Means**: algoritma clustering dengan memilih k titik centroid, tiap data ditempelkan ke centroid terdekat, dihitung ulang sampai stabil.
- **Elbow method**: memilih nilai k optimal dari grafik "siku", yaitu titik di mana penambahan k tidak lagi banyak menurunkan inersia.
- **Silhouette score**: mengukur seberapa cocok tiap titik dengan clusternya sendiri dibanding cluster lain. Makin dekat ke 1 makin bagus.
- **Hierarchical clustering**: membangun cluster dari bawah ke atas dengan menggabungkan titik yang paling mirip (agglomerative).
- **DBSCAN**: clustering berbasis kepadatan yang bisa menemukan cluster bentuk tak beraturan dan menandai titik aneh sebagai noise.

## Bab 11 - PCA dan Reduksi Dimensi

- **Curse of dimensionality**: masalah yang muncul saat fitur terlalu banyak, data jadi jarang tersebar dan komputasi berat.
- **PCA (Principal Component Analysis)**: teknik reduksi dimensi tanpa label yang mencari arah (komponen) dengan varians terbesar.
- **Variance kumulatif**: jumlah varians yang dijelaskan komponen terpilih, misal ambil komponen sampai 95% varians tercakup.
- **t-SNE**: teknik reduksi dimensi non-linear untuk visualisasi, bagus melihat pengelompokan data tapi hasilnya stokastik.
- **LDA (Linear Discriminant Analysis)**: reduksi dimensi yang memakai label kelas supaya jarak antar kelas maksimal.

## Bab 12 - Neural Networks

- **Perceptron / MLP**: perceptron adalah unit neuron tunggal, sedangkan MLP (Multi-Layer Perceptron) menumpuk banyak neuron dalam beberapa hidden layer.
- **Hidden layer**: lapisan di antara input dan output tempat jaringan belajar representasi pola.
- **Fungsi aktivasi**: fungsi non-linear di tiap neuron (misal relu, softmax) yang bikin jaringan bisa mempelajari pola tidak linear.
- **Layer Dense**: layer di mana tiap neuron terhubung ke semua neuron di layer sebelumnya.
- **Epoch / batch**: satu epoch berarti model melihat seluruh data latih satu kali. Data dibagi batch kecil supaya training efisien.
- **Optimizer dan loss**: optimizer (misal adam) mengatur cara bobot diperbarui, loss (misal sparse_categorical_crossentropy) mengukur seberapa salah prediksi.
- **Early stopping**: menghentikan pelatihan otomatis ketika val_loss berhenti membaik selama beberapa epoch (patience), supaya model tidak overfitting.

## Bab 13 - Convolutional Neural Networks

- **Konvolusi / filter**: operasi menggeser filter kecil ke seluruh citra untuk mendeteksi pola lokal seperti tepi dan tekstur.
- **Pooling (max pooling)**: merampingkan ukuran peta fitur dengan mengambil nilai maksimum tiap jendela, bikin model tahan terhadap pergeseran kecil.
- **Transfer learning**: memakai ulang bobot model yang sudah dilatih di dataset besar (misal ImageNet) lalu melatih classifier kecil di atasnya.
- **MobileNetV2**: arsitektur CNN ringan yang dipakai sebagai ekstraktor fitur di praktikum transfer learning.
- **CIFAR-10**: dataset 60.000 citra berwarna 32x32 dalam 10 kelas, dipakai sebagai subset kecil di praktikum.
