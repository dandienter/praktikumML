# Glosarium Praktikum Machine Learning

Kumpulan istilah dan tools yang dipakai di seluruh bab (Bab 1-14). Ditulis ringkas supaya
gampang dicari saat baca notebook atau PDF.

Cara membacanya: istilah disusun mengikuti urutan bab, jadi kalau lagi mengerjakan satu bab,
cukup buka bagian bab itu. Istilah yang dicetak **tebal** di dalam definisi adalah istilah
lain yang juga dijelaskan di glosarium ini.

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
- **Dataset**: kumpulan data yang dipakai untuk melatih dan menguji model, biasanya berbentuk tabel.
- **Fitur dan label**: fitur adalah kolom masukan yang dipakai model untuk menebak (misal panjang kelopak bunga), sedangkan label (target) adalah jawaban yang ingin ditebak (misal spesies bunga).
- **Supervised learning**: pembelajaran dengan data berlabel, contohnya klasifikasi (dataset Iris).
- **Unsupervised learning**: pembelajaran tanpa label, contohnya clustering.
- **Klasifikasi**: tugas menebak kategori/label dari suatu data.
- **Regresi**: tugas menebak nilai angka kontinu.
- **Clustering**: mengelompokkan data berdasarkan kemiripan tanpa label.
- **Train/test split**: membagi data menjadi data latih dan data uji supaya performa model diukur secara jujur.
- **Overfitting**: model terlalu menghafal data latih sehingga jelek di data baru.
- **Underfitting**: model terlalu sederhana sehingga gagal menangkap pola data.
- **Decision boundary**: garis/wilayah pemisah antar kelas yang dipelajari model. Bentuknya menunjukkan kompleksitas model: bergerigi (k kecil, overfitting) vs halus (k besar, underfitting).
- **n_neighbors (k)**: jumlah tetangga terdekat yang ikut voting di KNN. K kecil sensitif terhadap noise, k besar membuat batas keputusan lebih halus.

## Bab 2 - Workflow CRISP-DM dan Bias-Variance

- **CRISP-DM**: kerangka kerja proyek data mining dengan 6 fase: Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, Deployment.
- **Bias-variance tradeoff**: tarik-menarik antara bias tinggi (model terlalu sederhana) dan variance tinggi (model terlalu sensitif terhadap noise data). Model ideal seimbang di tengah.
- **Learning curve**: grafik skor train vs validasi terhadap ukuran data atau kompleksitas model, dipakai untuk mendiagnosis underfitting/overfitting.
- **Stratified sampling**: pembagian data yang menjaga proporsi tiap kelas di setiap bagian, penting untuk dataset yang tidak seimbang.
- **Validation set**: bagian data khusus untuk memilih model dan menyetel hyperparameter, terpisah dari data latih dan data uji.
- **Churn**: pelanggan yang berhenti memakai layanan (studi kasus dataset Telco).

## Bab 3 - Data Preparation

- **Missing value**: data yang hilang/kosong. Ditangani dengan hapus baris, imputasi statistik sederhana (mean/median), atau imputasi berbasis model (KNN Imputer).
- **Outlier**: data yang menyimpang jauh dari pola umum dan bisa merusak model. Dideteksi misalnya dengan Isolation Forest.
- **Imbalanced data**: kelas minoritas jauh lebih sedikit dari kelas mayoritas, bikin model bias menebak kelas mayoritas.
- **Imputasi**: teknik mengisi nilai yang hilang. **Mean imputation** mengisi dengan rata-rata kolom, **KNN Imputer** mengisi berdasarkan tetangga terdekat.
- **SMOTE**: oversampling yang membuat sampel sintetis kelas minoritas supaya jumlahnya seimbang.
- **Undersampling**: mengurangi sampel kelas mayoritas secara acak.
- **class_weight="balanced"**: opsi di scikit-learn yang memberi bobot lebih besar ke kelas minoritas tanpa mengubah data.
- **Isolation Forest**: algoritma deteksi outlier yang mengisolasi titik aneh lewat partisi acak. Parameter `contamination` mengatur perkiraan proporsi outlier di data.

## Bab 4 - Feature Engineering dan Pipeline

- **Feature construction**: membuat fitur baru dari fitur mentah agar pola lebih mudah dilihat model (misal rasio, binning).
- **Feature selection**: memilih subset fitur yang paling relevan. Ada tiga pendekatan: **filter** (skor statistik tanpa model, misal chi2), **wrapper** (evaluasi dengan model, misal RFE), **embedded** (seleksi saat training, misal Lasso/L1).
- **Pipeline**: menggabungkan preprocessing dan model jadi satu objek supaya alurnya konsisten dari training sampai deployment.
- **ColumnTransformer**: menerapkan transformasi berbeda ke kelompok kolom berbeda dalam satu langkah.
- **One-hot encoding**: cara mengubah fitur kategorikal menjadi kolom-kolom biner 0/1 (satu kolom per kategori) supaya bisa diproses model numerik.
- **Binning**: mengelompokkan nilai numerik kontinu ke dalam beberapa kategori rentang (misal umur menjadi Baru, Sedang, Tua).
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
- **Vektorisasi teks (CountVectorizer)**: mengubah dokumen teks menjadi vektor hitungan kata supaya bisa diproses model.
- **weights (uniform vs distance)**: di KNN, `uniform` memberi bobot sama ke semua tetangga, sedangkan `distance` memberi bobot lebih besar ke tetangga yang lebih dekat.

## Bab 7 - Evaluasi Model Klasifikasi

- **Confusion matrix**: tabel 2x2 berisi True Positive (TP), True Negative (TN), False Positive (FP), False Negative (FN).
- **Precision**: TP/(TP+FP), seberapa tepat prediksi positif model.
- **Recall**: TP/(TP+FN), seberapa banyak kasus positif yang berhasil ditangkap.
- **F1**: rata-rata harmonik precision dan recall, enak dipakai sebagai satu angka ringkasan.
- **ROC curve dan AUC**: kurva True Positive Rate vs False Positive Rate di berbagai threshold. AUC 1 berarti sempurna.
- **Stratified k-fold CV**: validasi silang yang tiap lipatannya menjaga proporsi kelas seperti data aslinya, lebih adil dari sekali split.
- **PR curve (Precision-Recall curve)**: kurva precision vs recall di berbagai threshold. Lebih informatif daripada ROC-AUC saat data tidak seimbang.
- **Threshold**: batas nilai probabilitas untuk memutuskan prediksi positif. Menggeser threshold menukar precision dengan recall.

## Bab 8 - Support Vector Machine

- **SVM**: classifier yang mencari hyperplane (garis pemisah) dengan margin selebar mungkin.
- **Support vector**: titik-titik data terdekat yang menentukan posisi garis pemisah.
- **Kernel trick**: trik memetakan data ke ruang berdimensi lebih tinggi supaya bisa dipisahkan secara linear. Kernel umum: linear, polynomial, RBF.
- **Parameter C**: mengatur tradeoff antara margin lebar dan kesalahan klasifikasi. C besar bikin model ketat (risiko overfitting).
- **Parameter gamma**: mengatur seberapa jauh pengaruh satu titik training. Gamma besar bikin boundary berliku mengikuti data.
- **GridSearchCV**: pencarian menyeluruh atas daftar kombinasi hyperparameter dengan cross-validation, untuk menemukan kombinasi terbaik secara sistematis.
- **Scaling**: SVM sensitif terhadap skala fitur, jadi fitur harus distandardisasi dulu sebelum training.

## Bab 9 - Ensemble Methods

- **Ensemble learning**: menggabungkan banyak model lemah menjadi satu model yang kuat dan stabil.
- **Bagging**: melatih banyak model di sampel bootstrap berbeda lalu dirata-rata/divoting. Contoh: Random Forest.
- **Boosting**: melatih model berurutan, tiap model baru fokus ke sampel yang salah diprediksi sebelumnya. Contoh: AdaBoost, Gradient Boosting.
- **Random Forest**: ensemble decision tree dengan bootstrap dan pemilihan fitur acak tiap split.
- **n_estimators**: jumlah pohon dalam ensemble, makin banyak biasanya makin stabil.
- **OOB score (Out-of-Bag)**: skor validasi gratis dari sampel yang tidak terpakai di tiap bootstrap, tanpa perlu data validasi terpisah.
- **Feature importance**: skor seberapa besar kontribusi tiap fitur terhadap keputusan ensemble, dihitung dari total penurunan impurity yang disumbangkan fitur itu.
- **Hard voting vs soft voting**: hard voting menggabungkan label hasil voting terbanyak tiap model, soft voting merata-ratakan probabilitas prediksi tiap model (biasanya lebih baik).
- **Learning rate**: seberapa besar langkah perbaikan tiap iterasi boosting. Kecil = belajar pelan tapi stabil, besar = cepat tapi bisa kelewatan optimum.

## Bab 10 - Clustering

- **Clustering**: unsupervised learning untuk mencari kelompok-kelompok alami di dalam data.
- **K-Means**: algoritma clustering dengan memilih k titik centroid, tiap data ditempelkan ke centroid terdekat, dihitung ulang sampai stabil.
- **Elbow method**: memilih nilai k optimal dari grafik "siku", yaitu titik di mana penambahan k tidak lagi banyak menurunkan inersia.
- **Silhouette score**: mengukur seberapa cocok tiap titik dengan clusternya sendiri dibanding cluster lain. Makin dekat ke 1 makin bagus.
- **Hierarchical clustering**: membangun cluster dari bawah ke atas dengan menggabungkan titik yang paling mirip (agglomerative).
- **DBSCAN**: clustering berbasis kepadatan yang bisa menemukan cluster bentuk tak beraturan dan menandai titik aneh sebagai noise.
- **k-means++**: cara inisialisasi centroid K-Means yang menyebar, jauh lebih stabil dan cepat konvergen daripada inisialisasi acak.
- **Inersia (inertia)**: jumlah jarak kuadrat tiap titik ke centroid clusternya. Dipakai di elbow method: pilih k di titik siku saat penurunan inersia melambat.
- **eps / min_samples**: dua parameter DBSCAN. `eps` adalah radius tetangga, `min_samples` jumlah minimum titik untuk membentuk cluster padat. Titik yang tidak masuk cluster mana pun ditandai sebagai noise.

## Bab 11 - PCA dan Reduksi Dimensi

- **Curse of dimensionality**: masalah yang muncul saat fitur terlalu banyak, data jadi jarang tersebar dan komputasi berat.
- **PCA (Principal Component Analysis)**: teknik reduksi dimensi tanpa label yang mencari arah (komponen) dengan varians terbesar.
- **Komponen utama (principal component)**: sumbu baru hasil PCA, berupa kombinasi linear fitur asli yang disusun dari penangkap varians terbesar ke terkecil.
- **Eigenvalue dan eigenvector**: pasangan angka dan arah yang menentukan komponen utama PCA. Eigenvector menunjukkan arah komponen, eigenvalue menunjukkan besar varians yang ditangkap arah itu.
- **Variance kumulatif**: jumlah varians yang dijelaskan komponen terpilih, misal ambil komponen sampai 95% varians tercakup.
- **t-SNE**: teknik reduksi dimensi non-linear untuk visualisasi, bagus melihat pengelompokan data tapi hasilnya stokastik.
- **LDA (Linear Discriminant Analysis)**: reduksi dimensi yang memakai label kelas supaya jarak antar kelas maksimal.
- **Scree plot**: grafik batang/garis varians tiap komponen PCA dari besar ke kecil. Jumlah komponen dipilih di titik siku atau sampai varians kumulatif mencapai target (misal 95%).

## Bab 12 - Neural Networks

- **Perceptron / MLP**: perceptron adalah unit neuron tunggal, sedangkan MLP (Multi-Layer Perceptron) menumpuk banyak neuron dalam beberapa hidden layer.
- **Bobot dan bias**: bobot adalah angka pengali tiap masukan neuron yang dipelajari saat training, bias adalah angka geser yang membantu neuron menyesuaikan ambang aktivasinya.
- **Hidden layer**: lapisan di antara input dan output tempat jaringan belajar representasi pola.
- **Fungsi aktivasi**: fungsi non-linear di tiap neuron (misal relu, softmax) yang bikin jaringan bisa mempelajari pola tidak linear.
- **Layer Dense**: layer di mana tiap neuron terhubung ke semua neuron di layer sebelumnya.
- **Forward pass**: aliran hitungan dari input ke output untuk menghasilkan prediksi (jumlah berbobot lalu aktivasi di tiap neuron).
- **Backpropagation**: algoritma menghitung kontribusi tiap bobot terhadap kesalahan prediksi, dengan menurunkan gradien loss dari layer output mundur ke input.
- **Gradient descent**: cara memperbarui bobot sedikit demi sedikit berlawanan arah gradien supaya loss makin kecil. Varian adaptifnya yang populer adalah optimizer Adam.
- **Epoch / batch**: satu epoch berarti model melihat seluruh data latih satu kali. Data dibagi batch kecil supaya training efisien. **Batch size** adalah jumlah sampel per batch (misal 32).
- **Optimizer dan loss**: optimizer (misal adam) mengatur cara bobot diperbarui, loss (misal sparse_categorical_crossentropy) mengukur seberapa salah prediksi.
- **Early stopping**: menghentikan pelatihan otomatis ketika val_loss berhenti membaik selama beberapa epoch (patience), supaya model tidak overfitting.
- **Softmax**: fungsi aktivasi di layer output untuk klasifikasi multi-kelas, mengubah skor menjadi probabilitas yang jumlahnya 1.
- **Learning rate**: ukuran langkah update bobot tiap iterasi. Terlalu besar bikin training tidak stabil, terlalu kecil bikin training lama.
- **val_loss**: nilai loss di data validasi. Dipantau untuk early stopping: training dihentikan saat val_loss berhenti membaik.

## Bab 13 - Convolutional Neural Networks

- **Konvolusi / filter**: operasi menggeser filter kecil ke seluruh citra untuk mendeteksi pola lokal seperti tepi dan tekstur.
- **Feature map**: hasil keluaran satu filter konvolusi, berupa peta yang menyorot di mana pola yang dicari filter itu muncul pada citra.
- **Stride**: jarak geser filter konvolusi tiap langkah. Stride besar membuat feature map lebih kecil dan hitungan lebih cepat.
- **Padding**: piksel tambahan (biasanya nol) di tepi citra supaya ukuran feature map tidak menyusut terlalu cepat dan informasi tepi tidak hilang.
- **Pooling (max pooling)**: merampingkan ukuran peta fitur dengan mengambil nilai maksimum tiap jendela, bikin model tahan terhadap pergeseran kecil.
- **Flatten**: mengubah feature map dua dimensi menjadi vektor satu dimensi supaya bisa masuk ke layer Dense.
- **Transfer learning**: memakai ulang bobot model yang sudah dilatih di dataset besar (misal ImageNet) lalu melatih classifier kecil di atasnya.
- **Fine-tuning**: melatih ulang sebagian atau seluruh bobot model transfer learning dengan learning rate kecil agar menyesuaikan ke data baru tanpa merusak pola yang sudah dipelajari.
- **MobileNetV2**: arsitektur CNN ringan yang dipakai sebagai ekstraktor fitur di praktikum transfer learning.
- **CIFAR-10**: dataset 60.000 citra berwarna 32x32 dalam 10 kelas, dipakai sebagai subset kecil di praktikum.
- **Data augmentation**: memperbanyak variasi data latih lewat transformasi acak (rotasi, flip, geser, zoom) supaya model lebih tahan terhadap variasi input.
- **Dropout**: teknik regularisasi yang mematikan neuron secara acak saat training, memaksa jaringan tidak bergantung pada neuron tertentu. Dropout 0,5 artinya setengah neuron dimatikan tiap iterasi.

## Bab 14 - Mini Project End-to-End

- **Serialisasi model**: menyimpan model yang sudah dilatih ke file (misal format pickle dengan joblib) supaya bisa dimuat dan dipakai lagi tanpa training ulang.
- **Deployment**: tahap memakai model di aplikasi nyata (misal web atau bot) agar bisa menerima data baru dan mengembalikan prediksi.
- **Model terbaik (pemilihan model)**: model yang dipilih dari hasil perbandingan bukan hanya dari akurasi tertinggi, tetapi juga metrik yang paling penting untuk kasusnya (pada proyek jantung, recall didahulukan karena false negative berbahaya).
