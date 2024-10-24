# Laporan Proyek Machine Learning - Adelia Octora Pristisahida

## Predictive Analytics - Harga rumah di California

Harga properti di California, khususnya rumah, terus mengalami perubahan yang signifikan dalam beberapa dekade terakhir. Faktor-faktor seperti pertumbuhan populasi, peningkatan biaya hidup, permintaan yang tinggi terhadap properti, serta kebijakan perumahan lokal dan negara bagian, telah mendorong fluktuasi harga yang substansial. Dalam konteks bisnis, pergerakan harga properti di California sangat mempengaruhi keputusan investasi. Para pengembang, investor, dan agen properti memerlukan pemahaman mendalam tentang tren harga rumah untuk menyusun strategi bisnis yang efektif. 

Di sisi lain, dengan semakin meningkatnya kebutuhan masyarakat akan perumahan, prediksi harga rumah menjadi alat penting bagi pemerintah dan penyusun kebijakan untuk merumuskan kebijakan perumahan yang inklusif dan berkelanjutan. Penggunaan teknologi seperti machine learning dan big data analytics untuk memprediksi harga rumah di masa depan dapat memberikan insight yang lebih akurat dan membantu pengambilan keputusan yang lebih baik. Melalui analisis prediksi harga rumah di California, kita dapat mengeksplorasi bagaimana berbagai faktor seperti lokasi geografis, ukuran properti, kondisi ekonomi lokal, tingkat pengangguran, serta kebijakan regional mempengaruhi dinamika pasar. 

## Business Understanding
### Problem Statements

Yang perlu dipahami dalam permasalahan ini adalah:
1. Bagaimana memprediksi nilai rumah berdasarkan karakteristik rumah dan lingkungan?
2. Bagaimana membantu pembeli dan penjual rumah menentukan harga yang wajar?
3. Bagaimana mengelola risiko kredit hipotek dengan lebih baik bagi bank atau lembaga keuangan?

Hal ini dapat membantu bank atau lembaga keuangan untuk memprediksi risiko gagal bayar kredit hipotek, sementara bagi perusahaan real estate, prediksi ini bisa meningkatkan volume penjualan.

Hal-hal tersebut dapat diketahui dengan bertanya pada stakeholder, antara lain, pembeli, lembaga keuangan, dan agen real estate. Namun dalam hal ini saya menggunakan dataset yang sudah siap digunakan yang ada di kaggle

### Goals

Menjelaskan tujuan dari pernyataan masalah:
1. Membangun model machine learning untuk memprediksi nilai median rumah di suatu area berdasarkan fitur-fitur tertentu seperti lokasi, ukuran rumah, jumlah kamar, dll.
2. Meminimalkan kesalahan prediksi harga rumah agar pembeli dan penjual mendapatkan estimasi yang lebih akurat.

### Solution statements
Solusi 1: Membangun Model Regresi dengan Beberapa Algoritma (K-NN, Random Forest, Gradient Boosting)
Solusi 2: Hyperparameter Tuning untuk Meningkatkan Kinerja Model (Grid Search, Random Search, atau Bayesian Optimization)

dalam proyek ini yang digunakan adalah mengikuti contoh dari materi yang telah diberikan antara lain, K-NN, Random Forest, dan Boosting Algoritme.

## Data Understanding
Dalam proyek ini, data yang digunakan adalah dataset properti yang berisi berbagai informasi tentang rumah di California, Amerika Serikat. Dataset ini mencakup fitur-fitur seperti lokasi geografis (longitude dan latitude), karakteristik rumah (jumlah kamar, jumlah kamar tidur, luas rumah, dan usia bangunan), serta data demografis dan ekonomi (populasi, jumlah rumah tangga, dan pendapatan median). Selain itu, dataset ini juga mencakup harga median rumah di setiap area yang digunakan sebagai target atau label untuk prediksi.

Jumlah Data (Baris dan Kolom): Dataset ini terdiri dari 20.640 baris dan 10 kolom. Setiap baris mewakili sebuah distrik di California, dan kolom-kolomnya memuat berbagai informasi yang berhubungan dengan harga rumah serta kondisi distrik tersebut.

Kondisi Data: Dataset ini cukup bersih dan lengkap, serta siap digunakan untuk analisis lebih lanjut. Data ini diambil dari Sensus 1990, dan mencakup informasi penting seperti lokasi geografis, jumlah kamar tidur, serta status sosio-ekonomi dari distrik-distrik yang diwakili.

Kaggle datasets : https://www.kaggle.com/datasets/camnugent/california-housing-prices

Uraian Seluruh Fitur pada Data:
1. longitude: Koordinat geografis (garis bujur) distrik.
2. latitude: Koordinat geografis (garis lintang) distrik.
3. housing_median_age: Usia rata-rata rumah di distrik tersebut.
4. total_rooms: Jumlah total ruangan di seluruh rumah di distrik tersebut.
5. total_bedrooms: Jumlah total kamar tidur di distrik tersebut.
6. population: Jumlah total populasi di distrik tersebut.
7. households: Jumlah total rumah tangga di distrik tersebut.
8. median_income: Pendapatan median tahunan rumah tangga di distrik (dalam puluhan ribu dolar AS).
9. median_house_value: Nilai median rumah di distrik tersebut (dalam dolar AS).
10. ocean_proximity: Kategori yang menunjukkan kedekatan distrik dengan lautan (misalnya, "NEAR OCEAN").

## Data Preparation
Pada tahap persiapan data, dilakukan beberapa teknik untuk memastikan data siap digunakan dalam proses modelisasi. Tahapan-tahapan yang dilakukan adalah sebagai berikut:
1. Handling Missing Values (Penanganan Nilai Hilang) dengan SimpleImputer: Fitur yang memiliki nilai hilang diatasi menggunakan teknik imputasi. Dalam hal ini, kita menggunakan SimpleImputer dari library sklearn untuk menggantikan nilai yang hilang dengan nilai rata-rata (mean) dari fitur tersebut. Ini dilakukan untuk memastikan tidak ada data yang hilang selama proses pelatihan model.
2. Encoding Fitur Kategori: Kolom ocean_proximity yang merupakan fitur kategori diubah menjadi nilai numerik menggunakan teknik encoding, seperti One-Hot Encoding. Ini dilakukan untuk mengubah fitur kategori menjadi format yang dapat dipahami oleh model machine learning.
3. Reduksi Dimensi dengan Principal Component Analysis (PCA): Untuk mengurangi dimensi data dan mengurangi noise, kami menggunakan Principal Component Analysis (PCA). Ini membantu dalam mengekstraksi fitur-fitur yang paling penting dari dataset, sekaligus mengurangi kompleksitas model.
4. Pembagian Dataset dengan Fungsi train_test_split: Dataset dibagi menjadi dua bagian, yaitu training set dan test set, dengan perbandingan 80:20 menggunakan fungsi train_test_split dari library sklearn. Training set digunakan untuk melatih model, sedangkan test set digunakan untuk menguji performa model.
5. Standarisasi (Standardization): Data yang sudah di-imputasi dan di-encode kemudian distandarisasi menggunakan StandardScaler dari library sklearn. Ini dilakukan agar fitur-fitur berada pada skala yang sama, sehingga membantu meningkatkan kinerja model machine learning.

## Modeling
Pada tahap ini, dilakukan pengembangan model prediksi harga rumah menggunakan tiga algoritma machine learning yang berbeda. Setiap algoritma dioptimalkan dengan parameter tertentu untuk mencapai hasil prediksi yang maksimal. Berikut adalah penjelasan tentang tahapan kerja dari masing-masing algoritma yang digunakan:

### K-Nearest Neighbors (K-NN):
Tahapan Kerja Algoritma: Algoritma K-NN bekerja dengan mencari sejumlah tetangga terdekat (dengan K sebagai jumlah tetangga) dari data yang ingin diprediksi. Jarak antara data baru dan data di training set dihitung menggunakan metrik seperti Euclidean distance. Berdasarkan mayoritas kelas tetangga tersebut, prediksi harga rumah dilakukan.
Parameter yang Digunakan:
n_neighbors: Jumlah tetangga terdekat (K). Parameter ini diatur pada nilai 10.
weights: Mengatur apakah semua tetangga memiliki pengaruh yang sama (uniform) atau bobot dihitung berdasarkan jaraknya (distance).

### Random Forest:
Tahapan Kerja Algoritma: Random Forest adalah algoritma ensemble yang bekerja dengan membangun beberapa decision trees dari subset data yang berbeda dan melakukan prediksi dengan cara menggabungkan hasil prediksi dari setiap pohon (biasanya melalui rata-rata). Setiap pohon keputusan dibangun dari fitur yang dipilih secara acak, sehingga menghasilkan model yang lebih robust.
Parameter yang Digunakan:
n_estimators: Jumlah pohon yang digunakan di dalam hutan. Pada model ini, nilai diatur pada 50.
max_depth: Kedalaman maksimum dari setiap pohon. Ini menentukan kompleksitas pohon dan diatur pada 16.
random_state: Parameter ini digunakan untuk menjaga konsistensi hasil ketika model dilatih ulang.

### Boosting Algorithm (Gradient Boosting):
Tahapan Kerja Algoritma: Gradient Boosting bekerja dengan membangun model secara berurutan, di mana setiap model baru mencoba mengoreksi kesalahan prediksi dari model sebelumnya. Algoritma ini menggunakan loss function untuk meminimalkan error, secara bertahap memperbaiki prediksi dengan menambahkan model-model baru.
Parameter yang Digunakan:
learning_rate: Ukuran langkah yang digunakan untuk memperbarui prediksi. Nilai kecil seperti 0.05 membantu menjaga kestabilan model.
random_state: Parameter untuk menjaga hasil yang konsisten.

## Evaluation
Interpretasi Hasil Berdasarkan Business Understanding
Pada bagian ini, kita tidak hanya menilai performa model berdasarkan metrik Mean Squared Error (MSE), tetapi juga mengevaluasi dampak dari model yang dikembangkan terhadap pemahaman bisnis dan apakah model tersebut berhasil menjawab problem statement dan tujuan proyek.

Cara Kerja MSE: MSE menghitung rata-rata dari selisih kuadrat antara nilai yang diprediksi oleh model dan nilai aktual. Metrik ini sensitif terhadap outlier karena kesalahan dipangkatkan dua, yang berarti semakin besar selisih antara prediksi dan nilai sebenarnya, semakin besar kontribusi kesalahan tersebut terhadap MSE. Semakin kecil nilai MSE, semakin baik performa model dalam memprediksi nilai target.

Interpretasi: Jika nilai MSE mendekati nol, model memiliki performa yang baik, karena ini berarti kesalahan prediksi sangat kecil. MSE yang lebih tinggi menunjukkan bahwa model memiliki kesalahan prediksi yang lebih besar, yang berarti model kurang mampu menangkap pola dalam data dengan baik.

MSE sering digunakan karena metrik ini mudah dihitung dan memberikan gambaran langsung tentang seberapa jauh prediksi model dari nilai sebenarnya. Namun, karena kuadrat dari kesalahan dihitung, MSE lebih sensitif terhadap kesalahan besar (outlier).

Dalam konteks Business Understanding, MSE digunakan untuk menilai akurasi prediksi model terhadap harga rumah. Di proyek ini, kita tidak hanya melihat MSE untuk membandingkan performa model secara teknis, tetapi juga mengevaluasi dampaknya terhadap pengambilan keputusan bisnis. Oleh karena itu, selain mengevaluasi model secara teknis, penting untuk memastikan apakah solusi yang dihasilkan (dalam hal ini model dengan MSE terendah) memberikan dampak positif terhadap tujuan bisnis dan problem statement yang dihadapi.


### K-Nearest Neighbors (K-NN):
MSE pada Train Set: 2,678,887.367064
MSE pada Test Set: 19,906,448.138128
Interpretasi: MSE pada test set menunjukkan bahwa model K-NN memiliki kesalahan prediksi yang cukup tinggi di luar sampel pelatihan. Artinya, model ini tidak terlalu baik dalam memprediksi harga rumah, terutama pada data baru yang tidak dilihat selama pelatihan. Ini mengindikasikan bahwa K-NN mungkin mengalami overfitting, dimana model bekerja baik pada data pelatihan tetapi tidak generalisasi dengan baik pada data test.

### Random Forest (RF):
MSE pada Train Set: 451,785.877382
MSE pada Test Set: 13,039,738.115209
Interpretasi: Random Forest memiliki MSE yang jauh lebih rendah pada train set dibandingkan K-NN, menunjukkan bahwa model ini mampu menangkap lebih banyak informasi dari data pelatihan. Namun, MSE pada test set masih cukup tinggi, meskipun lebih rendah dari K-NN. Ini menunjukkan bahwa Random Forest memiliki performa yang lebih baik daripada K-NN dalam memprediksi harga rumah pada data yang belum pernah dilihat, tetapi masih ada ruang untuk perbaikan dalam hal generalisasi.

### Boosting Algorithm (Gradient Boosting):
MSE pada Train Set: 4,085,425.297193
MSE pada Test Set: 8,911,992.892919
Interpretasi: Model Boosting menunjukkan performa terbaik di antara semua model dalam hal MSE pada test set. Meskipun MSE pada train set lebih tinggi dibandingkan Random Forest, model ini memiliki generalisasi yang lebih baik, sebagaimana ditunjukkan oleh MSE yang lebih rendah pada test set. Hal ini mengindikasikan bahwa Boosting mampu menangkap pola lebih baik dalam data dan lebih baik dalam menangani data baru.

Analisis Perbedaan Visualisasi
Berdasarkan hasil MSE, plot visualisasi yang sesuai dengan notebook akan menunjukkan bahwa Boosting Algorithm memiliki performa terbaik pada data test, diikuti oleh Random Forest, dan terakhir K-NN dengan MSE tertinggi.

Jika visualisasi di laporan berbeda dari hasil ini, kemungkinan ada kesalahan dalam menampilkan atau interpretasi plot. Pastikan bahwa visualisasi plot yang digunakan menampilkan MSE untuk train set dan test set secara terpisah dan sesuai dengan hasil yang tercatat dalam notebook.

### Kesimpulan
Setelah dianalisis ulang, kita dapat menyimpulkan bahwa berdasarkan MSE:
1. Boosting Algorithm memberikan prediksi yang paling akurat pada data test.
2. Random Forest bekerja lebih baik dari K-NN tetapi masih kurang optimal dibandingkan Boosting.
3. K-NN memiliki kesalahan yang paling besar dan mungkin mengalami overfitting.

Problem statement dalam proyek ini adalah untuk membuat prediksi harga rumah yang akurat di California, dengan tujuan membantu perusahaan real estate atau investor dalam pengambilan keputusan terkait pembelian, penjualan, atau pengembangan properti. Berdasarkan hasil evaluasi, Boosting Algorithm (Gradient Boosting) memberikan hasil prediksi yang paling akurat, sehingga berhasil menjawab problem statement dengan lebih baik dibandingkan model lain. Model ini dapat memberikan gambaran yang lebih baik tentang harga pasar rumah, yang penting bagi bisnis untuk merumuskan strategi yang lebih baik.

Tujuan dari proyek ini adalah untuk mengembangkan model prediksi harga rumah yang akurat dan dapat diandalkan. Dengan MSE yang lebih rendah pada Gradient Boosting, kita dapat menyimpulkan bahwa tujuan utama dari proyek ini berhasil dicapai. Model ini cukup efisien untuk digunakan dalam pengambilan keputusan bisnis, seperti estimasi harga rumah yang lebih realistis dan akurat dalam skenario dunia nyata.

Solusi yang direncanakan, yaitu menggunakan Gradient Boosting sebagai model prediksi utama, terbukti berdampak positif. Dengan tingkat akurasi yang lebih tinggi, perusahaan atau investor dapat membuat keputusan yang lebih baik terkait pembelian properti, pengaturan harga, dan evaluasi risiko. Hal ini juga akan mendukung strategi bisnis dalam jangka panjang dengan memberikan prediksi harga yang lebih dekat dengan realitas pasar, sehingga mengurangi risiko keputusan yang salah.

Referensi:
Breiman, L. (2001). Random forests. Machine learning, 45(1), 5-32.
Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. Annals of Statistics, 29(5), 1189–1232.
Tukey, J. W. (1977). Exploratory Data Analysis. Addison-Wesley.

**---Ini adalah bagian akhir laporan---**

