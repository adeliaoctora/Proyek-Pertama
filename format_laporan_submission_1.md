# Laporan Proyek Machine Learning - Adelia Octora Pristisahida

## Predictive Analytics - Harga rumah di California

Proyek ini dibuat dalam rangka mengerjakan tugas submission proyek pertama dari kelas belajar machine learning terapan di dicoding. Untuk mengerjakan tugas ini, harus memilih salah satu masalah yang ingin diselesaikan. Saya memilih prediksi harga rumah.

Sumber : https://www.kaggle.com/datasets/camnugent/california-housing-prices

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
Solusi 1: Membangun Model Regresi dengan Beberapa Algoritma (Linear Regression, Random Forest, Gradient Boosting)
Solusi 2: Hyperparameter Tuning untuk Meningkatkan Kinerja Model (Grid Search, Random Search, atau Bayesian Optimization)

dalam proyek ini yang digunakan adalah mengikuti contoh dari materi yang telah diberikan antara lain, K-NN, Random Forest, dan Boosting Algoritme.

## Data Understanding
Dalam proyek ini, data yang digunakan adalah dataset properti yang berisi berbagai informasi tentang rumah di California, Amerika Serikat. Dataset ini mencakup fitur-fitur seperti lokasi geografis (longitude dan latitude), karakteristik rumah (jumlah kamar, jumlah kamar tidur, luas rumah, dan usia bangunan), serta data demografis dan ekonomi (populasi, jumlah rumah tangga, dan pendapatan median). Selain itu, dataset ini juga mencakup harga median rumah di setiap area yang digunakan sebagai target atau label untuk prediksi.

Dataset yang digunakan dalam proyek ini tersedia secara publik dan dapat diunduh dari beberapa sumber, seperti:
California Housing Prices Dataset yang tersedia di Kaggle atau di Scikit-Learn yang menyediakan versi ringan dari dataset ini.
Dataset ini ideal untuk proyek prediksi harga rumah karena mencakup berbagai fitur yang dapat mempengaruhi harga rumah, seperti lokasi, kondisi lingkungan, dan ekonomi lokal. Data ini juga telah digunakan dalam berbagai proyek machine learning sehingga dapat menjadi titik awal yang baik untuk eksperimen model prediksi harga rumah.

Kaggle datasets : https://www.kaggle.com/datasets/camnugent/california-housing-prices

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
1. longitude: Ukuran seberapa jauh ke arah barat sebuah rumah; nilai yang lebih tinggi berarti lebih jauh ke barat.
2. latitude: Ukuran seberapa jauh ke arah utara sebuah rumah; nilai yang lebih tinggi berarti lebih jauh ke utara.
3. housingMedianAge: Usia rata-rata sebuah rumah dalam satu blok; angka yang lebih rendah menunjukkan bangunan yang lebih baru.
4. totalRooms: Jumlah total kamar dalam satu blok.
5. totalBedrooms: Jumlah total kamar tidur dalam satu blok.
6. population: Jumlah total orang yang tinggal dalam satu blok.
7. households: Jumlah total rumah tangga, sekelompok orang yang tinggal dalam satu unit rumah, untuk satu blok.
8. medianIncome: Pendapatan rata-rata rumah tangga dalam satu blok rumah (diukur dalam puluhan ribu Dolar AS).
9. medianHouseValue: Nilai rata-rata rumah untuk rumah tangga dalam satu blok (diukur dalam Dolar AS).
10. oceanProximity: Lokasi rumah terhadap kedekatannya dengan laut/samudra.

## Data Preparation
Teknik data preparation yang dilakukan, meliputi:
1. Encoding fitur kategori.
2. Reduksi dimensi dengan Principal Component Analysis (PCA).
3. Pembagian dataset dengan fungsi train_test_split dari library sklearn.
4. Standarisasi.

## Modeling
Pada tahap ini, model machine learning dikembangakn menggunakan tiga algoritma. Kemudian, dievaluasi performa masing-masing algoritma dan menentukan algoritma mana yang memberikan hasil prediksi terbaik. Ketiga algoritma yang digunakan, antara lain:
1. K-Nearest Neighbor
2. Random Forest
3. Boosting Algorithm

## Evaluation
Metrik yang digunakan pada prediksi ini adalah MSE atau Mean Squared Error yang menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi.

Hasil Model dengan nilai 118700:
1. KNN:
MSE: 77870
Ini menunjukkan bahwa model ini memiliki kesalahan yang signifikan dalam memprediksi harga rumah.
2. Random Forest:
MSE: 174590
Model ini jauh lebih baik dibandingkan Linear Regression, dengan kesalahan yang lebih kecil. Ini menunjukkan bahwa Random Forest mampu menangkap hubungan yang lebih kompleks dalam data.
3. Boosting :
MSE: 87506.4
Model Boosting menunjukkan performa terbaik di antara semua model yang diuji, dengan MSE terendah. Hal ini menunjukkan bahwa Gradient Boosting sangat efektif dalam menangkap pola yang ada di dalam data.

Interpretasi Hasil:
Dengan membandingkan MSE dari ketiga model, kita dapat menyimpulkan bahwa Gradient Boosting Regressor adalah model terbaik dalam hal akurasi prediksi harga rumah, diikuti oleh Random Forest Regressor dan terakhir Linear Regression.
Perbedaan signifikan dalam nilai MSE antara Linear Regression dan model ensemble (Random Forest dan Gradient Boosting) menunjukkan bahwa model linear sederhana tidak cukup untuk menangkap kompleksitas dalam data.


**---Ini adalah bagian akhir laporan---**
