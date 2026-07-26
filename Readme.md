# FINAL PROJECT PENGANTAR SAINS DATA

**Judul** : Analisis Performa IHSG Multi Time Frame
**Nama**  : Faris fadhlullah
**NIM**   : 25.11.6353

---

## 1. Deskripsi Singkat

Project ini menganalisis data historis saham IHSG (2001-2023) untuk 3 hal:

1. Membangun proxy index IHSG (market-cap weighted) dan membandingkan pergerakannya di 3 timeframe: daily, hourly, minute.
2. Menghitung & memvisualisasikan performa rata-rata tiap sektor di 3 jendela waktu (7 / 30 / 90 hari bursa terakhir).
3. Melatih model Machine Learning (scikit-learn) untuk memprediksi **arah index IHSG besok** (naik/turun), berdasarkan indikator teknikal (SMA, MACD, RSI, Bollinger %B, momentum, volume relatif) dari index proxy itu sendiri.

---

## 2. Struktur Folder Proyek

```
nama_project/
├─ utils.py             -> semua fungsi (load data, index proxy, indikator
│                          teknikal, training model, plotting)
├─ main.py              -> alur program utama, panggil fungsi di utils.py
├─ tickers.json          -> daftar saham: index_tickers (15 saham top
│                          market cap) & sector_tickers (~44 saham
│                          representasi 11 sektor)
├─ Dataset/
│  ├─ DaftarSaham.csv    -> metadata seluruh emiten (kode, sektor, market cap)
│  ├─ daily/*.csv        -> data harian per saham
│  ├─ hourly/*.csv       -> data per jam per saham
│  ├─ minutes/*.csv      -> data per menit per saham
│  └─ dataset.csv        -> dataset fitur ML (prediksi arah IHSG) hasil generate main.py
└─ output/
   ├─ 1_index_multi_timeframe.png
   ├─ 2_sector_performance.png
   ├─ 3_confusion_matrix.png
   ├─ 4_feature_importance.png
   └─ hasil_analisis.txt
```

---

## 3. Cara Menjalankan Program

1. Pastikan library berikut sudah terinstall:
   ```
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
2. Jalankan dari terminal / text editor (bukan Google Colab):
   ```
   python main.py
   ```
3. Semua grafik & ringkasan hasil otomatis tersimpan di folder `output/`.

---

## 4. Sumber Data

Dataset berasal dari Kaggle bernama **"Dataset Saham Indonesia"**
Link: https://www.kaggle.com/datasets/muamkh/ihsgstockdata
