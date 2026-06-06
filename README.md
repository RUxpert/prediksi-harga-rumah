# Prediksi Harga Rumah

Aplikasi web untuk memprediksi harga rumah di wilayah Depok menggunakan model **XGBoost** yang dioptimasi dengan **Optuna Hyperparameter Optimization**. Dibangun dengan Streamlit dan dapat diakses melalui browser.

## Demo

[![Open in Streamlit](https://img.shields.io/badge/Streamlit-Live_App-FF4B4B?logo=streamlit&logoColor=white)](https://regresio-harga-rumah.streamlit.app/)

🔗 Live Demo: https://regresio-harga-rumah.streamlit.app/

## Fitur

- Input 5 atribut properti (kamar tidur, kamar mandi, garasi, luas tanah, luas bangunan)
- Prediksi harga dalam format Rupiah
- Rentang estimasi harga berdasarkan MAPE model
- Detail input yang dapat di-expand

## Performa Model

| Metrik | Nilai |
|--------|-------|
| R² | 0.8264 |
| RMSE | Rp 301.132.091 |
| MAE | Rp 196.869.066 |
| MAPE | 16.53% |

> Model dilatih pada 13.339 data dan diuji pada 3.335 data properti wilayah Depok.

## Struktur Proyek

```
.
├── app.py                    # Aplikasi Streamlit
├── requirements.txt          # Dependensi Python
├── README.md
└── tuning_v1.1-copy/
    ├── model.pkl             # Model XGBoost terlatih
    ├── scaler.pkl            # RobustScaler untuk fitur
    └── metadata.json         # Metadata model & metrik
```

## Instalasi & Menjalankan Secara Lokal

**1. Clone repository**
```bash
git clone https://github.com/nandana05-tech/prediksi-harga-rumah.git
cd prediksi-harga-rumah
```

**2. Install dependensi**
```bash
pip install -r requirements.txt
```

**3. Jalankan aplikasi**
```bash
streamlit run app.py
```

Buka browser di `http://localhost:8501`

## Dataset

Data properti rumah di wilayah Depok yang mencakup 40.200 listing. Setelah pembersihan data dan penghapusan outlier (filter persentil 5–95), tersisa 16.674 baris yang digunakan untuk pelatihan.

**Atribut yang digunakan:**

| Atribut | Tipe | Keterangan |
|---------|------|-----------|
| Kamar Tidur | Int | Jumlah kamar tidur |
| Kamar Mandi | Int | Jumlah kamar mandi |
| Garasi | Int | Kapasitas garasi |
| Luas Tanah | Float | Luas tanah (m²) |
| Luas Bangunan | Float | Luas bangunan (m²) |

## Pipeline Model

```
Data Mentah → Parsing & Cleaning → Filter Outlier (IQR 5–95%)
    → Train-Test Split (80:20)
    → log1p(Harga) sebagai target
    → RobustScaler pada fitur
    → XGBoost + Optuna (300 trials, 5-fold CV)
    → Evaluasi dengan expm1(prediksi)
```

> Prediksi model dalam skala log dan diinvers dengan `numpy.expm1()` untuk mendapatkan harga dalam Rupiah.

## Dependensi

| Package | Versi |
|---------|-------|
| streamlit | 1.52.2 |
| pandas | 2.3.3 |
| numpy | 2.3.4 |
| scikit-learn | 1.7.2 |
| xgboost | 3.1.2 |
