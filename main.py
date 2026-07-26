"""
main.py - Final Project Pengantar Sains Data: Analisis Performa IHSG

3 analisa:
  1. Performa index proxy IHSG multi-timeframe (daily / hourly / minute)
  2. Performa sektor multi-timeframe (7 / 30 / 90 hari bursa terakhir)
  3. Machine Learning: prediksi arah IHSG besok (naik / turun)

Semua output (grafik & ringkasan teks) disimpan ke folder output/.

CATATAN: data historis ini hanya sampai awal Januari 2023 -> murni studi
kasus/latihan machine learning, BUKAN sinyal trading real-time atau
rekomendasi investasi.
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")
import utils

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
REPORT_LINES = []


def log(line=""):
    """Print ke layar sekaligus simpan ke laporan teks."""
    print(line)
    REPORT_LINES.append(line)


def analisa_1_index_multi_timeframe(index_tickers, weights):
    log("\n" + "-" * 70)
    log("[1] PERFORMA INDEX PROXY IHSG - MULTI TIMEFRAME")
    log("-" * 70)

    daily = utils.load_price_folder("Dataset/daily", index_tickers)
    hourly = utils.load_price_folder("Dataset/hourly", index_tickers)
    minute_full = utils.load_price_folder("Dataset/minutes", index_tickers)

    # timeframe minute: ambil 10 hari bursa terakhir saja biar grafik terbaca
    minute = {}
    for code, df in minute_full.items():
        last_dates = sorted(df["timestamp"].dt.date.unique())[-10:]
        minute[code] = df[df["timestamp"].dt.date.isin(last_dates)]

    idx_daily = utils.build_composite_index(daily, weights)
    idx_hourly = utils.build_composite_index(hourly, weights)
    idx_minute = utils.build_composite_index(minute, weights)

    utils.plot_index_multi_timeframe(
        idx_daily, idx_hourly, idx_minute,
        os.path.join(OUTPUT_DIR, "1_index_multi_timeframe.png")
    )

    log(f"DAILY  : {idx_daily.index.min().date()} s.d. {idx_daily.index.max().date()} ({len(idx_daily)} titik)")
    log(f"HOURLY : {idx_hourly.index.min()} s.d. {idx_hourly.index.max()} ({len(idx_hourly)} titik)")
    log(f"MINUTE : {idx_minute.index.min()} s.d. {idx_minute.index.max()} ({len(idx_minute)} titik, 10 hari terakhir)")
    log("Grafik disimpan -> output/1_index_multi_timeframe.png")

    return daily  # dipakai lagi di analisa 3


def analisa_2_sektor_multi_timeframe(sector_tickers, sector_map):
    log("\n" + "-" * 70)
    log("[2] PERFORMA SEKTOR - MULTI TIMEFRAME (7 / 30 / 90 hari bursa terakhir)")
    log("-" * 70)

    daily = utils.load_price_folder("Dataset/daily", sector_tickers)
    _, summary = utils.compute_sector_performance_multi(daily, sector_map, windows=(7, 30, 90))

    utils.plot_sector_performance(summary, os.path.join(OUTPUT_DIR, "2_sector_performance.png"))
    log(summary.round(2).to_string())
    log("Grafik disimpan -> output/2_sector_performance.png")


def analisa_3_prediksi_ihsg(daily_idx_data, weights):
    log("\n" + "-" * 70)
    log("[3] MACHINE LEARNING - PREDIKSI ARAH IHSG BESOK (NAIK / TURUN)")
    log("-" * 70)
    log("Fitur : indikator teknikal (SMA, MACD, RSI, Bollinger %B, momentum,")
    log("        volume relatif) dari index proxy IHSG hari ini.")
    log("Label : 1 jika index proxy BESOK lebih tinggi dari hari ini, 0 jika tidak.")

    idx_df = utils.build_index_dataframe(daily_idx_data, weights)
    dataset = utils.build_index_prediction_dataset(idx_df)

    log(f"\nTotal sampel (hari bursa)     : {len(dataset)}")
    log(f"Proporsi label naik (kelas 1) : {dataset['label_naik'].mean() * 100:.2f}%")

    result = utils.train_evaluate_model(dataset, test_size=0.2)

    log("\nPerbandingan performa model (data test = 20% terakhir berdasarkan tanggal):")
    log(f"{'Model':45s} {'Akurasi':>8s} {'Precision':>10s} {'Recall':>8s} {'F1':>8s} {'AUC':>8s}")
    for name, r in result["results"].items():
        log(f"{name:45s} {r['accuracy']*100:7.2f}% {r['precision']*100:9.2f}% "
            f"{r['recall']*100:7.2f}% {r['f1']*100:7.2f}% {r['auc']:8.3f}")

    best_name = "Random Forest"
    best = result["results"][best_name]
    utils.plot_confusion_matrix(result["y_test"], best["y_pred"], best_name,
                                 os.path.join(OUTPUT_DIR, "3_confusion_matrix.png"))
    utils.plot_feature_importance(best["model"], result["feature_cols"],
                                   os.path.join(OUTPUT_DIR, "4_feature_importance.png"))

    log(f"\nConfusion matrix disimpan  -> output/3_confusion_matrix.png")
    log(f"Feature importance disimpan -> output/4_feature_importance.png")


def main():
    log("=" * 70)
    log("FINAL PROJECT PENGANTAR SAINS DATA - ANALISIS PERFORMA IHSG")
    log("=" * 70)

    daftar = utils.load_daftar_saham("Dataset/DaftarSaham.csv")
    tickers = json.load(open("tickers.json"))
    index_tickers, sector_tickers = tickers["index_tickers"], tickers["sector_tickers"]

    weights = daftar.set_index("Code")["MarketCap"].to_dict()
    sector_map = daftar.set_index("Code")["Sector"].to_dict()

    log(f"\nSaham untuk proxy index IHSG : {len(index_tickers)}")
    log(f"Saham untuk performa sektor  : {len(sector_tickers)}")

    daily_idx_data = analisa_1_index_multi_timeframe(index_tickers, weights)
    analisa_2_sektor_multi_timeframe(sector_tickers, sector_map)
    analisa_3_prediksi_ihsg(daily_idx_data, weights)

    log("\n" + "=" * 70)
    log("CATATAN / DISCLAIMER")
    log("=" * 70)
    log("- Data historis (s.d. awal Jan 2023) untuk latihan machine learning,")
    log("  BUKAN sinyal trading real-time.")
    log("- Akurasi prediksi arah index umumnya mendekati baseline (tebak")
    log("  mayoritas) karena pergerakan index jangka pendek sangat dipengaruhi")
    log("  noise pasar & faktor eksternal di luar indikator teknikal.")
    log("- Proyek ini murni pembelajaran feature engineering + classification")
    log("  dengan scikit-learn, bukan rekomendasi investasi.")

    with open(os.path.join(OUTPUT_DIR, "hasil_analisis.txt"), "w") as f:
        f.write("\n".join(REPORT_LINES))
    print("\nSemua output tersimpan di folder 'output/'.")


if __name__ == "__main__":
    main()
