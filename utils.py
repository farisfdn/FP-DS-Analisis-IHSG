"""
utils.py - Fungsi bantu Final Project Pengantar Sains Data (Analisis IHSG)

Isinya dibagi mengikuti 3 analisa di main.py:
  1. build_composite_index()            -> index proxy IHSG multi-timeframe
  2. compute_sector_performance_multi()  -> performa sektor multi-timeframe
  3. add_technical_indicators(),
     build_index_prediction_dataset(),
     train_evaluate_model()             -> ML prediksi arah IHSG besok
Sisanya (load_*, plot_*) adalah fungsi pendukung.
"""

import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score,
)

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

FEATURE_COLS = ["sma_ratio", "macd", "macd_signal", "rsi_14", "bb_pctb", "momentum_3", "volume_rel"]


# =============================================================================
# LOAD DATA
# =============================================================================

def load_daftar_saham(path="Dataset/DaftarSaham.csv"):
    """Load metadata seluruh emiten (kode, sektor, market cap, dll)."""
    df = pd.read_csv(path)
    df["MarketCap"] = pd.to_numeric(df["MarketCap"], errors="coerce")
    return df


def load_price_csv(path):
    """Load 1 file OHLCV & buang baris close/open/high/low == 0 (jam
    istirahat / tidak ada transaksi -> bukan harga riil, kalau dibiarkan
    index/indikator teknikal bisa 'jatuh' ke 0 secara palsu)."""
    df = pd.read_csv(path, parse_dates=["timestamp"]).sort_values("timestamp")
    invalid = (df[["open", "high", "low", "close"]] == 0).any(axis=1)
    return df[~invalid].reset_index(drop=True)


def load_price_folder(folder, tickers):
    """Load banyak file sekaligus -> dict {kode_saham: dataframe}."""
    data = {}
    for t in tickers:
        fp = os.path.join(folder, f"{t}.csv")
        if os.path.exists(fp):
            data[t] = load_price_csv(fp)
    return data


# =============================================================================
# 1. INDEX PROXY IHSG (market-cap weighted, mirip metodologi IHSG asli)
# =============================================================================

def build_composite_index(price_dict, weights, price_col="close"):
    """Index proxy market-cap weighted, basis 100 di titik awal.
    price_dict: dict {kode: dataframe OHLCV} 1 timeframe (semua daily/
    hourly/minute). weights: dict {kode: bobot market cap}."""
    total_w = sum(weights.get(k, 0) for k in price_dict)
    frames = [
        df.set_index("timestamp")[price_col].rename(code) * (weights.get(code, 0) / total_w)
        for code, df in price_dict.items()
    ]
    combined = pd.concat(frames, axis=1).sort_index().ffill().dropna(how="all")
    index_level = combined.sum(axis=1)
    return index_level / index_level.iloc[0] * 100


def build_index_dataframe(daily_dict, weights):
    """1 dataframe time series index proxy IHSG: 'close' = index level,
    'volume' = total volume harian seluruh saham konstituen (untuk fitur ML)."""
    idx_close = build_composite_index(daily_dict, weights)
    vol_frames = [df.set_index("timestamp")["volume"].rename(c) for c, df in daily_dict.items()]
    idx_volume = pd.concat(vol_frames, axis=1).sort_index().fillna(0).sum(axis=1)
    idx_df = pd.DataFrame({"close": idx_close, "volume": idx_volume}).dropna()
    return idx_df.reset_index().rename(columns={"index": "timestamp"})


# =============================================================================
# 2. PERFORMA SEKTOR MULTI-TIMEFRAME
# =============================================================================

def compute_sector_performance_multi(daily_dict, sector_map, windows=(7, 30, 90)):
    """Return kumulatif tiap saham untuk beberapa jendela waktu (hari bursa),
    dirata-rata per sektor. Return: detail (per saham) & summary (pivot
    Sector x Window, rata-rata return %)."""
    rows = []
    for code, df in daily_dict.items():
        if code not in sector_map:
            continue
        for w in windows:
            if len(df) < w + 1:
                continue
            recent = df.tail(w + 1)
            ret = (recent["close"].iloc[-1] / recent["close"].iloc[0] - 1) * 100
            rows.append({"Code": code, "Sector": sector_map[code], "Window": f"{w}D", "Return_pct": ret})

    detail = pd.DataFrame(rows)
    summary = detail.groupby(["Sector", "Window"])["Return_pct"].mean().reset_index() \
                     .pivot(index="Sector", columns="Window", values="Return_pct")
    ordered_cols = [f"{w}D" for w in windows if f"{w}D" in summary.columns]
    summary = summary[ordered_cols].sort_values(ordered_cols[-1], ascending=False)
    return detail, summary


# =============================================================================
# 3. MACHINE LEARNING - PREDIKSI ARAH IHSG BESOK
# =============================================================================

def add_technical_indicators(df, price_col="close"):
    """Tambah indikator teknikal (SMA, MACD, RSI14, Bollinger %B, momentum,
    volume relatif). Semua hanya pakai data s.d. baris tsb (no look-ahead)."""
    df = df.copy()
    close = df[price_col]

    sma_5, sma_20 = close.rolling(5).mean(), close.rolling(20).mean()
    df["sma_ratio"] = sma_5 / sma_20

    ema_12, ema_26 = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    delta = close.diff()
    avg_gain = delta.clip(lower=0).rolling(14).mean()
    avg_loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi_14"] = 100 - (100 / (1 + avg_gain / avg_loss.replace(0, np.nan)))

    std_20 = close.rolling(20).std()
    upper, lower = sma_20 + 2 * std_20, sma_20 - 2 * std_20
    df["bb_pctb"] = (close - lower) / (upper - lower)

    df["momentum_3"] = close.pct_change(3) * 100
    df["volume_rel"] = df["volume"] / df["volume"].rolling(20).mean().replace(0, np.nan)

    return df


def build_index_prediction_dataset(idx_df):
    """Fitur = indikator teknikal index hari ini. Label = 1 jika index
    BESOK lebih tinggi dari hari ini, 0 jika tidak."""
    df = add_technical_indicators(idx_df)
    df["Date"] = df["timestamp"].dt.date
    df["close_besok"] = df["close"].shift(-1)
    df = df.dropna(subset=["close_besok"] + FEATURE_COLS)

    df["return_besok_pct"] = (df["close_besok"] / df["close"] - 1) * 100
    df["label_naik"] = (df["return_besok_pct"] > 0).astype(int)

    dataset = df[["Date"] + FEATURE_COLS + ["return_besok_pct", "label_naik"]]
    return dataset.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS).reset_index(drop=True)


def train_evaluate_model(dataset, test_size=0.2, random_state=42):
    """Latih Random Forest (model utama) + baseline Dummy & Logistic
    Regression untuk dibandingkan. Split berdasarkan urutan TANGGAL (bukan
    acak) supaya data test benar-benar 'masa depan' dari data train."""
    dataset = dataset.sort_values("Date").reset_index(drop=True)
    X, y = dataset[FEATURE_COLS], dataset["label_naik"]

    split = int(len(dataset) * (1 - test_size))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    scaler = StandardScaler()
    X_train_s, X_test_s = scaler.fit_transform(X_train), scaler.transform(X_test)

    models = {
        "Baseline (Dummy - selalu prediksi mayoritas)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=20, random_state=random_state, n_jobs=-1
        ),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        y_proba = model.predict_proba(X_test_s)[:, 1] if hasattr(model, "predict_proba") else y_pred
        results[name] = {
            "model": model, "y_pred": y_pred, "y_proba": y_proba,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "auc": roc_auc_score(y_test, y_proba) if len(set(y_test)) > 1 else float("nan"),
        }

    return {"results": results, "X_test": X_test, "y_test": y_test, "feature_cols": FEATURE_COLS}


# =============================================================================
# VISUALISASI
# =============================================================================

def plot_index_multi_timeframe(idx_daily, idx_hourly, idx_minute, save_path):
    fig, axes = plt.subplots(3, 1, figsize=(11, 10))
    data = [
        (idx_daily, "DAILY (histori penuh)", "#1f77b4"),
        (idx_hourly, "HOURLY", "#ff7f0e"),
        (idx_minute, "MINUTE (beberapa hari terakhir)", "#2ca02c"),
    ]
    for ax, (series, label, color) in zip(axes, data):
        ax.plot(series.index, series.values, color=color)
        ax.set_title(f"Proxy Index IHSG - Timeframe {label}")
        ax.set_ylabel("Index Level (basis 100)")
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_sector_performance(summary, save_path):
    """summary: pivot table Sector x Window (7D/30D/90D), isi = return %."""
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(summary, annot=True, fmt=".1f", cmap="RdYlGn", center=0,
                cbar_kws={"label": "Return (%)"}, ax=ax)
    ax.set_title("Performa Sektor Multi-Timeframe (rata-rata return %)")
    ax.set_xlabel("Jendela Waktu")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(y_test, y_pred, model_name, save_path):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Prediksi Turun", "Prediksi Naik"],
                yticklabels=["Aktual Turun", "Aktual Naik"], ax=ax)
    ax.set_title(f"Confusion Matrix - {model_name}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_feature_importance(model, feature_cols, save_path):
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    importances.plot(kind="barh", ax=ax, color="#4c72b0")
    ax.set_title("Feature Importance - Random Forest")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)