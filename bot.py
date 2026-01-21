import os
import time
import json
import requests
import numpy as np
import pandas as pd
from datetime import datetime
import mplfinance as mpf
from dotenv import load_dotenv

# TensorFlow-Setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ================== CONFIG & ENV ==================
load_dotenv() # Sucht nach einer Datei namens .env

TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY")
TELEGRAM_TOKEN     = os.getenv("TELEGRAM_TOKEN")
# Lädt CHAT_IDS als Liste (z.B. "123,456" -> ["123", "456"])
CHAT_IDS = [id.strip() for id in os.getenv("CHAT_IDS", "").split(",") if id.strip()]

SYMBOLS = {
    "BTC/USD": "Bitcoin",
    "ETH/USD": "Ethereum",
    "XRP/USD": "Ripple",
    "SOL/USD": "Solana",
    "TRX/USD": "Tron",
}

INTERVAL            = "1h"
SLEEP_MINUTES       = 60
OUTPUTSIZE          = 500
MODEL_EPOCHS        = 50
MODEL_BATCH_SIZE    = 16
RETRAIN_EVERY_CYCLE = 50
LOG_FILE            = "prediction_log_crypto_1h.json"

# API-Limit-Schutz (TwelveData Free: 8/min)
MAX_REQUESTS_PER_MIN = 7
_request_timestamps = []

# ================== UTILS ==================
def safe_filename(symbol: str) -> str:
    return symbol.replace("/", "_")

def rate_limit_guard():
    global _request_timestamps
    now = time.time()
    _request_timestamps = [t for t in _request_timestamps if now - t < 60.0]
    if len(_request_timestamps) >= MAX_REQUESTS_PER_MIN:
        sleep_for = 60.0 - (now - _request_timestamps[0]) + 0.2
        if sleep_for > 0:
            print(f"⏳ API-Limit nahe – pausiere {sleep_for:.1f}s …")
            time.sleep(sleep_for)
        now = time.time()
        _request_timestamps = [t for t in _request_timestamps if now - t < 60.0]

def twelvedata_get(path: str, params: dict):
    rate_limit_guard()
    url = f"https://api.twelvedata.com/{path}"
    r = requests.get(url, params=params, timeout=15)
    _request_timestamps.append(time.time())
    return r

# ================== DATA & FEATURES ==================
def get_historical_data(symbol: str, interval: str = INTERVAL, outputsize: int = OUTPUTSIZE):
    params = {
        "symbol": symbol, "interval": interval,
        "apikey": TWELVEDATA_API_KEY, "format": "JSON", "outputsize": outputsize,
    }
    try:
        r = twelvedata_get("time_series", params)
        data = r.json()
        if "values" in data:
            df = pd.DataFrame(data["values"])
            float_cols = ["open", "high", "low", "close"]
            if "volume" in df.columns: float_cols.append("volume")
            df[float_cols] = df[float_cols].astype(float)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df.sort_values("datetime", inplace=True)
            return df
        else:
            print(f"❌ Fehler bei {symbol}: {data.get('message')}")
    except Exception as e:
        print(f"❌ API-Fehler bei {symbol}: {e}")
    return None

def compute_rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = -delta.clip(upper=0).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def compute_macd(series: pd.Series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    return ema12 - ema26

def compute_features(df: pd.DataFrame):
    closes = df["close"]
    df["sma5"] = closes.rolling(window=5).mean()
    df["sma10"] = closes.rolling(window=10).mean()
    df["rsi"] = compute_rsi(closes)
    df["macd"] = compute_macd(closes)
    df["volatility"] = df["high"] - df["low"]
    df["label"] = (df["close"].shift(-1) > df["close"]).astype(int)
    return df.dropna()

def indicator_signals(df: pd.DataFrame):
    sig = {}
    sig["SMA"] = "📈 Bullish" if df["sma5"].iloc[-1] > df["sma10"].iloc[-1] else "📉 Bearish"
    rsi_last = df["rsi"].iloc[-1]
    sig["RSI"] = "📈 Bullish" if rsi_last < 30 else ("📉 Bearish" if rsi_last > 70 else "⚖️ Neutral")
    sig["MACD"] = "📈 Bullish" if df["macd"].iloc[-1] > 0 else "📉 Bearish"
    sig["Volatility"] = df["volatility"].iloc[-1]
    return sig

# ================== CHART ==================
def generate_candlestick_chart(df: pd.DataFrame, symbol: str):
    df_plot = df.copy().set_index("datetime").tail(50)
    chart_file = f"chart_{safe_filename(symbol)}.png"
    style = mpf.make_mpf_style(base_mpf_style="default", marketcolors=mpf.make_marketcolors(up='green', down='red', wick='gray', volume='in'))
    mpf.plot(df_plot, type='candle', style=style, volume="volume" in df_plot.columns, mav=(5, 10),
             title=f"{symbol} AI Analysis", savefig=dict(fname=chart_file, dpi=120))
    return chart_file

# ================== MODEL ==================
def build_model(input_dim: int):
    model = keras.Sequential([
        keras.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def train_model(df: pd.DataFrame, features: list, symbol: str):
    X, y = df[features].values, df["label"].values
    model = build_model(len(features))
    cb = keras.callbacks.EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
    model.fit(X, y, epochs=MODEL_EPOCHS, batch_size=MODEL_BATCH_SIZE, verbose=0, callbacks=[cb])
    model.save(f"model_{safe_filename(symbol)}_{INTERVAL}.keras")
    return model

def load_model(symbol: str):
    path = f"model_{safe_filename(symbol)}_{INTERVAL}.keras"
    return keras.models.load_model(path) if os.path.exists(path) else None

# ================== TELEGRAM & LOGS ==================
def send_telegram_message(msg: str, chart_path: str = None):
    msg += "\n\n🤖 Investalo AI Engine lernt weiter..."
    for chat_id in CHAT_IDS:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto" if chart_path else f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            data = {"chat_id": chat_id, "caption" if chart_path else "text": msg, "parse_mode": "Markdown"}
            if chart_path:
                with open(chart_path, "rb") as img: requests.post(url, data=data, files={"photo": img}, timeout=20)
            else:
                requests.post(url, data=data, timeout=20)
        except Exception as e: print(f"⚠️ Telegram Error: {e}")

def log_prediction(symbol, prediction, actual):
    entry = {"timestamp": datetime.now().isoformat(), "symbol": symbol, "prediction": int(prediction), "actual": int(actual)}
    data = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f: data = json.load(f)
    data.append(entry)
    with open(LOG_FILE, "w") as f: json.dump(data, f, indent=2)

# ================== MAIN LOOP ==================
if __name__ == "__main__":
    print(f"🔄 Deep Learning Crypto-Bot gestartet - Intervall: {SLEEP_MINUTES} Min")
    cycle, models, features_used, pending_predictions, last_signals = 0, {}, {}, {}, {}

    while True:
        print(f"\n⏰ {datetime.now().strftime('%H:%M:%S')} – Zyklus {cycle + 1}")
        for symbol, name in SYMBOLS.items():
            try:
                df = get_historical_data(symbol)
                if df is None or len(df) < 50: continue
                df = compute_features(df)
                features = ["sma5", "sma10", "rsi", "macd", "volatility"]

                if symbol not in models:
                    models[symbol] = load_model(symbol) or train_model(df, features, symbol)
                    features_used[symbol] = features

                # Prediction & Logging Logic
                latest = df[features].iloc[[-1]].values
                prob = models[symbol].predict(latest, verbose=0)[0][0]
                prediction = 1 if prob >= 0.5 else 0

                if symbol in pending_predictions and "actual_logged" not in pending_predictions[symbol]:
                    if len(df) >= 2:
                        log_prediction(symbol, pending_predictions[symbol]["prediction"], int(df["label"].iloc[-2]))
                        pending_predictions[symbol]["actual_logged"] = True
                
                pending_predictions[symbol] = {"prediction": prediction}
                ml_signal = "🚀 Bullish" if prediction == 1 else "🔥 Bearish"

                if last_signals.get(symbol) != ml_signal:
                    inds = indicator_signals(df)
                    msg = (f"*{name} ({symbol})*\n\n*KI-Signal:* {ml_signal}\n\n📊 *Indikatoren:*\n"
                           f"• SMA: {inds['SMA']}\n• RSI: {inds['RSI']}\n• MACD: {inds['MACD']}")
                    chart = generate_candlestick_chart(df, symbol)
                    send_telegram_message(msg, chart)
                    last_signals[symbol] = ml_signal

                if cycle % RETRAIN_EVERY_CYCLE == 0 and cycle > 0:
                    models[symbol] = train_model(df, features, symbol)

            except Exception as e: print(f"⚠️ Fehler {symbol}: {e}")

        cycle += 1
        time.sleep(SLEEP_MINUTES * 60)
