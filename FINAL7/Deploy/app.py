import os
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(MODEL_DIR, '.env')

if os.path.exists(ENV_PATH):
    with open(ENV_PATH, 'r', encoding='utf-8') as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

EXPECTED_FEATURE_COLS = [
    'open_ratio', 'high_ratio', 'low_ratio', 'range_ratio',
    'ffr', 'ffr_chg', 'nfp',
    'ret_this', 'ret_lag2',
]

import tensorflow as tf

model = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, 'lstm_gold.h5'),
    compile=False
)

with open(os.path.join(MODEL_DIR, 'scaler_X.pkl'), 'rb') as f:
    scaler_X = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'scaler_y.pkl'), 'rb') as f:
    scaler_y = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)

window_size  = meta['window_size']
feature_cols = meta.get('feature_cols', EXPECTED_FEATURE_COLS)
n_features   = len(feature_cols)

if feature_cols != EXPECTED_FEATURE_COLS:
    raise ValueError(f'Urutan feature_cols tidak sesuai artifact model: {feature_cols}')

app = Flask(__name__)

import time as _time
_ohlc_cache      = None
_ohlc_cache_time = 0
CACHE_TTL        = 300


# ── Routes ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html', meta=meta)


@app.route('/auto-ohlc')
def auto_ohlc():
    global _ohlc_cache, _ohlc_cache_time
    try:
        if _ohlc_cache and (_time.time() - _ohlc_cache_time) < CACHE_TTL:
            return jsonify(**_ohlc_cache)

        import yfinance as yf
        ticker = yf.Ticker('GC=F')
        hist   = ticker.history(period='3mo', interval='1wk')
        hist   = hist[['Open', 'High', 'Low', 'Close', 'Volume']].dropna().sort_index()
        required_candles = window_size + 2
        if len(hist) < required_candles:
            return jsonify(status='error', message=f'Data tidak cukup dari yfinance (butuh minimal {required_candles} minggu)')

        sequence = []
        sequence_rows = hist.iloc[-required_candles:]
        for i in range(len(sequence_rows)):
            row = sequence_rows.iloc[i]
            sequence.append({
                'date':   sequence_rows.index[i].strftime('%Y-%m-%d'),
                'open':   round(float(row['Open']), 2),
                'high':   round(float(row['High']), 2),
                'low':    round(float(row['Low']), 2),
                'close':  round(float(row['Close']), 2),
                'volume': int(row['Volume']),
            })

        history      = []
        history_rows = hist.iloc[-6:-1]
        for i in range(len(history_rows)):
            row = history_rows.iloc[i]
            history.append({
                'date':   history_rows.index[i].strftime('%Y-%m-%d'),
                'open':   round(float(row['Open']), 2),
                'high':   round(float(row['High']), 2),
                'low':    round(float(row['Low']), 2),
                'close':  round(float(row['Close']), 2),
                'volume': int(row['Volume']),
            })

        current = hist.iloc[-1]
        lag1    = history_rows.iloc[-1]
        lag2    = history_rows.iloc[-2]

        result = dict(
            status='ok',
            date=current.name.strftime('%Y-%m-%d'),
            open=round(float(current['Open']), 2),
            high=round(float(current['High']), 2),
            low=round(float(current['Low']), 2),
            close=round(float(current['Close']), 2),
            volume=int(current['Volume']),
            close_lag1=round(float(lag1['Close']), 2),
            close_lag2=round(float(lag2['Close']), 2),
            ohlc_sequence=sequence,
            history=history,
        )
        _ohlc_cache      = result
        _ohlc_cache_time = _time.time()

        return jsonify(**result)
    except Exception as e:
        return jsonify(status='error', message=f'yfinance error: {e}')


@app.route('/price-history')
def price_history():
    """Auto-fetch price history with volume for the price history table."""
    try:
        import yfinance as yf
        ticker = yf.Ticker('GC=F')
        hist   = ticker.history(period='6mo', interval='1wk')
        hist   = hist[['Open', 'High', 'Low', 'Close', 'Volume']].dropna().sort_index()

        rows = []
        for i in range(len(hist)):
            row = hist.iloc[i]
            rows.append({
                'date':    hist.index[i].strftime('%Y-%m-%d'),
                'open':    round(float(row['Open']), 2),
                'high':    round(float(row['High']), 2),
                'low':     round(float(row['Low']), 2),
                'close':   round(float(row['Close']), 2),
                'volume':  int(row['Volume']),
            })

        # Most recent first, only last 10 weeks
        rows.reverse()
        return jsonify(status='ok', rows=rows[:10])
    except Exception as e:
        return jsonify(status='error', message=f'price-history error: {e}')


def build_feature_row(open_price, high_price, low_price, close_this, close_lag1,
                      close_lag2, ffr, ffr_prev, nfp):
    if close_lag1 <= 0 or close_lag2 <= 0:
        raise ValueError('Close Lag 1 dan Close Lag 2 harus lebih besar dari 0')

    ffr_chg = ffr - ffr_prev
    return np.array([[
        open_price / close_lag1,
        high_price / close_lag1,
        low_price  / close_lag1,
        (high_price - low_price) / close_lag1,
        ffr,
        ffr_chg,
        nfp,
        close_this / close_lag1,
        close_lag1 / close_lag2,
    ]], dtype=float)


def build_live_window(ohlc_sequence, ffr, ffr_prev, nfp):
    if not isinstance(ohlc_sequence, list):
        raise ValueError('ohlc_sequence harus berupa list candle mingguan')

    required_candles = window_size + 2
    if len(ohlc_sequence) < required_candles:
        raise ValueError(f'Data OHLC tidak cukup untuk sequence {window_size} minggu')

    rows = []
    candles = ohlc_sequence[-required_candles:]
    for i in range(2, len(candles)):
        candle = candles[i]
        lag1 = candles[i - 1]
        lag2 = candles[i - 2]

        row_ffr_prev = ffr_prev if i == len(candles) - 1 else ffr

        rows.append(build_feature_row(
            float(candle['open']),
            float(candle['high']),
            float(candle['low']),
            float(candle['close']),
            float(lag1['close']),
            float(lag2['close']),
            ffr,
            row_ffr_prev,
            nfp,
        )[0])

    if len(rows) != window_size:
        raise ValueError(f'Ukuran sequence tidak sesuai: {len(rows)}')

    return np.array(rows, dtype=float), candles[-1], candles[-2], candles[-3]


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        ffr       = float(data['ffr'])
        ffr_prev  = float(data['ffr_prev'])
        nfp       = float(data['nfp'])
        window_raw, current_candle, lag1_candle, lag2_candle = build_live_window(
            data.get('ohlc_sequence'),
            ffr,
            ffr_prev,
            nfp,
        )

        open_price   = float(current_candle['open'])
        high_price   = float(current_candle['high'])
        low_price    = float(current_candle['low'])
        close_this   = float(current_candle['close'])
        close_prev   = float(lag1_candle['close'])
        close_2w_ago = float(lag2_candle['close'])

        print("\n" + "="*60)
        print("DEBUG PREDICT - Raw Inputs:")
        print(f"  open={open_price}, high={high_price}, low={low_price}")
        print(f"  close={close_this}, close_lag1={close_prev}, close_lag2={close_2w_ago}")
        print(f"  ffr={ffr}, ffr_prev={ffr_prev}, nfp={nfp}")

        window = scaler_X.transform(window_raw)

        pred_scaled = model.predict(
            window.reshape(1, window_size, n_features), verbose=0
        )
        pred_ratio = float(scaler_y.inverse_transform(pred_scaled)[0][0])
        pred_price = close_prev * pred_ratio

        print(f"DEBUG PREDICT - pred_ratio={pred_ratio:.6f}, pred_price=${pred_price:.2f}")
        print("="*60 + "\n")

        # Hitung arah: return dari lag1→pred diapply ke close_this
        lag1_to_pred_return = pred_price - close_prev
        adjusted_price      = close_this + lag1_to_pred_return

        is_up     = bool(adjusted_price > close_this)
        pred_dir  = 'Naik' if is_up else 'Turun'
        delta     = adjusted_price - close_this
        delta_pct = delta / close_this * 100

        return jsonify(
            status='ok',
            pred=round(pred_price, 2),
            adjusted_price=round(adjusted_price, 2),
            pred_ratio=round(pred_ratio, 6),
            is_up=is_up,
            pred_dir=pred_dir,
            last_close=round(close_this, 2),
            close_lag1=round(close_prev, 2),
            delta=round(delta, 2),
            delta_pct=round(delta_pct, 4),
        )
    except Exception as e:
        return jsonify(status='error', message=str(e))


# ── Model History (live backtest) ──────────────────────────────────────────

@app.route('/model-history')
def model_history():
    try:
        import yfinance as yf
        import requests
        import pandas as pd

        FRED_API_KEY = os.environ.get('FRED_API_KEY')
        if not FRED_API_KEY:
            return jsonify(status="error", message="FRED_API_KEY belum diset di environment atau file .env")

        def get_fred_series(series_id, start="2024-01-01"):
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {"series_id": series_id, "api_key": FRED_API_KEY, "file_type": "json", "observation_start": start}
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            df = pd.DataFrame(r.json()["observations"])
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            return df[["date", "value"]].dropna().sort_values("date").reset_index(drop=True)

        ticker = yf.Ticker("GC=F")
        ohlc = ticker.history(period="8mo", interval="1wk")[["Open", "High", "Low", "Close"]].dropna().sort_index()
        if len(ohlc) < 20:
            return jsonify(status="error", message="Data yfinance tidak cukup")

        ohlc_df = ohlc.reset_index()
        ohlc_df.columns = ["date", "open", "high", "low", "close"]
        ohlc_df["date"] = pd.to_datetime(ohlc_df["date"]).dt.tz_localize(None)

        ffr_raw = get_fred_series("DFEDTARU")
        ffr = ffr_raw.rename(columns={"value": "ffr"})

        payems_raw = get_fred_series("PAYEMS")
        payems = payems_raw.rename(columns={"value": "payems"})
        payems["nfp"] = payems["payems"].diff()

        macro = pd.merge(ffr, payems[["date", "nfp"]], on="date", how="outer")
        macro = macro.sort_values("date").reset_index(drop=True)
        macro[["ffr", "nfp"]] = macro[["ffr", "nfp"]].ffill()
        macro = macro.dropna(subset=["ffr", "nfp"]).reset_index(drop=True)

        df = pd.merge_asof(ohlc_df.sort_values("date"), macro.sort_values("date"), on="date", direction="backward")
        df = df.dropna(subset=["ffr", "nfp"]).reset_index(drop=True)

        if len(df) < 10:
            return jsonify(status="error", message="Data tidak cukup setelah merge")

        df["close_lag1"] = df["close"].shift(1)
        df["close_lag2"] = df["close"].shift(2)
        df["open_ratio"] = df["open"] / df["close_lag1"]
        df["high_ratio"] = df["high"] / df["close_lag1"]
        df["low_ratio"] = df["low"] / df["close_lag1"]
        df["range_ratio"] = (df["high"] - df["low"]) / df["close_lag1"]
        df["ffr_chg"] = df["ffr"].diff().fillna(0)
        df["ret_this"] = df["close"] / df["close_lag1"]
        df["ret_lag2"] = df["close_lag1"] / df["close_lag2"]
        df["y_ratio"] = df["close"].shift(-1) / df["close_lag1"]

        df = df.dropna().reset_index(drop=True)

        if len(df) < window_size + 1:
            return jsonify(status="error", message=f"Data terlalu sedikit ({len(df)} baris)")

        X_raw = df[feature_cols].values
        X_scaled = scaler_X.transform(X_raw)

        X_seq = []
        for i in range(window_size, len(X_scaled)):
            X_seq.append(X_scaled[i - window_size : i])

        X_seq = np.array(X_seq)

        pred_scaled = model.predict(X_seq, verbose=0)
        pred_ratio = scaler_y.inverse_transform(pred_scaled).flatten()

        close_ref = df["close_lag1"].iloc[window_size - 1 :].values

        rows = []
        for i in range(len(pred_ratio)):
            actual_close_next = float(df["close"].iloc[window_size + i])
            pred_price = float(close_ref[i] * pred_ratio[i])
            abs_error = abs(pred_price - actual_close_next)
            abs_error_pct = (abs_error / actual_close_next * 100) if actual_close_next else 0
            pred_dir = "Naik" if pred_ratio[i] > 1.0 else "Turun"
            actual_dir = "Naik" if actual_close_next > float(close_ref[i]) else "Turun"
            status = "Benar" if pred_dir == actual_dir else "Salah"

            rows.append({
                "date": str(df["date"].iloc[window_size + i].date()),
                "close_lag1": round(float(close_ref[i]), 2),
                "pred_close": round(pred_price, 2),
                "actual_close": round(actual_close_next, 2),
                "abs_error": round(abs_error, 2),
                "abs_error_pct": round(abs_error_pct, 2),
                "pred_direction": pred_dir,
                "actual_direction": actual_dir,
                "status": status,
            })

        last_10 = rows[-10:]
        total = len(last_10)
        correct = sum(1 for r in last_10 if r["status"] == "Benar")
        accuracy = (correct / total * 100) if total else 0
        mae = (sum(r["abs_error"] for r in last_10) / total) if total else 0
        mape = (sum(r["abs_error_pct"] for r in last_10) / total) if total else 0
        rmse = (sum(r["abs_error"] ** 2 for r in last_10) / total) ** 0.5 if total else 0

        return jsonify(
            status="ok",
            rows=last_10,
            total=total,
            correct=correct,
            accuracy=round(accuracy, 2),
            mae=round(mae, 2),
            mape=round(mape, 2),
            rmse=round(rmse, 2),
        )
    except Exception as e:
        return jsonify(status="error", message=f"model-history error: {e}")


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(debug=False, host='0.0.0.0', port=port)
