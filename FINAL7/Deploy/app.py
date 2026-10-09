import os
import pickle
import numpy as np
import requests as _requests
from datetime import datetime, timezone
from flask import Flask, render_template, request, jsonify

_YF_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'application/json',
}

def _yf_ohlcv(period='3mo', interval='1wk'):
    url = 'https://query1.finance.yahoo.com/v8/finance/chart/GC%3DF'
    params = {'range': period, 'interval': interval, 'includePrePost': 'false'}
    r = _requests.get(url, params=params, headers=_YF_HEADERS, timeout=30)
    r.raise_for_status()
    chart = r.json()['chart']['result'][0]
    timestamps = chart['timestamp']
    q = chart['indicators']['quote'][0]
    rows = []
    for i, ts in enumerate(timestamps):
        o, h, l, c, v = q['open'][i], q['high'][i], q['low'][i], q['close'][i], q['volume'][i]
        if None in (o, h, l, c):
            continue
        rows.append({
            'date': datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d'),
            'open': round(float(o), 2),
            'high': round(float(h), 2),
            'low':  round(float(l), 2),
            'close': round(float(c), 2),
            'volume': int(v) if v is not None else 0,
        })
    return rows

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _lstm_layer(X_seq, W, U, b, return_sequences=False):
    units = W.shape[1] // 4
    h = np.zeros(units)
    c = np.zeros(units)
    outputs = []
    for t in range(X_seq.shape[0]):
        z = X_seq[t] @ W + h @ U + b
        i = _sigmoid(z[:units])
        f = _sigmoid(z[units:2*units])
        c_hat = np.tanh(z[2*units:3*units])
        o = _sigmoid(z[3*units:])
        c = f * c + i * c_hat
        h = o * np.tanh(c)
        if return_sequences:
            outputs.append(h.copy())
    return np.array(outputs) if return_sequences else h

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

_w = np.load(os.path.join(MODEL_DIR, 'lstm_weights.npz'))
_lstm1_W, _lstm1_U, _lstm1_b = _w['lstm1_W'], _w['lstm1_U'], _w['lstm1_b']
_lstm2_W, _lstm2_U, _lstm2_b = _w['lstm2_W'], _w['lstm2_U'], _w['lstm2_b']
_dense_W, _dense_b           = _w['dense_W'], _w['dense_b']

def _model_predict(x_scaled):
    h_seq   = _lstm_layer(x_scaled, _lstm1_W, _lstm1_U, _lstm1_b, return_sequences=True)
    h_final = _lstm_layer(h_seq,    _lstm2_W, _lstm2_U, _lstm2_b, return_sequences=False)
    return float(h_final @ _dense_W + _dense_b)

def _batch_predict(X_seqs):
    return np.array([_model_predict(x) for x in X_seqs])

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
_ohlc_cache          = None
_ohlc_cache_time     = 0
_history_cache       = None
_history_cache_time  = 0
CACHE_TTL            = 300
HISTORY_CACHE_TTL    = 600


@app.route('/')
def index():
    return render_template('index.html', meta=meta)


@app.route('/auto-ohlc')
def auto_ohlc():
    global _ohlc_cache, _ohlc_cache_time
    try:
        if _ohlc_cache and (_time.time() - _ohlc_cache_time) < CACHE_TTL:
            return jsonify(**_ohlc_cache)

        rows = _yf_ohlcv(period='3mo', interval='1wk')
        required_candles = window_size + 2
        if len(rows) < required_candles:
            return jsonify(status='error', message=f'Data tidak cukup dari Yahoo Finance (butuh minimal {required_candles} minggu, dapat {len(rows)})')

        sequence = rows[-required_candles:]
        history  = rows[-6:-1]
        current  = rows[-1]
        lag1     = rows[-2]
        lag2     = rows[-3]

        result = dict(
            status='ok',
            date=current['date'],
            open=current['open'],
            high=current['high'],
            low=current['low'],
            close=current['close'],
            volume=current['volume'],
            close_lag1=lag1['close'],
            close_lag2=lag2['close'],
            ohlc_sequence=sequence,
            history=history,
        )
        _ohlc_cache      = result
        _ohlc_cache_time = _time.time()

        return jsonify(**result)
    except Exception as e:
        return jsonify(status='error', message=f'Yahoo Finance error: {e}')


@app.route('/price-history')
def price_history():
    try:
        rows = _yf_ohlcv(period='6mo', interval='1wk')
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
        lag1   = candles[i - 1]
        lag2   = candles[i - 2]
        row_ffr_prev = ffr_prev if i == len(candles) - 1 else ffr
        rows.append(build_feature_row(
            float(candle['open']), float(candle['high']), float(candle['low']),
            float(candle['close']), float(lag1['close']), float(lag2['close']),
            ffr, row_ffr_prev, nfp,
        )[0])
    if len(rows) != window_size:
        raise ValueError(f'Ukuran sequence tidak sesuai: {len(rows)}')
    return np.array(rows, dtype=float), candles[-1], candles[-2], candles[-3]


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        ffr      = float(data['ffr'])
        ffr_prev = float(data['ffr_prev'])
        nfp      = float(data['nfp'])
        window_raw, current_candle, lag1_candle, lag2_candle = build_live_window(
            data.get('ohlc_sequence'), ffr, ffr_prev, nfp,
        )

        close_this   = float(current_candle['close'])
        close_prev   = float(lag1_candle['close'])

        window = scaler_X.transform(window_raw)
        pred_scaled_val = _model_predict(window)
        pred_ratio = float(scaler_y.inverse_transform([[pred_scaled_val]])[0][0])
        pred_price = close_prev * pred_ratio

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


@app.route('/model-history')
def model_history():
    global _history_cache, _history_cache_time
    try:
        if _history_cache and (_time.time() - _history_cache_time) < HISTORY_CACHE_TTL:
            return jsonify(**_history_cache)

        import pandas as pd

        FRED_API_KEY = os.environ.get('FRED_API_KEY')
        if not FRED_API_KEY:
            return jsonify(status="error", message="FRED_API_KEY belum diset di environment atau file .env")

        def get_fred_series(series_id, start="2024-01-01"):
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {"series_id": series_id, "api_key": FRED_API_KEY, "file_type": "json", "observation_start": start}
            r = _requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            df = pd.DataFrame(r.json()["observations"])
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            return df[["date", "value"]].dropna().sort_values("date").reset_index(drop=True)

        raw_rows = _yf_ohlcv(period='8mo', interval='1wk')
        if len(raw_rows) < 20:
            return jsonify(status="error", message="Data Yahoo Finance tidak cukup")

        ohlc_df = pd.DataFrame(raw_rows)
        ohlc_df["date"] = pd.to_datetime(ohlc_df["date"])

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

        df["close_lag1"]  = df["close"].shift(1)
        df["close_lag2"]  = df["close"].shift(2)
        df["open_ratio"]  = df["open"] / df["close_lag1"]
        df["high_ratio"]  = df["high"] / df["close_lag1"]
        df["low_ratio"]   = df["low"]  / df["close_lag1"]
        df["range_ratio"] = (df["high"] - df["low"]) / df["close_lag1"]
        df["ffr_chg"]     = df["ffr"].diff().fillna(0)
        df["ret_this"]    = df["close"] / df["close_lag1"]
        df["ret_lag2"]    = df["close_lag1"] / df["close_lag2"]
        df["y_ratio"]     = df["close"].shift(-1) / df["close_lag1"]
        df = df.dropna().reset_index(drop=True)

        if len(df) < window_size + 1:
            return jsonify(status="error", message=f"Data terlalu sedikit ({len(df)} baris)")

        X_raw    = df[feature_cols].values
        X_scaled = scaler_X.transform(X_raw)

        X_seqs = np.array([X_scaled[i - window_size:i] for i in range(window_size, len(X_scaled))])
        pred_scaled_batch = _batch_predict(X_seqs)
        pred_ratio = scaler_y.inverse_transform(pred_scaled_batch.reshape(-1, 1)).flatten()

        close_ref = df["close_lag1"].iloc[window_size:].values

        rows = []
        for i in range(len(pred_ratio)):
            actual_close_next = float(df["close"].iloc[window_size + i])
            pred_price = float(close_ref[i] * pred_ratio[i])
            abs_error = abs(pred_price - actual_close_next)
            abs_error_pct = (abs_error / actual_close_next * 100) if actual_close_next else 0
            pred_dir   = "Naik" if pred_ratio[i] > 1.0 else "Turun"
            actual_dir = "Naik" if actual_close_next > float(close_ref[i]) else "Turun"
            status     = "Benar" if pred_dir == actual_dir else "Salah"
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

        last_10  = rows[-10:]
        total    = len(last_10)
        correct  = sum(1 for r in last_10 if r["status"] == "Benar")
        accuracy = (correct / total * 100) if total else 0
        mae      = (sum(r["abs_error"] for r in last_10) / total) if total else 0
        mape     = (sum(r["abs_error_pct"] for r in last_10) / total) if total else 0
        rmse     = (sum(r["abs_error"] ** 2 for r in last_10) / total) ** 0.5 if total else 0

        res_data = dict(
            status="ok", rows=last_10, total=total, correct=correct,
            accuracy=round(accuracy, 2), mae=round(mae, 2),
            mape=round(mape, 2), rmse=round(rmse, 2),
        )
        _history_cache      = res_data
        _history_cache_time = _time.time()
        return jsonify(**res_data)
    except Exception as e:
        return jsonify(status="error", message=f"model-history error: {e}")


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(debug=False, host='0.0.0.0', port=port)
