from flask import Flask, render_template, jsonify, request
import requests, time, traceback, os
from datetime import datetime
import pytz

app = Flask(__name__)

# =================== CONFIG ===================
API_KEY = os.environ.get("TWELVE_DATA_API_KEY", "b7ea33d435964da0b0a65b1c6a029891")
PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "EUR/JPY", "AUD/USD"]
INTERVAL = "5min"
OUTPUTSIZE = 200
LOCAL_TZ = pytz.timezone("Asia/Kolkata")

# ZigZag / ATR
ATR_PERIOD = 14
ZZ_ATR_MULT = 1.0
MAX_SWINGS = 60

# CWRV
FIB_MIN = 0.382
FIB_MAX = 0.618
MIN_BODY_PERCENT = 0.30  # breakout candle body vs range

# Sideways
SIDEWAYS_BARS = 12
SIDEWAYS_OVERLAP_COUNT = 7

# Backtest
BACKTEST_SIGNALS = 30

# Short cache to avoid API spam
CACHE_TTL = 20  # seconds
_cache = {"candles": {}, "ts": {}}


# =================== HELPERS ===================
def log(msg):
    try:
        print(f"[{datetime.utcnow().isoformat()}] {msg}")
    except Exception:
        pass

def http_get(url, params=None, timeout=10):
    try:
        return requests.get(url, params=params, timeout=timeout)
    except Exception as e:
        log(f"http_get error: {e}")
        return None


# =================== DATA ===================
def fetch_candles(symbol, interval=INTERVAL, outputsize=OUTPUTSIZE):
    """
    Twelve Data fetch. Returns oldest->newest candle list with keys t,o,h,l,c.
    Always returns None on failure (never throws).
    """
    try:
        now = time.time()
        if symbol in _cache["candles"] and (now - _cache["ts"].get(symbol, 0)) < CACHE_TTL:
            return _cache["candles"][symbol]

        base = "https://api.twelvedata.com/time_series"
        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "apikey": API_KEY,
            "format": "JSON"
        }
        r = http_get(base, params=params, timeout=12)
        if r is None:
            return None
        data = r.json()

        if "values" not in data:
            # Try without slash (e.g., EURUSD)
            alt = symbol.replace("/", "")
            if alt != symbol:
                params["symbol"] = alt
                r = http_get(base, params=params, timeout=12)
                if r is None:
                    return None
                data = r.json()

        if "values" not in data:
            log(f"fetch_candles: bad response for {symbol}: {data}")
            return None

        raw = list(reversed(data["values"]))  # oldest -> newest
        candles = []
        for v in raw:
            try:
                candles.append({
                    "t": v.get("datetime"),
                    "o": float(v["open"]),
                    "h": float(v["high"]),
                    "l": float(v["low"]),
                    "c": float(v["close"]),
                })
            except Exception:
                continue

        if not candles:
            return None

        _cache["candles"][symbol] = candles
        _cache["ts"][symbol] = now
        return candles
    except Exception as e:
        log(f"fetch_candles exception {symbol}: {e}")
        return None


# =================== INDICATORS ===================
def compute_atr(highs, lows, closes, period=ATR_PERIOD):
    try:
        if len(closes) < 2:
            return 0.0
        trs = []
        for i in range(1, len(closes)):
            tr = max(highs[i]-lows[i