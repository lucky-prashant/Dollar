from flask import Flask, render_template, jsonify
import requests
from datetime import datetime
import pytz

app = Flask(__name__)

API_KEY = "b7ea33d435964da0b0a65b1c6a029891"
PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "EUR/JPY", "AUD/USD"]

# Accuracy tracking
total_trades = 0
correct_trades = 0

def fetch_candles(symbol, interval="5min", count=30):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={count}&apikey={API_KEY}"
    r = requests.get(url)
    data = r.json()
    if "values" not in data:
        return None
    candles = list(reversed(data["values"]))
    return candles

def detect_cwrv_123(candles):
    # Simple placeholder detection logic
    if len(candles) < 5:
        return None, False
    closes = [float(c["close"]) for c in candles]
    if closes[-1] > closes[-2] > closes[-3]:
        return "CALL", True
    elif closes[-1] < closes[-2] < closes[-3]:
        return "PUT", True
    return "SIDEWAYS", False

def detect_elliott_wave(candles):
    closes = [float(c["close"]) for c in candles]
    # Very simplified wave detection
    if closes[-1] > closes[-2] > closes[-3]:
        return "Wave 3"
    elif closes[-1] < closes[-2] < closes[-3]:
        return "Wave C"
    return "Wave ?"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze")
def analyze():
    global total_trades, correct_trades
    results = []
    for pair in PAIRS:
        candles = fetch_candles(pair)
        if not candles:
            results.append({
                "pair": pair,
                "signal": "NO DATA",
                "trade": "No Trade",
                "accuracy": 0,
                "cwrv": "Not detected",
                "wave": "N/A"
            })
            continue

        signal, detected = detect_cwrv_123(candles)
        wave = detect_elliott_wave(candles)

        trade_status = "Take Trade" if detected and signal != "SIDEWAYS" else "No Trade"
        cwrv_status = "Detected" if detected else "Not detected"

        # Accuracy calc example: assume correct if detected and matches a mock rule
        total_trades += 1
        if detected:
            correct_trades += 1
        accuracy = round((correct_trades / total_trades) * 100, 2)

        results.append({
            "pair": pair,
            "signal": signal,
            "trade": trade_status,
            "accuracy": accuracy,
            "cwrv": cwrv_status,
            "wave": wave
        })

    return jsonify(results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
