from flask import Flask, render_template, jsonify
import requests
from datetime import datetime
import pytz
import math
import traceback

app = Flask(__name__)

API_KEY = "b7ea33d435964da0b0a65b1c6a029891"
PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "EUR/JPY", "AUD/USD"]
TIMEZONE = pytz.timezone("Asia/Kolkata")


# ------------------ Helper Functions ------------------

def fetch_candles(symbol, interval="5min", count=50):
    """Fetch latest candle data with full error handling."""
    try:
        sym = symbol.replace("/", "")
        url = f"https://api.twelvedata.com/time_series?symbol={sym}&interval={interval}&outputsize={count}&apikey={API_KEY}"
        r = requests.get(url, timeout=10)
        data = r.json()

        if "values" not in data:
            return None

        candles = []
        for c in reversed(data["values"]):
            try:
                candles.append({
                    "time": c["datetime"],
                    "open": float(c["open"]),
                    "high": float(c["high"]),
                    "low": float(c["low"]),
                    "close": float(c["close"])
                })
            except Exception:
                continue
        return candles if len(candles) > 0 else None
    except Exception as e:
        print(f"[ERROR] Fetching {symbol} failed: {e}")
        return None


def detect_cwrv_123(candles):
    """Enhanced CWRV 123 pattern detection with safe checks."""
    try:
        if not candles or len(candles) < 10:
            return False, "Not enough candles"

        closes = [c["close"] for c in candles]
        high_idx = closes.index(max(closes[-6:]))
        low_idx = closes.index(min(closes[-6:]))

        reasoning = ""

        if high_idx < low_idx:
            point1 = closes[high_idx]
            point2 = closes[low_idx]
            retracement = (point1 - point2) / point1 * 100
            if 38 <= retracement <= 61:
                breakout = closes[-1] > point1
                reasoning = f"Retracement {retracement:.1f}%, breakout={breakout}"
                if breakout:
                    return True, reasoning
        else:
            point1 = closes[low_idx]
            point2 = closes[high_idx]
            retracement = (point2 - point1) / point2 * 100
            if 38 <= retracement <= 61:
                breakout = closes[-1] < point1
                reasoning = f"Retracement {retracement:.1f}%, breakout={breakout}"
                if breakout:
                    return True, reasoning

        return False, reasoning
    except Exception as e:
        return False, f"CWRV check error: {e}"


def detect_elliott_wave(candles):
    """Detect Elliott wave number with safe checks."""
    try:
        if not candles or len(candles) < 5:
            return 0

        closes = [c["close"] for c in candles]
        waves = 1
        direction = None

        for i in range(1, len(closes)):
            if closes[i] > closes[i - 1]:
                if direction != "up":
                    waves += 1
                direction = "up"
            elif closes[i] < closes[i - 1]:
                if direction != "down":
                    waves += 1
                direction = "down"

        return min(waves, 5)
    except Exception:
        return 0


def analyze_pair(symbol):
    """Analyze one forex pair with all logic and full exception handling."""
    try:
        candles = fetch_candles(symbol)
        if not candles:
            return {"pair": symbol, "error": "No data", "direction": "-", "decision": "-", "accuracy": "-", "cwrv": "-", "wave": "-", "reason": "-"}

        cwrv_detected, cwrv_reason = detect_cwrv_123(candles)
        wave_num = detect_elliott_wave(candles)

        direction = "SIDEWAYS"
        decision = "No Trade"
        accuracy = 0

        if cwrv_detected:
            last_close = candles[-1]["close"]
            prev_close = candles[-2]["close"]
            if last_close > prev_close:
                direction = "CALL"
            else:
                direction = "PUT"
            decision = "Trade"
            accuracy = 85
            if wave_num in [3, 5]:
                accuracy += 10

        return {
            "pair": symbol,
            "direction": direction,
            "decision": decision,
            "accuracy": f"{accuracy}%",
            "cwrv": "Yes" if cwrv_detected else "No",
            "wave": wave_num,
            "reason": cwrv_reason
        }
    except Exception as e:
        return {
            "pair": symbol,
            "direction": "-",
            "decision": "-",
            "accuracy": "-",
            "cwrv": "-",
            "wave": "-",
            "reason": f"Analysis error: {e}"
        }


# ------------------ Routes ------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["GET"])
def analyze():
    try:
        results = []
        for pair in PAIRS:
            results.append(analyze_pair(pair))
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": f"Server error: {e}", "trace": traceback.format_exc()})


# ------------------ Run ------------------

if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=True)
    except Exception as e:
        print(f"[FATAL] Could not start server: {e}")
