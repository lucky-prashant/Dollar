# app.py - robust drop-in replacement
from flask import Flask, render_template, jsonify, request
import requests, time, math, traceback
from datetime import datetime
import pytz, os

app = Flask(__name__)

# ---------------- CONFIG ----------------
API_KEY = "b7ea33d435964da0b0a65b1c6a029891"
PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "EUR/JPY", "AUD/USD"]
INTERVAL = "5min"
OUTPUTSIZE = 200
LOCAL_TZ = pytz.timezone("Asia/Kolkata")

# ZigZag / ATR params
ATR_PERIOD = 14
ZZ_ATR_MULT = 1.0
MAX_SWINGS = 60

# CWRV params
FIB_MIN = 0.382
FIB_MAX = 0.618
MIN_BODY_PERCENT = 0.30

# Sideways detection
SIDEWAYS_BARS = 12
SIDEWAYS_OVERLAP_COUNT = 7

# Backtest accuracy
BACKTEST_SIGNALS = 30

# Cache
CACHE_TTL = 20
_cache = {"candles": {}, "ts": {}}

# ---------------- Logging helper ----------------
def log(s):
    try:
        print(f"[{datetime.utcnow().isoformat()}] {s}")
    except Exception:
        pass

# ---------------- HTTP helper ----------------
def http_get(url, params=None, timeout=10):
    try:
        return requests.get(url, params=params, timeout=timeout)
    except Exception as e:
        log(f"http_get error: {e}")
        return None

# ---------------- Candles ----------------
def fetch_candles(symbol, interval=INTERVAL, outputsize=OUTPUTSIZE):
    """
    Get candles from TwelveData. Returns oldest->newest list or None.
    """
    try:
        now = time.time()
        # short cache
        cached = _cache["candles"].get(symbol)
        ts = _cache["ts"].get(symbol, 0)
        if cached and (now - ts) < CACHE_TTL:
            return cached

        base = "https://api.twelvedata.com/time_series"
        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "apikey": API_KEY,
            "format": "JSON",
        }
        r = http_get(base, params=params, timeout=10)
        if r is None:
            return None
        data = r.json()
        # fallback without slash if API expects that
        if "values" not in data:
            alt = symbol.replace("/", "")
            if alt != symbol:
                params["symbol"] = alt
                r = http_get(base, params=params, timeout=10)
                if r is None:
                    return None
                data = r.json()
        if "values" not in data:
            log(f"fetch_candles: no 'values' for {symbol} -> {data}")
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
                    "c": float(v["close"])
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

# ---------------- Simple indicators ----------------
def compute_atr(highs, lows, closes, period=ATR_PERIOD):
    try:
        if len(closes) < 2: return 0.0
        trs = []
        for i in range(1, len(closes)):
            tr = max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
            trs.append(tr)
        if not trs: return 0.0
        p = min(period, len(trs))
        return sum(trs[-p:]) / p
    except Exception:
        return 0.0

# ---------------- ZigZag swings ----------------
def zigzag_swings(candles, atr_mult=ZZ_ATR_MULT, max_keep=MAX_SWINGS):
    try:
        n = len(candles)
        if n < ATR_PERIOD + 3:
            return []
        highs = [c["h"] for c in candles]
        lows = [c["l"] for c in candles]
        closes = [c["c"] for c in candles]
        a = compute_atr(highs, lows, closes, ATR_PERIOD)
        if a <= 0:
            return []

        threshold = a * atr_mult
        swings = []
        direction = None
        cur_peak = highs[0]; cur_peak_idx = 0
        cur_trough = lows[0]; cur_trough_idx = 0

        for i in range(1, n):
            h, l = highs[i], lows[i]
            if h >= cur_peak:
                cur_peak = h; cur_peak_idx = i
            if l <= cur_trough:
                cur_trough = l; cur_trough_idx = i

            if direction is None:
                if cur_peak - cur_trough >= threshold:
                    direction = "up" if closes[-1] >= closes[0] else "down"
                    if direction == "up":
                        swings.append({"idx": cur_trough_idx, "type": "L", "price": cur_trough, "time": candles[cur_trough_idx]["t"]})
                    else:
                        swings.append({"idx": cur_peak_idx, "type": "H", "price": cur_peak, "time": candles[cur_peak_idx]["t"]})
                continue

            if direction == "up":
                if cur_peak - l >= threshold:
                    swings.append({"idx": cur_peak_idx, "type": "H", "price": cur_peak, "time": candles[cur_peak_idx]["t"]})
                    direction = "down"
                    cur_trough = l; cur_trough_idx = i
            else:
                if h - cur_trough >= threshold:
                    swings.append({"idx": cur_trough_idx, "type": "L", "price": cur_trough, "time": candles[cur_trough_idx]["t"]})
                    direction = "up"
                    cur_peak = h; cur_peak_idx = i

        if direction == "up":
            swings.append({"idx": cur_peak_idx, "type": "H", "price": cur_peak, "time": candles[cur_peak_idx]["t"]})
        elif direction == "down":
            swings.append({"idx": cur_trough_idx, "type": "L", "price": cur_trough, "time": candles[cur_trough_idx]["t"]})

        # dedupe and sort
        unique = {}
        for s in swings:
            unique[(s["idx"], s["type"])] = s
        swings = sorted(unique.values(), key=lambda x: x["idx"])
        return swings[-max_keep:]
    except Exception as e:
        log(f"zigzag error: {e}")
        return []

# ---------------- Market structure ----------------
def market_structure(swings):
    try:
        if len(swings) < 5:
            return "sideways", "few swings"
        highs = [s for s in swings if s["type"] == "H"]
        lows = [s for s in swings if s["type"] == "L"]
        if len(highs) >= 3 and len(lows) >= 3:
            if highs[-3]["price"] < highs[-2]["price"] < highs[-1]["price"] and lows[-3]["price"] < lows[-2]["price"] < lows[-1]["price"]:
                return "up", "HH & HL"
            if highs[-3]["price"] > highs[-2]["price"] > highs[-1]["price"] and lows[-3]["price"] > lows[-2]["price"] > lows[-1]["price"]:
                return "down", "LH & LL"
        return "sideways", "mixed"
    except Exception as e:
        log(f"market_structure error: {e}")
        return "sideways", "error"

# ---------------- Patterns ----------------
def detect_pattern(candles):
    try:
        if not candles or len(candles) < 2: return None
        a, b = candles[-2], candles[-1]
        # engulfing
        if a["c"] < a["o"] and b["c"] > b["o"] and b["c"] > a["o"]:
            return "bullish_engulf"
        if a["c"] > a["o"] and b["c"] < b["o"] and b["c"] < a["o"]:
            return "bearish_engulf"
        # pin
        body = abs(b["c"] - b["o"])
        up_wick = b["h"] - max(b["c"], b["o"])
        lo_wick = min(b["c"], b["o"]) - b["l"]
        if body == 0: body = 1e-9
        if lo_wick > 2*body and lo_wick > up_wick:
            return "pin_bottom"
        if up_wick > 2*body and up_wick > lo_wick:
            return "pin_top"
        return None
    except Exception:
        return None

def sideways_filter(candles):
    try:
        if not candles or len(candles) < SIDEWAYS_BARS: return False
        last = candles[-SIDEWAYS_BARS:]
        overlaps = 0
        for i in range(1, len(last)):
            a, b = last[i-1], last[i]
            a_lo, a_hi = min(a["o"], a["c"]), max(a["o"], a["c"])
            b_lo, b_hi = min(b["o"], b["c"]), max(b["o"], b["c"])
            if (b_lo >= a_lo and b_hi <= a_hi) or (a_lo >= b_lo and a_hi <= b_hi):
                overlaps += 1
        return overlaps >= SIDEWAYS_OVERLAP_COUNT
    except Exception:
        return False

# ---------------- Elliott heuristic ----------------
def elliott_wave_label(swings):
    try:
        if len(swings) < 5: return "unknown"
        seq = swings[-5:]
        types = [s["type"] for s in seq]
        if types == ["L","H","L","H","L"]:
            w1 = seq[1]["price"] - seq[0]["price"]
            w3 = seq[3]["price"] - seq[2]["price"]
            rule2 = seq[2]["price"] > seq[0]["price"]
            rule3 = abs(w3) > abs(w1) * 1.05
            rule4 = seq[4]["price"] > (seq[0]["price"] + (seq[1]["price"] - seq[0]["price"]) * 0.10)
            if rule2 and rule3 and rule4:
                return "5" if seq[-1]["type"] == "L" else "4"
        if types == ["H","L","H","L","H"]:
            w1 = seq[0]["price"] - seq[1]["price"]
            w3 = seq[2]["price"] - seq[3]["price"]
            rule2 = seq[2]["price"] < seq[0]["price"]
            rule3 = abs(w3) > abs(w1) * 1.05
            rule4 = seq[4]["price"] < (seq[0]["price"] - (seq[0]["price"] - seq[1]["price"]) * 0.10)
            if rule2 and rule3 and rule4:
                return "5" if seq[-1]["type"] == "H" else "4"
        return "unknown"
    except Exception as e:
        log(f"elliott error: {e}")
        return "unknown"

# ---------------- CWRV 1-2-3 ----------------
def find_123_points_from_swings(swings, lookback=8):
    try:
        if len(swings) < 3: return None, None, None, "not enough swings"
        start = max(0, len(swings) - lookback)
        for i in range(start, len(swings)-2):
            s1, s2, s3 = swings[i], swings[i+1], swings[i+2]
            if s1["type"] == "L" and s2["type"] == "H" and s3["type"] == "L" and s3["price"] > s1["price"]:
                return s1, s2, s3, "L-H-L"
            if s1["type"] == "H" and s2["type"] == "L" and s3["type"] == "H" and s3["price"] < s1["price"]:
                return s1, s2, s3, "H-L-H"
        return None, None, None, "no 1-2-3"
    except Exception as e:
        return None, None, None, f"error {e}"

def fib_retrace(p1, p2, p3):
    try:
        denom = (p2 - p1)
        if denom == 0: return 0.0
        return abs((p2 - p3) / denom)
    except Exception:
        return 0.0

def validate_123(p1, p2, p3, candles, trend):
    try:
        if not (p1 and p2 and p3): return False, "missing points"
        if not (p1["idx"] < p2["idx"] < p3["idx"]): return False, "bad chronology"
        fib = fib_retrace(p1["price"], p2["price"], p3["price"])
        if not (FIB_MIN <= fib <= FIB_MAX): return False, f"fib {fib:.3f} outside"
        last = candles[-1]
        if trend == "up":
            if last["c"] <= p2["price"]: return False, "no breakout above p2"
        elif trend == "down":
            if last["c"] >= p2["price"]: return False, "no breakout below p2"
        body = abs(last["c"] - last["o"])
        rng = last["h"] - last["l"] if last["h"] - last["l"] > 0 else 1e-9
        if (body / rng) < MIN_BODY_PERCENT: return False, f"body too small {body/rng:.2f}"
        if (len(candles) - p3["idx"]) > 40: return False, "p3 too old"
        return True, f"fib={fib:.3f}; body={body/rng:.2f}"
    except Exception as e:
        return False, f"validate error {e}"

def backtest_accuracy(candles, swings, trend):
    try:
        if len(candles) < 40 or len(swings) < 6: return 100
        wins = []
        for i in range(25, len(candles)-1):
            hist = candles[:i+1]
            sw = zigzag_swings(hist)
            st, _ = market_structure(sw)
            if st != trend: continue
            p1,p2,p3,_ = find_123_points_from_swings(sw)
            ok, _ = validate_123(p1,p2,p3,hist,st)
            if not ok: continue
            pred = "CALL" if st == "up" else "PUT"
            nxt = candles[i+1]
            win = (nxt["c"] > nxt["o"] and pred == "CALL") or (nxt["c"] < nxt["o"] and pred == "PUT")
            wins.append(win)
        if not wins: return 100
        recent = wins[-BACKTEST_SIGNALS:]
        return int(round(sum(1 for w in recent if w) / len(recent) * 100))
    except Exception as e:
        log(f"backtest error: {e}")
        return 100

# ---------------- Analysis per pair ----------------
def analyze_pair(symbol):
    out = {
        "pair": symbol,
        "signal": "SIDEWAYS",
        "status": "NO TRADE",
        "accuracy": 100,
        "cwrv": "No",
        "cwrv_conf": 0,
        "wave": "unknown",
        "candles": [],
        "why": ""
    }
    try:
        candles = fetch_candles(symbol)
        if not candles or len(candles) < 30:
            out["why"] = "insufficient data"
            return out

        out["candles"] = candles[-60:]
        swings = zigzag_swings(candles)
        trend, tr = market_structure(swings)
        is_sideways = sideways_filter(candles)
        p1,p2,p3,findmsg = find_123_points_from_swings(swings)
        valid, valmsg = validate_123(p1,p2,p3,candles,trend) if (p1 and p2 and p3) else (False, findmsg)
        pat = detect_pattern(candles)
        wave = elliott_wave_label(swings)
        accuracy = backtest_accuracy(candles, swings, trend)

        conf = 0
        if valid: conf += 60
        if pat in ("bullish_engulf", "pin_bottom") and trend == "up": conf += 10
        if pat in ("bearish_engulf", "pin_top") and trend == "down": conf += 10
        if wave == "3": conf += 15
        if wave == "5": conf -= 15
        conf += int((accuracy - 70) * 0.3)
        conf = max(0, min(100, int(round(conf))))

        if trend == "up" and valid and not is_sideways:
            signal = "CALL"
        elif trend == "down" and valid and not is_sideways:
            signal = "PUT"
        else:
            signal = "SIDEWAYS"

        if signal == "SIDEWAYS" or is_sideways:
            status = "NO TRADE"
        else:
            if conf >= 70 and accuracy >= 75:
                status = "TRADE"
            elif conf >= 50 and accuracy >= 60:
                status = "RISKY"
            else:
                status = "NO TRADE"

        out.update({
            "signal": signal,
            "status": status,
            "accuracy": accuracy,
            "cwrv": "Yes" if valid else "No",
            "cwrv_conf": conf,
            "wave": wave,
            "why": f"trend={trend} ({tr}); find={findmsg}; validate={valmsg}; pat={pat}; swings={len(swings)}"
        })
        return out
    except Exception as e:
        log(f"analyze_pair error {symbol}: {e}\n{traceback.format_exc()}")
        out["why"] = f"analysis error: {e}"
        return out

# ---------------- Routes ----------------
@app.route("/")
def index():
    try:
        return render_template("index.html", pairs=PAIRS)
    except Exception:
        return "<h1>CWRV App</h1><p>Use /analyze</p>"

@app.route("/analyze", methods=["GET"])
def analyze():
    try:
        results = []
        for s in PAIRS:
            res = analyze_pair(s)
            results.append(res if isinstance(res, dict) else {"pair": s, "signal": "SIDEWAYS", "status": "NO TRADE", "accuracy": 100, "cwrv": "No", "wave": "unknown", "why": "invalid"})
        return jsonify({"results": results})
    except Exception as e:
        log(f"/analyze error: {e}\n{traceback.format_exc()}")
        return jsonify({"results": [], "error": str(e)}), 500

# Debug: expose swings and p1/p2/p3 for a pair
@app.route("/debug", methods=["GET"])
def debug():
    pair = request.args.get("pair", PAIRS[0])
    try:
        candles = fetch_candles(pair)
        if not candles:
            return jsonify({"error": "no data"})
        swings = zigzag_swings(candles)
        p1,p2,p3,findmsg = find_123_points_from_swings(swings)
        return jsonify({
            "pair": pair,
            "swings": swings,
            "p1": p1, "p2": p2, "p3": p3,
            "find": findmsg
        })
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()})

# ---------------- Run ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    log(f"Starting app on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
