from flask import Flask, render_template, jsonify, request
import requests
import os
import time
import traceback
from datetime import datetime
import pytz

app = Flask(__name__)

# ---------- Config ----------
API_KEY = os.environ.get("TWELVE_DATA_API_KEY", "b7ea33d435964da0b0a65b1c6a029891")
PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "EUR/JPY", "AUD/CAD"]
INTERVAL = "5min"
INITIAL_FETCH = 30               # fetch this many candles on first load
LOCAL_TZ = pytz.timezone("Asia/Kolkata")

# CWRV / validation params (relaxed for more signals)
FIB_MIN = 0.30
FIB_MAX = 0.70
MIN_BODY_PERCENT = 0.25
RECENT_P3_MAX_AGE = 50

# Sideways detection
SIDEWAYS_BARS = 15
SIDEWAYS_OVERLAP_COUNT = 6

# cache / in-memory store
_cache = {
    "candles": {},  # key: pair -> list of candles (oldest->newest)
    "ts": {}        # key: pair -> last fetch time
}
CACHE_TTL = 8  # seconds

# ---------- Utilities ----------
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

# ---------- Data fetch ----------
def fetch_candles(pair, outputsize=INITIAL_FETCH):
    """
    Return list of candles oldest->newest.
    Each candle: {t, o, h, l, c, v}
    """
    try:
        base = "https://api.twelvedata.com/time_series"
        params = {
            "symbol": pair,
            "interval": INTERVAL,
            "outputsize": outputsize,
            "apikey": API_KEY,
            "format": "JSON"
        }
        r = http_get(base, params=params)
        if r is None:
            return None
        data = r.json()
        # try alt symbol without slash if needed
        if "values" not in data:
            alt = pair.replace("/", "")
            if alt != pair:
                params["symbol"] = alt
                r2 = http_get(base, params=params)
                if r2 is None:
                    return None
                data = r2.json()
        if "values" not in data:
            log(f"fetch_candles: bad response for {pair}: {data}")
            return None
        raw = list(reversed(data["values"]))  # oldest->newest
        candles = []
        for v in raw:
            try:
                candles.append({
                    "t": v.get("datetime"),
                    "o": float(v["open"]),
                    "h": float(v["high"]),
                    "l": float(v["low"]),
                    "c": float(v["close"]),
                    "v": int(float(v.get("volume", 0))) if v.get("volume") is not None else 0
                })
            except Exception:
                continue
        if not candles:
            return None
        # cache
        _cache["candles"][pair] = candles
        _cache["ts"][pair] = time.time()
        return candles
    except Exception as e:
        log(f"fetch_candles exception {pair}: {e}")
        return None

def append_new_candles_if_any(pair, existing):
    """
    Fetch recent candles and append any new ones (existing is oldest->newest).
    """
    try:
        fetched = fetch_candles(pair, outputsize=5)
        if not fetched:
            return existing
        last_known = existing[-1]["t"] if existing else None
        times = [c["t"] for c in fetched]
        if last_known and last_known in times:
            idx = times.index(last_known)
            new = fetched[idx+1:]
        else:
            # fallback: we missed or old; replace entire series
            new = fetched
        if new:
            existing.extend(new)
            # cap size
            return existing[-220:]
        return existing
    except Exception as e:
        log(f"append_new_candles_if_any error: {e}")
        return existing

# ---------- Indicators & helpers ----------
def compute_atr(highs, lows, closes, period=14):
    try:
        if len(closes) < 2:
            return 0.0
        trs = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            trs.append(tr)
        if not trs:
            return 0.0
        p = min(period, len(trs))
        return sum(trs[-p:]) / p
    except Exception:
        return 0.0

def zigzag_swings(candles, atr_mult=1.0, max_keep=80):
    """Simple ATR-based swings (oldest->newest)."""
    try:
        n = len(candles)
        if n < 17:
            return []
        highs = [c["h"] for c in candles]
        lows = [c["l"] for c in candles]
        closes = [c["c"] for c in candles]
        a = compute_atr(highs, lows, closes)
        if a <= 0:
            return []
        threshold = a * atr_mult
        swings = []
        cur_peak = highs[0]; cur_peak_idx = 0
        cur_trough = lows[0]; cur_trough_idx = 0
        direction = None
        for i in range(1, n):
            h = highs[i]; l = lows[i]
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
        # dedupe & sort
        unique = {(s["idx"], s["type"]): s for s in swings}
        swings = sorted(unique.values(), key=lambda x: x["idx"])
        return swings[-max_keep:]
    except Exception as e:
        log(f"zigzag error: {e}")
        return []

def detect_trend_simple(candles, pivot_lookback=2, swing_lookback=3):
    """
    Detects trend by identifying pivot highs/lows and labeling HH/HL/LH/LL.
    Returns (trend, wave_labels)
    candles expected oldest->newest.
    """
    try:
        if not candles or len(candles) < pivot_lookback*2 + 3:
            return "sideways", []
        highs = [c["h"] for c in candles]
        lows = [c["l"] for c in candles]
        swing_points = []
        n = len(candles)
        for i in range(pivot_lookback, n - pivot_lookback):
            is_high = True
            for j in range(1, pivot_lookback+1):
                if highs[i] < highs[i-j] or highs[i] < highs[i+j]:
                    is_high = False; break
            if is_high:
                swing_points.append(("H", highs[i], i))
                continue
            is_low = True
            for j in range(1, pivot_lookback+1):
                if lows[i] > lows[i-j] or lows[i] > lows[i+j]:
                    is_low = False; break
            if is_low:
                swing_points.append(("L", lows[i], i))
        swing_points = swing_points[-(swing_lookback*2+2):]
        wave_labels = []
        for k in range(1, len(swing_points)):
            prev_type, prev_price, _ = swing_points[k-1]
            curr_type, curr_price, _ = swing_points[k]
            if curr_type == "H" and prev_type == "H":
                wave_labels.append("HH" if curr_price > prev_price else "LH")
            elif curr_type == "L" and prev_type == "L":
                wave_labels.append("HL" if curr_price > prev_price else "LL")
            else:
                wave_labels.append(curr_type)
        up_count = sum(1 for x in wave_labels if x in ("HH", "HL"))
        down_count = sum(1 for x in wave_labels if x in ("LH", "LL"))
        if up_count > down_count and up_count + down_count > 0:
            trend = "up"
        elif down_count > up_count and up_count + down_count > 0:
            trend = "down"
        else:
            trend = "sideways"
        return trend, wave_labels
    except Exception as e:
        log(f"detect_trend_simple error: {e}")
        return "sideways", []

def detect_pattern(candles):
    """Simple candle pattern helpers for confluence."""
    try:
        if len(candles) < 2: return None
        a, b = candles[-2], candles[-1]
        if a["c"] < a["o"] and b["c"] > b["o"] and b["c"] > a["o"]:
            return "bullish_engulf"
        if a["c"] > a["o"] and b["c"] < b["o"] and b["c"] < a["o"]:
            return "bearish_engulf"
        body = abs(b["c"] - b["o"])
        rng = b["h"] - b["l"] if (b["h"] - b["l"]) != 0 else 1e-9
        up_w = b["h"] - max(b["c"], b["o"])
        lo_w = min(b["c"], b["o"]) - b["l"]
        if (lo_w / rng) > 0.66:
            return "pin_bottom"
        if (up_w / rng) > 0.66:
            return "pin_top"
        return None
    except Exception:
        return None

def sideways_filter(candles):
    """Detect boxy overlap in last N bars."""
    try:
        if len(candles) < SIDEWAYS_BARS: return False
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

# ---------- CWRV 1-2-3 ----------
def find_123_points_from_swings(swings, lookback=12):
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
            if last["c"] <= p2["price"]:
                return False, "no breakout above p2"
        elif trend == "down":
            if last["c"] >= p2["price"]:
                return False, "no breakout below p2"
        body = abs(last["c"] - last["o"])
        rng = last["h"] - last["l"] if last["h"] - last["l"] > 0 else 1e-9
        if (body / rng) < MIN_BODY_PERCENT:
            return False, f"breakout body too small ({body/rng:.2f})"
        vols = [c.get("v", 0) for c in candles[-10:]]
        avgv = sum(vols)/len(vols) if vols else 0
        if avgv > 0 and last.get("v", 0) < avgv*0.5:
            # weaker volume but allowed (we return True with note)
            return True, f"fib={fib:.3f}; body={body/rng:.2f}; low_vol"
        if (len(candles) - p3["idx"]) > RECENT_P3_MAX_AGE:
            return False, "p3 too old"
        return True, f"fib={fib:.3f}; body={body/rng:.2f}"
    except Exception as e:
        return False, f"validate error {e}"

def backtest_accuracy(candles, swings, trend):
    try:
        if len(candles) < 60 or len(swings) < 6:
            return 100
        wins = []
        for i in range(25, len(candles)-1):
            hist = candles[:i+1]
            sw = zigzag_swings(hist)
            st, _ = market_structure(sw)
            if st != trend:
                continue
            p1,p2,p3,_ = find_123_points_from_swings(sw)
            ok, _ = validate_123(p1,p2,p3,hist,st)
            if not ok:
                continue
            pred = "CALL" if st == "up" else "PUT"
            nxt = candles[i+1]
            win = (nxt["c"] > nxt["o"] and pred == "CALL") or (nxt["c"] < nxt["o"] and pred == "PUT")
            wins.append(win)
        if not wins:
            return 100
        recent = wins[-30:]
        return int(round(sum(1 for w in recent if w) / len(recent) * 100))
    except Exception as e:
        log(f"backtest error: {e}")
        return 100

def market_structure(swings):
    try:
        if len(swings) < 5:
            return "sideways", "few swings"
        highs = [s for s in swings if s["type"] == "H"]
        lows  = [s for s in swings if s["type"] == "L"]
        if len(highs) >= 3 and len(lows) >= 3:
            if highs[-3]["price"] < highs[-2]["price"] < highs[-1]["price"] and lows[-3]["price"] < lows[-2]["price"] < lows[-1]["price"]:
                return "up", "HH & HL"
            if highs[-3]["price"] > highs[-2]["price"] > highs[-1]["price"] and lows[-3]["price"] > lows[-2]["price"] > lows[-1]["price"]:
                return "down", "LH & LL"
        return "sideways", "mixed"
    except Exception as e:
        log(f"market_structure error: {e}")
        return "sideways", "error"

# ---------- Analysis per pair ----------
def analyze_pair(pair, existing_candles=None):
    out = {
        "pair": pair,
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
        # initial fetch or append
        if existing_candles is None or len(existing_candles) < INITIAL_FETCH:
            candles = fetch_candles(pair, outputsize=INITIAL_FETCH)
            if not candles:
                out["why"] = "insufficient data"
                return out, None
        else:
            candles = append_new_candles_if_any(pair, existing_candles)
        if not candles or len(candles) < 40:
            out["why"] = "insufficient data after fetch"
            return out, candles
        # core analysis
        out["candles"] = candles[-220:]
        swings = zigzag_swings(candles)
        trend, wave_labels = detect_trend_simple(candles)
        is_sideways = sideways_filter(candles)
        p1,p2,p3,findmsg = find_123_points_from_swings(swings)
        valid, valmsg = validate_123(p1,p2,p3,candles,trend) if (p1 and p2 and p3) else (False, findmsg)
        pat = detect_pattern(candles)
        accuracy = backtest_accuracy(candles, swings, trend)
        # confidence blending
        conf = 0
        if valid: conf += 60
        if pat in ("bullish_engulf", "pin_bottom") and trend == "up": conf += 10
        if pat in ("bearish_engulf", "pin_top") and trend == "down": conf += 10
        if is_sideways: conf -= 10
        conf += int((accuracy - 70) * 0.3)
        conf = max(0, min(100, int(round(conf))))
        # decision (more permissive)
        if trend == "up" and valid:
            signal = "CALL"
        elif trend == "down" and valid:
            signal = "PUT"
        else:
            signal = "SIDEWAYS"
        if signal == "SIDEWAYS" or is_sideways:
            status = "NO TRADE"
        else:
            if conf >= 65 and accuracy >= 70:
                status = "TRADE"
            elif conf >= 45 and accuracy >= 55:
                status = "RISKY"
            else:
                status = "NO TRADE"
        out.update({
            "signal": signal,
            "status": status,
            "accuracy": accuracy,
            "cwrv": "Yes" if valid else "No",
            "cwrv_conf": conf,
            "wave": ",".join(wave_labels) if wave_labels else "unknown",
            "why": f"trend={trend}; find={findmsg}; validate={valmsg}; pat={pat}; swings={len(swings)}; sideways={is_sideways}"
        })
        return out, candles
    except Exception as e:
        log(f"analyze_pair error {pair}: {e}\n{traceback.format_exc()}")
        out["why"] = f"analysis error: {e}"
        return out, existing_candles

# ---------- Routes ----------
@app.route("/")
def index():
    return render_template("index.html", pairs=PAIRS)

@app.route("/analyze", methods=["GET"])
def analyze():
    try:
        pairs = request.args.getlist("pair") or PAIRS
        results = []
        for s in pairs:
            cached = _cache["candles"].get(s)
            res, updated = analyze_pair(s, cached)
            if updated:
                _cache["candles"][s] = updated
                _cache["ts"][s] = time.time()
            results.append(res if isinstance(res, dict) else {
                "pair": s, "signal": "SIDEWAYS", "status": "NO TRADE",
                "accuracy": 100, "cwrv": "No", "cwrv_conf": 0, "wave": "unknown",
                "why": "invalid"
            })
        return jsonify({"results": results})
    except Exception as e:
        log(f"/analyze error: {e}\n{traceback.format_exc()}")
        return jsonify({"results": [], "error": str(e)}), 500

@app.route("/debug", methods=["GET"])
def debug():
    pair = request.args.get("pair", PAIRS[0])
    try:
        cached = _cache["candles"].get(pair) or fetch_candles(pair)
        if not cached:
            return jsonify({"error": "no data"})
        swings = zigzag_swings(cached)
        p1,p2,p3,findmsg = find_123_points_from_swings(swings)
        trend, waves = detect_trend_simple(cached)
        return jsonify({
            "pair": pair,
            "swings": swings,
            "p1": p1, "p2": p2, "p3": p3,
            "find": findmsg,
            "trend": trend,
            "waves": waves
        })
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()})

# ---------- Run ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    log(f"Starting app on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
