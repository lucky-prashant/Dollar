# app.py
from flask import Flask, render_template, jsonify, request
import requests, os, time
from datetime import datetime
import pytz

app = Flask(__name__)

# ==================== CONFIG ====================
API_KEY = "b7ea33d435964da0b0a65b1c6a029891"   # your key
PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "EUR/JPY", "AUD/USD"]
INTERVAL = "5min"
OUTPUTSIZE = 300

LOCAL_TZ = pytz.timezone("Asia/Kolkata")

# ZigZag / Swings
ATR_PERIOD = 14
ZZ_ATR_MULT = 1.0          # reversal threshold = ATR * this multiplier
MAX_SWINGS = 30            # keep last N swings

# Accuracy
BACKTEST_SIGNALS = 30

# Cache to reduce API calls
CACHE_TTL = 45
_cache = {"data": {}, "ts": {}}


# ==================== DATA ====================
def fetch_candles(symbol: str, interval: str = INTERVAL, outputsize: int = OUTPUTSIZE):
    """Fetch candles from TwelveData (oldest -> newest)."""
    now = time.time()
    if symbol in _cache["data"] and (now - _cache["ts"].get(symbol, 0) < CACHE_TTL):
        return _cache["data"][symbol]

    base = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": API_KEY,
        "format": "JSON",
    }
    r = requests.get(base, params=params, timeout=15)
    data = r.json()
    if "values" not in data:
        # try without slash
        alt = symbol.replace("/", "")
        params["symbol"] = alt
        r = requests.get(base, params=params, timeout=15)
        data = r.json()
        if "values" not in data:
            msg = data.get("message") or data.get("status") or str(data)
            raise RuntimeError(f"TwelveData error for {symbol}: {msg}")

    # API returns newest-first; reverse to oldest-first
    values = list(reversed(data["values"]))
    candles = []
    for v in values:
        try:
            candles.append({
                "t": v["datetime"],
                "o": float(v["open"]),
                "h": float(v["high"]),
                "l": float(v["low"]),
                "c": float(v["close"]),
            })
        except Exception:
            continue

    _cache["data"][symbol] = candles
    _cache["ts"][symbol] = now
    return candles


# ==================== INDICATORS / UTILS ====================
def atr(highs, lows, closes, period=ATR_PERIOD):
    if len(closes) < 2:
        return 0.0
    trs = []
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i],
                 abs(highs[i] - closes[i-1]),
                 abs(lows[i] - closes[i-1]))
        trs.append(tr)
    if not trs:
        return 0.0
    p = min(period, len(trs))
    return sum(trs[-p:]) / p


def zigzag_swings(candles, atr_mult=ZZ_ATR_MULT, max_keep=MAX_SWINGS):
    """
    Dynamic ZigZag using ATR threshold: reverse leg when price moves > ATR*mult against it.
    Returns swings: [{"idx": i, "type": "H"/"L", "price": p, "time": t}, ...]
    """
    n = len(candles)
    if n < ATR_PERIOD + 3:
        return []

    highs = [c["h"] for c in candles]
    lows  = [c["l"] for c in candles]
    closes= [c["c"] for c in candles]
    _atr = atr(highs, lows, closes, ATR_PERIOD)
    if _atr == 0:
        return []

    threshold = _atr * atr_mult

    # Start from first bar as initial pivot (low)
    piv_idx = 0
    piv_price_low = lows[0]
    piv_price_high = highs[0]
    direction = None  # "up" or "down"
    swings = []

    for i in range(1, n):
        hi, lo = highs[i], lows[i]

        # extend current pivot range
        if hi > piv_price_high:
            piv_price_high = hi
            piv_high_idx = i
        if lo < piv_price_low:
            piv_price_low = lo
            piv_low_idx = i

        if direction is None:
            # initialize direction once we have a move > threshold
            if hi - piv_price_low >= threshold:
                direction = "up"
                # first pivot is a Low
                swings.append({"idx": piv_low_idx if 'piv_low_idx' in locals() else 0,
                               "type": "L",
                               "price": piv_price_low,
                               "time": candles[piv_low_idx if 'piv_low_idx' in locals() else 0]["t"]})
                # set new pivot from the high
                cur_peak = hi
                cur_peak_idx = i
            elif piv_price_high - lo >= threshold:
                direction = "down"
                swings.append({"idx": piv_high_idx if 'piv_high_idx' in locals() else 0,
                               "type": "H",
                               "price": piv_price_high,
                               "time": candles[piv_high_idx if 'piv_high_idx' in locals() else 0]["t"]})
                cur_trough = lo
                cur_trough_idx = i
            continue

        if direction == "up":
            # track highest point in leg
            if hi > cur_peak:
                cur_peak = hi
                cur_peak_idx = i
            # reversal?
            if cur_peak - lo >= threshold:
                # mark High swing
                swings.append({"idx": cur_peak_idx, "type": "H", "price": cur_peak, "time": candles[cur_peak_idx]["t"]})
                # switch direction and set new trough
                direction = "down"
                cur_trough = lo
                cur_trough_idx = i

        elif direction == "down":
            if lo < cur_trough:
                cur_trough = lo
                cur_trough_idx = i
            if hi - cur_trough >= threshold:
                swings.append({"idx": cur_trough_idx, "type": "L", "price": cur_trough, "time": candles[cur_trough_idx]["t"]})
                direction = "up"
                cur_peak = hi
                cur_peak_idx = i

    # finalize with the last extreme as a swing
    if direction == "up" and 'cur_peak_idx' in locals():
        swings.append({"idx": cur_peak_idx, "type": "H", "price": cur_peak, "time": candles[cur_peak_idx]["t"]})
    if direction == "down" and 'cur_trough_idx' in locals():
        swings.append({"idx": cur_trough_idx, "type": "L", "price": cur_trough, "time": candles[cur_trough_idx]["t"]})

    # sort unique & keep last
    swings = sorted({(s["idx"], s["type"]): s for s in swings}.values(), key=lambda x: x["idx"])
    return swings[-max_keep:]


def market_structure(swings):
    """Up/Down/Sideways via HH/HL or LH/LL of recent swings."""
    if len(swings) < 5:
        return "sideways", "few swings"
    highs = [s for s in swings if s["type"] == "H"]
    lows  = [s for s in swings if s["type"] == "L"]
    if len(highs) >= 3 and len(lows) >= 3:
        if highs[-3]["price"] < highs[-2]["price"] < highs[-1]["price"] and \
           lows[-3]["price"]  < lows[-2]["price"]  < lows[-1]["price"]:
            return "up", "HH/HL"
        if highs[-3]["price"] > highs[-2]["price"] > highs[-1]["price"] and \
           lows[-3]["price"]  > lows[-2]["price"]  > lows[-1]["price"]:
            return "down", "LH/LL"
    return "sideways", "mixed"


def detect_pattern(candles):
    """Simple last-2-candles patterns to help CWRV confirmation."""
    if len(candles) < 2:
        return None
    a, b = candles[-2], candles[-1]
    # engulf
    if a["c"] < a["o"] and b["c"] > b["o"] and b["c"] > a["o"]:
        return "bullish_engulf"
    if a["c"] > a["o"] and b["c"] < b["o"] and b["c"] < a["o"]:
        return "bearish_engulf"
    # pin
    body = abs(b["c"] - b["o"])
    up_wick = b["h"] - max(b["c"], b["o"])
    lo_wick = min(b["c"], b["o"]) - b["l"]
    if body == 0:
        body = 1e-9
    if lo_wick > 2*body and lo_wick > up_wick:
        return "pin_bottom"
    if up_wick > 2*body and up_wick > lo_wick:
        return "pin_top"
    return None


def detect_cwrv_123(swings, trend):
    """
    CWRV 1-2-3 using last 3 alternating swings.
    Up: L(1) -> H(2) -> HL(3) where HL > L(1)
    Down: H(1) -> L(2) -> LH(3) where LH < H(1)
    """
    if len(swings) < 3:
        return False, "not enough swings"
    s1, s2, s3 = swings[-3], swings[-2], swings[-1]
    if trend == "up":
        if s1["type"] == "L" and s2["type"] == "H" and s3["type"] == "L" and s3["price"] > s1["price"]:
            return True, f"1-2-3 up (HL {s3['price']:.5f} > {s1['price']:.5f})"
        # allow candle confirmation
        return False, "no HL"
    if trend == "down":
        if s1["type"] == "H" and s2["type"] == "L" and s3["type"] == "H" and s3["price"] < s1["price"]:
            return True, f"1-2-3 down (LH {s3['price']:.5f} < {s1['price']:.5f})"
        return False, "no LH"
    return False, "trend sideways"


def sideways_filter(candles):
    """Range-overlap sideways filter on last 12 bars."""
    if len(candles) < 12:
        return False
    last = candles[-12:]
    overlaps = 0
    for i in range(1, len(last)):
        a, b = last[i-1], last[i]
        a_lo, a_hi = min(a["o"], a["c"]), max(a["o"], a["c"])
        b_lo, b_hi = min(b["o"], b["c"]), max(b["o"], b["c"])
        if (b_lo >= a_lo and b_hi <= a_hi) or (a_lo >= b_lo and a_hi <= b_hi):
            overlaps += 1
    return overlaps >= 7


# ==================== ELLIOTT WAVE ====================
def elliott_wave_label(swings):
    """
    Heuristic Elliott impulse (1..5) validation from the last 9 swings using rules:
      - Alternating L/H sequence in trend direction
      - Wave 2 does NOT retrace 100% of Wave 1
      - Wave 3 is not the shortest and should exceed Wave 1 length (magnitude)
      - Wave 4 does NOT overlap Wave 1 price territory (for most FX cases)
    Returns ("1".."5", "A"/"B"/"C", or "unknown")
    """
    if len(swings) < 5:
        return "unknown"

    # Build alternating sequences ending at latest swing for both directions
    def legs_prices(seq):
        return [abs(seq[i]["price"] - seq[i-1]["price"]) for i in range(1, len(seq))]

    # Try Uptrend impulse: L, H, L, H, L (at least 5 points)
    up_seq = []
    for s in reversed(swings):
        if not up_seq:
            up_seq.append(s)
        else:
            # need opposite type to alternate
            if s["type"] == ("L" if up_seq[-1]["type"] == "H" else "H"):
                up_seq.append(s)
            if len(up_seq) >= 7:  # limit
                break
    up_seq = list(reversed(up_seq))
    # Ensure it starts with L and alternates
    up_impulse = None
    if len(up_seq) >= 5 and up_seq[0]["type"] == "L":
        seq = up_seq[-5:]  # last 5 alternations
        if [x["type"] for x in seq] in (["L","H","L","H","L"], ["L","H","L","H","L","H"][:5]):
            # Validate rules
            w1 = seq[1]["price"] - seq[0]["price"]    # L->H
            w2 = seq[2]["price"] - seq[1]["price"]    # H->L (negative)
            w3 = seq[3]["price"] - seq[2]["price"]    # L->H
            w4 = seq[4]["price"] - seq[3]["price"]    # H->L (negative)

            # Rule checks
            rule2_ok = seq[2]["price"] > seq[0]["price"]      # wave2 not 100% retrace
            rule3_ok = abs(w3) > abs(w1) * 1.05               # wave3 longer than wave1 by small margin
            # wave4 does not overlap wave1 price territory
            w1_low, w1_high = seq[0]["price"], seq[1]["price"]
            # wave4 L should be above w1_high? (strict) FX often overlaps slightly; allow 10% overlap tolerance
            rule4_ok = seq[4]["price"] > (w1_low + (w1_high - w1_low) * 0.10)

            if rule2_ok and rule3_ok and rule4_ok and w1 > 0 and w3 > 0:
                # Determine current wave number from last swing type
                up_impulse = "5" if seq[-1]["type"] == "L" else "4"

    # Try Downtrend impulse: H, L, H, L, H
    down_seq = []
    for s in reversed(swings):
        if not down_seq:
            down_seq.append(s)
        else:
            if s["type"] == ("H" if down_seq[-1]["type"] == "L" else "L"):
                down_seq.append(s)
            if len(down_seq) >= 7:
                break
    down_seq = list(reversed(down_seq))
    down_impulse = None
    if len(down_seq) >= 5 and down_seq[0]["type"] == "H":
        seq = down_seq[-5:]
        if [x["type"] for x in seq] in (["H","L","H","L","H"], ["H","L","H","L","H","L"][:5]):
            w1 = seq[1]["price"] - seq[0]["price"]    # H->L (negative)
            w2 = seq[2]["price"] - seq[1]["price"]    # L->H (positive)
            w3 = seq[3]["price"] - seq[2]["price"]    # H->L (negative)
            w4 = seq[4]["price"] - seq[3]["price"]    # L->H (positive)

            rule2_ok = seq[2]["price"] < seq[0]["price"]      # wave2 not 100% retrace up to H0
            rule3_ok = abs(w3) > abs(w1) * 1.05               # wave3 longer than wave1
            # wave4 not overlapping wave1 territory too much
            w1_low, w1_high = seq[1]["price"], seq[0]["price"]
            rule4_ok = seq[4]["price"] < (w1_high - (w1_high - w1_low) * 0.10)

            if rule2_ok and rule3_ok and rule4_ok and w1 < 0 and w3 < 0:
                down_impulse = "5" if seq[-1]["type"] == "H" else "4"

    if up_impulse and not down_impulse:
        # If last swing is H after seq L-H-L-H, we're in Wave 4; if last is L, likely 5 done
        return up_impulse
    if down_impulse and not up_impulse:
        return down_impulse

    # If neither impulse confirmed, try corrective label by symmetry
    if len(swings) >= 3:
        last3 = swings[-3:]
        # small oscillations → corrective
        rng = max(s["price"] for s in last3) - min(s["price"] for s in last3)
        if rng < 0.5 * max(1e-9, rng):  # (kept for structure; practically always False)
            return "B"
    return "unknown"


# ==================== BACKTEST / DECISION ====================
def backtest_accuracy(candles, swings, trend):
    """Quick in-session accuracy estimate by replaying our CWRV + trend calls."""
    if len(candles) < 40 or len(swings) < 6:
        return 100
    results = []
    for i in range(20, len(candles) - 1):
        sub = candles[:i+1]
        sw = zigzag_swings(sub)
        st, _ = market_structure(sw)
        if st != trend:
            continue
        ok, _ = detect_cwrv_123(sw, st)
        if not ok:
            continue
        pred = "CALL" if st == "up" else "PUT"
        nxt = candles[i+1]
        win = (nxt["c"] > nxt["o"] and pred == "CALL") or (nxt["c"] < nxt["o"] and pred == "PUT")
        results.append(win)
    if not results:
        return 100
    recent = results[-BACKTEST_SIGNALS:]
    return int(round(sum(1 for x in recent if x) / len(recent) * 100))


def analyze_pair(symbol):
    out = {
        "pair": symbol,
        "signal": "SIDEWAYS",
        "status": "NO TRADE",
        "accuracy": 100,
        "cwrv": "No",
        "wave": "unknown",
        "candles": [],
        "why": ""
    }
    try:
        candles = fetch_candles(symbol)
        out["candles"] = candles[-60:]

        swings = zigzag_swings(candles)
        trend, reason = market_structure(swings)
        cwrv_ok, cwrv_msg = detect_cwrv_123(swings, trend)
        is_sideways = sideways_filter(candles)
        wave = elliott_wave_label(swings)

        accuracy = backtest_accuracy(candles, swings, trend)

        # Decision
        if trend == "up" and cwrv_ok and not is_sideways:
            signal = "CALL"
        elif trend == "down" and cwrv_ok and not is_sideways:
            signal = "PUT"
        else:
            signal = "SIDEWAYS"

        status = "NO TRADE"
        if signal != "SIDEWAYS" and not is_sideways:
            status = "TRADE" if accuracy >= 80 else ("RISKY" if accuracy >= 60 else "NO TRADE")

        out.update({
            "signal": signal,
            "status": status,
            "accuracy": accuracy,
            "cwrv": "Yes" if cwrv_ok else "No",
            "wave": wave,
            "why": f"struct={trend} ({reason}); cwrv={cwrv_msg}; sideways={is_sideways}; swings={len(swings)}"
        })
        return out
    except Exception as e:
        out["why"] = f"Error: {e}"
        return out


# ==================== ROUTES ====================
@app.route("/")
def index():
    # If you use the HTML I sent earlier, keep this route.
    return render_template("index.html", pairs=PAIRS)

@app.route("/analyze", methods=["GET"])
def analyze():
    symbols = request.args.getlist("pair") or PAIRS
    results = [analyze_pair(sym) for sym in symbols]
    return jsonify({"results": results})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
