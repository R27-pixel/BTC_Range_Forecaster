"""
BTC 1-Hour Range Forecaster — Backend (FastAPI)
Implements GBM with Student-t fat tails + volatility clustering.
Data source: Binance public API (no key required).
"""

import json
import os
import time
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─── Config ───────────────────────────────────────────────────────────────────
BINANCE_BASE = "https://data-api.binance.vision/api/v3"
SYMBOL       = "BTCUSDT"
INTERVAL     = "1h"
HISTORY_BARS = 500
BACKTEST_BARS= 720   # ~30 days of 1h bars
N_SIMS       = 10_000
CI_LEVEL     = 0.95
PREDICTIONS_FILE = Path("predictions.jsonl")

app = FastAPI(title="BTC Range Forecaster")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Binance Data ──────────────────────────────────────────────────────────────

def fetch_klines(symbol: str = SYMBOL, interval: str = INTERVAL, limit: int = HISTORY_BARS) -> List[dict]:
    """Fetch OHLCV klines from Binance public API."""
    url = f"{BINANCE_BASE}/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        raw = r.json()
    except Exception as e:
        raise RuntimeError(f"Binance fetch failed: {e}")

    bars = []
    for k in raw:
        bars.append({
            "open_time": int(k[0]),
            "open": float(k[1]),
            "high": float(k[2]),
            "low":  float(k[3]),
            "close":float(k[4]),
            "volume":float(k[5]),
            "close_time": int(k[6]),
        })
    return bars


def closes_from_bars(bars: List[dict]) -> np.ndarray:
    return np.array([b["close"] for b in bars], dtype=np.float64)


# ─── GBM Model ────────────────────────────────────────────────────────────────

def log_returns(closes: np.ndarray) -> np.ndarray:
    return np.log(closes[1:] / closes[:-1])


def estimate_params(closes: np.ndarray, window: int = 24):
    """
    Estimate drift (mu) and volatility (sigma) from closes.
    Uses recent window for volatility clustering (short lookback = sensitive).
    Applies Student-t df for fat-tail correction.
    """
    rets = log_returns(closes)
    mu   = float(np.mean(rets))

    # Volatility clustering: use recent window for sigma
    recent = rets[-window:]
    sigma  = float(np.std(recent, ddof=1))

    # Student-t degrees-of-freedom: fit to full return history
    # Clamp nu to [3, 30] — BTC empirically sits around 3-5
    from scipy import stats as sp_stats
    try:
        nu, _, _ = sp_stats.t.fit(rets, floc=0)
        nu = float(np.clip(nu, 3.0, 30.0))
    except Exception:
        nu = 4.0

    return mu, sigma, nu


def simulate_gbm(S0: float, mu: float, sigma: float, nu: float,
                 n_sims: int = N_SIMS, dt: float = 1.0) -> np.ndarray:

    scale = math.sqrt((nu - 2.0) / nu) if nu > 2 else 1.0
    Z = np.random.standard_t(df=nu, size=n_sims) * scale

    # stochastic volatility
    vol_shock = np.random.normal(1.0, 0.2, size=n_sims)
    sigma_dynamic = sigma * vol_shock

    #calibrated scaling (tunable)
    CALIBRATION = 1.18
    sigma_dynamic = sigma_dynamic * CALIBRATION

    return S0 * np.exp(
        (mu - 0.5 * sigma_dynamic ** 2) * dt
        + sigma_dynamic * math.sqrt(dt) * Z
    )

def predict_range(closes: np.ndarray, alpha: float = 1.0 - CI_LEVEL,
                  vol_window: int = 24) -> dict:
    """
    Full prediction pipeline: estimate params → simulate → extract CI.
    Returns dict with low, high, current price, and metadata.
    """
    S0         = float(closes[-1])
    mu, sigma, nu = estimate_params(closes, window=vol_window)
    terminal   = simulate_gbm(S0, mu, sigma, nu)
    low        = float(np.percentile(terminal, 100 * alpha / 2))
    high       = float(np.percentile(terminal, 100 * (1 - alpha / 2)))
    width      = high - low

    return {
        "current_price": S0,
        "low_95":  round(low, 2),
        "high_95": round(high, 2),
        "width":   round(width, 2),
        "mu":      round(mu, 8),
        "sigma":   round(sigma, 6),
        "nu":      round(nu, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ─── Winkler Score ─────────────────────────────────────────────────────────────

def winkler_score(low: float, high: float, actual: float, alpha: float = 0.05) -> float:
    width = high - low
    if actual < low:
        return width + (2 / alpha) * (low - actual)
    elif actual > high:
        return width + (2 / alpha) * (actual - high)
    return width


# ─── Backtest ─────────────────────────────────────────────────────────────────

def run_backtest(bars: List[dict], min_train: int = 100, vol_window: int = 24):
    """
    Walk-forward backtest over provided bars.
    No-peek guaranteed: prediction at bar i uses only bars[0..i].
    """
    closes = closes_from_bars(bars)
    n      = len(closes)
    results = []

    for i in range(min_train, n - 1):
        train_closes = closes[: i + 1]   # strictly up to bar i
        S0           = float(train_closes[-1])

        mu, sigma, nu = estimate_params(train_closes, window=vol_window)
        terminal      = simulate_gbm(S0, mu, sigma, nu)

        low  = float(np.percentile(terminal, 2.5))
        high = float(np.percentile(terminal, 97.5))
        actual = float(closes[i + 1])     # bar i+1 — revealed after prediction

        hit      = int(low <= actual <= high)
        winkler  = winkler_score(low, high, actual)

        results.append({
            "bar_index": i,
            "timestamp": bars[i + 1]["open_time"],
            "actual":    round(actual, 2),
            "low_95":    round(low, 2),
            "high_95":   round(high, 2),
            "width_95":  round(high - low, 2),
            "coverage_95": hit,
            "winkler":   round(winkler, 4),
        })

    return results


def backtest_metrics(results: List[dict]) -> dict:
    if not results:
        return {}
    coverage  = np.mean([r["coverage_95"] for r in results])
    avg_width = np.mean([r["width_95"] for r in results])
    avg_wink  = np.mean([r["winkler"] for r in results])
    return {
        "n_predictions":  len(results),
        "coverage_95":    round(float(coverage), 4),
        "mean_width_95":  round(float(avg_width), 2),
        "mean_winkler_95":round(float(avg_wink), 4),
    }


# ─── Persistence (Part C) ─────────────────────────────────────────────────────

def save_prediction(pred: dict):
    with open(PREDICTIONS_FILE, "a") as f:
        f.write(json.dumps(pred) + "\n")


def load_predictions() -> List[dict]:
    if not PREDICTIONS_FILE.exists():
        return []
    rows = []
    with open(PREDICTIONS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def fill_actuals(preds: List[dict], bars: List[dict]) -> List[dict]:
    """
    For each saved prediction, try to fill in the actual price if that bar has
    now closed (matched by open_time of the next bar).
    """
    time_to_close = {b["open_time"]: b["close"] for b in bars}
    enriched = []
    for p in preds:
        p2 = dict(p)
        target_ts = p.get("target_open_time")
        if target_ts and target_ts in time_to_close:
            p2["actual"] = time_to_close[target_ts]
            p2["hit"]    = int(p2["low_95"] <= p2["actual"] <= p2["high_95"])
        enriched.append(p2)
    return enriched


@app.get("/history")
def history():
    """
    Return prediction history with actuals filled in.
    """
    try:
        bars = fetch_klines(limit=HISTORY_BARS + 100)  # extra for filling actuals
    except RuntimeError as e:
        raise HTTPException(502, str(e))

    preds = load_predictions()
    history = fill_actuals(preds, bars)
    return {"history": history}


# ─── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}


@app.get("/current")
def current_prediction():
    """
    Fetch latest bars, run model, return current prediction + last 50 OHLCV bars.
    Saves prediction to JSONL for Part C history.
    """
    try:
        bars = fetch_klines(limit=HISTORY_BARS)
    except RuntimeError as e:
        raise HTTPException(502, str(e))

    closes   = closes_from_bars(bars)
    pred     = predict_range(closes)
    last_50  = bars[-50:]

    # Save for history (Part C)
    to_save = dict(pred)
    to_save["target_open_time"] = last_50[-1]["close_time"] + 1   # next bar's open
    save_prediction(to_save)

    return {
        "prediction": pred,
        "bars":       last_50,
    }


@app.get("/backtest")
def backtest_endpoint(limit: int = BACKTEST_BARS):
    """
    Run 30-day backtest. Saves results to backtest_results.jsonl.
    NOTE: This is slow (~720 iterations). Call once; results are cached.
    """
    cache_file = Path("backtest_results.jsonl")

    # Return cached results if fresh (< 4 hours old)
    if cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < 4 * 3600:
            rows = []
            with open(cache_file) as f:
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line.strip()))
            metrics = backtest_metrics(rows)
            return {"metrics": metrics, "results": rows[-50:], "cached": True}

    try:
        bars = fetch_klines(limit=limit + 10)  # fetch extra to have test data
    except RuntimeError as e:
        raise HTTPException(502, str(e))

    results = run_backtest(bars)
    metrics = backtest_metrics(results)

    # Save JSONL
    with open(cache_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    return {"metrics": metrics, "results": results[-50:], "cached": False}


@app.get("/history")
def prediction_history():
    """Return all saved predictions with actuals filled in where available (Part C)."""
    try:
        bars = fetch_klines(limit=200)
    except RuntimeError:
        bars = []

    preds    = load_predictions()
    enriched = fill_actuals(preds, bars)
    return {"history": enriched[-100:]}  # last 100


@app.get("/metrics")
def metrics_only():
    """Return backtest metrics from cached file if available."""
    cache_file = Path("backtest_results.jsonl")
    if not cache_file.exists():
        return {"error": "No backtest run yet. Call /backtest first."}
    rows = []
    with open(cache_file) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line.strip()))
    return backtest_metrics(rows)
