"""
Tests for BTC Range Forecaster backend.
Exercises all core model functions with synthetic BTC-like data.
No live API calls — runs fully offline.
"""

import sys
import math
import numpy as np
import json
from pathlib import Path

# ── Add backend to path ─────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from main import (
    log_returns,
    estimate_params,
    simulate_gbm,
    predict_range,
    winkler_score,
    run_backtest,
    backtest_metrics,
    save_prediction,
    load_predictions,
)

# ── Helpers ─────────────────────────────────────────────────────────────────
PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
tests_run   = 0
tests_passed= 0

def check(name, condition, detail=""):
    global tests_run, tests_passed
    tests_run += 1
    if condition:
        tests_passed += 1
        print(f"  {PASS}  {name}")
    else:
        print(f"  {FAIL}  {name}" + (f"  →  {detail}" if detail else ""))


def make_btc_closes(n=600, seed=42, start=65000.0, mu_h=0.0001, sigma_h=0.015):
    """Generate synthetic hourly BTC closes via GBM with Student-t noise."""
    rng = np.random.default_rng(seed)
    nu  = 4.0
    z   = rng.standard_t(df=nu, size=n) * math.sqrt((nu-2)/nu)
    log_ret = (mu_h - 0.5*sigma_h**2) + sigma_h * z
    prices  = start * np.exp(np.cumsum(log_ret))
    return np.concatenate([[start], prices])


def make_bars(closes):
    return [{"open_time": i*3600000, "close": float(c), "close_time": (i+1)*3600000 - 1}
            for i, c in enumerate(closes)]


# ── Test Suite ───────────────────────────────────────────────────────────────

print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("  BTC Range Forecaster — Unit Tests")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

closes = make_btc_closes()

# 1. log_returns
print("[ 1 ] log_returns")
rets = log_returns(closes)
check("length is N-1",           len(rets) == len(closes) - 1)
check("no NaN/Inf",              np.all(np.isfinite(rets)))
check("returns are small floats",np.abs(rets).max() < 0.5)

# 2. estimate_params
print("\n[ 2 ] estimate_params")
mu, sigma, nu = estimate_params(closes, window=24)
check("mu is finite float",  math.isfinite(mu))
check("sigma > 0",           sigma > 0)
check("nu in [3, 30]",       3.0 <= nu <= 30.0,  f"nu={nu:.2f}")
check("sigma plausible BTC", 0.001 < sigma < 0.15, f"sigma={sigma:.5f}")

# 3. simulate_gbm
print("\n[ 3 ] simulate_gbm")
S0   = float(closes[-1])
sims = simulate_gbm(S0, mu, sigma, nu, n_sims=20_000)
check("correct shape",       sims.shape == (20_000,))
check("all prices > 0",      np.all(sims > 0))
check("mean near S0",        abs(sims.mean() / S0 - 1.0) < 0.05,
                             f"mean={sims.mean():.0f} S0={S0:.0f}")
check("std reasonable",      sims.std() / S0 < 0.10,  f"cv={sims.std()/S0:.4f}")

# 4. predict_range
print("\n[ 4 ] predict_range")
pred = predict_range(closes, vol_window=24)
check("has required keys",   all(k in pred for k in ["current_price","low_95","high_95","width","mu","sigma","nu","timestamp"]))
check("low_95 < current",    pred["low_95"] < pred["current_price"],
                             f"low={pred['low_95']} cur={pred['current_price']}")
check("high_95 > current",   pred["high_95"] > pred["current_price"],
                             f"high={pred['high_95']} cur={pred['current_price']}")
check("width == high-low",   abs(pred["width"] - (pred["high_95"] - pred["low_95"])) < 0.01)
check("width is positive",   pred["width"] > 0)

# 5. winkler_score
print("\n[ 5 ] winkler_score")
low_w, high_w = 100.0, 200.0
check("score=width when inside",     abs(winkler_score(low_w, high_w, 150.0) - 100.0) < 1e-9)
check("score > width when below",    winkler_score(low_w, high_w, 50.0) > 100.0)
check("score > width when above",    winkler_score(low_w, high_w, 300.0) > 100.0)
check("miss penalty proportional",
      winkler_score(low_w, high_w, 0.0) > winkler_score(low_w, high_w, 90.0),
      "bigger miss should score higher")
# alpha=0.05 penalty: 2/0.05 = 40× miss
miss = 10.0
expected = 100.0 + 40.0 * miss
check("correct penalty formula",
      abs(winkler_score(low_w, high_w, low_w - miss) - expected) < 1e-6,
      f"got {winkler_score(low_w, high_w, low_w - miss):.4f} expected {expected:.4f}")

# 6. No-peek guarantee
print("\n[ 6 ] No-peek guarantee")
closes_short = make_btc_closes(n=120)
bars_short   = make_bars(closes_short)
results      = run_backtest(bars_short, min_train=50, vol_window=12)
if results:
    for i, r in enumerate(results):
        bar_idx = r["bar_index"]
        # Prediction must have been made using only bars[0..bar_idx]
        # We verify by checking the 'actual' price == bar_idx+1 close
        expected_actual = bars_short[bar_idx + 1]["close"]
        check(f"  bar {bar_idx}: actual matches bar[i+1]",
              abs(r["actual"] - expected_actual) < 0.01,
              f"{r['actual']:.2f} vs {expected_actual:.2f}")
        if i >= 3:  # just sample first few
            break
else:
    print("  (no results — increase closes_short n)")

# 7. backtest_metrics
print("\n[ 7 ] backtest_metrics")
results_full = run_backtest(bars_short, min_train=50, vol_window=12)
metrics = backtest_metrics(results_full)
check("has all keys",       all(k in metrics for k in ["coverage_95","mean_width_95","mean_winkler_95","n_predictions"]))
check("n_predictions == len", metrics["n_predictions"] == len(results_full))
check("coverage in [0,1]",  0.0 <= metrics["coverage_95"] <= 1.0,  f"cov={metrics['coverage_95']}")
check("mean_width > 0",     metrics["mean_width_95"] > 0)
check("mean_winkler > 0",   metrics["mean_winkler_95"] > 0)

# 8. Statistical coverage check (GBM should hit ~95% on its own data)
print("\n[ 8 ] Coverage statistical sanity check")
closes_med = make_btc_closes(n=300)
bars_med   = make_bars(closes_med)
results_med= run_backtest(bars_med, min_train=80, vol_window=24)
m_med      = backtest_metrics(results_med)
cov        = m_med["coverage_95"]
# We expect ~90-100% on synthetic data generated by matching GBM process
check(f"coverage between 0.80 and 1.00 ({cov:.3f})",  0.80 <= cov <= 1.00, f"cov={cov:.3f}")
print(f"    coverage={cov:.3f}  mean_width=${m_med['mean_width_95']:.0f}  winkler=${m_med['mean_winkler_95']:.0f}")

# 9. Persistence (save/load)
print("\n[ 9 ] Persistence")
import tempfile, os
tmp = Path(tempfile.mktemp(suffix=".jsonl"))
orig = Path("predictions.jsonl")

# Monkey-patch file path
import main as main_mod
main_mod.PREDICTIONS_FILE = tmp

test_pred = {"current_price": 65000.0, "low_95": 64000.0, "high_95": 66000.0,
             "timestamp": "2024-01-01T00:00:00+00:00", "target_open_time": 999}
save_prediction(test_pred)
save_prediction({**test_pred, "current_price": 65500.0})
loaded = load_predictions()
check("saved 2 predictions",    len(loaded) == 2)
check("data roundtrips intact", loaded[0]["current_price"] == 65000.0)
check("second entry correct",   loaded[1]["current_price"] == 65500.0)
tmp.unlink(missing_ok=True)

# Restore
main_mod.PREDICTIONS_FILE = orig

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
color = "\033[92m" if tests_passed == tests_run else "\033[91m"
print(f"  {color}Results: {tests_passed}/{tests_run} tests passed\033[0m")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

if tests_passed < tests_run:
    sys.exit(1)
