# ₿ BTC Range Forecaster

A real-time Bitcoin (BTC/USDT) price range prediction system using
**Geometric Brownian Motion (GBM) with Student-t innovations and stochastic volatility calibration.**

---

## 🌐 Live Demo

* 🔗 Frontend: https://gregarious-crostata-e7a581.netlify.app
* 🔗 Backend API: https://web-production-596e4.up.railway.app

---

## ⚙️ Architecture

```
Frontend (Netlify)
        ↓
FastAPI Backend (Railway)
        ↓
Binance Public API
```

---

## 🧠 Model Overview

* **GBM (Geometric Brownian Motion)** for price evolution
* **Student-t distribution** for fat tails (crypto volatility)
* **Stochastic volatility scaling** to capture clustering
* **Monte Carlo simulation (10,000 paths)**
* **95% Confidence Interval prediction**

---

## 📊 Key Metrics

* Coverage (95% target): ~0.94
* Adaptive interval width
* Winkler score for calibration quality

---

## 📡 API Endpoints

| Endpoint    | Description                    |
| ----------- | ------------------------------ |
| `/health`   | Health check                   |
| `/current`  | Live prediction + last 50 bars |
| `/backtest` | Walk-forward backtest          |
| `/metrics`  | Cached metrics                 |
| `/history`  | Stored predictions             |

---

## 🚀 Local Setup

```bash
pip install -r requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## ☁️ Deployment

### Backend (Railway)

Start command:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```

### Frontend (Netlify)

* Static HTML deployed via drag & drop
* Connected to backend via REST API

---

## 📈 Data Source

Binance Public API (no API key required):

```
https://data-api.binance.vision/api/v3/klines
```

---

## 💡 Key Improvements

* Introduced stochastic volatility scaling to improve tail coverage
* Calibrated model to reduce underestimation of extreme moves
* Achieved near-target coverage (~94%) without excessive widening

---

## 🧪 Evaluation

* Walk-forward backtesting (no data leakage)
* Predictions evaluated only after candle close
* Persistent logging of predictions and outcomes

---

## 👨‍💻 Author

R27
