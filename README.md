# BTC Range Forecaster — Backend

FastAPI backend implementing GBM + Student-t forecasting for BTC/USDT 1h bars.

## Setup

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Endpoints

| Endpoint     | Description |
|-------------|-------------|
| `GET /health` | Healthcheck |
| `GET /current` | Live prediction + last 50 bars |
| `GET /backtest?limit=720` | Run 30-day backtest (cached after first run) |
| `GET /metrics` | Cached backtest metrics only |
| `GET /history` | All saved predictions (Part C persistence) |

## Deploy (free)

**Railway / Render / Fly.io:**
```bash
# Railway
railway init && railway up

# Or Render: connect GitHub repo, set start command:
uvicorn main:app --host 0.0.0.0 --port $PORT
```

## Data Source

Uses Binance public API — no API key required:
```
https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=1h&limit=500
```
