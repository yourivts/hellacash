# ⚡ HellaCash — Autonomous Bitvavo Trading Bot

A self-contained crypto trading bot for [Bitvavo](https://bitvavo.com) with:
- **Multi-strategy engine** (hybrid, trend-following, mean-reversion, breakout)
- **Adaptive learning** — tunes signal weights from trade history
- **Sentiment analysis** — Reddit + Twitter/X + RSS news feeds
- **Capital preservation** — ATR stops, drawdown circuit breakers, Kelly position sizing
- **Clean web dashboard** — live charts, analytics, trade history

---

## Quick Start

### 1. Clone & configure

```bash
cp .env.example .env
# Edit .env — add your Bitvavo API keys and (optionally) Reddit/Twitter credentials
```

### 2. Start PostgreSQL

```bash
docker-compose up db -d
```

Or use your own PostgreSQL and set `DATABASE_URL` in `.env`.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the bot

```bash
python -m bot.main
```

Dashboard: **http://localhost:8000**

---

## Paper Trading (Default)

`PAPER_TRADING=true` is the default. All orders are **simulated** — no real money is touched.

To go live: set `PAPER_TRADING=false` in `.env`. Make sure you've run in paper mode for at least 2 weeks and are satisfied with results.

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `BITVAVO_API_KEY` | — | Bitvavo API key |
| `BITVAVO_API_SECRET` | — | Bitvavo API secret |
| `PAPER_TRADING` | `true` | Paper/live mode |
| `MAX_PAIRS` | `25` | Number of pairs to trade |
| `MAX_DRAWDOWN_PCT` | `8.0` | Emergency halt threshold (%) |
| `MAX_POSITION_SIZE_PCT` | `20.0` | Max single position size |
| `MAX_DAILY_LOSS_EUR` | `200.0` | Daily loss limit |
| `MIN_SIGNAL_CONFIDENCE` | `0.60` | Min signal strength to trade |
| `REDDIT_CLIENT_ID` | — | Reddit API credentials |
| `TWITTER_BEARER_TOKEN` | — | Twitter API v2 Bearer Token |
| `DATABASE_URL` | postgresql+asyncpg://… | PostgreSQL connection |

---

## Architecture

```
bot/
├── main.py           # Orchestrator — asyncio event loop + uvicorn
├── config.py         # Settings from .env
├── exchange/         # Bitvavo REST + WebSocket client
├── indicators/       # RSI, MACD, Bollinger, ATR, EMA, volume, composite
├── strategy/         # hybrid + trend + range + breakout + regime router
├── risk/             # Engine gate, Kelly sizer, ATR stops, drawdown guard
├── sentiment/        # Reddit/Twitter/RSS scrapers + VADER/FinBERT NLP
├── learning/         # Trade analyzer, signal evaluator, param optimizer
├── portfolio/        # Real-time position + P&L tracker
├── events/           # asyncio pub/sub bus
└── data/             # SQLAlchemy ORM + repositories

api/                  # FastAPI REST + WebSocket hub
frontend/             # Single-page dashboard (no build step)
```

---

## Risk Management

All risk limits are **config-only** — the learning system cannot touch them:

1. Signal confidence must exceed `MIN_SIGNAL_CONFIDENCE` (0.60)
2. Portfolio drawdown < `MAX_DRAWDOWN_PCT` (8%) — hard halt if breached
3. Daily realized loss < `MAX_DAILY_LOSS_EUR` — stops new trades for the day
4. Position size ≤ `MAX_POSITION_SIZE_PCT` of portfolio
5. ≤ 5 concurrent open positions
6. Expected ROI must exceed 2× trading fees + 0.3%
7. ATR-based stop-loss on every position (entry − 2×ATR)
8. 3:1 reward/risk minimum (take-profit = entry + 3×ATR risk)

---

## Sentiment Sources

| Source | Method | Weight |
|---|---|---|
| News | CoinDesk/Cointelegraph/Decrypt RSS + FinBERT | 40% |
| Reddit | PRAW — r/CryptoCurrency, r/Bitcoin, r/ethereum | 35% |
| Twitter/X | tweepy v2 (snscrape fallback) | 25% |

Sentiment runs every 10 minutes. Scores range from -1.0 (very bearish) to +1.0 (very bullish).

---

## Adaptive Learning

- After each trade closes → features + outcome saved to `learning_events`
- Indicator accuracy tracked → low-accuracy indicators get lower weights
- After every 50 trades → Bayesian parameter optimizer tunes signal weights
- Weekly → `RegimeClassifier` retrained on new candle data

---

## Docker

```bash
docker-compose up --build
```

Runs PostgreSQL + bot in containers. Dashboard at http://localhost:8000.
