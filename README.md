# Trading Strategy Dashboard

A Streamlit dashboard comparing three algorithmic trading strategies running on separate Alpaca paper accounts.

## Strategies

| Strategy | Method |
|---|---|
| **RL Trader** | Deep RL (PPO) — momentum-weighted multi-asset allocation, weekly rebalance |
| **Regime Trader** | HMM regime detection — allocation shifts between bull / cautious / defensive tiers |
| **Claudebot** | AI swing trader — 5-factor scoring, trailing stops, max 3 trades/week |

## Tabs

- **Live Portfolio** — real-time equity, positions, and P&L from each Alpaca account
- **Historical Performance** — daily equity curves and performance metrics pulled from Alpaca's portfolio history API
- **Backtest Comparison** — run all 3 strategy proxies against any historical date range to compare performance without waiting months for live results

## Running locally

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your Alpaca API keys
streamlit run app.py
```

## Deployment

Deployed on [Streamlit Community Cloud](https://share.streamlit.io). Secrets (API keys) are configured in the Streamlit Cloud dashboard under **App settings → Secrets**.
