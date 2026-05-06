# Trading Strategy Dashboard

A Streamlit dashboard comparing three algorithmic trading strategies running on separate Alpaca paper accounts.

## Agents

---

## Comparison

| | Regime Trader | Claude Trader | RL Trader |
|---|---|---|---|
| **Core model** | HMM regime classification | Claude LLM + rule scoring | PPO / A2C / SAC |
| **Signal** | Forward HMM inference + technical filter | Web research + 5-factor score (≥7 to enter) | Learned policy weights |
| **Execution** | Live loop / daily cron | Scheduled ephemeral VMs | Daily cron (offline training) |
| **Trailing stop** | ATR × 1.5–3.0 (regime-dependent) | 10% GTC; tightens at +15% / +20% | Learned via reward; hard caps enforced |
| **Max position** | 8% | 8% | 8% |
| **Max exposure** | 80% | 80% | 80% |
| **Data** | OHLCV + macro (VIX, yield, credit) | OHLCV + Tavily web research | OHLCV + macro (same as Regime) |
| **Transparency** | SQLite state DB | Git Markdown memory files | SQLite equity + trade log |

---

### [Regime Trader](https://github.com/HOSH19/Regime_Trader.git)

Fully systematic long-only trader driven by a Hidden Markov Model that classifies latent market regimes.

**How it works:**
- Trains an HMM (3–7 states, BIC-selected, Student-t emissions) on price + macro features (VIX, yield spread, credit stress)
- Forward algorithm only. No lookahead; a stability filter requires a new regime to persist ≥3 bars before acting
- Regime states are ranked by historical return (BEAR → NEUTRAL → BULL) and each maps to one of three strategy tiers (low / mid / high volatility)
- Within each tier, a technical filter (RSI momentum or Bollinger mean-reversion) gates individual entries
- Sizing via half-Kelly with correlation-aware position capping

**Risk controls:** ATR trailing stops (1.5–3.0× multiplier, regime-dependent); circuit breakers at 2%/3% daily and 5%/7% weekly drawdown; 10% peak-to-trough lock requiring manual restart.

**Key params:** max 8% per position, 80% max gross exposure, 1% risk per trade, ≥0.55 regime confidence required.

---

### [Claude Trader](https://github.com/HOSH19/Claude_Trader.git)

Autonomous swing trader orchestrated by Claude Code scheduled remote agents — no persistent server, just ephemeral VMs running on a cron schedule.

**How it works:**
- Five daily routines (pre-market, open, midday, close, weekly review) each spin up a VM, pull the latest state from git, run Claude LLM with web research via Tavily, and commit decisions back to `main` as Markdown memory files
- Candidates from a 20-symbol universe (10 tech, 4 finance, 2 energy, 2 healthcare, 2 ETFs) are scored 0–10 across five factors: catalyst strength, sector momentum, technical setup, volume confirmation, risk/reward — must score ≥7 to enter
- Real GTC orders placed via Alpaca; 10% trailing stop on every position; tightens to 7% at +15% gain and 5% at +20%

**Risk controls:** -7% hard stop; portfolio halt if equity drops 10% from session start; sector blacklist after 2 consecutive losses; max 3 new trades per week.

**Key params:** max 10 open positions, 8% per position, 80% target deployment.

---

### [RL Trader](https://github.com/HOSH19/RL_Trader.git)

Multi-asset portfolio allocator trained via reinforcement learning (PPO / A2C / SAC) using walk-forward cross-validation.

**How it works:**
- Gymnasium `TradingEnv`: observation is 65–80 TA + macro features per asset plus current weights, equity, and cash; action is a softmax over portfolio weights (including a cash slot)
- Walk-forward training: 252-bar train window, 126-bar test window, 126-bar step; weights warm-started across folds; early stopping monitors OOS Sharpe
- Reward combines log return, drawdown penalty, transaction cost, and concentration penalty (all tunable λ)
- At inference, the best checkpoint for the current date's fold is loaded daily; a hard risk wrapper clips any weight that breaches position limits before orders are submitted

**Risk controls:** 8% max single position, 20% minimum cash, 80% max gross exposure; 3%/10% drawdown halt (daily/peak-to-trough).

**Key params:** MLP policy [256, 128], 0.03% transaction cost + 0.05% slippage modelled in training, 500k–1M timesteps per fold.

---

## Dashboard tabs

- **Live Portfolio** — real-time equity, positions, and P&L from each Alpaca account
- **Historical Performance** — daily equity curves and performance metrics from Alpaca's portfolio history API
- **Backtest Comparison** — run all three strategy proxies against any historical date range

## Running locally

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your Alpaca API keys
streamlit run app.py
```

## Deployment

Deployed on [Streamlit Community Cloud](https://share.streamlit.io).