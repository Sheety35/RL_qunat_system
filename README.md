# Stocks RL — Existing Work

Reinforcement-learning experiments for stock and portfolio trading using TensorFlow/Keras, gymnasium, and yfinance data. Two trainers (single-stock and multi-stock) plus testing/diagnostics scripts.

## Files

| File | Purpose |
|---|---|
| `rl_model.py` | **Single-stock DQN trainer.** Trains a Deep Q-Network on AAPL daily data (2019-01-01 to 2024-01-01). 3 actions (Hold/Buy/Sell), 12-dim observation (balance, holdings, price, volume, SMA-5/20, RSI, risk metrics). $1,000 starting capital, 40% max-loss stop, 30% profit target. Saves to `trained_models/`. |
| `multi_rl.py` | **Multi-stock portfolio DQN trainer.** 6 stocks (AAPL, GOOGL, MSFT, TSLA, AMZN, NVDA). 5 actions per stock (Hold / Buy / Sell / Buy More / Sell Half) = 30 total actions. Adds diversification bonus, position-concentration penalty (max 25 % per stock), Huber loss. $10,000 starting capital. Saves to `portfolio_models/`. |
| `test_portfolio_model.py` | Loads the most recent portfolio `.h5` model and backtests it on the last N days of fresh yfinance data. Produces `portfolio_backtest_results.png`. |
| `debug_portfolio_model.py` | Inspects model behaviour on fresh data — counts action choices, dumps Q-values, exposes why the policy under-trades. |
| `test.py` | Companion diagnostics: deep Q-value analysis, observation-space sanity check, action-execution test, fix recommendations. |
| `trained_models/` | Saved single-stock model (`best_model_20250813_000605.h5`) plus metadata. |
| `portfolio_models/` | Saved portfolio model (`best_portfolio_model_20250813_011128.h5`) plus metadata. |
| `portfolio_backtest_results.png` | Last backtest output chart. |
| `venv/` | Local Python virtual environment (TensorFlow, gymnasium, yfinance, pandas, matplotlib). |

## Architecture (current)

```
yfinance daily OHLCV
        |
        v
StockTradingEnvironment / PortfolioTradingEnvironment   (gymnasium env)
        |
        v
DQNAgent / PortfolioDQNAgent                            (Keras MLP, target net, epsilon-greedy)
        |
        v
.h5 model + .pkl metadata
        |
        v
Backtest on hold-out / live yfinance window
```

## How to run

```powershell
# 1. activate the existing venv
.\venv\Scripts\Activate.ps1

# 2. train single-stock model (long — ~1500 episodes)
python rl_model.py

# 3. train portfolio model (longer — ~2000 episodes, 6 stocks)
python multi_rl.py

# 4. backtest the saved portfolio model on the most recent ~90 days
python test_portfolio_model.py

# 5. diagnose model behaviour
python debug_portfolio_model.py
python test.py
```

## Known issues (from the diagnostic scripts themselves)

The comments inside `test.py` already flag these:

1. **Policy over-prefers Hold.** Reward structure rewards capital preservation more than active trading, so the model parks in cash.
2. **No opportunity-cost penalty.** Sitting idle while the market rallies isn't punished.
3. **Sparse diversification incentive.** Bonus exists but is small vs the concentration penalty.
4. **Action space may be unbalanced.** 30 discrete actions with skewed Q-values; some action indices barely fire.
5. **Observation space is too local.** Only the stock's own SMA/RSI — no market-relative or momentum features, no volatility, no regime indicator.
6. **Single training window (2019–2024).** Covers a long bull run + COVID crash, but no test of bear-market regimes or rate-shock periods.
7. **Daily bars only.** Real-time / intraday trading is not actually supported by the current data pipeline.

## Status

**Working:** the training loop runs, models save, backtests execute, diagnostics catch the behavioural problems.
**Not working as a profitable trader:** see issues above, and `STOCK_TRADING_GUIDE.md` for the full assessment + recommended next steps.
