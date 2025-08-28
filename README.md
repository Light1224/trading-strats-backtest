# Trading Strategies Backtest & Optimisation

This project is a quantitative trading research framework designed for analysing and optimising systematic strategies on Indian equities. It combines multiple technical trading strategies into an ensemble model and evaluates them through a realistic backtesting simulator built from scratch.

## Key Features

- Strategies Implemented
- MACD, RSI, Bollinger Bands, Moving Average Crossover, Stochastic Oscillator, Trend Following, Breakout, Mean Reversion.
Backtesting Simulator (research-grade)
Supports long and short trades.
Volatility-adjusted position sizing to control risk dynamically.
Kelly criterion (fractional Kelly) applied for optimal capital allocation.
Drawdown controls to reduce risk after portfolio losses.
Leverage constraints and risk budgeting across open positions.
Trade-by-trade execution with equity curve tracking and full trade logs.

## Optimisation
Uses Bayesian optimisation to tune ensemble strategy weights and adaptive buy/sell thresholds.
Enables dynamic adjustment across different market regimes.
Risk & Performance Analytics
Computes Sharpe, Sortino, Calmar ratios, volatility, skew, kurtosis.
Logs trade statistics (win rate, profit factor, average P&L).
Integrated Streamlit dashboard for interactive analysis.

## Results Across Market Regimes

- The simulator was tested across 15+ years of Indian equity data, covering multiple market environments:

2007–2017 (Financial Crisis + Recovery + Bull Run)
- Total Return: 777.91%
- CAGR: 24.28%
- Sharpe Ratio: 1.01
- Sortino Ratio: 1.45
- Max Drawdown: -45.62%
- Calmar Ratio: 0.53
- Volatility (ann.): 25.12%
- Total Trades: 3,295
- Win Rate: 78.86%

<img width="1214" height="920" alt="image" src="https://github.com/user-attachments/assets/35bc1e7f-7552-4fe4-aff3-3bf06169b511" />

2020–2025 (Post-COVID + Recent Bull Market)
- Total Return: 173.89%
- CAGR: 22.31%
- Sharpe Ratio: 0.84
- Sortino Ratio: 0.95
- Max Drawdown: -46.39%
- Calmar Ratio: 0.48
- Volatility (ann.): 29.95%
- Total Trades: 1,927
- Win Rate: 73.91%


<img width="607" height="460" alt="image" src="https://github.com/user-attachments/assets/4cb51426-8463-4e01-9238-b7d4013eca6d" />

2010 - 2020
- Total Return: 92.51%
- CAGR: 6.78%
- Sharpe Ratio: 0.40
- Sortino Ratio: 0.53
- Max Drawdown: -50.88%
- Calmar Ratio: 0.13
- Volatility (ann.): 24.38%
- Total Trades: 4919
- Win Rate: 73.34%
- Average Trade P&L: -16.25
- Returns Skew: 0.2611
- Returns Kurtosis: 10.8918

<img width="607" height="460" alt="image" src="https://github.com/user-attachments/assets/e1a9b01e-b14b-4778-a9cb-d69aacef7d1e" />

