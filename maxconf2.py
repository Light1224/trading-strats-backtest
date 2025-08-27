import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization

pd.set_option('future.no_silent_downcasting', True)
np.seterr(all='ignore')  # ignore warnings for numeric issues

# --- Define 8 base strategies ---

def macd_strategy(data):
    data = data.sort_values('Date').reset_index(drop=True)
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    buy = (data['MACD'] > data['Signal']) & (data['MACD'].shift(1) <= data['Signal'].shift(1))
    sell = (data['MACD'] < data['Signal']) & (data['MACD'].shift(1) >= data['Signal'].shift(1))
    return buy.fillna(False).astype(int), sell.fillna(False).astype(int)

def rsi_strategy(data, period=14, overbought=70, oversold=30):
    data = data.sort_values('Date').reset_index(drop=True)
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
    rs = gain / loss
    rsi = 100 - 100 / (1 + rs)
    buy = (rsi < oversold) & (rsi.shift(1) >= oversold)
    sell = (rsi > overbought) & (rsi.shift(1) <= overbought)
    return buy.fillna(False).astype(int), sell.fillna(False).astype(int)

def bollinger_bands_strategy(data, window=20, n_std=2):
    data = data.sort_values('Date').reset_index(drop=True)
    ma = data['Close'].rolling(window=window).mean()
    std = data['Close'].rolling(window=window).std()
    upper = ma + n_std * std
    lower = ma - n_std * std
    buy = (data['Close'].shift(1) < lower.shift(1)) & (data['Close'] > lower)
    sell = (data['Close'].shift(1) > upper.shift(1)) & (data['Close'] < upper)
    return buy.fillna(False).astype(int), sell.fillna(False).astype(int)

def ma_crossover_strategy(data, short_win=50, long_win=200):
    data = data.sort_values('Date').reset_index(drop=True)
    sma_short = data['Close'].rolling(window=short_win).mean()
    sma_long = data['Close'].rolling(window=long_win).mean()
    buy = (sma_short > sma_long) & (sma_short.shift(1) <= sma_long.shift(1))
    sell = (sma_short < sma_long) & (sma_short.shift(1) >= sma_long.shift(1))
    return buy.fillna(False).astype(int), sell.fillna(False).astype(int)

def stochastic_oscillator_strategy(data, k_period=14, d_period=3):
    data = data.sort_values('Date').reset_index(drop=True)
    low_min = data['Low'].rolling(k_period).min()
    high_max = data['High'].rolling(k_period).max()
    k = 100 * (data['Close'] - low_min) / (high_max - low_min)
    d = k.rolling(d_period).mean()
    buy = (k > d) & (k.shift(1) <= d.shift(1)) & (k < 20)
    sell = (k < d) & (k.shift(1) >= d.shift(1)) & (k > 80)
    return buy.fillna(False).astype(int), sell.fillna(False).astype(int)

def adx_strategy(data, n=14, threshold=25):
    data = data.sort_values('Date').reset_index(drop=True)
    high_diff = data['High'] - data['High'].shift(1)
    low_diff = data['Low'].shift(1) - data['Low']
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    tr1 = data['High'] - data['Low']
    tr2 = abs(data['High'] - data['Close'].shift(1))
    tr3 = abs(data['Low'] - data['Close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(n).mean()
    plus_di = 100 * (pd.Series(plus_dm).rolling(n).sum() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(n).sum() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(n).mean()
    buy = (adx > threshold) & (plus_di > minus_di) & (plus_di.shift(1) <= minus_di.shift(1))
    sell = (adx > threshold) & (plus_di < minus_di) & (plus_di.shift(1) >= minus_di.shift(1))
    return buy.fillna(False).astype(int), sell.fillna(False).astype(int)

def momentum_strategy(data, window=10):
    data = data.sort_values('Date').reset_index(drop=True)
    mom = data['Close'] - data['Close'].shift(window)
    buy = mom > 0
    sell = mom < 0
    return buy.fillna(False).astype(int), sell.fillna(False).astype(int)

# List of base 8 strategies
base_strategies = [
    macd_strategy,
    rsi_strategy,
    bollinger_bands_strategy,
    ma_crossover_strategy,
    stochastic_oscillator_strategy,
    adx_strategy,
    momentum_strategy,
]

# Create shifted variants to extend strategy list to 20 total
def shifted_strategy(base_func, shift):
    def strat(data):
        buy, sell = base_func(data)
        buy = buy.shift(shift).fillna(0).astype(int)
        sell = sell.shift(shift).fillna(0).astype(int)
        return buy, sell
    return strat

strategy_funcs = base_strategies.copy()
shift_values = [1, 2, 3]  # 3 shifts * 8 strategies = 24 but we truncate to 20 strategies total
for shift in shift_values:
    for base_func in base_strategies:
        if len(strategy_funcs) >= 20:
            break
        strategy_funcs.append(shifted_strategy(base_func, shift))
    if len(strategy_funcs) >= 20:
        break

# Names just for reference
strategy_names = [f"Strategy_{i+1}" for i in range(len(strategy_funcs))]

# --- Functions for scoring and trading ---

def calculate_combined_score(df, signals, weights, window=5):
    combined_score = np.zeros(len(df))
    for w, (buy, sell) in zip(weights, signals):
        combined_score += w * (buy - sell)
    combined_score = pd.Series(combined_score).rolling(window=window, min_periods=1).sum().fillna(0).values
    return combined_score

def simulate_trades(df, score, buy_thresh, sell_thresh, initial_capital=100000):
    pos = 0
    cash = initial_capital
    shares = 0
    portfolio_vals = []
    for i in range(len(df)):
        price = df['Close'].iloc[i]
        s = score[i]
        if pos == 0 and s >= buy_thresh:
            shares = cash / price
            cash = 0
            pos = 1
        elif pos == 1 and s <= sell_thresh:
            cash = shares * price
            shares = 0
            pos = 0
        portfolio_vals.append(cash + shares * price)
    return np.array(portfolio_vals)

def portfolio_performance(portfolio_vals, dates):
    returns = np.diff(portfolio_vals) / portfolio_vals[:-1]
    avg_daily = np.mean(returns)
    std_daily = np.std(returns)
    sharpe = (avg_daily / std_daily) * np.sqrt(252) if std_daily != 0 else np.nan
    total_return = portfolio_vals[-1] / portfolio_vals[0] - 1
    days = (dates.iloc[-1] - dates.iloc[0]).days
    years = days / 365.25 if days > 0 else 1
    cagr = (portfolio_vals[-1] / portfolio_vals[0]) ** (1/years) - 1
    max_dd = np.max(np.maximum.accumulate(portfolio_vals) - portfolio_vals) / np.max(np.maximum.accumulate(portfolio_vals))
    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
        "Avg Daily Return": avg_daily,
        "Daily Return Std": std_daily
    }

def black_box_function(**params):
    weights = np.array([params[f'w{i}'] for i in range(len(strategy_funcs))])
    buy = params['buy_thresh']
    sell = params['sell_thresh']
    combined_portfolio = None
    combined_dates = None

    for file in stock_files:
        path = os.path.join(data_folder, file)
        df = pd.read_csv(path, parse_dates=['Date'])
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['Date', 'Close'], inplace=True)
        df.reset_index(drop=True, inplace=True)

        signals = []
        for strat in strategy_funcs:
            buy_sig, sell_sig = strat(df.copy())
            buy_sig = buy_sig.fillna(0).astype(int)
            sell_sig = sell_sig.fillna(0).astype(int)
            signals.append((buy_sig.values, sell_sig.values))

        score = calculate_combined_score(df, signals, weights, window=5)
        vals = simulate_trades(df, score, buy, sell)

        if combined_portfolio is None:
            combined_portfolio = vals / len(stock_files)
            combined_dates = df['Date']
        else:
            min_len = min(len(combined_portfolio), len(vals))
            combined_portfolio = combined_portfolio[:min_len] + vals[:min_len] / len(stock_files)
            combined_dates = combined_dates[:min_len]

    stats = portfolio_performance(combined_portfolio, combined_dates)
    return stats["Total Return"]

if __name__=="__main__":
    data_folder = "data"
    stats_folder = "stats"
    os.makedirs(stats_folder, exist_ok=True)

    stock_files = [f for f in os.listdir(data_folder) if f.endswith(".csv")]

    pbounds = {f'w{i}': (0, 1) for i in range(len(strategy_funcs))}
    pbounds['buy_thresh'] = (0, 10)
    pbounds['sell_thresh'] = (-10, 0)

    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        random_state=42,
        verbose=2,
    )

    print("Starting Bayesian Optimization (100 iterations)...")
    optimizer.maximize(init_points=20, n_iter=80)  # total 100 iterations

    best_params = optimizer.max["params"]
    best_weights = np.array([best_params[f'w{i}'] for i in range(len(strategy_funcs))])
    buy_thresh = best_params["buy_thresh"]
    sell_thresh = best_params["sell_thresh"]

    print(f"\nBest found parameters:\nBuy threshold: {buy_thresh:.3f}, Sell threshold: {sell_thresh:.3f}")

    all_stats = []
    all_portfolios = []

    for file in stock_files:
        stock_name = file.replace(".csv", "")
        path = os.path.join(data_folder, file)
        df = pd.read_csv(path, parse_dates=["Date"])
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["Date", "Close"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        signals = []
        for strat in strategy_funcs:
            b, s = strat(df.copy())
            signals.append((b.fillna(0).astype(int).values, s.fillna(0).astype(int).values))

        score = calculate_combined_score(df, signals, best_weights, window=5)
        vals = simulate_trades(df, score, buy_thresh, sell_thresh)
        stats = portfolio_performance(vals, df["Date"])
        stats["Stock"] = stock_name
        all_stats.append(stats)
        all_portfolios.append((stock_name, df["Date"], vals))

    summary_df = pd.DataFrame(all_stats)
    cols = ["Stock", "Total Return", "CAGR", "Sharpe Ratio", "Max Drawdown", "Avg Daily Return", "Daily Return Std"]
    summary_df = summary_df[cols].sort_values("Total Return", ascending=False)
    summary_path = os.path.join(stats_folder, "final_optimized_portfolio_stats.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"\nSaved portfolio stats to: {summary_path}")
    print(summary_df.to_string(index=False))

    # Aggregate equal-weight portfolio for full available timespan
    merged = None
    for (name, dates, vals) in all_portfolios:
        pf = pd.DataFrame({"Date": dates, f"Portfolio_{name}": vals})
        if merged is None:
            merged = pf
        else:
            merged = pd.merge(merged, pf, on="Date", how="outer")

    merged.sort_values("Date", inplace=True)
    merged.fillna(method="ffill", inplace=True)
    merged.fillna(method="bfill", inplace=True)
    pf_cols = [c for c in merged.columns if c.startswith("Portfolio_")]
    merged["Overall_Portfolio"] = merged[pf_cols].mean(axis=1)

    overall_stats = portfolio_performance(merged["Overall_Portfolio"].values, merged["Date"])
    print("\nOverall Portfolio Performance (Equal-Weighted):")
    for k, v in overall_stats.items():
        print(f"  {k}: {v:.4%}")

    plt.figure(figsize=(12, 6))
    plt.plot(merged["Date"], merged["Overall_Portfolio"])
    plt.title("Overall Equal-Weighted Portfolio Value Over Time")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plt.show()

    daily_rets = np.diff(merged["Overall_Portfolio"]) / merged["Overall_Portfolio"][:-1]
    plt.figure(figsize=(12, 6))
    plt.plot(merged["Date"].iloc[1:], daily_rets)
    plt.title("Overall Equal-Weighted Portfolio Daily Returns Over Time")
    plt.xlabel("Date")
    plt.ylabel("Daily Return")
    plt.grid(True)
    plt.show()

