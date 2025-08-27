import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization

pd.set_option('future.no_silent_downcasting', True)
np.seterr(all='ignore')  # suppress warnings

# --- Trading Strategies ---
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

# Add 15 shifted versions of these 5 strategies to get 20 total strategies
def shifted_strategy(base_func, shift):
    def strat(data):
        buy, sell = base_func(data)
        buy = buy.shift(shift).fillna(0).astype(int)
        sell = sell.shift(shift).fillna(0).astype(int)
        return buy, sell
    return strat

base_strategies = [
    macd_strategy,
    rsi_strategy,
    bollinger_bands_strategy,
    ma_crossover_strategy,
    stochastic_oscillator_strategy,
]

strategy_funcs = base_strategies.copy()
shift_values = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

for shift in shift_values:
    for base_strat in base_strategies:
        if len(strategy_funcs) >= 20:
            break
        strategy_funcs.append(shifted_strategy(base_strat, shift))
    if len(strategy_funcs) >= 20:
        break

strategy_names = [f"Strategy_{i+1}" for i in range(len(strategy_funcs))]

# --- Portfolio and optimization functions ---

def calculate_combined_score(df, signals, weights, window=5):
    combined_score = np.zeros(len(df))
    for w, (buy, sell) in zip(weights, signals):
        combined_score += w * (buy - sell)
    combined_score = pd.Series(combined_score).rolling(window=window, min_periods=1).sum().fillna(0).values
    return combined_score

def simulate_trades(df, combined_score, buy_thresh, sell_thresh, initial_capital=100000,
                    max_risk_per_trade=0.05, stop_loss_pct=0.1, take_profit_pct=0.2,
                    commission_pct=0.001, slippage_pct=0.001):
    position = 0
    cash = initial_capital
    shares = 0
    portfolio_values = []
    entry_price = None

    for i in range(len(df)):
        price = df['Close'].iloc[i]
        buy_price = price * (1 + slippage_pct)
        sell_price = price * (1 - slippage_pct)
        score = combined_score[i]

        if position == 0 and score >= buy_thresh:
            alloc_cash = cash * max_risk_per_trade
            shares = alloc_cash / buy_price if buy_price > 0 else 0
            cash -= shares * buy_price
            cash -= alloc_cash * commission_pct  # commission on buy
            position = 1
            entry_price = buy_price
        elif position == 1:
            # Stop loss or take profit triggered
            if entry_price is not None:
                if price <= entry_price * (1 - stop_loss_pct) or price >= entry_price * (1 + take_profit_pct):
                    cash += shares * sell_price
                    cash -= shares * sell_price * commission_pct  # commission on sell
                    shares = 0
                    position = 0
                    entry_price = None
                # Else sell signal triggered
                elif score <= sell_thresh:
                    cash += shares * sell_price
                    cash -= shares * sell_price * commission_pct  # commission on sell
                    shares = 0
                    position = 0
                    entry_price = None
            else:
                # Fallback to sell if no entry_price for some reason
                if score <= sell_thresh:
                    cash += shares * sell_price
                    cash -= shares * sell_price * commission_pct
                    shares = 0
                    position = 0
                    entry_price = None

        portfolio_values.append(cash + shares * price)

    return np.array(portfolio_values)

def portfolio_performance(portfolio_values, dates):
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    avg_daily_return = np.mean(returns)
    std_daily_return = np.std(returns)
    sharpe = (avg_daily_return / std_daily_return) * np.sqrt(252) if std_daily_return != 0 else np.nan
    total_return = portfolio_values[-1] / portfolio_values[0] - 1
    days = (dates.iloc[-1] - dates.iloc[0]).days
    years = days / 365.25 if days > 0 else 1
    cagr = (portfolio_values[-1] / portfolio_values[0]) ** (1/years) - 1
    max_dd = np.max(np.maximum.accumulate(portfolio_values) - portfolio_values) / np.max(np.maximum.accumulate(portfolio_values))
    return {
        'Total Return': total_return,
        'CAGR': cagr,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_dd,
        'Average Daily Return': avg_daily_return,
        'Daily Return StdDev': std_daily_return,
    }

def black_box_function(**params):
    weights = np.array([params[f'w{i}'] for i in range(len(strategy_funcs))])
    buy_thresh = params['buy_thresh']
    sell_thresh = params['sell_thresh']
    max_risk = params.get('max_risk_per_trade', 0.05)
    stop_loss = params.get('stop_loss_pct', 0.1)
    take_profit = params.get('take_profit_pct', 0.2)
    commission = params.get('commission_pct', 0.001)
    slippage = params.get('slippage_pct', 0.001)

    portfolio_agg = None
    dates_ref = None

    for stock_file in stock_files:
        path = os.path.join(data_folder, stock_file)
        df = pd.read_csv(path, parse_dates=['Date'])
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['Date', 'Close'], inplace=True)
        df.reset_index(drop=True, inplace=True)

        signals = []
        for strat_func in strategy_funcs:
            buy, sell = strat_func(df.copy())
            buy = buy.fillna(0).astype(int)
            sell = sell.fillna(0).astype(int)
            signals.append((buy.values, sell.values))

        combined_score = calculate_combined_score(df, signals, weights, window=5)
        portfolio_vals = simulate_trades(df, combined_score, buy_thresh, sell_thresh,
                                         initial_capital=100000, max_risk_per_trade=max_risk,
                                         stop_loss_pct=stop_loss, take_profit_pct=take_profit,
                                         commission_pct=commission, slippage_pct=slippage)

        if portfolio_agg is None:
            portfolio_agg = portfolio_vals / len(stock_files)
            dates_ref = df['Date']
        else:
            min_len = min(len(portfolio_agg), len(portfolio_vals))
            portfolio_agg = portfolio_agg[:min_len] + portfolio_vals[:min_len] / len(stock_files)
            dates_ref = dates_ref[:min_len]

    stats = portfolio_performance(portfolio_agg, dates_ref)
    return stats['Total Return']

if __name__ == "__main__":
    data_folder = 'data'
    stats_folder = 'stats'
    os.makedirs(stats_folder, exist_ok=True)

    stock_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]

    pbounds = {f'w{i}': (0, 1) for i in range(len(strategy_funcs))}
    pbounds.update({
        'buy_thresh': (0, 10),
        'sell_thresh': (-10, 0),
        'max_risk_per_trade': (0.01, 0.1),
        'stop_loss_pct': (0.01, 0.2),
        'take_profit_pct': (0.05, 0.4),
        'commission_pct': (0.0, 0.005),
        'slippage_pct': (0.0, 0.005)
    })

    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        random_state=42,
        verbose=2
    )

    print("Starting Bayesian optimization with 100 iterations and risk controls...")
    optimizer.maximize(init_points=20, n_iter=80)

    best_params = optimizer.max['params']
    best_weights = np.array([best_params[f'w{i}'] for i in range(len(strategy_funcs))])
    buy_thresh = best_params['buy_thresh']
    sell_thresh = best_params['sell_thresh']
    max_risk = best_params['max_risk_per_trade']
    stop_loss = best_params['stop_loss_pct']
    take_profit = best_params['take_profit_pct']
    commission = best_params['commission_pct']
    slippage = best_params['slippage_pct']

    print("\nBest parameters found:")
    print(f" Buy threshold: {buy_thresh:.4f}")
    print(f" Sell threshold: {sell_thresh:.4f}")
    print(f" Max risk per trade: {max_risk:.4f}")
    print(f" Stop loss pct: {stop_loss:.4f}")
    print(f" Take profit pct: {take_profit:.4f}")
    print(f" Commission pct: {commission:.4f}")
    print(f" Slippage pct: {slippage:.4f}")

    all_stats = []
    all_portfolios = []

    for stock_file in stock_files:
        stock_name = stock_file.replace('.csv', '')
        path = os.path.join(data_folder, stock_file)
        df = pd.read_csv(path, parse_dates=['Date'])
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['Date', 'Close'], inplace=True)
        df.reset_index(drop=True, inplace=True)

        signals = []
        for strat_func in strategy_funcs:
            buy, sell = strat_func(df.copy())
            buy = buy.fillna(0).astype(int)
            sell = sell.fillna(0).astype(int)
            signals.append((buy.values, sell.values))

        combined_score = calculate_combined_score(df, signals, best_weights, window=5)
        portfolio_vals = simulate_trades(df, combined_score, buy_thresh, sell_thresh,
                                         initial_capital=100000, max_risk_per_trade=max_risk,
                                         stop_loss_pct=stop_loss, take_profit_pct=take_profit,
                                         commission_pct=commission, slippage_pct=slippage)
        stats = portfolio_performance(portfolio_vals, df['Date'])
        stats['Stock'] = stock_name
        all_stats.append(stats)
        all_portfolios.append((stock_name, df['Date'], portfolio_vals))

    summary_df = pd.DataFrame(all_stats)
    cols = ['Stock', 'Total Return', 'CAGR', 'Sharpe Ratio', 'Max Drawdown', 'Average Daily Return', 'Daily Return StdDev']
    summary_df = summary_df[cols].sort_values('Total Return', ascending=False)
    summary_path = os.path.join(stats_folder, 'final_optimized_portfolio_stats.csv')
    summary_df.to_csv(summary_path, index=False)

    print(f"\nSaved portfolio stats to: {summary_path}")
    print(summary_df.to_string(index=False))

    # Merge all portfolios by outer joining dates for full timeline plot
    merged = None
    for name, dates, vals in all_portfolios:
        temp_df = pd.DataFrame({'Date': dates, f'Portfolio_{name}': vals})
        if merged is None:
            merged = temp_df
        else:
            merged = pd.merge(merged, temp_df, on='Date', how='outer')

    merged.sort_values('Date', inplace=True)
    merged.ffill(inplace=True)
    merged.bfill(inplace=True)

    portfolio_cols = [col for col in merged.columns if col.startswith('Portfolio_')]
    merged['Overall_Portfolio'] = merged[portfolio_cols].mean(axis=1)

    overall_stats = portfolio_performance(merged['Overall_Portfolio'].values, merged['Date'])
    print("\nOverall equal-weighted portfolio performance:")
    for k, v in overall_stats.items():
        print(f"  {k}: {v:.4%}")

    plt.figure(figsize=(12,6))
    plt.plot(merged['Date'], merged['Overall_Portfolio'])
    plt.title('Overall Equal-Weighted Portfolio Value Over Time (With Risk Controls)')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.grid(True)
    plt.show()

    daily_returns = np.diff(merged['Overall_Portfolio'].values) / merged['Overall_Portfolio'].values[:-1]
    plt.figure(figsize=(12,6))
    plt.plot(merged['Date'].iloc[1:], daily_returns)
    plt.title('Overall Equal-Weighted Portfolio Daily Returns Over Time (With Risk Controls)')
    plt.xlabel('Date')
    plt.ylabel('Daily Return')
    plt.grid(True)
    plt.show()

