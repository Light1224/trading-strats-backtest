import os
import pandas as pd
import numpy as np

pd.set_option('future.no_silent_downcasting', True)

# --- Trading Strategies ---
def macd_strategy(data):
    data = data.sort_values('Date').reset_index(drop=True)
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['Buy_Signal'] = (data['MACD'] > data['Signal']) & (data['MACD'].shift(1) <= data['Signal'].shift(1))
    data['Sell_Signal'] = (data['MACD'] < data['Signal']) & (data['MACD'].shift(1) >= data['Signal'].shift(1))
    return data[['Date', 'Buy_Signal', 'Sell_Signal']]

def rsi_strategy(data, period=14, overbought=70, oversold=30):
    data = data.sort_values('Date').reset_index(drop=True)
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
    rs = gain / loss
    data['RSI'] = 100 - 100 / (1 + rs)
    data['Buy_Signal'] = (data['RSI'] < oversold) & (data['RSI'].shift(1) >= oversold)
    data['Sell_Signal'] = (data['RSI'] > overbought) & (data['RSI'].shift(1) <= overbought)
    return data[['Date', 'Buy_Signal', 'Sell_Signal']]

def bollinger_bands_strategy(data, window=20, n_std=2):
    data = data.sort_values('Date').reset_index(drop=True)
    data['MA'] = data['Close'].rolling(window=window).mean()
    data['STD'] = data['Close'].rolling(window=window).std()
    data['Upper'] = data['MA'] + n_std * data['STD']
    data['Lower'] = data['MA'] - n_std * data['STD']
    data['Buy_Signal'] = (data['Close'].shift(1) < data['Lower'].shift(1)) & (data['Close'] > data['Lower'])
    data['Sell_Signal'] = (data['Close'].shift(1) > data['Upper'].shift(1)) & (data['Close'] < data['Upper'])
    return data[['Date', 'Buy_Signal', 'Sell_Signal']]

def ma_crossover_strategy(data, short_win=50, long_win=200):
    data = data.sort_values('Date').reset_index(drop=True)
    data['SMA_short'] = data['Close'].rolling(window=short_win).mean()
    data['SMA_long'] = data['Close'].rolling(window=long_win).mean()
    data['Buy_Signal'] = (data['SMA_short'] > data['SMA_long']) & (data['SMA_short'].shift(1) <= data['SMA_long'].shift(1))
    data['Sell_Signal'] = (data['SMA_short'] < data['SMA_long']) & (data['SMA_short'].shift(1) >= data['SMA_long'].shift(1))
    return data[['Date', 'Buy_Signal', 'Sell_Signal']]

def stochastic_oscillator_strategy(data, k_period=14, d_period=3):
    data = data.sort_values('Date').reset_index(drop=True)
    low_min = data['Low'].rolling(k_period).min()
    high_max = data['High'].rolling(k_period).max()
    data['%K'] = 100 * (data['Close'] - low_min) / (high_max - low_min)
    data['%D'] = data['%K'].rolling(d_period).mean()
    data['Buy_Signal'] = (data['%K'] > data['%D']) & (data['%K'].shift(1) <= data['%D'].shift(1)) & (data['%K'] < 20)
    data['Sell_Signal'] = (data['%K'] < data['%D']) & (data['%K'].shift(1) >= data['%D'].shift(1)) & (data['%K'] > 80)
    return data[['Date', 'Buy_Signal', 'Sell_Signal']]

strategies = {
    'MACD': macd_strategy,
    'RSI': rsi_strategy,
    'Bollinger Bands': bollinger_bands_strategy,
    'MA Crossover': ma_crossover_strategy,
    'Stochastic Oscillator': stochastic_oscillator_strategy,
}

def calculate_combined_score(df, signal_dfs, window=5):
    combined = pd.DataFrame({'Date': df['Date'], 'Close': df['Close']})
    score_series = pd.Series(np.zeros(len(df)), index=df.index)

    for signal_df in signal_dfs:
        signal_df = signal_df.set_index(df.index)
        buy_int = signal_df['Buy_Signal'].astype(int)
        sell_int = signal_df['Sell_Signal'].astype(int)
        score_series += buy_int - sell_int

    combined['Score'] = score_series.rolling(window=window, min_periods=1).sum()
    return combined

def simulate_trades_from_score(df, buy_thresh, sell_thresh, initial_capital=100000):
    df = df.copy().reset_index(drop=True)
    df['Position'] = 0

    for i in range(len(df)):
        if i == 0:
            df.at[i, 'Position'] = 0
            continue
        prev_pos = df.at[i-1, 'Position']
        score = df.at[i, 'Score']
        if score >= buy_thresh:
            df.at[i, 'Position'] = 1
        elif score <= sell_thresh:
            df.at[i, 'Position'] = 0
        else:
            df.at[i, 'Position'] = prev_pos

    df['Returns'] = df['Close'].pct_change().fillna(0)
    df['Strategy_Returns'] = df['Position'] * df['Returns']
    df['Portfolio_Value'] = initial_capital * (1 + df['Strategy_Returns']).cumprod()
    df['Portfolio_Return'] = df['Portfolio_Value'].pct_change().fillna(0)
    return df

def calculate_return_statistics(portfolio_df):
    total_return = portfolio_df['Portfolio_Value'].iloc[-1] / portfolio_df['Portfolio_Value'].iloc[0] - 1
    daily_returns = portfolio_df['Portfolio_Return']
    avg_return = daily_returns.mean()
    std_return = daily_returns.std()
    sharpe_ratio = (avg_return / std_return) * np.sqrt(252) if std_return != 0 else np.nan
    max_drawdown = ((portfolio_df['Portfolio_Value'].cummax() - portfolio_df['Portfolio_Value']) / portfolio_df['Portfolio_Value'].cummax()).max()
    num_trades = portfolio_df['Position'].diff().abs().sum()
    winning_trades = ((daily_returns > 0) & (portfolio_df['Position'] == 1)).sum()
    losing_trades = ((daily_returns < 0) & (portfolio_df['Position'] == 1)).sum()
    stats = {
        'Total Return': total_return,
        'Average Daily Return': avg_return,
        'Return StdDev': std_return,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Number of Trades': num_trades,
        'Winning Trades (days)': winning_trades,
        'Losing Trades (days)': losing_trades
    }
    return stats

def main():
    data_folder = 'data'
    stats_folder = 'stats'
    os.makedirs(stats_folder, exist_ok=True)

    buy_thresholds = range(0, 11)         # from 0 to 10 inclusive
    sell_thresholds = range(0, -11, -1)  # from 0 down to -10 inclusive

    best_portfolio_stats = []
    overall_positions = []  # store all portfolios for overall portfolio analysis
    overall_returns = []

    stock_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    for stock_file in stock_files:
        stock_name = stock_file.replace('.csv', '')
        filepath = os.path.join(data_folder, stock_file)
        df = pd.read_csv(filepath, parse_dates=['Date'])
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['Date', 'Close']).reset_index(drop=True)

        print(f"Processing stock: {stock_name}")

        signal_dfs = []
        for strat_name, strat_func in strategies.items():
            sig_df = strat_func(df.copy())
            signal_dfs.append(sig_df)

        combined_df = calculate_combined_score(df, signal_dfs, window=5)

        best_stats = None
        best_params = None
        best_portfolio = None

        for buy_thres in buy_thresholds:
            for sell_thres in sell_thresholds:
                portfolio_df = simulate_trades_from_score(combined_df, buy_thres, sell_thres)
                stats = calculate_return_statistics(portfolio_df)
                if (best_stats is None) or (stats['Total Return'] > best_stats['Total Return']):
                    best_stats = stats
                    best_params = (buy_thres, sell_thres)
                    best_portfolio = portfolio_df

        print(f"Best thresholds for {stock_name}: Buy >= {best_params[0]}, Sell <= {best_params[1]}")
        for k, v in best_stats.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

        best_stats.update({'Stock': stock_name, 'Buy Threshold': best_params[0], 'Sell Threshold': best_params[1]})
        best_portfolio_stats.append(best_stats)

        best_portfolio = best_portfolio.copy()
        best_portfolio['Stock'] = stock_name
        overall_positions.append(best_portfolio)
        overall_returns.append(best_stats['Total Return'])

    summary_df = pd.DataFrame(best_portfolio_stats)
    summary_df = summary_df[['Stock', 'Buy Threshold', 'Sell Threshold', 'Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Number of Trades']]
    summary_df = summary_df.sort_values(by='Total Return', ascending=False)

    save_path = os.path.join(stats_folder, 'best_thresholds_portfolio_performance.csv')
    summary_df.to_csv(save_path, index=False)
    print("\nSummary saved to:", save_path)
    print(summary_df.to_string(index=False))

    # Construct overall portfolio by equally weighting stocks daily
    if overall_positions:
        merged_portfolio = None
        for pf in overall_positions:
            pf_temp = pf[['Date', 'Portfolio_Value']].copy()
            pf_temp.rename(columns={'Portfolio_Value': f'Portfolio_{pf["Stock"].iloc[0]}'}, inplace=True)
            if merged_portfolio is None:
                merged_portfolio = pf_temp
            else:
                merged_portfolio = pd.merge(merged_portfolio, pf_temp, on='Date', how='outer')
        merged_portfolio.fillna(method='ffill', inplace=True)
        merged_portfolio.fillna(method='bfill', inplace=True)
        # Average portfolio value of all stocks per day
        merged_portfolio['Overall_Portfolio_Value'] = merged_portfolio.iloc[:, 1:].mean(axis=1)
        overall_return = merged_portfolio['Overall_Portfolio_Value'].iloc[-1] / merged_portfolio['Overall_Portfolio_Value'].iloc[0] - 1
        print(f"\nOverall equal-weighted portfolio return across all stocks: {overall_return:.4f}")

if __name__ == "__main__":
    main()

