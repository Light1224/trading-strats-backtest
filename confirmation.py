import os
import pandas as pd
import numpy as np

pd.set_option('future.no_silent_downcasting', True)

# --- Trading Strategy Implementations ---

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
    gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
    rs = gain / loss
    data['RSI'] = 100 - 100 / (1 + rs)
    data['Buy_Signal'] = (data['RSI'] < oversold) & (data['RSI'].shift(1) >= oversold)
    data['Sell_Signal'] = (data['RSI'] > overbought) & (data['RSI'].shift(1) <= overbought)
    return data[['Date', 'Buy_Signal', 'Sell_Signal']]

def bollinger_bands_strategy(data, window=20, n_std=2):
    data = data.sort_values('Date').reset_index(drop=True)
    data['MA'] = data['Close'].rolling(window).mean()
    data['STD'] = data['Close'].rolling(window).std()
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

# (Other strategies also defined similarly / omitted here for brevity)

strategies = {
    'MACD': macd_strategy,
    'RSI': rsi_strategy,
    'Bollinger Bands': bollinger_bands_strategy,
    'Moving Average Crossover': ma_crossover_strategy,
    'Stochastic Oscillator': stochastic_oscillator_strategy,
    # add other strategies if needed
}

# --- Calculate combined score (sum of +1 per buy and -1 per sell) ---

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

def simulate_trades_from_score(df, initial_capital=100000):
    df = df.copy().reset_index(drop=True)
    df['Position'] = 0
    # Define holding position based on sign of Score (simple: >0 long, <=0 flat)
    df.loc[df['Score'] > 0, 'Position'] = 1
    df.loc[df['Score'] <= 0, 'Position'] = 0
    df['Position'] = df['Position'].shift(1).fillna(0)  # position held from prior day
    
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
    rolling_window = 5
    initial_capital = 100000  # Starting capital

    stock_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]

    for stock_file in stock_files:
        stock_name = stock_file.replace('.csv', '')
        filepath = os.path.join(data_folder, stock_file)
        df = pd.read_csv(filepath, parse_dates=['Date'])
        for col in ['Open','High','Low','Close','Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['Date','Close']).reset_index(drop=True)

        # Generate signals for all strategies
        signal_dfs = []
        print(f"\nProcessing stock: {stock_name}")
        for strat_name, strat_func in strategies.items():
            sig_df = strat_func(df.copy())
            signal_dfs.append(sig_df)
            print(f"  Strategy {strat_name} generated {sig_df['Buy_Signal'].sum()} buy and {sig_df['Sell_Signal'].sum()} sell signals.")

        # Calculate combined score and rolling sum
        combined_df = calculate_combined_score(df, signal_dfs, window=rolling_window)

        # Simulate trades based on score
        portfolio_df = simulate_trades_from_score(combined_df, initial_capital=initial_capital)

        # Calculate and print return statistics
        stats = calculate_return_statistics(portfolio_df)
        print(f"Performance stats for combined strategy score (rolling window {rolling_window} days):")
        for key, val in stats.items():
            if isinstance(val, float):
                print(f"  {key}: {val:.4f}")
            else:
                print(f"  {key}: {val}")

        # Optionally: Save portfolio_df or stats to CSV or plot portfolio value and returns here

if __name__ == "__main__":
    main()

