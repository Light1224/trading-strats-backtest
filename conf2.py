import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting

# --- Strategy Implementations ---

def macd_strategy(data):
    data = data.sort_values('Date').reset_index(drop=True)
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['Buy_Signal'] = (data['MACD'] > data['Signal']) & (data['MACD'].shift(1) <= data['Signal'].shift(1))
    data['Sell_Signal'] = (data['MACD'] < data['Signal']) & (data['MACD'].shift(1) >= data['Signal'].shift(1))
    return data[['Date','Buy_Signal','Sell_Signal']]

def rsi_strategy(data, period=14, overbought=70, oversold=30):
    data = data.sort_values('Date').reset_index(drop=True)
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
    rs = gain / loss
    data['RSI'] = 100 - 100 / (1 + rs)
    data['Buy_Signal'] = (data['RSI'] < oversold) & (data['RSI'].shift(1) >= oversold)
    data['Sell_Signal'] = (data['RSI'] > overbought) & (data['RSI'].shift(1) <= overbought)
    return data[['Date','Buy_Signal','Sell_Signal']]

def bollinger_bands_strategy(data, window=20, n_std=2):
    data = data.sort_values('Date').reset_index(drop=True)
    data['MA'] = data['Close'].rolling(window).mean()
    data['STD'] = data['Close'].rolling(window).std()
    data['Upper'] = data['MA'] + n_std * data['STD']
    data['Lower'] = data['MA'] - n_std * data['STD']
    data['Buy_Signal'] = (data['Close'].shift(1) < data['Lower'].shift(1)) & (data['Close'] > data['Lower'])
    data['Sell_Signal'] = (data['Close'].shift(1) > data['Upper'].shift(1)) & (data['Close'] < data['Upper'])
    return data[['Date','Buy_Signal','Sell_Signal']]

def ma_crossover_strategy(data, short_win=50, long_win=200):
    data = data.sort_values('Date').reset_index(drop=True)
    data['SMA_short'] = data['Close'].rolling(window=short_win).mean()
    data['SMA_long'] = data['Close'].rolling(window=long_win).mean()
    data['Buy_Signal'] = (data['SMA_short'] > data['SMA_long']) & (data['SMA_short'].shift(1) <= data['SMA_long'].shift(1))
    data['Sell_Signal'] = (data['SMA_short'] < data['SMA_long']) & (data['SMA_short'].shift(1) >= data['SMA_long'].shift(1))
    return data[['Date','Buy_Signal','Sell_Signal']]

def stochastic_oscillator_strategy(data, k_period=14, d_period=3):
    data = data.sort_values('Date').reset_index(drop=True)
    low_min = data['Low'].rolling(k_period).min()
    high_max = data['High'].rolling(k_period).max()
    data['%K'] = 100 * (data['Close'] - low_min) / (high_max - low_min)
    data['%D'] = data['%K'].rolling(d_period).mean()
    data['Buy_Signal'] = (data['%K'] > data['%D']) & (data['%K'].shift(1) <= data['%D'].shift(1)) & (data['%K'] < 20)
    data['Sell_Signal'] = (data['%K'] < data['%D']) & (data['%K'].shift(1) >= data['%D'].shift(1)) & (data['%K'] > 80)
    return data[['Date','Buy_Signal','Sell_Signal']]

def calculate_adx(data, n=14):
    data = data.sort_values('Date').reset_index(drop=True)
    high_diff = data['High'] - data['High'].shift(1)
    low_diff = data['Low'].shift(1) - data['Low']
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
    tr1 = data['High'] - data['Low']
    tr2 = abs(data['High'] - data['Close'].shift(1))
    tr3 = abs(data['Low'] - data['Close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(n, min_periods=1).mean()
    plus_di = 100 * (pd.Series(plus_dm).rolling(n, min_periods=1).sum() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(n, min_periods=1).sum() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(n, min_periods=1).mean()
    data['+DI'] = plus_di.values
    data['-DI'] = minus_di.values
    data['ADX'] = adx.values
    return data

def trend_following_strategy(data, adx_threshold=25):
    data = calculate_adx(data)
    data['Buy_Signal'] = (data['ADX'] > adx_threshold) & (data['+DI'] > data['-DI']) & (data['+DI'].shift(1) <= data['-DI'].shift(1))
    data['Sell_Signal'] = (data['ADX'] > adx_threshold) & (data['+DI'] < data['-DI']) & (data['+DI'].shift(1) >= data['-DI'].shift(1))
    return data[['Date','Buy_Signal','Sell_Signal']]

def breakout_strategy(data, lookback=20):
    data = data.sort_values('Date').reset_index(drop=True)
    data['Highest_High'] = data['High'].shift(1).rolling(lookback).max()
    data['Lowest_Low'] = data['Low'].shift(1).rolling(lookback).min()
    data['Buy_Signal'] = data['Close'] > data['Highest_High']
    data['Sell_Signal'] = data['Close'] < data['Lowest_Low']
    return data[['Date','Buy_Signal','Sell_Signal']]

def mean_reversion_strategy(data, window=20, num_std=2):
    data = data.sort_values('Date').reset_index(drop=True)
    data['MA'] = data['Close'].rolling(window=window).mean()
    data['STD'] = data['Close'].rolling(window=window).std()
    data['Upper_BB'] = data['MA'] + num_std * data['STD']
    data['Lower_BB'] = data['MA'] - num_std * data['STD']
    data['Buy_Signal'] = (data['Close'].shift(1) < data['Lower_BB'].shift(1)) & (data['Close'] > data['Lower_BB'])
    data['Sell_Signal'] = (data['Close'].shift(1) > data['Upper_BB'].shift(1)) & (data['Close'] < data['Upper_BB'])
    return data[['Date','Buy_Signal','Sell_Signal']]

strategies = {
    'MACD': macd_strategy,
    'RSI': rsi_strategy,
    'Bollinger Bands': bollinger_bands_strategy,
    'MA Crossover': ma_crossover_strategy,
    'Stochastic Oscillator': stochastic_oscillator_strategy,
    'Trend Following': trend_following_strategy,
    'Breakout': breakout_strategy,
    'Mean Reversion': mean_reversion_strategy,
}

# --- Combined Score ---

def calculate_combined_score(df, strategy_results, window=5):
    combined = pd.DataFrame({'Date': df['Date'], 'Close': df['Close']})
    total_signals = pd.Series(0, index=df.index, dtype=float)

    for strat_df in strategy_results:
        strat_df = strat_df.set_index(df.index)
        buy_int = strat_df['Buy_Signal'].astype(int)
        sell_int = strat_df['Sell_Signal'].astype(int)
        total_signals += buy_int - sell_int

    combined['Score'] = total_signals.rolling(window=window, min_periods=1).sum()
    return combined

# --- 3D Plot ---

def plot_signals_3d(df, stock_name):
    signal_mask = (df['Buy_Signal'] | df['Sell_Signal'])
    signal_df = df[signal_mask].copy()
    
    if signal_df.empty:
        print(f"No signals for {stock_name}")
        return

    x = np.arange(len(signal_df))
    y = signal_df['Close'].values
    z = signal_df['Score'].values
    c = ['green' if b else 'red' for b in signal_df['Buy_Signal']]

    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=c, s=50, alpha=0.8)
    ax.set_xlabel('Signal Index')
    ax.set_ylabel('Close Price')
    ax.set_zlabel('Combined Score')
    ax.set_title(f'{stock_name} Signals: Price vs Score')
    plt.show()

# --- Main Execution ---

def main():
    data_folder = 'data'
    rolling_window = 5
    stock_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]

    for stock_file in stock_files:
        stock_name = stock_file.replace('.csv','')
        filepath = os.path.join(data_folder, stock_file)
        df = pd.read_csv(filepath, parse_dates=['Date'])
        for col in ['Open','High','Low','Close','Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['Date','Close','High','Low']).reset_index(drop=True)

        # Run all strategies
        strategy_results = [func(df.copy()) for func in strategies.values()]

        # Calculate combined score
        combined_df = calculate_combined_score(df, strategy_results, window=rolling_window)

        # Add any-signal flags
        any_signals = pd.DataFrame(False, index=df.index, columns=['Buy_Signal','Sell_Signal'])
        for strat_df in strategy_results:
            any_signals['Buy_Signal'] |= strat_df['Buy_Signal']
            any_signals['Sell_Signal'] |= strat_df['Sell_Signal']
        combined_df['Buy_Signal'] = any_signals['Buy_Signal']
        combined_df['Sell_Signal'] = any_signals['Sell_Signal']

        # 3D plot
        plot_signals_3d(combined_df, stock_name)

if __name__ == "__main__":
    main()

