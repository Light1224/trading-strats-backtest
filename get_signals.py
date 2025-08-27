import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Strategy Implementations ---

def macd_strategy(data):
    data = data.sort_values('Date').reset_index(drop=True)
    data['EMA12'] = data['Close'].ewm(span=12).mean()
    data['EMA26'] = data['Close'].ewm(span=26).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal'] = data['MACD'].ewm(span=9).mean()
    data['Buy'] = (data['MACD'] > data['Signal']) & (data['MACD'].shift() <= data['Signal'].shift())
    data['Sell'] = (data['MACD'] < data['Signal']) & (data['MACD'].shift() >= data['Signal'].shift())
    return data

def rsi_strategy(data, period=14, overbought=70, oversold=30):
    data = data.sort_values('Date').reset_index(drop=True)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    data['RSI'] = 100 - 100 / (1 + rs)
    data['Buy'] = (data['RSI'] > oversold) & (data['RSI'].shift() <= oversold)
    data['Sell'] = (data['RSI'] < overbought) & (data['RSI'].shift() >= overbought)
    return data

def bollinger_bands_strategy(data, window=20, n_std=2):
    data = data.sort_values('Date').reset_index(drop=True)
    data['MA'] = data['Close'].rolling(window).mean()
    data['STD'] = data['Close'].rolling(window).std()
    data['Upper'] = data['MA'] + n_std * data['STD']
    data['Lower'] = data['MA'] - n_std * data['STD']
    data['Buy'] = (data['Close'].shift() < data['Lower'].shift()) & (data['Close'] > data['Lower'])
    data['Sell'] = (data['Close'].shift() > data['Upper'].shift()) & (data['Close'] < data['Upper'])
    return data

def ma_crossover_strategy(data, short_win=50, long_win=200):
    data = data.sort_values('Date').reset_index(drop=True)
    data['SMA_short'] = data['Close'].rolling(window=short_win).mean()
    data['SMA_long'] = data['Close'].rolling(window=long_win).mean()
    data['Buy'] = (data['SMA_short'] > data['SMA_long']) & (data['SMA_short'].shift() <= data['SMA_long'].shift())
    data['Sell'] = (data['SMA_short'] < data['SMA_long']) & (data['SMA_short'].shift() >= data['SMA_long'].shift())
    return data

def stochastic_oscillator_strategy(data, k_period=14, d_period=3):
    data = data.sort_values('Date').reset_index(drop=True)
    low_min = data['Low'].rolling(k_period).min()
    high_max = data['High'].rolling(k_period).max()
    data['%K'] = 100 * (data['Close'] - low_min) / (high_max - low_min)
    data['%D'] = data['%K'].rolling(d_period).mean()
    data['Buy'] = (data['%K'] > data['%D']) & (data['%K'].shift() <= data['%D'].shift()) & (data['%K'] < 20)
    data['Sell'] = (data['%K'] < data['%D']) & (data['%K'].shift() >= data['%D'].shift()) & (data['%K'] > 80)
    return data

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
    atr = tr.rolling(n).mean()
    plus_di = 100 * (pd.Series(plus_dm).rolling(n).sum() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(n).sum() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(n).mean()
    data['+DI'] = plus_di.values
    data['-DI'] = minus_di.values
    data['ADX'] = adx.values
    return data

def trend_following_strategy(data, adx_threshold=25):
    data = calculate_adx(data)
    data['Buy'] = (data['ADX'] > adx_threshold) & (data['+DI'] > data['-DI']) & (data['+DI'].shift() <= data['-DI'].shift())
    data['Sell'] = (data['ADX'] > adx_threshold) & (data['+DI'] < data['-DI']) & (data['+DI'].shift() >= data['-DI'].shift())
    return data

def breakout_strategy(data, lookback=20):
    data = data.sort_values('Date').reset_index(drop=True)
    data['Highest_High'] = data['High'].shift(1).rolling(lookback).max()
    data['Lowest_Low'] = data['Low'].shift(1).rolling(lookback).min()
    data['Buy'] = data['Close'] > data['Highest_High']
    data['Sell'] = data['Close'] < data['Lowest_Low']
    data['Buy_Signal'] = data['Buy'] & (~data['Buy'].shift(1).fillna(False))
    data['Sell_Signal'] = data['Sell'] & (~data['Sell'].shift(1).fillna(False))
    return data

def mean_reversion_strategy(data, window=20, num_std=2):
    data = data.sort_values('Date').reset_index(drop=True)
    data['MA'] = data['Close'].rolling(window).mean()
    data['STD'] = data['Close'].rolling(window).std()
    data['Upper_BB'] = data['MA'] + num_std * data['STD']
    data['Lower_BB'] = data['MA'] - num_std * data['STD']
    data['Buy'] = (data['Close'].shift(1) < data['Lower_BB'].shift(1)) & (data['Close'] > data['Lower_BB'])
    data['Sell'] = (data['Close'].shift(1) > data['Upper_BB'].shift(1)) & (data['Close'] < data['Upper_BB'])
    return data

# --- Plotting Function ---

def plot_signals(data, strategy_name, stock_name, save_folder='signal_plots'):
    os.makedirs(save_folder, exist_ok=True)

    plt.figure(figsize=(14,7))
    plt.plot(data['Date'], data['Close'], label='Close Price', color='black')
    
    buys = data.index[data['Buy'] == True].tolist()
    sells = data.index[data['Sell'] == True].tolist()

    plt.scatter(data.loc[buys,'Date'], data.loc[buys,'Close'], marker='^', color='green', label='Buy Signal', s=80)
    plt.scatter(data.loc[sells,'Date'], data.loc[sells,'Close'], marker='v', color='red', label='Sell Signal', s=80)

    plt.title(f'{strategy_name} Signals for {stock_name}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    filename = f"{strategy_name}_{stock_name}.png".replace(' ', '_')
    plt.savefig(os.path.join(save_folder, filename))
    plt.show()
    plt.close()

# --- Main Execution Flow ---

def load_and_prepare(filepath):
    data = pd.read_csv(filepath, parse_dates=['Date'])
    # Clean and convert numeric columns
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.dropna(subset=['Date', 'Close', 'High', 'Low'])
    return data.reset_index(drop=True)

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

def main():
    data_folder = 'data'
    for filename in os.listdir(data_folder):
        if filename.endswith('.csv'):
            stock_name = filename.replace('.csv','')
            filepath = os.path.join(data_folder, filename)
            data = load_and_prepare(filepath)

            for strat_name, strat_func in strategies.items():
                strat_df = strat_func(data.copy())
                plot_signals(strat_df, strat_name, stock_name)

if __name__ == "__main__":
    main()

