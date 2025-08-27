import os
import pandas as pd
import numpy as np

# --- Trading Strategies ---

def macd_strategy(data):
    data = data.sort_values('Date').reset_index(drop=True)
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['Buy'] = (data['MACD'] > data['Signal']) & (data['MACD'].shift(1) <= data['Signal'].shift(1))
    data['Sell'] = (data['MACD'] < data['Signal']) & (data['MACD'].shift(1) >= data['Signal'].shift(1))
    data['Buy_Signal'] = data['Buy'] & (~data['Buy'].shift(1).fillna(False).astype(bool))
    data['Sell_Signal'] = data['Sell'] & (~data['Sell'].shift(1).fillna(False).astype(bool))
    return data

def rsi_strategy(data, period=14, overbought=70, oversold=30):
    data = data.sort_values('Date').reset_index(drop=True)
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
    rs = gain / loss
    data['RSI'] = 100 - 100 / (1 + rs)
    data['Buy'] = (data['RSI'] > oversold) & (data['RSI'].shift(1) <= oversold)
    data['Sell'] = (data['RSI'] < overbought) & (data['RSI'].shift(1) >= overbought)
    data['Buy_Signal'] = data['Buy'] & (~data['Buy'].shift(1).fillna(False).astype(bool))
    data['Sell_Signal'] = data['Sell'] & (~data['Sell'].shift(1).fillna(False).astype(bool))
    return data

def bollinger_bands_strategy(data, window=20, n_std=2):
    data = data.sort_values('Date').reset_index(drop=True)
    data['MA'] = data['Close'].rolling(window=window).mean()
    data['STD'] = data['Close'].rolling(window=window).std()
    data['Upper'] = data['MA'] + n_std * data['STD']
    data['Lower'] = data['MA'] - n_std * data['STD']
    data['Buy'] = (data['Close'].shift(1) < data['Lower'].shift(1)) & (data['Close'] > data['Lower'])
    data['Sell'] = (data['Close'].shift(1) > data['Upper'].shift(1)) & (data['Close'] < data['Upper'])
    data['Buy_Signal'] = data['Buy'] & (~data['Buy'].shift(1).fillna(False).astype(bool))
    data['Sell_Signal'] = data['Sell'] & (~data['Sell'].shift(1).fillna(False).astype(bool))
    return data

def ma_crossover_strategy(data, short_win=50, long_win=200):
    data = data.sort_values('Date').reset_index(drop=True)
    data['SMA_short'] = data['Close'].rolling(window=short_win).mean()
    data['SMA_long'] = data['Close'].rolling(window=long_win).mean()
    data['Buy'] = (data['SMA_short'] > data['SMA_long']) & (data['SMA_short'].shift(1) <= data['SMA_long'].shift(1))
    data['Sell'] = (data['SMA_short'] < data['SMA_long']) & (data['SMA_short'].shift(1) >= data['SMA_long'].shift(1))
    data['Buy_Signal'] = data['Buy'] & (~data['Buy'].shift(1).fillna(False).astype(bool))
    data['Sell_Signal'] = data['Sell'] & (~data['Sell'].shift(1).fillna(False).astype(bool))
    return data

def stochastic_oscillator_strategy(data, k_period=14, d_period=3):
    data = data.sort_values('Date').reset_index(drop=True)
    low_min = data['Low'].rolling(k_period).min()
    high_max = data['High'].rolling(k_period).max()
    data['%K'] = 100 * (data['Close'] - low_min) / (high_max - low_min)
    data['%D'] = data['%K'].rolling(d_period).mean()
    data['Buy'] = (data['%K'] > data['%D']) & (data['%K'].shift(1) <= data['%D'].shift(1)) & (data['%K'] < 20)
    data['Sell'] = (data['%K'] < data['%D']) & (data['%K'].shift(1) >= data['%D'].shift(1)) & (data['%K'] > 80)
    data['Buy_Signal'] = data['Buy'] & (~data['Buy'].shift(1).fillna(False).astype(bool))
    data['Sell_Signal'] = data['Sell'] & (~data['Sell'].shift(1).fillna(False).astype(bool))
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
    data['Buy'] = (data['ADX'] > adx_threshold) & (data['+DI'] > data['-DI']) & (data['+DI'].shift(1) <= data['-DI'].shift(1))
    data['Sell'] = (data['ADX'] > adx_threshold) & (data['+DI'] < data['-DI']) & (data['+DI'].shift(1) >= data['-DI'].shift(1))
    data['Buy_Signal'] = data['Buy'] & (~data['Buy'].shift(1).fillna(False).astype(bool))
    data['Sell_Signal'] = data['Sell'] & (~data['Sell'].shift(1).fillna(False).astype(bool))
    return data

def breakout_strategy(data, lookback=20):
    data = data.sort_values('Date').reset_index(drop=True)
    data['Highest_High'] = data['High'].shift(1).rolling(lookback).max()
    data['Lowest_Low'] = data['Low'].shift(1).rolling(lookback).min()
    data['Buy'] = data['Close'] > data['Highest_High']
    data['Sell'] = data['Close'] < data['Lowest_Low']
    data['Buy_Signal'] = data['Buy'] & (~data['Buy'].shift(1).fillna(False).astype(bool))
    data['Sell_Signal'] = data['Sell'] & (~data['Sell'].shift(1).fillna(False).astype(bool))
    return data

def mean_reversion_strategy(data, window=20, num_std=2):
    data = data.sort_values('Date').reset_index(drop=True)
    data['MA'] = data['Close'].rolling(window=window).mean()
    data['STD'] = data['Close'].rolling(window=window).std()
    data['Upper_BB'] = data['MA'] + num_std * data['STD']
    data['Lower_BB'] = data['MA'] - num_std * data['STD']
    data['Buy'] = (data['Close'].shift(1) < data['Lower_BB'].shift(1)) & (data['Close'] > data['Lower_BB'])
    data['Sell'] = (data['Close'].shift(1) > data['Upper_BB'].shift(1)) & (data['Close'] < data['Upper_BB'])
    data['Buy_Signal'] = data['Buy'] & (~data['Buy'].shift(1).fillna(False).astype(bool))
    data['Sell_Signal'] = data['Sell'] & (~data['Sell'].shift(1).fillna(False).astype(bool))
    return data

# Dictionary mapping
strategies = {
    'MACD': macd_strategy,
    'RSI': rsi_strategy,
    'Bollinger Bands': bollinger_bands_strategy,
    'Moving Average Crossover': ma_crossover_strategy,
    'Stochastic Oscillator': stochastic_oscillator_strategy,
    'Trend Following': trend_following_strategy,
    'Breakout': breakout_strategy,
    'Mean Reversion': mean_reversion_strategy,
}

# Calculate rolling score (+1 buy, -1 sell) over 'window' days
def calculate_rolling_score(data, window, buy_col='Buy_Signal', sell_col='Sell_Signal'):
    data['BuyInt'] = data[buy_col].astype(int)
    data['SellInt'] = data[sell_col].astype(int)
    rolling_buys = data['BuyInt'].rolling(window=window, min_periods=1).sum()
    rolling_sells = data['SellInt'].rolling(window=window, min_periods=1).sum()
    return rolling_buys - rolling_sells

# Compute statistics on score series
def score_stats(score_series):
    stats = {
        'Total days': len(score_series),
        'Positive score days': (score_series > 0).sum(),
        'Neutral score days': (score_series == 0).sum(),
        'Negative score days': (score_series < 0).sum(),
        'Average score': score_series.mean(),
        'Max score': score_series.max(),
        'Min score': score_series.min(),
        'Score volatility (std dev)': score_series.std(),
    }
    return stats

def main():
    data_folder = 'data'
    stats_folder = 'stats'
    os.makedirs(stats_folder, exist_ok=True)

    rolling_windows = range(5, 21)  # 5 to 20 days window

    all_results = []

    stock_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]

    for stock_file in stock_files:
        stock_name = stock_file.replace('.csv', '')
        filepath = os.path.join(data_folder, stock_file)
        df = pd.read_csv(filepath, parse_dates=['Date'])
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['Date', 'Close', 'High', 'Low']).reset_index(drop=True)

        for strat_name, strat_func in strategies.items():
            print(f"Processing {stock_name} with {strat_name}")
            strat_df = strat_func(df.copy())

            # Ensure Buy_Signal, Sell_Signal exist and are boolean arrays
            if 'Buy_Signal' not in strat_df.columns or strat_df['Buy_Signal'].isnull().all():
                strat_df['Buy_Signal'] = strat_df['Buy'] & (~strat_df['Buy'].shift(1).fillna(False).astype(bool))
            if 'Sell_Signal' not in strat_df.columns or strat_df['Sell_Signal'].isnull().all():
                strat_df['Sell_Signal'] = strat_df['Sell'] & (~strat_df['Sell'].shift(1).fillna(False).astype(bool))
            strat_df['Buy_Signal'] = strat_df['Buy_Signal'].astype(bool)
            strat_df['Sell_Signal'] = strat_df['Sell_Signal'].astype(bool)

            for window in rolling_windows:
                rolling_score = calculate_rolling_score(strat_df, window)
                combined_df = pd.DataFrame({'Close': df.set_index('Date')['Close']}).join(rolling_score.rename('Score')).fillna(0)

                stats = score_stats(combined_df['Score'])
                stats.update({'Stock': stock_name, 'Strategy': strat_name, 'Rolling Window': window})
                all_results.append(stats)

    results_df = pd.DataFrame(all_results)
    results_df = results_df[['Stock', 'Strategy', 'Rolling Window', 'Total days', 'Positive score days',
                             'Neutral score days', 'Negative score days', 'Average score', 'Max score',
                             'Min score', 'Score volatility (std dev)']]

    output_path = os.path.join(stats_folder, 'rolling_signal_score_stats.csv')
    results_df.to_csv(output_path, index=False)
    print(f"All done. Stats saved to {output_path}")

if __name__ == "__main__":
    main()

