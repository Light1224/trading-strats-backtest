import os
import pandas as pd
import numpy as np

def bollinger_band_strategy(data, window=20, num_std=2):
    if data.empty or 'Date' not in data.columns or 'Close' not in data.columns:
        return None

    data = data.sort_values('Date').reset_index(drop=True)

    for col in ['Close', 'Open', 'High', 'Low', 'Volume']:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    data = data.dropna(subset=['Close', 'Date'])
    if data.empty:
        return None

    # Calculate Bollinger Bands
    data['MA'] = data['Close'].rolling(window=window, min_periods=window).mean()
    data['STD'] = data['Close'].rolling(window=window, min_periods=window).std()
    data['Upper_Band'] = data['MA'] + num_std * data['STD']
    data['Lower_Band'] = data['MA'] - num_std * data['STD']

    data = data.dropna(subset=['MA', 'STD', 'Upper_Band', 'Lower_Band'])

    data['Buy_Signal'] = np.nan
    data['Sell_Signal'] = np.nan

    for i in range(1, len(data)):
        # Buy signal when price crosses below lower band then closes back inside
        if data.iloc[i-1]['Close'] < data.iloc[i-1]['Lower_Band'] and data.iloc[i]['Close'] > data.iloc[i]['Lower_Band']:
            data.loc[data.index[i], 'Buy_Signal'] = data.iloc[i]['Close']
        # Sell signal when price crosses above upper band then closes back inside
        elif data.iloc[i-1]['Close'] > data.iloc[i-1]['Upper_Band'] and data.iloc[i]['Close'] < data.iloc[i]['Upper_Band']:
            data.loc[data.index[i], 'Sell_Signal'] = data.iloc[i]['Close']

    buy_prices = data[['Date', 'Buy_Signal']].dropna().reset_index()
    sell_prices = data[['Date', 'Sell_Signal']].dropna().reset_index()

    trades = []
    sell_idx = 0
    for _, buy_row in buy_prices.iterrows():
        while sell_idx < len(sell_prices) and sell_prices.loc[sell_idx, 'index'] <= buy_row['index']:
            sell_idx += 1
        if sell_idx < len(sell_prices):
            sell_row = sell_prices.loc[sell_idx]
            if sell_row['Date'] > buy_row['Date']:
                trades.append({
                    'Buy_Date': buy_row['Date'],
                    'Sell_Date': sell_row['Date'],
                    'Buy_Price': buy_row['Buy_Signal'],
                    'Sell_Price': sell_row['Sell_Signal'],
                    'Buy_Index': buy_row['index'],
                    'Sell_Index': sell_row['index']
                })
                sell_idx += 1
        else:
            break

    df_trades = pd.DataFrame(trades)
    if df_trades.empty:
        return None

    df_trades['Return'] = (df_trades['Sell_Price'] - df_trades['Buy_Price']) / df_trades['Buy_Price']
    df_trades['Duration'] = df_trades['Sell_Index'] - df_trades['Buy_Index']
    df_trades['Profit'] = df_trades['Return'] > 0

    total_trades = len(df_trades)
    winning_trades = df_trades['Profit'].sum()
    losing_trades = total_trades - winning_trades
    win_rate = winning_trades / total_trades if total_trades > 0 else np.nan
    average_return = df_trades['Return'].mean()
    median_return = df_trades['Return'].median()
    max_return = df_trades['Return'].max()
    min_return = df_trades['Return'].min()
    average_duration = df_trades['Duration'].mean()
    median_duration = df_trades['Duration'].median()
    max_duration = df_trades['Duration'].max()
    min_duration = df_trades['Duration'].min()
    total_return = (df_trades['Return'] + 1).prod() - 1 if total_trades > 0 else np.nan

    cumulative_returns = (data['Close'].pct_change().fillna(0) + 1).cumprod() - 1
    max_drawdown = (cumulative_returns.cummax() - cumulative_returns).max()
    profit_factor = df_trades.loc[df_trades['Profit'], 'Return'].sum() / -df_trades.loc[~df_trades['Profit'], 'Return'].sum() if losing_trades > 0 else np.nan
    expectancy = df_trades['Return'].mean() * win_rate - df_trades['Return'].mean() * (1 - win_rate) if total_trades > 0 else np.nan
    std_return = df_trades['Return'].std()

    stats = {
        "File": None,  # to be set outside function
        "Total trades": total_trades,
        "Winning trades": winning_trades,
        "Losing trades": losing_trades,
        "Win rate": win_rate,
        "Average return per trade": average_return,
        "Median return per trade": median_return,
        "Max return": max_return,
        "Min return": min_return,
        "Average trade duration (days)": average_duration,
        "Median trade duration (days)": median_duration,
        "Max trade duration (days)": max_duration,
        "Min trade duration (days)": min_duration,
        "Total cumulative return": total_return,
        "Maximum drawdown": max_drawdown,
        "Profit factor": profit_factor,
        "Expectancy": expectancy,
        "Return volatility (std dev)": std_return,
    }
    return stats

folder_path = 'data'
save_folder = 'stats'
import os
os.makedirs(save_folder, exist_ok=True)

results = []

for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        csv_data = pd.read_csv(file_path, parse_dates=['Date'])
        stat = bollinger_band_strategy(csv_data)
        if stat:
            stat['File'] = filename
            results.append(stat)

if results:
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by='Win rate', ascending=False)
    save_path = os.path.join(save_folder, 'bollinger_stats.csv')
    df_results.to_csv(save_path, index=False)
else:
    print("No valid results to save.")

