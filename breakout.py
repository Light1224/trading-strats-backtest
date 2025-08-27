import os
import pandas as pd

def breakout_strategy(data, lookback=20):
    data = data.sort_values('Date').reset_index(drop=True)
    data['Highest_High'] = data['High'].shift(1).rolling(window=lookback).max()
    data['Lowest_Low'] = data['Low'].shift(1).rolling(window=lookback).min()

    data['Buy'] = data['Close'] > data['Highest_High']
    data['Sell'] = data['Close'] < data['Lowest_Low']

    data['Buy_Signal'] = data['Buy'] & (~data['Buy'].shift(1).fillna(False))
    data['Sell_Signal'] = data['Sell'] & (~data['Sell'].shift(1).fillna(False))

    return data

def simulate_trades(data):
    buys = data[data['Buy_Signal']].index
    sells = data[data['Sell_Signal']].index.tolist()
    trades = []
    sell_idx = 0

    for buy_idx in buys:
        while sell_idx < len(sells) and sells[sell_idx] <= buy_idx:
            sell_idx += 1
        if sell_idx < len(sells):
            sell_pos = sells[sell_idx]
            trades.append({
                'Buy Date': data.loc[buy_idx, 'Date'],
                'Sell Date': data.loc[sell_pos, 'Date'],
                'Buy Price': data.loc[buy_idx, 'Close'],
                'Sell Price': data.loc[sell_pos, 'Close'],
                'Return': (data.loc[sell_pos, 'Close'] - data.loc[buy_idx, 'Close']) / data.loc[buy_idx, 'Close'],
                'Duration': (data.loc[sell_pos, 'Date'] - data.loc[buy_idx, 'Date']).days
            })
            sell_idx += 1
    return pd.DataFrame(trades)

def breakout_analysis(lookback=20):
    folder = 'data'
    save_folder = 'stats'
    os.makedirs(save_folder, exist_ok=True)
    results = []

    for filename in os.listdir(folder):
        if filename.endswith('.csv'):
            filepath = os.path.join(folder, filename)
            data = pd.read_csv(filepath, parse_dates=['Date'])
            # Convert relevant columns to numeric and clean
            for col in ['Close', 'High', 'Low']:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            data = data.dropna(subset=['Close', 'High', 'Low', 'Date'])
            data = data.reset_index(drop=True)
            if data.empty:
                print(f"Skipping {filename}, empty after cleaning")
                continue

            data = breakout_strategy(data, lookback)
            trades = simulate_trades(data)
            if trades.empty:
                print(f"No trades for {filename}")
                continue

            total_trades = len(trades)
            winning_trades = trades[trades['Return'] > 0].shape[0]
            losing_trades = total_trades - winning_trades
            win_rate = winning_trades / total_trades if total_trades > 0 else float('nan')
            avg_return = trades['Return'].mean()
            median_return = trades['Return'].median()
            max_return = trades['Return'].max()
            min_return = trades['Return'].min()
            avg_duration = trades['Duration'].mean()
            median_duration = trades['Duration'].median()
            max_duration = trades['Duration'].max()
            min_duration = trades['Duration'].min()
            total_cum_return = (trades['Return'] + 1).prod() - 1
            cumulative_returns = (data['Close'].pct_change().fillna(0) + 1).cumprod() - 1
            max_drawdown = (cumulative_returns.cummax() - cumulative_returns).max()
            profit_factor = trades.loc[trades['Return'] > 0, 'Return'].sum() / -trades.loc[trades['Return'] <= 0, 'Return'].sum() if losing_trades > 0 else float('nan')
            expectancy = avg_return * win_rate - avg_return * (1 - win_rate)
            std_return = trades['Return'].std()

            stats = {
                'File': filename,
                'Total trades': total_trades,
                'Winning trades': winning_trades,
                'Losing trades': losing_trades,
                'Win rate': win_rate,
                'Average return per trade': avg_return,
                'Median return per trade': median_return,
                'Max return': max_return,
                'Min return': min_return,
                'Average trade duration': avg_duration,
                'Median trade duration': median_duration,
                'Max trade duration': max_duration,
                'Min trade duration': min_duration,
                'Total cumulative return': total_cum_return,
                'Maximum drawdown': max_drawdown,
                'Profit factor': profit_factor,
                'Expectancy': expectancy,
                'Return volatility': std_return,
            }
            results.append(stats)

            print(f"\n{filename} first 10 trades:")
            print(trades.head(10).to_string(index=False))

    if results:
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values(by='Win rate', ascending=False)
        save_path = os.path.join(save_folder, 'breakout_strategy_stats.csv')
        df_results.to_csv(save_path, index=False)
        print(f"\nSummary saved to {save_path}")

if __name__ == "__main__":
    breakout_analysis()

