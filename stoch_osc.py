import os
import pandas as pd
import numpy as np

def stochastic_oscillator_analysis(k_period=14, d_period=3):
    folder = 'data'
    save_folder = 'stats'
    os.makedirs(save_folder, exist_ok=True)
    results = []

    for filename in os.listdir(folder):
        if filename.endswith('.csv'):
            path = os.path.join(folder, filename)
            data = pd.read_csv(path, parse_dates=['Date'])

            # Convert and clean price data
            data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
            data['Low'] = pd.to_numeric(data['Low'], errors='coerce')
            data['High'] = pd.to_numeric(data['High'], errors='coerce')
            data = data.dropna(subset=['Close', 'Low', 'High', 'Date'])
            data = data.sort_values('Date').reset_index(drop=True)
            if data.empty:
                print(f"Skipping {filename} empty after cleaning")
                continue
            
            low_min = data['Low'].rolling(window=k_period).min()
            high_max = data['High'].rolling(window=k_period).max()
            data['%K'] = 100 * (data['Close'] - low_min) / (high_max - low_min)
            data['%D'] = data['%K'].rolling(window=d_period).mean()

            data['Buy'] = (data['%K'] > data['%D']) & (data['%K'].shift() <= data['%D'].shift()) & (data['%K'] < 20)
            data['Sell'] = (data['%K'] < data['%D']) & (data['%K'].shift() >= data['%D'].shift()) & (data['%K'] > 80)

            buys = data[data['Buy']].index
            sells = data[data['Sell']].index

            trades = []
            sell_idx = 0
            for buy_idx in buys:
                while sell_idx < len(sells) and sells[sell_idx] <= buy_idx:
                    sell_idx += 1
                if sell_idx < len(sells):
                    sell_pos = sells[sell_idx]
                    buy_price = data.loc[buy_idx, 'Close']
                    sell_price = data.loc[sell_pos, 'Close']
                    ret = (sell_price - buy_price) / buy_price
                    duration = (data.loc[sell_pos, 'Date'] - data.loc[buy_idx, 'Date']).days
                    trades.append({'Buy Date': data.loc[buy_idx, 'Date'], 'Sell Date': data.loc[sell_pos, 'Date'],
                                   'Buy Price': buy_price, 'Sell Price': sell_price,
                                   'Return': ret, 'Duration': duration})
                    sell_idx += 1
            
            df_trades = pd.DataFrame(trades)
            if df_trades.empty:
                print(f"No trades for {filename}")
                continue

            total_trades = len(df_trades)
            winning_trades = df_trades[df_trades['Return'] > 0].shape[0]
            losing_trades = total_trades - winning_trades
            win_rate = winning_trades / total_trades if total_trades > 0 else np.nan
            avg_return = df_trades['Return'].mean()
            median_return = df_trades['Return'].median()
            max_return = df_trades['Return'].max()
            min_return = df_trades['Return'].min()
            avg_duration = df_trades['Duration'].mean()
            median_duration = df_trades['Duration'].median()
            max_duration = df_trades['Duration'].max()
            min_duration = df_trades['Duration'].min()
            total_cum_return = (df_trades['Return'] + 1).prod() - 1
            cumulative_returns = (data['Close'].pct_change().fillna(0) + 1).cumprod() - 1
            max_drawdown = (cumulative_returns.cummax() - cumulative_returns).max()
            profit_factor = df_trades.loc[df_trades['Return'] > 0, 'Return'].sum() / -df_trades.loc[df_trades['Return'] <= 0, 'Return'].sum()
            expectancy = avg_return * win_rate - avg_return * (1 - win_rate)
            std_return = df_trades['Return'].std()

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
            print(df_trades.head(10).to_string(index=False))

    if results:
        df_summary = pd.DataFrame(results)
        df_summary = df_summary.sort_values(by='Win rate', ascending=False)
        save_path = os.path.join(save_folder, 'stochastic_oscillator_stats.csv')
        df_summary.to_csv(save_path, index=False)
        print(f"\nSummary statistics saved to {save_path}")

if __name__ == "__main__":
    stochastic_oscillator_analysis()

