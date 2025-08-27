import os
import pandas as pd
import numpy as np
import talib

DATA_FOLDER = "data"
RESULTS_FILE = "results.csv"
TRADES_FOLDER = "trades"

os.makedirs(TRADES_FOLDER, exist_ok=True)

# --------------------
# Your Optimised Strategy
# --------------------
def run_strategy(data, stock_name, weights, threshold):
    # Normalize column names
    data.columns = [c.strip().lower() for c in data.columns]

    # Pick close column (handle "close" vs "adj close")
    if "close" in data.columns:
        close = pd.to_numeric(data["close"], errors="coerce")
    elif "adj close" in data.columns:
        close = pd.to_numeric(data["adj close"], errors="coerce")
    else:
        raise ValueError(f"No close price column in {stock_name}")

    # Drop rows where close is NaN
    data = data.assign(close=close).dropna(subset=["close"]).reset_index(drop=True)

    # Handle date
    if "date" in data.columns:
        dates = pd.to_datetime(data["date"], errors="coerce")
    else:
        dates = pd.Series(range(len(data)))  # fallback index

    close = data["close"].values

    # Example indicators (replace with your 6+shifted ones)
    sma10 = talib.SMA(close, timeperiod=10)
    sma50 = talib.SMA(close, timeperiod=50)
    rsi = talib.RSI(close, timeperiod=14)

    # Score with optimised weights
    score = (
        weights.get("sma_signal", 1.0) * np.where(sma10 > sma50, 1, -1)
        + weights.get("rsi_signal", 1.0) * np.where(rsi < 30, 1, np.where(rsi > 70, -1, 0))
    )

    data["score"] = score

    position = 0
    entry_price = 0
    returns = []
    trade_log = []

    for i in range(len(data)):
        s = score[i]
        price = close[i]
        date = dates.iloc[i].date() if i < len(dates) else i

        # BUY condition
        if s > threshold and position == 0:
            position = 1
            entry_price = price
            trade_log.append(f"BUY {stock_name} at {price:.2f} on {date}")

        # SELL condition
        elif s < -threshold and position == 1:
            position = 0
            ret = (price - entry_price) / entry_price
            returns.append(ret)
            trade_log.append(f"SELL {stock_name} at {price:.2f} on {date} | Return {ret*100:.2f}%")

    # Close open position at end
    if position == 1:
        final_ret = (close[-1] - entry_price) / entry_price
        returns.append(final_ret)
        trade_log.append(f"CLOSE {stock_name} at {close[-1]:.2f} on {dates.iloc[-1].date()} | Return {final_ret*100:.2f}%")

    # Save trade log
    with open(os.path.join(TRADES_FOLDER, f"{stock_name}_trades.txt"), "w") as f:
        for log in trade_log:
            f.write(log + "\n")

    total_return = (np.prod([1+r for r in returns]) - 1) * 100 if returns else 0
    return total_return


# --------------------
# Run on All Files
# --------------------
results = []

optimised_weights = {
    "sma_signal": 0.8,
    "rsi_signal": 1.2,
}
optimised_threshold = 0.5   # <-- your optimised threshold

for file in os.listdir(DATA_FOLDER):
    if file.endswith(".csv"):
        stock_name = file.replace(".csv", "")
        path = os.path.join(DATA_FOLDER, file)

        try:
            data = pd.read_csv(path)
            total_return = run_strategy(data, stock_name, optimised_weights, optimised_threshold)
            results.append((stock_name, total_return))
            print(f"{stock_name}: {total_return:.2f}%")
        except Exception as e:
            print(f"Skipping {stock_name}: {e}")

# Save summary
df = pd.DataFrame(results, columns=["Stock", "Return %"])
df.to_csv(RESULTS_FILE, index=False)
print(f"\nFinal results saved to {RESULTS_FILE}")

