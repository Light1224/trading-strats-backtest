#!/usr/bin/env python3
"""
Aggressive portfolio backtester using your optimized params (exactly).
Saves:
 - backtest_logs/all_trades.csv
 - backtest_logs/portfolio_curve.csv
 - backtest_logs/portfolio_curve.png
 - stats/final_optimized_portfolio_stats.csv

Usage: python3 aggressive_backtester.py
"""

import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ---------------- USER INPUT ----------------
DATA_FOLDER = "data"
LOG_FOLDER = "backtest_logs"
STATS_FOLDER = "stats"
BEST_WEIGHTS_PATH = "best_weights.npy"  # optional: if present it'll be used

os.makedirs(LOG_FOLDER, exist_ok=True)
os.makedirs(STATS_FOLDER, exist_ok=True)

start_date_str = input("Start date (YYYY-MM-DD): ").strip()
end_date_str = input("End date (YYYY-MM-DD): ").strip()
initial_capital = float(input("Initial capital (e.g. 100000): ").strip())

START_DATE = pd.to_datetime(start_date_str)
END_DATE = pd.to_datetime(end_date_str)

# ---------------- YOUR OPTIMIZED PARAMETERS (exact) ----------------
BEST_PARAMS = {
    "buy_thresh": 1.8451,
    "sell_thresh": -4.6955,
    "max_risk_per_trade": 0.1000,
    "stop_loss_pct": 0.2000,
    "take_profit_pct": 0.4000,
    "commission_pct": 0.0000,
    "slippage_pct": 0.0000,
}

# Aggressive/realism config (you can tweak)
MIN_QTY_FALLBACK_ALLOC_FRAC = 0.10   # aggressive fallback allocate 10% of equity if risk-size yields 0
MAX_HOLD_DAYS = 252 * 2             # force close after 2 years (None to disable)
SCORE_WINDOW = 5                    # rolling window used for combined score
MIN_AVG_VOLUME_FOR_TRADE = 10       # keep small to not filter too much (set higher to skip illiquid)

# ---------------- Strategy definitions (your 5 base + shifted) ----------------
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

def shifted_strategy(base_func, shift):
    def strat(data):
        buy, sell = base_func(data)
        buy = buy.shift(shift).fillna(0).astype(int)
        sell = sell.shift(shift).fillna(0).astype(int)
        return buy, sell
    return strat

BASE_STRATS = [
    macd_strategy,
    rsi_strategy,
    bollinger_bands_strategy,
    ma_crossover_strategy,
    stochastic_oscillator_strategy,
]

STRAT_FUNCS = BASE_STRATS.copy()
for shift in range(1, 16):
    for f in BASE_STRATS:
        if len(STRAT_FUNCS) >= 20:
            break
        STRAT_FUNCS.append(shifted_strategy(f, shift))
    if len(STRAT_FUNCS) >= 20:
        break

# ---------------- Utilities ----------------
def load_and_clean_csv(path, start=None, end=None):
    # parse Date without infer_datetime_format (deprecated)
    df = pd.read_csv(path, parse_dates=['Date'])
    df.columns = [c.strip() for c in df.columns]
    if 'Close' not in df.columns and 'Adj Close' in df.columns:
        df['Close'] = df['Adj Close']
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Date', 'Close']).sort_values('Date').reset_index(drop=True)
    if start is not None:
        df = df[df['Date'] >= start]
    if end is not None:
        df = df[df['Date'] <= end]
    df = df.reset_index(drop=True)
    return df

def load_best_weights(n):
    if os.path.exists(BEST_WEIGHTS_PATH):
        try:
            w = np.load(BEST_WEIGHTS_PATH)
            w = np.asarray(w, dtype=float)
            if w.size == n:
                return w
            else:
                print(f"[WARN] best_weights.npy size {w.size} != {n}, using equal weights")
        except Exception as e:
            print("[WARN] failed to load best_weights.npy:", e)
    return np.ones(n) / n

def calculate_signals(df):
    signals = []
    for f in STRAT_FUNCS:
        b, s = f(df.copy())
        b = np.asarray(b, dtype=float).flatten()
        s = np.asarray(s, dtype=float).flatten()
        m = len(df)
        if len(b) < m:
            b = np.pad(b, (0, m - len(b)), constant_values=0)
        if len(s) < m:
            s = np.pad(s, (0, m - len(s)), constant_values=0)
        signals.append((b, s))
    return signals

def combined_score(signals, weights, window=SCORE_WINDOW):
    n = len(signals[0][0])
    comb = np.zeros(n, dtype=float)
    for w, (b, s) in zip(weights, signals):
        comb += w * (b - s)
    comb = pd.Series(comb).rolling(window=window, min_periods=1).sum().fillna(0).values
    return comb

def scale_score_to_thresholds(score, buy_thresh, sell_thresh):
    s = np.array(score, dtype=float)
    pos = s[s > 0]
    neg = s[s < 0]
    scale_pos = 1.0
    scale_neg = 1.0
    if pos.size > 2:
        p90 = np.percentile(pos, 90)
        if p90 > 0 and abs(buy_thresh) > abs(p90):
            scale_pos = abs(buy_thresh) / (p90 + 1e-12)
    if neg.size > 2:
        p10 = np.percentile(neg, 10)
        if p10 < 0 and abs(sell_thresh) > abs(p10):
            scale_neg = abs(sell_thresh) / (abs(p10) + 1e-12)
    scaler = max(1.0, scale_pos, scale_neg)
    return s * scaler, scaler

def compute_qty(entry_price, equity, max_risk_per_trade, stop_loss_pct):
    if entry_price <= 0 or equity <= 0:
        return 0
    risk_cash = equity * max_risk_per_trade
    if stop_loss_pct > 1e-9:
        per_share_risk = entry_price * stop_loss_pct
        qty = math.floor(risk_cash / (per_share_risk + 1e-12))
    else:
        alloc_cash = equity * max_risk_per_trade
        qty = math.floor(alloc_cash / (entry_price + 1e-12))
    return max(0, int(qty))

# ---------------- Load data & precompute ----------------
files = sorted([f for f in os.listdir(DATA_FOLDER) if f.lower().endswith('.csv')])
if not files:
    raise SystemExit("No CSV files in data/")

weights = load_best_weights(len(STRAT_FUNCS))
params = BEST_PARAMS.copy()
print("Using exact optimized params:", params)
print("Using weights sum:", weights.sum())

stocks = {}
calendar = set()
print("Loading and preparing stocks...")

for fname in files:
    name = os.path.splitext(fname)[0]
    path = os.path.join(DATA_FOLDER, fname)
    try:
        df = load_and_clean_csv(path, START_DATE, END_DATE)
    except Exception as e:
        print(f"Skipping {name}: load error {e}")
        continue
    if df.empty:
        continue
    signals = calculate_signals(df)
    raw = combined_score(signals, weights, window=SCORE_WINDOW)
    scaled, scaler = scale_score_to_thresholds(raw, params['buy_thresh'], params['sell_thresh'])
    df = df.assign(RawScore=raw, Score=scaled, ScalerUsed=float(scaler))
    df.attrs['avg_vol'] = df['Volume'].dropna().mean() if 'Volume' in df.columns else np.nan
    stocks[name] = df
    calendar.update(df['Date'].dt.date.values)

if not stocks:
    raise SystemExit("No valid stocks after loading")

calendar = sorted(list(calendar))
calendar = [pd.to_datetime(d) for d in calendar]

# ---------------- Portfolio simulation (aggressive) ----------------
print("Starting aggressive portfolio simulation...")
cash = float(initial_capital)
positions = {}   # stock -> dict {qty, entry_price, entry_date, peak_price}
trade_rows = []
portfolio_history = []

for current_date in calendar:
    # gather today's candidate buy/sell info
    buy_cands = []
    sell_cands = []
    # update last_price for positions and check exits
    for stock, df in stocks.items():
        rows = np.where(df['Date'].dt.date == current_date.date())[0]
        if rows.size == 0:
            continue
        i = int(rows[0])
        price = float(df.at[i, 'Close'])
        score = float(df.at[i, 'Score'])
        avg_vol = df.attrs.get('avg_vol', np.nan)

        # if held -> check exit conditions first
        if stock in positions:
            pos = positions[stock]
            entry_price = pos['entry_price']
            holding_days = (current_date - pos['entry_date']).days
            stop_price = entry_price * (1 - params['stop_loss_pct'])
            take_price = entry_price * (1 + params['take_profit_pct'])
            hit_stop = price <= stop_price
            hit_take = price >= take_price
            timed_exit = (MAX_HOLD_DAYS is not None) and (holding_days >= MAX_HOLD_DAYS)
            signal_sell = score <= params['sell_thresh']
            if hit_stop or hit_take or timed_exit or signal_sell:
                sell_cands.append((stock, price, 'STOP' if hit_stop else ('TAKE' if hit_take else ('TIME' if timed_exit else 'SIGNAL'))))
            else:
                # update last price & peak
                positions[stock]['last_price'] = price
                if price > positions[stock]['peak_price']:
                    positions[stock]['peak_price'] = price
        else:
            # not held -> consider buy if score >= buy_thresh and liquidity ok
            if score >= params['buy_thresh']:
                if not np.isnan(avg_vol) and avg_vol < MIN_AVG_VOLUME_FOR_TRADE:
                    # skip illiquid
                    continue
                buy_cands.append((stock, price, score, i))

    # Execute sells (free cash)
    for stock, price, reason in sorted(sell_cands, key=lambda x: x[0]):
        if stock not in positions:
            continue
        pos = positions.pop(stock)
        qty = pos['qty']
        exit_price = price * (1 - params['slippage_pct'])
        notional = qty * exit_price
        commission = notional * params['commission_pct']
        proceeds = notional - commission
        cash += proceeds
        trade_rows.append({
            "Date": current_date, "Stock": stock, "Action": "SELL", "Qty": qty,
            "Price": exit_price, "Proceeds": proceeds, "CashAfter": cash, "Reason": reason,
            "EntryDate": pos['entry_date'], "EntryPrice": pos['entry_price'], "HoldingDays": (current_date - pos['entry_date']).days
        })
        print(f"[{current_date.date()}] SELL {stock} qty={qty} @ {exit_price:.4f} reason={reason} cash={cash:.2f}")

    # Execute buys: aggressive - attempt to open all buys while cash allows
    # Rank by score descending to prioritize highest conviction
    buy_cands.sort(key=lambda x: x[2], reverse=True)
    for stock, price, score, idx in buy_cands:
        # compute qty from risk-size using current equity snapshot
        held_value = sum(positions[s]['qty'] * positions[s]['last_price'] for s in positions) if positions else 0.0
        equity_snapshot = cash + held_value
        entry_price_applied = price * (1 + params['slippage_pct'])
        qty = compute_qty(entry_price_applied, equity_snapshot, params['max_risk_per_trade'], params['stop_loss_pct'])
        if qty <= 0:
            fallback_cash = max(initial_capital * MIN_QTY_FALLBACK_ALLOC_FRAC, equity_snapshot * MIN_QTY_FALLBACK_ALLOC_FRAC)
            qty = math.floor(fallback_cash / (entry_price_applied + 1e-12))
        if qty <= 0:
            continue
        notional = qty * entry_price_applied
        commission = notional * params['commission_pct']
        total_cost = notional + commission
        if total_cost > cash:
            affordable_qty = math.floor(cash / (entry_price_applied * (1 + params['commission_pct']) + 1e-12))
            if affordable_qty <= 0:
                continue
            qty = int(affordable_qty)
            notional = qty * entry_price_applied
            commission = notional * params['commission_pct']
            total_cost = notional + commission
        # open position
        cash -= total_cost
        positions[stock] = {
            "qty": qty,
            "entry_price": entry_price_applied,
            "entry_date": current_date,
            "peak_price": price,
            "last_price": price
        }
        trade_rows.append({
            "Date": current_date, "Stock": stock, "Action": "BUY", "Qty": qty,
            "Price": entry_price_applied, "Proceeds": -total_cost, "CashAfter": cash, "Reason": "SCORE",
            "EntryDate": current_date, "EntryPrice": entry_price_applied, "HoldingDays": 0
        })
        print(f"[{current_date.date()}] BUY  {stock} qty={qty} @ {entry_price_applied:.4f} cash_after={cash:.2f} score={score:.3f}")

    # portfolio snapshot for the day
    held_value = sum(positions[s]['qty'] * positions[s]['last_price'] for s in positions) if positions else 0.0
    equity = cash + held_value
    portfolio_history.append({"Date": current_date, "Cash": cash, "HeldValue": held_value, "Equity": equity, "OpenPositions": len(positions)})

# End of calendar loop: close remaining positions at last available price
last_date = calendar[-1]
for stock, pos in list(positions.items()):
    df = stocks[stock]
    exit_price = float(df['Close'].iloc[-1]) * (1 - params['slippage_pct'])
    notional = pos['qty'] * exit_price
    commission = notional * params['commission_pct']
    proceeds = notional - commission
    cash += proceeds
    trade_rows.append({
        "Date": last_date, "Stock": stock, "Action": "SELL", "Qty": pos['qty'],
        "Price": exit_price, "Proceeds": proceeds, "CashAfter": cash, "Reason": "END_CLOSE",
        "EntryDate": pos['entry_date'], "EntryPrice": pos['entry_price'], "HoldingDays": (last_date - pos['entry_date']).days
    })
    print(f"[{last_date.date()}] END_CLOSE SELL {stock} qty={pos['qty']} @ {exit_price:.4f} cash={cash:.2f}")
    positions.pop(stock, None)

# final equity snapshot (day after last)
held_value = 0.0
equity = cash + held_value
portfolio_history.append({"Date": last_date + pd.Timedelta(days=1), "Cash": cash, "HeldValue": 0.0, "Equity": equity, "OpenPositions": 0})

# ---------------- Save trades and portfolio curve ----------------
trades_df = pd.DataFrame(trade_rows)
if not trades_df.empty:
    trades_df = trades_df[[
        "Date", "Stock", "Action", "Qty", "Price", "Proceeds", "CashAfter", "Reason", "EntryDate", "EntryPrice", "HoldingDays"
    ]]
    trades_df.to_csv(os.path.join(LOG_FOLDER, "all_trades.csv"), index=False)
    print(f"\nSaved trades -> {os.path.join(LOG_FOLDER, 'all_trades.csv')}")
else:
    print("No trades executed.")

port_df = pd.DataFrame(portfolio_history)
port_df.to_csv(os.path.join(LOG_FOLDER, "portfolio_curve.csv"), index=False)
print(f"Saved portfolio curve -> {os.path.join(LOG_FOLDER, 'portfolio_curve.csv')}")

# ---------------- Plot portfolio curve ----------------
plt.figure(figsize=(12,6))
plt.plot(pd.to_datetime(port_df['Date']), port_df['Equity'], label='Equity')
plt.title('Portfolio Equity Curve')
plt.xlabel('Date')
plt.ylabel('Equity (currency)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(LOG_FOLDER, "portfolio_curve.png"))
plt.show()
print(f"Saved plot -> {os.path.join(LOG_FOLDER, 'portfolio_curve.png')}")

# ---------------- Compute and save returns statistics ----------------
def perf_stats_from_curve(df):
    arr = np.array(df['Equity'], dtype=float)
    if len(arr) < 2:
        return {}
    rets = np.diff(arr) / arr[:-1]
    avg = np.nanmean(rets)
    std = np.nanstd(rets)
    sharpe = (avg / std) * np.sqrt(252) if std > 0 else np.nan
    total_return = arr[-1] / arr[0] - 1
    days = (pd.to_datetime(df['Date'].iloc[-1]) - pd.to_datetime(df['Date'].iloc[0])).days
    years = days / 365.25 if days > 0 else 1.0
    cagr = (arr[-1] / arr[0]) ** (1.0 / years) - 1 if arr[0] > 0 else np.nan
    running_max = np.maximum.accumulate(arr)
    max_dd = float(np.nanmax((running_max - arr) / running_max))
    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
        "Avg Daily Return": avg,
        "Daily Std": std
    }

perf = perf_stats_from_curve(port_df)
summary = [{"Stock": "PORTFOLIO", **perf}]
summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(STATS_FOLDER, "final_optimized_portfolio_stats.csv"), index=False)
print(f"Saved summary -> {os.path.join(STATS_FOLDER, 'final_optimized_portfolio_stats.csv')}")
print("\nPortfolio Performance:")
for k,v in perf.items():
    if isinstance(v, float):
        if k == "Sharpe":
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v:.2%}")
    else:
        print(f"  {k}: {v}")

print("\nDone.")

