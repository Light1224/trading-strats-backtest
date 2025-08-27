#!/usr/bin/env python3
"""
Backtester that uses your 20-strategy ensemble and your optimized thresholds,
but ensures signals scale so trades actually fire. Saves trade logs and equity
curves into backtest_logs/, and a summary into stats/.
"""

import os
import math
import pandas as pd
import numpy as np
from typing import Tuple, List

# -----------------------
# USER CONFIG (change if needed)
# -----------------------
DATA_FOLDER = "data"
LOG_FOLDER = "backtest_logs"
STATS_FOLDER = "stats"
START_DATE = None   # e.g. "2000-01-01" or None to use all
END_DATE = None     # e.g. "2022-12-31" or None to use all
INITIAL_CAPITAL = 100000.0

# Your optimized parameters (from your earlier run) — used as defaults
BEST_PARAMS = {
    "buy_thresh": 1.8451,
    "sell_thresh": -4.6955,
    "max_risk_per_trade": 0.1000,
    "stop_loss_pct": 0.2000,
    "take_profit_pct": 0.4000,
    "commission_pct": 0.0000,
    "slippage_pct": 0.0000,
}

# Optional: if you saved optimized weights from the Bayesian run as best_weights.npy
BEST_WEIGHTS_PATH = "best_weights.npy"  # if exists it will be used; otherwise equal weights

# Safety & realism tweaks
MIN_QTY_FALLBACK_ALLOC_FRAC = 0.05  # if computed qty==0, allocate 5% equity as fallback
MAX_HOLD_DAYS = 252 * 2  # force close after ~2 years to avoid never-selling positions (set to None to disable)

# -----------------------
# Strategy definitions (same as your original script)
# -----------------------
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

# Build 20 strategies (5 base + shifted)
STRAT_FUNCS = BASE_STRATS.copy()
for shift in range(1, 16):
    for f in BASE_STRATS:
        if len(STRAT_FUNCS) >= 20:
            break
        STRAT_FUNCS.append(shifted_strategy(f, shift))
    if len(STRAT_FUNCS) >= 20:
        break

# -----------------------
# Utilities: data cleaning + weights loader
# -----------------------
def load_and_clean_csv(path: str, start_date=None, end_date=None) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['Date'], dayfirst=False, infer_datetime_format=True)
    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    # choose Close or Adj Close
    if 'Close' not in df.columns and 'Adj Close' in df.columns:
        df['Close'] = df['Adj Close']
    # coerce numeric
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Date', 'Close']).sort_values('Date').reset_index(drop=True)
    if start_date:
        df = df[df['Date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['Date'] <= pd.to_datetime(end_date)]
    df = df.reset_index(drop=True)
    return df

def load_best_weights(n: int) -> np.ndarray:
    if os.path.exists(BEST_WEIGHTS_PATH):
        try:
            w = np.load(BEST_WEIGHTS_PATH)
            w = np.asarray(w, dtype=float)
            if w.shape[0] == n:
                return w
            else:
                print(f"[WARN] best_weights.npy length {w.shape[0]} != {n}; using equal weights.")
        except Exception as e:
            print("[WARN] failed to load best_weights.npy:", e)
    return np.ones(n) / n

# -----------------------
# Scoring, scaling, simulation, logging
# -----------------------
def calculate_signals(df: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
    signals = []
    for f in STRAT_FUNCS:
        b, s = f(df.copy())
        b = np.asarray(b, dtype=float).flatten()
        s = np.asarray(s, dtype=float).flatten()
        # ensure lengths match df
        m = len(df)
        if len(b) < m:
            b = np.pad(b, (0, m - len(b)), constant_values=0)
        if len(s) < m:
            s = np.pad(s, (0, m - len(s)), constant_values=0)
        signals.append((b, s))
    return signals

def calculate_combined_score(df: pd.DataFrame, signals: List[Tuple[np.ndarray, np.ndarray]], weights: np.ndarray, window=5) -> np.ndarray:
    n = len(df)
    combined = np.zeros(n, dtype=float)
    for w, (b, s) in zip(weights, signals):
        combined += w * (b - s)
    # rolling sum (window)
    comb_series = pd.Series(combined).rolling(window=window, min_periods=1).sum().fillna(0)
    return comb_series.values

def scale_score_to_thresholds(score: np.ndarray, buy_thresh: float, sell_thresh: float) -> Tuple[np.ndarray, float]:
    """
    Scale score array if necessary so that the 90th percentile of positive scores >= buy_thresh
    and the 10th percentile of negative scores <= sell_thresh. Returns scaled score and scaler.
    This preserves relative information but enables threshold crossing.
    """
    s = np.array(score, dtype=float)
    pos = s[s > 0]
    neg = s[s < 0]
    scale_pos = 1.0
    scale_neg = 1.0
    # compute percentiles safely
    if pos.size > 2:
        p90 = np.percentile(pos, 90)
        if p90 <= 0:
            scale_pos = 1.0
        else:
            scale_pos = abs(buy_thresh) / (p90 + 1e-12) if abs(buy_thresh) > abs(p90) else 1.0
    if neg.size > 2:
        p10 = np.percentile(neg, 10)  # will be negative
        if p10 >= 0:
            scale_neg = 1.0
        else:
            scale_neg = abs(sell_thresh) / (abs(p10) + 1e-12) if abs(sell_thresh) > abs(p10) else 1.0
    scaler = max(1.0, scale_pos, scale_neg)
    s_scaled = s * scaler
    return s_scaled, scaler

def compute_qty(entry_price: float, equity: float, max_risk_per_trade: float, stop_loss_pct: float) -> int:
    """
    Compute integer quantity using risk per trade.
    risk_cash = equity * max_risk_per_trade
    per_share_risk = entry_price * stop_loss_pct
    qty = floor(risk_cash / per_share_risk)
    fallback to MIN_QTY_FALLBACK_ALLOC_FRAC if result 0.
    """
    if entry_price <= 0 or equity <= 0:
        return 0
    risk_cash = equity * max_risk_per_trade
    if stop_loss_pct > 1e-9:
        per_share_risk = entry_price * stop_loss_pct
        qty = math.floor(risk_cash / (per_share_risk + 1e-12))
    else:
        alloc = equity * max_risk_per_trade
        qty = math.floor(alloc / (entry_price + 1e-12))
    return max(0, int(qty))

def simulate_trades_and_log(df: pd.DataFrame, combined_score: np.ndarray, params: dict,
                            initial_capital: float, stock_name: str, out_folder: str) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Runs the simulation for a single stock and returns:
      - equity_curve (np.ndarray)
      - daily_log (pd.DataFrame): Date, Close, Score, Action, Shares, Cash, Equity, Reason
      - trades_log (pd.DataFrame): per-trade rows (entry & exit combined per row)
    """
    n = len(df)
    cash = float(initial_capital)
    shares = 0
    entry_price = None
    entry_index = None
    equity_curve = []
    daily_rows = []
    trades = []  # will store dict per completed trade

    buy_thresh = params['buy_thresh']
    sell_thresh = params['sell_thresh']
    max_risk = params['max_risk_per_trade']
    stop_loss = params['stop_loss_pct']
    take_profit = params['take_profit_pct']
    commission = params['commission_pct']
    slippage = params['slippage_pct']

    for i in range(n):
        date = df['Date'].iloc[i]
        price = float(df['Close'].iloc[i])
        score = float(combined_score[i])
        action = "HOLD"
        reason = ""

        buy_price = price * (1 + slippage)
        sell_price = price * (1 - slippage)

        # ENTRY
        if shares == 0 and score >= buy_thresh:
            # determine qty
            qty = compute_qty(buy_price, cash + shares * price, max_risk, stop_loss)
            if qty <= 0:
                # fallback allocation using a fraction of equity
                fallback_alloc = max(INITIAL_CAPITAL * MIN_QTY_FALLBACK_ALLOC_FRAC, (cash + shares * price) * MIN_QTY_FALLBACK_ALLOC_FRAC)
                qty = math.floor(fallback_alloc / (buy_price + 1e-12))
            if qty > 0:
                notional = qty * buy_price
                commission_cost = notional * commission
                total_cost = notional + commission_cost
                if total_cost <= cash + 1e-9:
                    cash -= total_cost
                    shares += qty
                    entry_price = buy_price
                    entry_index = i
                    action = "BUY"
                    reason = f"score>={buy_thresh:.4f}"
                    # record an entry stub in trades (exit fields fill on sell)
                    trades.append({
                        "entry_date": date, "entry_index": i, "entry_price": entry_price, "qty": qty,
                        "entry_cost": total_cost, "exit_date": None, "exit_index": None,
                        "exit_price": None, "exit_proceeds": None, "pnl": None, "return_pct": None,
                        "exit_reason": None, "holding_days": None
                    })
                else:
                    action = "NO_BUY_insufficient_cash"
                    reason = "insufficient_cash_for_qty"
            else:
                action = "NO_BUY_qty0"
                reason = "qty_computed_0"

        # EXIT checks
        elif shares > 0:
            # compute stop and take
            stop_price = entry_price * (1 - stop_loss)
            take_price = entry_price * (1 + take_profit)
            hit_stop = price <= stop_price
            hit_take = price >= take_price
            signal_sell = score <= sell_thresh

            # forced time exit
            holding_days = (df['Date'].iloc[i] - df['Date'].iloc[entry_index]).days if entry_index is not None else 0
            timed_exit = (MAX_HOLD_DAYS is not None) and (holding_days >= MAX_HOLD_DAYS)

            if hit_stop or hit_take or signal_sell or timed_exit:
                exit_price = sell_price
                notional = shares * exit_price
                commission_cost = notional * commission
                proceeds = notional - commission_cost
                cash += proceeds
                # fill last trade exit details
                for t in reversed(trades):
                    if t['exit_date'] is None:
                        t['exit_date'] = date
                        t['exit_index'] = i
                        t['exit_price'] = exit_price
                        t['exit_proceeds'] = proceeds
                        t['pnl'] = proceeds - t['entry_cost']
                        t['return_pct'] = (exit_price / t['entry_price']) - 1 if t['entry_price'] and t['entry_price']>0 else None
                        t['exit_reason'] = "STOP" if hit_stop else ("TAKE" if hit_take else ("SIGNAL" if signal_sell else "TIME"))
                        t['holding_days'] = (t['exit_index'] - t['entry_index']) if t['entry_index'] is not None and t['exit_index'] is not None else None
                        break
                shares = 0
                entry_price = None
                entry_index = None
                action = "SELL"
                reason = t['exit_reason']

        equity = cash + shares * price
        equity_curve.append(equity)

        daily_rows.append({
            "Date": date, "Close": price, "Score": score, "Action": action,
            "Reason": reason, "Cash": cash, "Shares": shares, "Equity": equity
        })

    # end loop - if position still open, close at last price
    if shares > 0:
        date = df['Date'].iloc[-1]
        price = float(df['Close'].iloc[-1])
        exit_price = price * (1 - slippage)
        notional = shares * exit_price
        commission_cost = notional * commission
        proceeds = notional - commission_cost
        cash += proceeds
        for t in reversed(trades):
            if t['exit_date'] is None:
                t['exit_date'] = date
                t['exit_index'] = n - 1
                t['exit_price'] = exit_price
                t['exit_proceeds'] = proceeds
                t['pnl'] = proceeds - t['entry_cost']
                t['return_pct'] = (exit_price / t['entry_price']) - 1 if t['entry_price'] and t['entry_price']>0 else None
                t['exit_reason'] = "END_CLOSE"
                t['holding_days'] = (t['exit_index'] - t['entry_index']) if t['entry_index'] is not None else None
                break
        shares = 0
        equity = cash
        equity_curve[-1] = equity
        if daily_rows:
            daily_rows[-1]['Cash'] = cash
            daily_rows[-1]['Shares'] = shares
            daily_rows[-1]['Equity'] = equity

    # Build dataframes, save logs
    daily_df = pd.DataFrame(daily_rows)
    trades_df = pd.DataFrame(trades)

    # Ensure types
    if not trades_df.empty:
        trades_df = trades_df[[
            "entry_date", "exit_date", "entry_price", "exit_price", "qty",
            "entry_cost", "exit_proceeds", "pnl", "return_pct", "holding_days", "exit_reason"
        ]]

    # Save logs
    os.makedirs(out_folder, exist_ok=True)
    daily_df.to_csv(os.path.join(out_folder, f"{stock_name}_daily.csv"), index=False)
    trades_df.to_csv(os.path.join(out_folder, f"{stock_name}_trades.csv"), index=False)

    return np.array(equity_curve), daily_df, trades_df

# -----------------------
# Performance metrics
# -----------------------
def portfolio_performance(portfolio_values: np.ndarray, dates: pd.Series) -> dict:
    if portfolio_values is None or len(portfolio_values) < 2:
        return {'Total Return': 0.0, 'CAGR': 0.0, 'Sharpe Ratio': np.nan, 'Max Drawdown': 0.0, 'Average Daily Return': 0.0, 'Daily Return StdDev': 0.0}
    rets = np.diff(portfolio_values) / portfolio_values[:-1]
    avg = np.nanmean(rets)
    std = np.nanstd(rets)
    sharpe = (avg / std) * np.sqrt(252) if std > 0 else np.nan
    total_return = portfolio_values[-1] / portfolio_values[0] - 1
    days = (dates.iloc[-1] - dates.iloc[0]).days
    years = days / 365.25 if days > 0 else 1.0
    cagr = (portfolio_values[-1] / portfolio_values[0]) ** (1/years) - 1 if portfolio_values[0] > 0 else 0.0
    running_max = np.maximum.accumulate(portfolio_values)
    max_dd = float(np.nanmax((running_max - portfolio_values) / running_max))
    return {
        'Total Return': float(total_return),
        'CAGR': float(cagr),
        'Sharpe Ratio': float(sharpe) if not np.isnan(sharpe) else np.nan,
        'Max Drawdown': float(max_dd),
        'Average Daily Return': float(avg),
        'Daily Return StdDev': float(std)
    }

# -----------------------
# Main runner
# -----------------------
def main():
    os.makedirs(LOG_FOLDER, exist_ok=True)
    os.makedirs(STATS_FOLDER, exist_ok=True)

    files = sorted([f for f in os.listdir(DATA_FOLDER) if f.lower().endswith('.csv')])
    if not files:
        print("No CSVs found in", DATA_FOLDER)
        return

    weights = load_best_weights(len(STRAT_FUNCS))
    print("Using weights (first 8 shown):", weights[:8], " (sum=", weights.sum(), ")")
    # use exact best params
    params = BEST_PARAMS.copy()

    all_stats = []
    merged_equity_df = None
    portfolio_cols = []

    for fname in files:
        name = os.path.splitext(fname)[0]
        path = os.path.join(DATA_FOLDER, fname)
        try:
            df = load_and_clean_csv(path, START_DATE, END_DATE)
        except Exception as e:
            print(f"Skipping {fname}: failed to read/clean ({e})")
            continue
        if df.empty:
            print(f"Skipping {fname}: no data after cleaning")
            continue

        signals = calculate_signals(df)
        raw_score = calculate_combined_score(df, signals, weights, window=5)

        # scale per-stock so thresholds are reachable while preserving relative info
        scaled_score, scaler = scale_score_to_thresholds(raw_score, params['buy_thresh'], params['sell_thresh'])
        if scaler > 1.0:
            print(f"[{name}] applied score scaler {scaler:.3f} so thresholds are reachable")

        equity_curve, daily_df, trades_df = simulate_trades_and_log(
            df, scaled_score, params, INITIAL_CAPITAL, stock_name=name, out_folder=LOG_FOLDER
        )

        stats = portfolio_performance(equity_curve, df['Date'])
        stats['Stock'] = name
        stats['ScalerUsed'] = scaler
        stats['NumTrades'] = len(trades_df) if trades_df is not None else 0
        all_stats.append(stats)

        # save equity curve per stock
        eq_df = pd.DataFrame({"Date": df['Date'], f"Equity_{name}": equity_curve})
        eq_out = os.path.join(LOG_FOLDER, f"{name}_equity.csv")
        eq_df.to_csv(eq_out, index=False)

        # merge for overall
        if merged_equity_df is None:
            merged_equity_df = eq_df.copy()
        else:
            merged_equity_df = pd.merge(merged_equity_df, eq_df, on='Date', how='outer')
        portfolio_cols.append(f"Equity_{name}")

        print(f"[{name}] trades: {len(trades_df)}  Total Return: {stats['Total Return']:.2%}  CAGR: {stats['CAGR']:.2%}")

    # write summary
    summary_df = pd.DataFrame(all_stats)
    if not summary_df.empty:
        cols = ['Stock', 'ScalerUsed', 'NumTrades', 'Total Return', 'CAGR', 'Sharpe Ratio', 'Max Drawdown', 'Average Daily Return', 'Daily Return StdDev']
        summary_df = summary_df[cols].sort_values('Total Return', ascending=False)
        summary_path = os.path.join(STATS_FOLDER, 'final_optimized_portfolio_stats.csv')
        summary_df.to_csv(summary_path, index=False)
        print("\nSaved summary ->", summary_path)
        print(summary_df.head(40).to_string(index=False))

    # overall equity
    if merged_equity_df is not None:
        merged_equity_df = merged_equity_df.sort_values('Date').reset_index(drop=True)
        merged_equity_df[portfolio_cols] = merged_equity_df[portfolio_cols].ffill().bfill()
        merged_equity_df['Overall_Equity'] = merged_equity_df[portfolio_cols].mean(axis=1)
        overall_path = os.path.join(LOG_FOLDER, "overall_equity.csv")
        merged_equity_df[['Date', 'Overall_Equity']].to_csv(overall_path, index=False)
        overall_stats = portfolio_performance(merged_equity_df['Overall_Equity'].values, merged_equity_df['Date'])
        print("\nOverall portfolio stats:")
        for k, v in overall_stats.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.2%}" if k != 'Sharpe Ratio' else f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        print("Saved overall equity ->", overall_path)

if __name__ == "__main__":
    np.seterr(all='ignore')
    pd.set_option('mode.chained_assignment', None)
    main()

