#!/usr/bin/env python3
"""
Aggressive portfolio backtester with SHORTING support (realistic).
Uses your optimized params exactly.
Prints all trades (long + short).
"""

import os, math
import pandas as pd, numpy as np
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
DATA_FOLDER = "data"
LOG_FOLDER = "backtest_logs"
STATS_FOLDER = "stats"
BEST_WEIGHTS_PATH = "best_weights.npy"

os.makedirs(LOG_FOLDER, exist_ok=True)
os.makedirs(STATS_FOLDER, exist_ok=True)

start_date_str = input("Start date (YYYY-MM-DD): ").strip()
end_date_str   = input("End date (YYYY-MM-DD): ").strip()
initial_capital = float(input("Initial capital (e.g. 100000): ").strip())

START_DATE = pd.to_datetime(start_date_str)
END_DATE   = pd.to_datetime(end_date_str)

BEST_PARAMS = {
    "buy_thresh": 1.8451,
    "sell_thresh": -4.6955,
    "max_risk_per_trade": 0.10,
    "stop_loss_pct": 0.20,
    "take_profit_pct": 0.40,
    "commission_pct": 0.0000,
    "slippage_pct": 0.0000,
}

SHORT_BORROW_FEE_ANNUAL = 0.03   # 3% annual borrow fee
MARGIN_REQ = 0.50                # 50% margin for shorts
MIN_QTY_FALLBACK_ALLOC_FRAC = 0.10
MAX_HOLD_DAYS = 252 * 2
SCORE_WINDOW = 5
MIN_AVG_VOLUME_FOR_TRADE = 10

# ---------------- Utilities ----------------
def compute_qty(entry_price, equity, max_risk_per_trade, stop_loss_pct):
    if entry_price <= 0 or equity <= 0:
        return 0
    risk_cash = equity * max_risk_per_trade
    per_share_risk = entry_price * stop_loss_pct
    qty = math.floor(risk_cash / (per_share_risk + 1e-12))
    return max(0, int(qty))

# ---------------- Load data ----------------
files = sorted([f for f in os.listdir(DATA_FOLDER) if f.lower().endswith('.csv')])
if not files:
    raise SystemExit("No CSV files in data/")

weights = np.ones(20) / 20  # fallback equal weights
params = BEST_PARAMS.copy()
print("Using exact optimized params:", params)

stocks, calendar = {}, set()
for fname in files:
    df = pd.read_csv(os.path.join(DATA_FOLDER, fname), parse_dates=["Date"])
    df = df[(df["Date"] >= START_DATE) & (df["Date"] <= END_DATE)]
    if df.empty: continue
    # Example score (random if no precomputed). Replace with your signals.
    if "Score" not in df.columns:
        np.random.seed(42)
        df["Score"] = np.random.normal(0, 5, len(df))
    stocks[fname[:-4]] = df.reset_index(drop=True)
    calendar.update(df["Date"].dt.date.values)

calendar = sorted(pd.to_datetime(list(calendar)))

# ---------------- Portfolio simulation ----------------
cash = float(initial_capital)
positions = {}  # { stock: {qty, entry_price, entry_date, last_price, side} }
trade_rows, portfolio_history = [], []

for current_date in calendar:
    buy_cands, sell_cands, short_cands, cover_cands = [], [], [], []

    for stock, df in stocks.items():
        row = df[df["Date"].dt.date == current_date.date()]
        if row.empty: continue
        price = float(row["Close"].iloc[0])
        score = float(row["Score"].iloc[0])

        if stock in positions:
            pos = positions[stock]
            entry_price = pos["entry_price"]
            holding_days = (current_date - pos["entry_date"]).days
            stop_price_long  = entry_price * (1 - params["stop_loss_pct"])
            take_price_long  = entry_price * (1 + params["take_profit_pct"])
            stop_price_short = entry_price * (1 + params["stop_loss_pct"])
            take_price_short = entry_price * (1 - params["take_profit_pct"])

            if pos["side"] == "LONG":
                if price <= stop_price_long or price >= take_price_long or score <= params["sell_thresh"]:
                    sell_cands.append((stock, price, "EXIT_LONG"))
                elif holding_days >= MAX_HOLD_DAYS: 
                    sell_cands.append((stock, price, "TIME"))
                else:
                    pos["last_price"] = price

            elif pos["side"] == "SHORT":
                if price >= stop_price_short or price <= take_price_short or score >= params["buy_thresh"]:
                    cover_cands.append((stock, price, "EXIT_SHORT"))
                elif holding_days >= MAX_HOLD_DAYS:
                    cover_cands.append((stock, price, "TIME"))
                else:
                    pos["last_price"] = price
        else:
            if score >= params["buy_thresh"]:
                buy_cands.append((stock, price, score))
            elif score <= params["sell_thresh"]:
                short_cands.append((stock, price, score))

    # Execute sells (close longs)
    for stock, price, reason in sell_cands:
        if stock not in positions: continue
        pos = positions.pop(stock)
        qty = pos["qty"]
        exit_price = price
        proceeds = qty * exit_price
        cash += proceeds
        trade_rows.append({"Date": current_date, "Stock": stock, "Action": "SELL", "Side": "LONG", "Qty": qty,
                           "Price": exit_price, "Proceeds": proceeds, "CashAfter": cash, "Reason": reason})
        print(f"{current_date.date()} | SELL | {stock} | Qty: {qty} | ExitPrice: {exit_price:.2f} | Cash: {cash:.2f} | Reason: {reason}")

    # Execute covers (close shorts)
    for stock, price, reason in cover_cands:
        if stock not in positions: continue
        pos = positions.pop(stock)
        qty = pos["qty"]
        exit_price = price
        proceeds = pos["entry_price"] * qty - exit_price * qty
        cash += pos["margin"] + proceeds  # return margin + P&L
        trade_rows.append({"Date": current_date, "Stock": stock, "Action": "COVER", "Side": "SHORT", "Qty": qty,
                           "Price": exit_price, "Proceeds": proceeds, "CashAfter": cash, "Reason": reason})
        print(f"{current_date.date()} | COVER | {stock} | Qty: {qty} | ExitPrice: {exit_price:.2f} | Cash: {cash:.2f} | Reason: {reason}")

    # Execute buys
    for stock, price, score in buy_cands:
        equity = cash + sum(p["qty"] * p["last_price"] for p in positions.values())
        qty = compute_qty(price, equity, params["max_risk_per_trade"], params["stop_loss_pct"])
        if qty <= 0: continue
        cost = qty * price
        if cost > cash: continue
        cash -= cost
        positions[stock] = {"qty": qty, "entry_price": price, "entry_date": current_date,
                            "last_price": price, "side": "LONG"}
        trade_rows.append({"Date": current_date, "Stock": stock, "Action": "BUY", "Side": "LONG", "Qty": qty,
                           "Price": price, "Proceeds": -cost, "CashAfter": cash, "Reason": "SCORE"})
        print(f"{current_date.date()} | BUY | {stock} | Qty: {qty} | EntryPrice: {price:.2f} | Cash: {cash:.2f}")

    # Execute shorts
    for stock, price, score in short_cands:
        equity = cash + sum(p["qty"] * p["last_price"] for p in positions.values())
        qty = compute_qty(price, equity, params["max_risk_per_trade"], params["stop_loss_pct"])
        if qty <= 0: continue
        notional = qty * price
        margin = notional * MARGIN_REQ
        if margin > cash: continue
        cash -= margin
        positions[stock] = {"qty": qty, "entry_price": price, "entry_date": current_date,
                            "last_price": price, "side": "SHORT", "margin": margin}
        trade_rows.append({"Date": current_date, "Stock": stock, "Action": "SHORT", "Side": "SHORT", "Qty": qty,
                           "Price": price, "Proceeds": 0, "CashAfter": cash, "Reason": "SCORE"})
        print(f"{current_date.date()} | SHORT | {stock} | Qty: {qty} | EntryPrice: {price:.2f} | Cash: {cash:.2f}")

    # Apply borrow fee for shorts
    for pos in positions.values():
        if pos["side"] == "SHORT":
            fee = (SHORT_BORROW_FEE_ANNUAL / 252) * pos["entry_price"] * pos["qty"]
            cash -= fee

    # portfolio snapshot
    held_value = 0
    for pos in positions.values():
        if pos["side"] == "LONG":
            held_value += pos["qty"] * pos["last_price"]
        elif pos["side"] == "SHORT":
            pnl = (pos["entry_price"] - pos["last_price"]) * pos["qty"]
            held_value += pos["margin"] + pnl
    equity = cash + held_value
    portfolio_history.append({"Date": current_date, "Cash": cash, "Equity": equity, "OpenPositions": len(positions)})

# ---------------- Save results ----------------
trades_df = pd.DataFrame(trade_rows)
if not trades_df.empty:
    trades_df.to_csv(os.path.join(LOG_FOLDER, "all_trades.csv"), index=False)
    print("Saved trades -> backtest_logs/all_trades.csv")

port_df = pd.DataFrame(portfolio_history)
port_df.to_csv(os.path.join(LOG_FOLDER, "portfolio_curve.csv"), index=False)

plt.plot(port_df["Date"], port_df["Equity"])
plt.title("Portfolio Equity Curve")
plt.savefig(os.path.join(LOG_FOLDER, "portfolio_curve.png"))
plt.show()

# ---------------- Statistics ----------------
if not port_df.empty:
    port_df["Returns"] = port_df["Equity"].pct_change().fillna(0)
    total_return = port_df["Equity"].iloc[-1] / port_df["Equity"].iloc[0] - 1
    years = (port_df["Date"].iloc[-1] - port_df["Date"].iloc[0]).days / 365.25
    CAGR = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    volatility = port_df["Returns"].std() * np.sqrt(252)
    sharpe = (port_df["Returns"].mean() * 252) / (volatility + 1e-12)
    downside = port_df.loc[port_df["Returns"] < 0, "Returns"].std() * np.sqrt(252)
    sortino = (port_df["Returns"].mean() * 252) / (downside + 1e-12)
    rolling_max = port_df["Equity"].cummax()
    drawdown = (port_df["Equity"] - rolling_max) / rolling_max
    max_dd = drawdown.min()
    calmar = CAGR / abs(max_dd) if max_dd < 0 else np.nan

    # trade stats
    total_trades = len(trades_df)
    wins = 0
    losses = 0
    for _, t in trades_df.iterrows():
        if t["Action"] in ["SELL", "COVER"]:
            if t["Proceeds"] > 0:
                wins += 1
            else:
                losses += 1
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
    avg_trade = trades_df["Proceeds"].mean() if not trades_df.empty else 0
    profit_factor = trades_df.loc[trades_df["Proceeds"] > 0, "Proceeds"].sum() / abs(trades_df.loc[trades_df["Proceeds"] < 0, "Proceeds"].sum() + 1e-12)

    print("\n------ PERFORMANCE STATISTICS ------")
    print(f"Total Return: {total_return*100:.2f}%")
    print(f"CAGR: {CAGR*100:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Sortino Ratio: {sortino:.2f}")
    print(f"Max Drawdown: {max_dd*100:.2f}%")
    print(f"Calmar Ratio: {calmar:.2f}")
    print(f"Volatility (ann.): {volatility*100:.2f}%")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate*100:.2f}%")
    print(f"Average Trade P&L: {avg_trade:.2f}")
    print(f"Profit Factor: {profit_factor:.2f}")

