#!/usr/bin/env python3
"""
Aggressive portfolio backtester with SHORTING support (realistic).
Uses your optimized params exactly.
Prints all trades (long + short).

Enhanced with:
- Volatility targeting (scales per-trade risk by recent realized vol)
- Drawdown control (reduces risk / halts new entries when drawdown deepens)
- Risk budget (caps total at-risk capital across open positions)
- Max positions & leverage caps

These changes only affect capital allocation / sizing / entry gating.
Core trade logic & exits remain the same.
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
    "max_risk_per_trade": 0.10,   # base risk per trade (fraction of equity)
    "stop_loss_pct": 0.20,
    "take_profit_pct": 0.40,
    "commission_pct": 0.001,
    "slippage_pct": 0.0010,
}

# Risk management / sizing hyperparams (new)
TARGET_VOL = 0.20                 # target annual volatility (20%)
VOL_LOOKBACK_DAYS = 63            # ~quarter realized vol window for scaling
MIN_VOL_SCALE = 0.5               # don't scale below this
MAX_VOL_SCALE = 2.0               # don't scale above this
MAX_TOTAL_RISK = 0.30             # don't risk more than 30% of equity across open positions
MAX_POSITIONS = 20                # hard cap on open simultaneous positions
MAX_LEVERAGE = 2.0                # max notional / equity

# Drawdown-based controls
DD_WARNING = 0.20                 # start reducing when drawdown > 20%
DD_REDUCE = 0.40                  # further reduce when dd > 40%
DD_HALT = 0.60                    # halt new entries if drawdown > 60%

SHORT_BORROW_FEE_ANNUAL = 0.03    # 3% annual borrow fee
MARGIN_REQ = 0.50                 # 50% margin for shorts
MIN_QTY_FALLBACK_ALLOC_FRAC = 0.10
MAX_HOLD_DAYS = 252 * 2
SCORE_WINDOW = 5
MIN_AVG_VOLUME_FOR_TRADE = 10

# ---------------- Utilities ----------------
def compute_qty(entry_price, equity, max_risk_per_trade, stop_loss_pct):
    """Compute quantity that risks `max_risk_per_trade` fraction of equity with given stop."""
    if entry_price <= 0 or equity <= 0:
        return 0
    risk_cash = equity * max_risk_per_trade
    per_share_risk = entry_price * stop_loss_pct
    qty = math.floor(risk_cash / (per_share_risk + 1e-12))
    return max(0, int(qty))

def cap_between(x, lo, hi):
    return max(lo, min(hi, x))

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
positions = {}  # { stock: {qty, entry_price, entry_date, last_price, side, margin(if short)} }
trade_rows, portfolio_history = [], []

# helper to compute current equity snapshot
def compute_held_value(positions):
    held_value = 0.0
    for pos in positions.values():
        if pos["side"] == "LONG":
            held_value += pos["qty"] * pos["last_price"]
        elif pos["side"] == "SHORT":
            pnl = (pos["entry_price"] - pos["last_price"]) * pos["qty"]
            held_value += pos.get("margin", 0.0) + pnl
    return held_value

for current_date in calendar:
    buy_cands, sell_cands, short_cands, cover_cands = [], [], [], []

    # compute current equity from last snapshot
    held_value = compute_held_value(positions)
    equity = cash + held_value

    # compute realized vol from recent portfolio history (for volatility targeting)
    hist_df = pd.DataFrame(portfolio_history)
    if not hist_df.empty:
        # Use equity-based returns for volatility estimate
        hist_df = hist_df.sort_values("Date").reset_index(drop=True)
        # include today's not-yet-appended equity to compute realized vol: use last recorded equity (if any)
        ret_series = hist_df["Equity"].pct_change().dropna()
        recent_ret = ret_series.iloc[-VOL_LOOKBACK_DAYS:] if len(ret_series) >= VOL_LOOKBACK_DAYS else ret_series
        if len(recent_ret) >= 2:
            realized_vol = recent_ret.std() * np.sqrt(252)
        else:
            realized_vol = np.std(ret_series) * np.sqrt(252) if len(ret_series) > 0 else TARGET_VOL
    else:
        realized_vol = TARGET_VOL

    # vol scaling: if realized vol > target, reduce risk, else increase
    vol_scale = TARGET_VOL / (realized_vol + 1e-12)
    vol_scale = cap_between(vol_scale, MIN_VOL_SCALE, MAX_VOL_SCALE)

    # drawdown check: compute peak and current drawdown
    if not hist_df.empty:
        peak = hist_df["Equity"].cummax().iloc[-1]
        current_dd = 0.0 if peak <= 0 else (equity - peak) / peak
    else:
        current_dd = 0.0

    # drawdown factor: multiply base risk_per_trade by this
    if current_dd > -DD_HALT:
        # current_dd is negative when in drawdown; convert to positive magnitude
        dd_mag = abs(min(0.0, current_dd))
    else:
        dd_mag = abs(min(0.0, current_dd))
    # default factor = 1; reduce when dd exceeds thresholds
    dd_factor = 1.0
    if dd_mag >= DD_HALT:
        dd_factor = 0.0    # halt new entries
    elif dd_mag >= DD_REDUCE:
        dd_factor = 0.25   # sharply reduce risk
    elif dd_mag >= DD_WARNING:
        dd_factor = 0.6    # moderate reduction

    # Effective max risk per trade after vol targeting and drawdown control
    effective_risk_per_trade = params["max_risk_per_trade"] * vol_scale * dd_factor
    effective_risk_per_trade = cap_between(effective_risk_per_trade, 0.005, 0.5)  # sensible bounds

    # compute current total at-risk capital across open positions (sum of (qty*entry_price*stop_loss_pct))
    total_at_risk = 0.0
    total_notional = 0.0
    for p in positions.values():
        if p["side"] == "LONG":
            total_at_risk += p["qty"] * p["entry_price"] * params["stop_loss_pct"]
            total_notional += p["qty"] * p["last_price"]
        else:  # SHORT
            # for shorts, at-risk (if stop loss triggers) is similar magnitude
            total_at_risk += p["qty"] * p["entry_price"] * params["stop_loss_pct"]
            total_notional += p["qty"] * p["last_price"]

    # gate for too many positions or excessive leverage: don't take new entries if constraints exceeded
    allow_new_entries = True
    if len(positions) >= MAX_POSITIONS:
        allow_new_entries = False
    if equity > 0 and (total_notional / (equity + 1e-12)) >= MAX_LEVERAGE:
        allow_new_entries = False
    if total_at_risk >= equity * MAX_TOTAL_RISK:
        allow_new_entries = False
    if effective_risk_per_trade <= 0.0:
        allow_new_entries = False

    # iterate stocks
    for stock, df in stocks.items():
        row = df[df["Date"].dt.date == current_date.date()]
        if row.empty: continue
        price = float(row["Close"].iloc[0])
        score = float(row["Score"].iloc[0])

        # ensure positions store last_price even if not changing
        if stock in positions:
            positions[stock]["last_price"] = price

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
            # only consider new entries if allowed by global controls
            if not allow_new_entries:
                continue

            if score >= params["buy_thresh"]:
                # compute qty using effective risk per trade (volatility & drawdown adjusted)
                equity_now = equity
                base_qty = compute_qty(price, equity_now, effective_risk_per_trade, params["stop_loss_pct"])
                qty = base_qty

                # enforce total_at_risk budget: reduce qty until total_at_risk + this_trade_risk <= MAX_TOTAL_RISK * equity
                max_allowed_risk_cash = equity_now * MAX_TOTAL_RISK
                this_trade_risk_cash = qty * price * params["stop_loss_pct"]
                # if this trade would exceed budget, scale qty down
                if total_at_risk + this_trade_risk_cash > max_allowed_risk_cash and this_trade_risk_cash > 0:
                    allowed_risk_remaining = max(0.0, max_allowed_risk_cash - total_at_risk)
                    qty = int(math.floor(allowed_risk_remaining / (price * params["stop_loss_pct"] + 1e-12)))

                # ensure cost fits available cash (for long)
                cost = qty * price
                if qty > 0 and cost <= cash:
                    # final check: don't exceed max positions
                    if len(positions) < MAX_POSITIONS and qty > 0:
                        buy_cands.append((stock, price, score, qty))
            elif score <= params["sell_thresh"]:
                equity_now = equity
                base_qty = compute_qty(price, equity_now, effective_risk_per_trade, params["stop_loss_pct"])
                qty = base_qty

                # enforce risk budget and margin requirements
                notional = qty * price
                margin = notional * MARGIN_REQ
                max_allowed_risk_cash = equity_now * MAX_TOTAL_RISK
                this_trade_risk_cash = qty * price * params["stop_loss_pct"]
                if total_at_risk + this_trade_risk_cash > max_allowed_risk_cash and this_trade_risk_cash > 0:
                    allowed_risk_remaining = max(0.0, max_allowed_risk_cash - total_at_risk)
                    qty = int(math.floor(allowed_risk_remaining / (price * params["stop_loss_pct"] + 1e-12)))
                    notional = qty * price
                    margin = notional * MARGIN_REQ

                if qty > 0 and margin <= cash:
                    if len(positions) < MAX_POSITIONS and qty > 0:
                        short_cands.append((stock, price, score, qty))

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
        cash += pos.get("margin", 0.0) + proceeds  # return margin + P&L
        trade_rows.append({"Date": current_date, "Stock": stock, "Action": "COVER", "Side": "SHORT", "Qty": qty,
                           "Price": exit_price, "Proceeds": proceeds, "CashAfter": cash, "Reason": reason})
        print(f"{current_date.date()} | COVER | {stock} | Qty: {qty} | ExitPrice: {exit_price:.2f} | Cash: {cash:.2f} | Reason: {reason}")

    # Execute buys (new longs) - note buy_cands now contains qty
    for item in buy_cands:
        stock, price, score, qty = item
        if qty <= 0: continue
        cost = qty * price
        if cost > cash:  # skip if not enough cash
            continue
        cash -= cost
        positions[stock] = {"qty": qty, "entry_price": price, "entry_date": current_date,
                            "last_price": price, "side": "LONG"}
        # update total_at_risk and total_notional bookkeeping
        total_at_risk += qty * price * params["stop_loss_pct"]
        total_notional += qty * price
        trade_rows.append({"Date": current_date, "Stock": stock, "Action": "BUY", "Side": "LONG", "Qty": qty,
                           "Price": price, "Proceeds": -cost, "CashAfter": cash, "Reason": "SCORE"})
        print(f"{current_date.date()} | BUY  | {stock} | Qty: {qty} | EntryPrice: {price:.2f} | Cash: {cash:.2f}")

    # Execute shorts (new)
    for item in short_cands:
        stock, price, score, qty = item
        if qty <= 0: continue
        notional = qty * price
        margin = notional * MARGIN_REQ
        if margin > cash:
            continue
        cash -= margin
        positions[stock] = {"qty": qty, "entry_price": price, "entry_date": current_date,
                            "last_price": price, "side": "SHORT", "margin": margin}
        total_at_risk += qty * price * params["stop_loss_pct"]
        total_notional += qty * price
        trade_rows.append({"Date": current_date, "Stock": stock, "Action": "SHORT", "Side": "SHORT", "Qty": qty,
                           "Price": price, "Proceeds": 0, "CashAfter": cash, "Reason": "SCORE"})
        print(f"{current_date.date()} | SHORT | {stock} | Qty: {qty} | EntryPrice: {price:.2f} | Cash: {cash:.2f}")

    # Apply borrow fee for shorts
    for pos in positions.values():
        if pos["side"] == "SHORT":
            fee = (SHORT_BORROW_FEE_ANNUAL / 252) * pos["entry_price"] * pos["qty"]
            cash -= fee

    # portfolio snapshot (update last prices already set above)
    held_value = compute_held_value(positions)
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
    port_df = port_df.sort_values("Date").reset_index(drop=True)
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
    gross_profit = 0.0
    gross_loss = 0.0
    for _, t in trades_df.iterrows():
        if t["Action"] in ["SELL", "COVER"]:
            if t["Proceeds"] > 0:
                wins += 1
                gross_profit += t["Proceeds"]
            else:
                losses += 1
                gross_loss += t["Proceeds"]
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
    avg_trade = trades_df["Proceeds"].mean() if not trades_df.empty else 0
    profit_factor = gross_profit / (abs(gross_loss) + 1e-12)

    # extra risk metrics
    max_daily_loss = port_df["Returns"].min()
    avg_daily_return = port_df["Returns"].mean()
    skew = port_df["Returns"].skew()
    kurtosis = port_df["Returns"].kurtosis()

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
    print(f"Max Daily Return (neg): {max_daily_loss:.4f}")
    print(f"Avg Daily Return: {avg_daily_return:.6f}")
    print(f"Returns Skew: {skew:.4f}")
    print(f"Returns Kurtosis: {kurtosis:.4f}")

    # save a small stats summary to disk
    stats_summary = {
        "Total Return %": total_return * 100,
        "CAGR %": CAGR * 100,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown %": max_dd * 100,
        "Calmar": calmar,
        "Volatility %": volatility * 100,
        "Total Trades": total_trades,
        "Win Rate %": win_rate * 100,
        "Avg Trade P&L": avg_trade,
        "Profit Factor": profit_factor,
        "Skew": skew,
        "Kurtosis": kurtosis,
    }
    pd.DataFrame([stats_summary]).to_csv(os.path.join(STATS_FOLDER, "performance_stats.csv"), index=False)
    print(f"Saved stats -> {os.path.join(STATS_FOLDER, 'performance_stats.csv')}")

