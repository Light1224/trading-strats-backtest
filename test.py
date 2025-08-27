#!/usr/bin/env python3
import os, math
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

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
    "commission_pct": 0.001,
    "slippage_pct": 0.0010,
}

TARGET_VOL = 0.20
VOL_LOOKBACK_DAYS = 63
MIN_VOL_SCALE = 0.5
MAX_VOL_SCALE = 2.0
MAX_TOTAL_RISK = 0.30
MAX_POSITIONS = 20
MAX_LEVERAGE = 2.0

DD_WARNING = 0.20
DD_REDUCE = 0.40
DD_HALT = 0.60

SHORT_BORROW_FEE_ANNUAL = 0.03
MARGIN_REQ = 0.50
MIN_QTY_FALLBACK_ALLOC_FRAC = 0.10
MAX_HOLD_DAYS = 252 * 2
SCORE_WINDOW = 5
MIN_AVG_VOLUME_FOR_TRADE = 10

KELLY_FRACTION = 0.5
RISK_PARITY_LOOKBACK = 20
TRAIL_ATR_MULT = 2.5
MAX_DAILY_LOSS = 0.10

def compute_qty(entry_price, equity, max_risk_per_trade, stop_loss_pct):
    if entry_price <= 0 or equity <= 0:
        return 0
    risk_cash = equity * max_risk_per_trade
    per_share_risk = entry_price * stop_loss_pct
    qty = math.floor(risk_cash / (per_share_risk + 1e-12))
    return max(0, int(qty))

def cap_between(x, lo, hi):
    return max(lo, min(hi, x))

def kelly_position(win_rate, payoff, loss_rate, kelly_fraction=1.0):
    if payoff <= 0 or loss_rate == 0: return kelly_fraction
    kelly = win_rate - (1 - win_rate) / payoff
    return cap_between(kelly * kelly_fraction, 0, 1)

def compute_atr(df, window=14):
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window).mean().fillna(0)

def compute_correlation_matrix(stocks, current_date, lookback=20):
    price_data = {}
    for stock, df in stocks.items():
        df_hist = df[df["Date"] <= current_date].tail(lookback)
        if len(df_hist) < lookback: continue
        price_data[stock] = df_hist["Close"].astype(float).values
    if len(price_data) < 2: return None
    price_df = pd.DataFrame(price_data)
    return price_df.pct_change().corr()

files = sorted([f for f in os.listdir(DATA_FOLDER) if f.lower().endswith('.csv')])
if not files:
    raise SystemExit("No CSV files in data/")

weights = np.ones(20) / 20
params = BEST_PARAMS.copy()
print("Using exact optimized params:", params)

stocks, calendar = {}, set()
for fname in files:
    df = pd.read_csv(os.path.join(DATA_FOLDER, fname), parse_dates=["Date"])
    df = df[(df["Date"] >= START_DATE) & (df["Date"] <= END_DATE)]
    if df.empty: continue
    for col in ["Close", "High", "Low"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "High" not in df.columns or df["High"].isnull().all():
        df["High"] = df["Close"]
    if "Low" not in df.columns or df["Low"].isnull().all():
        df["Low"] = df["Close"]
    if "Score" not in df.columns:
        np.random.seed(42)
        df["Score"] = np.random.normal(0, 5, len(df))
    df["ATR"] = compute_atr(df)
    stocks[fname[:-4]] = df.reset_index(drop=True)
    calendar.update(df["Date"].dt.date.values)

calendar = sorted(pd.to_datetime(list(calendar)))

cash = float(initial_capital)
positions = {}
trade_rows, portfolio_history = [], []
daily_loss = 0.0
last_equity = initial_capital

def compute_held_value(positions):
    held_value = 0.0
    for pos in positions.values():
        if pos["side"] == "LONG":
            held_value += pos["qty"] * pos["last_price"]
        elif pos["side"] == "SHORT":
            pnl = (pos["entry_price"] - pos["last_price"]) * pos["qty"]
            held_value += pos.get("margin", 0.0) + pnl
    return held_value

def process_stock(args):
    (stock, df, current_date, positions, params, allow_new_entries, corr_matrix, open_stocks, 
     effective_risk_per_trade, total_at_risk, cash, equity, total_notional) = args
    buy_cand = None
    sell_cand = None
    short_cand = None
    cover_cand = None

    row = df[df["Date"].dt.date == current_date.date()]
    if row.empty: return (buy_cand, sell_cand, short_cand, cover_cand, stock, None)
    price = float(row["Close"].iloc[0])
    score = float(row["Score"].iloc[0])
    atr = float(row["ATR"].iloc[0])

    if stock in positions:
        pos = positions[stock]
        entry_price = pos["entry_price"]
        holding_days = (current_date - pos["entry_date"]).days
        stop_price_long  = entry_price * (1 - params["stop_loss_pct"])
        take_price_long  = entry_price * (1 + params["take_profit_pct"])
        stop_price_short = entry_price * (1 + params["stop_loss_pct"])
        take_price_short = entry_price * (1 - params["take_profit_pct"])

        if pos["side"] == "LONG":
            trail_stop = max(stop_price_long, price - TRAIL_ATR_MULT * atr)
            if price <= trail_stop or price >= take_price_long or score <= params["sell_thresh"]:
                sell_cand = (stock, price, "EXIT_LONG")
            elif holding_days >= MAX_HOLD_DAYS:
                sell_cand = (stock, price, "TIME")
        elif pos["side"] == "SHORT":
            trail_stop = min(stop_price_short, price + TRAIL_ATR_MULT * atr)
            if price >= trail_stop or price <= take_price_short or score >= params["buy_thresh"]:
                cover_cand = (stock, price, "EXIT_SHORT")
            elif holding_days >= MAX_HOLD_DAYS:
                cover_cand = (stock, price, "TIME")
    else:
        if not allow_new_entries:
            return (buy_cand, sell_cand, short_cand, cover_cand, stock, price)
        if corr_matrix is not None and open_stocks:
            if stock in corr_matrix.index:
                valid_open = [s for s in open_stocks if s in corr_matrix.columns]
                if valid_open:
                    corrs = corr_matrix.loc[stock, valid_open]
                    if any(abs(corrs) > 0.85):
                        return (buy_cand, sell_cand, short_cand, cover_cand, stock, price)
        risk_contrib = []
        for s, d in stocks.items():
            d_hist = d[d["Date"] <= current_date].tail(RISK_PARITY_LOOKBACK)
            if len(d_hist) < RISK_PARITY_LOOKBACK: continue
            returns = d_hist["Close"].pct_change().dropna()
            risk_contrib.append(returns.std())
        avg_risk = np.mean(risk_contrib) if risk_contrib else 1.0
        asset_risk = row["Close"].pct_change().std() if len(row) > 1 else avg_risk
        risk_parity_scale = avg_risk / (asset_risk + 1e-12)
        risk_parity_scale = cap_between(risk_parity_scale, 0.5, 2.0)
        win_rate, payoff = 0.5, 2.0
        kelly_frac = kelly_position(win_rate, payoff, 1 - win_rate, KELLY_FRACTION)
        if score >= params["buy_thresh"]:
            equity_now = equity
            base_qty = compute_qty(price, equity_now, effective_risk_per_trade * risk_parity_scale * kelly_frac, params["stop_loss_pct"])
            qty = base_qty
            max_allowed_risk_cash = equity_now * MAX_TOTAL_RISK
            this_trade_risk_cash = qty * price * params["stop_loss_pct"]
            if total_at_risk + this_trade_risk_cash > max_allowed_risk_cash and this_trade_risk_cash > 0:
                allowed_risk_remaining = max(0.0, max_allowed_risk_cash - total_at_risk)
                qty = int(math.floor(allowed_risk_remaining / (price * params["stop_loss_pct"] + 1e-12)))
            cost = qty * price
            if qty > 0 and cost <= cash:
                buy_cand = (stock, price, score, qty)
        elif score <= params["sell_thresh"]:
            equity_now = equity
            base_qty = compute_qty(price, equity_now, effective_risk_per_trade * risk_parity_scale * kelly_frac, params["stop_loss_pct"])
            qty = base_qty
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
                short_cand = (stock, price, score, qty)
    return (buy_cand, sell_cand, short_cand, cover_cand, stock, price)

for current_date in calendar:
    buy_cands, sell_cands, short_cands, cover_cands = [], [], [], []

    held_value = compute_held_value(positions)
    equity = cash + held_value

    hist_df = pd.DataFrame(portfolio_history)
    if not hist_df.empty:
        hist_df = hist_df.sort_values("Date").reset_index(drop=True)
        ret_series = hist_df["Equity"].pct_change().dropna()
        recent_ret = ret_series.iloc[-VOL_LOOKBACK_DAYS:] if len(ret_series) >= VOL_LOOKBACK_DAYS else ret_series
        if len(recent_ret) >= 2:
            realized_vol = recent_ret.std() * np.sqrt(252)
        else:
            realized_vol = np.std(ret_series) * np.sqrt(252) if len(ret_series) > 0 else TARGET_VOL
    else:
        realized_vol = TARGET_VOL

    vol_scale = TARGET_VOL / (realized_vol + 1e-12)
    vol_scale = cap_between(vol_scale, MIN_VOL_SCALE, MAX_VOL_SCALE)

    if not hist_df.empty:
        peak = hist_df["Equity"].cummax().iloc[-1]
        current_dd = 0.0 if peak <= 0 else (equity - peak) / peak
    else:
        current_dd = 0.0

    if current_dd > -DD_HALT:
        dd_mag = abs(min(0.0, current_dd))
    else:
        dd_mag = abs(min(0.0, current_dd))
    dd_factor = 1.0
    if dd_mag >= DD_HALT:
        dd_factor = 0.0
    elif dd_mag >= DD_REDUCE:
        dd_factor = 0.25
    elif dd_mag >= DD_WARNING:
        dd_factor = 0.6

    effective_risk_per_trade = params["max_risk_per_trade"] * vol_scale * dd_factor
    effective_risk_per_trade = cap_between(effective_risk_per_trade, 0.005, 0.5)

    total_at_risk = 0.0
    total_notional = 0.0
    for p in positions.values():
        if p["side"] == "LONG":
            total_at_risk += p["qty"] * p["entry_price"] * params["stop_loss_pct"]
            total_notional += p["qty"] * p["last_price"]
        else:
            total_at_risk += p["qty"] * p["entry_price"] * params["stop_loss_pct"]
            total_notional += p["qty"] * p["last_price"]

    allow_new_entries = True
    if len(positions) >= MAX_POSITIONS:
        allow_new_entries = False
    if equity > 0 and (total_notional / (equity + 1e-12)) >= MAX_LEVERAGE:
        allow_new_entries = False
    if total_at_risk >= equity * MAX_TOTAL_RISK:
        allow_new_entries = False
    if effective_risk_per_trade <= 0.0:
        allow_new_entries = False
    if not hist_df.empty and (equity - last_equity) < -MAX_DAILY_LOSS * last_equity:
        allow_new_entries = False

    corr_matrix = compute_correlation_matrix(stocks, current_date, lookback=RISK_PARITY_LOOKBACK)
    open_stocks = set(positions.keys())

    args_list = []
    for stock, df in stocks.items():
        args_list.append((stock, df, current_date, positions, params, allow_new_entries, corr_matrix, open_stocks, 
                          effective_risk_per_trade, total_at_risk, cash, equity, total_notional))

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_stock, args_list))

    for buy_cand, sell_cand, short_cand, cover_cand, stock, price in results:
        if stock in positions and price is not None:
            positions[stock]["last_price"] = price
        if buy_cand: buy_cands.append(buy_cand)
        if sell_cand: sell_cands.append(sell_cand)
        if short_cand: short_cands.append(short_cand)
        if cover_cand: cover_cands.append(cover_cand)

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

    for stock, price, reason in cover_cands:
        if stock not in positions: continue
        pos = positions.pop(stock)
        qty = pos["qty"]
        exit_price = price
        proceeds = pos["entry_price"] * qty - exit_price * qty
        cash += pos.get("margin", 0.0) + proceeds
        trade_rows.append({"Date": current_date, "Stock": stock, "Action": "COVER", "Side": "SHORT", "Qty": qty,
                           "Price": exit_price, "Proceeds": proceeds, "CashAfter": cash, "Reason": reason})
        print(f"{current_date.date()} | COVER | {stock} | Qty: {qty} | ExitPrice: {exit_price:.2f} | Cash: {cash:.2f} | Reason: {reason}")

    for item in buy_cands:
        stock, price, score, qty = item
        if qty <= 0: continue
        cost = qty * price
        if cost > cash:
            continue
        cash -= cost
        positions[stock] = {"qty": qty, "entry_price": price, "entry_date": current_date,
                            "last_price": price, "side": "LONG"}
        total_at_risk += qty * price * params["stop_loss_pct"]
        total_notional += qty * price
        trade_rows.append({"Date": current_date, "Stock": stock, "Action": "BUY", "Side": "LONG", "Qty": qty,
                           "Price": price, "Proceeds": -cost, "CashAfter": cash, "Reason": "SCORE"})
        print(f"{current_date.date()} | BUY  | {stock} | Qty: {qty} | EntryPrice: {price:.2f} | Cash: {cash:.2f}")

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

    for pos in positions.values():
        if pos["side"] == "SHORT":
            fee = (SHORT_BORROW_FEE_ANNUAL / 252) * pos["entry_price"] * pos["qty"]
            cash -= fee

    held_value = compute_held_value(positions)
    equity = cash + held_value
    portfolio_history.append({"Date": current_date, "Cash": cash, "Equity": equity, "OpenPositions": len(positions)})
    daily_loss = equity - last_equity
    last_equity = equity

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

