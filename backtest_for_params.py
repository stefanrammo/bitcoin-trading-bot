#!/usr/bin/env python
"""
Backtest for Trading Strategy Parameters using CSV Data

This script loads historical candlestick data from a CSV file,
calculates technical indicators (SMA and RSI) using the parameters in BEST_PARAMS,
and performs a backtest of a strategy that takes both long and short trades.

Strategy (using candle close values only):
  - Long Entry (BUY): if the candle’s close > SMA and RSI < rsi_entry.
  - Short Entry (SELL): if the candle’s close < SMA and RSI > rsi_exit.

Once in a trade, exit on the first candle that meets ANY of the following exit conditions:
  For LONG trades:
      • Candle’s close falls below its SMA OR RSI rises above rsi_exit,
      • OR the candle’s close is less than or equal to the stop‑loss level,
      • OR the candle’s close is greater than or equal to the take‑profit level.
      (Stop Loss = entry_price * (1 – stop_loss_pct); TP = entry_price * (1 + take_profit_pct))

  For SHORT trades:
      • Candle’s close rises above its SMA OR RSI falls below rsi_entry,
      • OR the candle’s close is greater than or equal to the stop‑loss level,
      • OR the candle’s close is less than or equal to the take‑profit level.
      (Stop Loss = entry_price * (1 + stop_loss_pct); TP = entry_price * (1 - take_profit_pct))

When a trade is closed, the profit is computed using the leverage:
  - For a long trade: profit = (exit_price – entry_price) * leverage
  - For a short trade: profit = (entry_price – exit_price) * leverage

Fees are applied on both entry and exit (using a fee rate of 0.04% per side),
so that the net profit is reduced by:
    fee_cost = (entry_price + exit_price) * leverage * fee_rate

The balance is updated additively with the net profit.

At the end, the script prints:
  • Total trades, wins, losses, and average PnL,
  • Final account value (starting from $10,000)
It also plots:
  • A price chart with entry/exit markers,
  • An equity curve.

Requirements:
    pip install pandas numpy matplotlib PyQt5

Usage:
    python backtest_for_params.py
"""

import matplotlib
matplotlib.use('Qt5Agg')  # Use Qt5Agg backend for interactive plots on Ubuntu
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import logging
from datetime import datetime

# -----------------------------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("backtest.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# -----------------------------------------------------------------------------
# Constants and BEST_PARAMS (from the best trial adjusted from your Optuna backtest)
# -----------------------------------------------------------------------------
FEE_RATE = 0.0004  # 0.04% fee per side

BEST_PARAMS = {
    "sma_period": 76,
    "rsi_period": 5,
    "rsi_entry": 22,    # For LONG entry: RSI below 22
    "rsi_exit": 60,     # For SHORT entry: RSI above 60
    "stop_loss_pct": 0.0478,   # 4.78% stop loss
    "take_profit_pct": 0.1305, # 13.05% take profit
    "leverage": 3
}

# -----------------------------------------------------------------------------
# Technical Indicator Functions
# -----------------------------------------------------------------------------
def compute_sma(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return prices.rolling(window=period).mean()

def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# -----------------------------------------------------------------------------
# Backtesting Function (candle-close based simulation)
# -----------------------------------------------------------------------------
def run_backtest():
    csv_path = "data/btc_15min_data.csv"
    logger.info(f"Loading CSV data from {csv_path}")

    # Read CSV and rename columns for consistency
    df = pd.read_csv(csv_path)
    df.rename(columns={
        "Open Time": "open_time",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "Close Time": "close_time",
        "Quote Asset Volume": "quote_asset_volume",
        "Number of Trades": "num_trades",
        "Taker Buy Base Volume": "taker_buy_base_volume",
        "Taker Buy Quote Volume": "taker_buy_quote_volume",
        "Ignore": "ignore"
    }, inplace=True)

    # Convert time columns to datetime
    df["open_time"] = pd.to_datetime(df["open_time"])
    df["close_time"] = pd.to_datetime(df["close_time"])

    # Ensure numeric types for price columns
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["high"] = pd.to_numeric(df["high"], errors="coerce")
    df["low"] = pd.to_numeric(df["low"], errors="coerce")

    # Calculate technical indicators
    df["sma"] = compute_sma(df["close"], BEST_PARAMS["sma_period"])
    df["rsi"] = compute_rsi(df["close"], BEST_PARAMS["rsi_period"])

    # Drop rows with NaN (from rolling windows)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Backtesting variables
    starting_equity = 10000.0
    equity = starting_equity
    trades = []
    trade_dates = []  # Record exit times for equity curve
    equity_curve = []
    trade_count = 0
    wins = 0
    losses = 0
    in_position = False
    entry_price = None
    entry_time = None
    trade_type = None  # "BUY" for long, "SELL" for short

    # Iterate through the candles row by row
    for idx, row in df.iterrows():
        current_price = row["close"]
        current_sma = row["sma"]
        current_rsi = row["rsi"]
        current_time = row["close_time"]

        # If not in a trade, check for an entry signal
        if not in_position:
            if current_price > current_sma and current_rsi < BEST_PARAMS["rsi_entry"]:
                in_position = True
                trade_type = "BUY"
                entry_price = current_price
                entry_time = current_time
                trade_count += 1
            elif current_price < current_sma and current_rsi > BEST_PARAMS["rsi_exit"]:
                in_position = True
                trade_type = "SELL"
                entry_price = current_price
                entry_time = current_time
                trade_count += 1

            equity_curve.append(equity)
            trade_dates.append(current_time)
            continue

        # If in a trade, compute exit thresholds and check exit conditions
        if trade_type == "BUY":
            stop_loss = entry_price * (1 - BEST_PARAMS["stop_loss_pct"])
            take_profit = entry_price * (1 + BEST_PARAMS["take_profit_pct"])
            if (current_price < current_sma or current_rsi > BEST_PARAMS["rsi_exit"] or
                current_price <= stop_loss or current_price >= take_profit):
                exit_price = current_price
                exit_time = current_time
                raw_profit = (exit_price - entry_price) * BEST_PARAMS["leverage"]
                fee_cost = (entry_price + exit_price) * BEST_PARAMS["leverage"] * FEE_RATE
                net_profit = raw_profit - fee_cost
                net_pnl_pct = (net_profit / (entry_price * BEST_PARAMS["leverage"])) * 100
                equity += net_profit
                if net_pnl_pct > 0:
                    wins += 1
                else:
                    losses += 1
                trades.append({
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "trade_type": trade_type,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_pct": net_pnl_pct,
                    "equity": equity
                })
                in_position = False

        elif trade_type == "SELL":
            stop_loss = entry_price * (1 + BEST_PARAMS["stop_loss_pct"])
            take_profit = entry_price * (1 - BEST_PARAMS["take_profit_pct"])
            if (current_price > current_sma or current_rsi < BEST_PARAMS["rsi_entry"] or
                current_price >= stop_loss or current_price <= take_profit):
                exit_price = current_price
                exit_time = current_time
                raw_profit = (entry_price - exit_price) * BEST_PARAMS["leverage"]
                fee_cost = (entry_price + exit_price) * BEST_PARAMS["leverage"] * FEE_RATE
                net_profit = raw_profit - fee_cost
                net_pnl_pct = (net_profit / (entry_price * BEST_PARAMS["leverage"])) * 100
                equity += net_profit
                if net_pnl_pct > 0:
                    wins += 1
                else:
                    losses += 1
                trades.append({
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "trade_type": trade_type,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_pct": net_pnl_pct,
                    "equity": equity
                })
                in_position = False

        equity_curve.append(equity)
        trade_dates.append(current_time)

    # Backtest summary
    total_trades = len(trades)
    avg_pnl = np.mean([t["pnl_pct"] for t in trades]) if trades else 0
    total_profit_pct = (equity / starting_equity - 1) * 100

    logger.info(f"Backtest completed. Total trades: {total_trades}")
    logger.info(f"Wins: {wins}, Losses: {losses}, Average PnL: {avg_pnl:.2f}%")
    logger.info(f"Total Profit: {total_profit_pct:.2f}%, Final Account Value: ${equity:.2f}")

    # Plot Price Chart with trade markers
    plt.figure(figsize=(14, 6))
    plt.plot(df["close_time"], df["close"], label="Close Price", color="blue", alpha=0.5)
    for trade in trades:
        if trade["trade_type"] == "BUY":
            plt.plot(trade["entry_time"], trade["entry_price"], marker="^", color="green", markersize=10)
            plt.plot(trade["exit_time"], trade["exit_price"], marker="v", color="red", markersize=10)
        else:
            plt.plot(trade["entry_time"], trade["entry_price"], marker="v", color="red", markersize=10)
            plt.plot(trade["exit_time"], trade["exit_price"], marker="^", color="green", markersize=10)
    plt.title("BTC Price with Trade Markers")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

    # Plot Equity Curve
    plt.figure(figsize=(10, 4))
    plt.plot(trade_dates, equity_curve, marker="o", linestyle="-", color="purple")
    plt.title("Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Equity ($)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_backtest()
