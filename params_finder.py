#!/usr/bin/env python
import pandas as pd
import numpy as np
import optuna
from datetime import datetime, timedelta

# Fee rate constant (0.04% per side)
FEE_RATE = 0.0004


def calculate_indicators(df, sma_period, rsi_period):
    # Calculate Simple Moving Average (SMA)
    df['SMA'] = df['Close'].rolling(window=sma_period).mean()

    # Calculate RSI (using a standard method)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    roll_up = gain.rolling(window=rsi_period, min_periods=rsi_period).mean()
    roll_down = loss.rolling(window=rsi_period, min_periods=rsi_period).mean()
    rs = roll_up / roll_down
    df['RSI'] = np.where(roll_down == 0, 100, 100 - (100 / (1 + rs)))

    return df


def simulate_trading(df, sma_period, rsi_period,
                     rsi_entry, rsi_exit,
                     stop_loss_pct, take_profit_pct, leverage):
    """
    Simulate a trading strategy that takes both long and short positions.

    When not in a position:
      - Enter long if current price > SMA and RSI < rsi_entry.
      - Enter short if current price < SMA and RSI > rsi_exit.

    For long trades:
      - Exit when price falls below SMA, RSI rises above rsi_exit,
        or when stop loss / take profit is hit.
      - Stop loss = entry_price * (1 - stop_loss_pct)
      - Take profit = entry_price * (1 + take_profit_pct)
      - Raw profit = (current_price - entry_price) * leverage
      - Fee cost = (entry_price + current_price) * leverage * FEE_RATE
      - Net profit = raw profit - fee cost

    For short trades:
      - Exit when price rises above SMA, RSI falls below rsi_entry,
        or when stop loss / take profit is hit.
      - Stop loss = entry_price * (1 + stop_loss_pct)
      - Take profit = entry_price * (1 - take_profit_pct)
      - Raw profit = (entry_price - current_price) * leverage
      - Fee cost = (entry_price + current_price) * leverage * FEE_RATE
      - Net profit = raw profit - fee cost

    A maximum drawdown penalty is applied if drawdown exceeds 25%.

    Additionally, the function counts the number of completed trades.

    Returns:
      df_copy: DataFrame with added indicator and simulation columns.
      final_balance: final account balance after simulation.
      max_drawdown: maximum drawdown experienced.
      trade_count: total number of completed trades.
    """
    df_copy = df.copy()
    df_copy = calculate_indicators(df_copy, sma_period, rsi_period)

    balance = 10000.0
    in_position = False
    entry_price = 0.0
    running_max = balance  # Track running maximum balance

    positions = []  # Record: 1 for long, -1 for short, 0 for flat
    signals = []  # Record trade signals: 1 = enter long, -1 = exit long, -2 = enter short, 2 = exit short, 0 = nothing
    balances = []
    trade_count = 0
    current_position = 0  # 1 for long, -1 for short, 0 for flat

    for i in range(len(df_copy)):
        sma_val = df_copy['SMA'].iloc[i]
        rsi_val = df_copy['RSI'].iloc[i]
        current_price = df_copy['Close'].iloc[i]

        # If indicators are not available, just record the current state.
        if pd.isna(sma_val) or pd.isna(rsi_val):
            positions.append(current_position)
            signals.append(0)
            balances.append(balance)
            continue

        signal = 0

        # Check for entry signals if not in a trade
        if not in_position:
            # Long Entry: price above SMA and RSI below rsi_entry
            if current_price > sma_val and rsi_val < rsi_entry:
                in_position = True
                current_position = 1
                entry_price = current_price
                signal = 1  # Enter long
            # Short Entry: price below SMA and RSI above rsi_exit
            elif current_price < sma_val and rsi_val > rsi_exit:
                in_position = True
                current_position = -1
                entry_price = current_price
                signal = -2  # Enter short

        # If in a trade, check for exit conditions
        elif in_position:
            if current_position == 1:  # In a long trade
                stop_loss = entry_price * (1 - stop_loss_pct)
                take_profit = entry_price * (1 + take_profit_pct)
                if (current_price < sma_val or rsi_val > rsi_exit or
                        current_price <= stop_loss or current_price >= take_profit):
                    # Calculate net profit for long trade
                    raw_profit = (current_price - entry_price) * leverage
                    fee_cost = (entry_price + current_price) * leverage * FEE_RATE
                    net_profit = raw_profit - fee_cost
                    balance = balance + net_profit
                    in_position = False
                    signal = -1  # Exit long
                    trade_count += 1
                    current_position = 0
            elif current_position == -1:  # In a short trade
                stop_loss = entry_price * (1 + stop_loss_pct)
                take_profit = entry_price * (1 - take_profit_pct)
                if (current_price > sma_val or rsi_val < rsi_entry or
                        current_price >= stop_loss or current_price <= take_profit):
                    # Calculate net profit for short trade
                    raw_profit = (entry_price - current_price) * leverage
                    fee_cost = (entry_price + current_price) * leverage * FEE_RATE
                    net_profit = raw_profit - fee_cost
                    balance = balance + net_profit
                    in_position = False
                    signal = 2  # Exit short
                    trade_count += 1
                    current_position = 0

        positions.append(current_position)
        signals.append(signal)
        balances.append(balance)
        running_max = max(running_max, balance)
        current_drawdown = (running_max - balance) / running_max
        if current_drawdown > 0.25:
            return df_copy, -10000, current_drawdown, trade_count

    df_copy['Position'] = positions
    df_copy['Signal'] = signals
    df_copy['Balance'] = balances

    balance_series = pd.Series(balances)
    running_max_series = balance_series.cummax()
    drawdowns = (running_max_series - balance_series) / running_max_series
    max_drawdown = drawdowns.max()

    return df_copy, balance, max_drawdown, trade_count


def objective(trial):
    sma_period = trial.suggest_int("sma_period", 10, 100)
    rsi_period = trial.suggest_int("rsi_period", 5, 30)
    rsi_entry = trial.suggest_int("rsi_entry", 20, 40)
    rsi_exit = trial.suggest_int("rsi_exit", 60, 80)
    stop_loss_pct = trial.suggest_float("stop_loss_pct", 0.005, 0.05)
    risk_reward_ratio = trial.suggest_float("risk_reward_ratio", 1.0, 5.0)
    take_profit_pct = stop_loss_pct * risk_reward_ratio
    leverage = trial.suggest_int("leverage", 1, 10)

    global df
    one_year_ago = df.index.max() - pd.DateOffset(years=1)
    filtered_df = df.loc[df.index >= one_year_ago]
    if filtered_df.empty:
        filtered_df = df

    _, final_balance, _, trade_count = simulate_trading(
        filtered_df, sma_period, rsi_period,
        rsi_entry, rsi_exit,
        stop_loss_pct, take_profit_pct, leverage
    )

    # Automatically penalize any trial with fewer than 10 completed trades.
    if trade_count < 10:
        return -10000
    return final_balance


if __name__ == "__main__":
    data_file = "data/btc_15min_data.csv"
    df = pd.read_csv(data_file, parse_dates=["Open Time"])

    # Clean column headers and ensure 'Close' is numeric.
    df.columns = df.columns.str.strip()
    df.sort_values(by="Open Time", inplace=True)
    df.set_index("Open Time", inplace=True)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    one_year_ago = df.index.max() - pd.DateOffset(years=1)
    df_filtered = df.loc[df.index >= one_year_ago]
    if not df_filtered.empty:
        df = df_filtered

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=500)

    best_trial = study.best_trial
    formatted_params = {
        "sma_period": best_trial.params["sma_period"],
        "rsi_period": best_trial.params["rsi_period"],
        "rsi_entry": best_trial.params["rsi_entry"],
        "rsi_exit": best_trial.params["rsi_exit"],
        "stop_loss_pct": f"{best_trial.params['stop_loss_pct']:.2%}",
        "risk_reward_ratio": f"{best_trial.params['risk_reward_ratio']:.2f}",
        "leverage": best_trial.params["leverage"]
    }
    best_take_profit_pct = best_trial.params["stop_loss_pct"] * best_trial.params["risk_reward_ratio"]

    print("Best trial:")
    print(f"Final balance: {best_trial.value:.2f}")
    print("Best parameters:")
    for key, value in formatted_params.items():
        print(f"  {key}: {value}")
    print(f"Calculated take_profit_pct: {best_take_profit_pct:.2%}")

    result_df, final_bal, max_dd, trade_count = simulate_trading(
        df,
        best_trial.params["sma_period"],
        best_trial.params["rsi_period"],
        best_trial.params["rsi_entry"],
        best_trial.params["rsi_exit"],
        best_trial.params["stop_loss_pct"],
        best_take_profit_pct,
        best_trial.params["leverage"]
    )
    print(f"Final balance with best parameters: {final_bal:.2f}")
    print(f"Maximum Drawdown: {max_dd * 100:.2f}%")
    print(f"Total Trades: {trade_count}")
    print(result_df.tail(10))
