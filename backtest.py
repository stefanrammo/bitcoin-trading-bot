import pandas as pd
import numpy as np
import optuna
from datetime import datetime, timedelta


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
    df_copy = df.copy()
    df_copy = calculate_indicators(df_copy, sma_period, rsi_period)

    balance = 10000.0
    in_position = False
    entry_price = 0.0
    running_max = balance  # Track running maximum balance

    positions = []
    signals = []
    balances = []

    # Strategy:
    # Long Entry: if current price is above the SMA and RSI is below rsi_entry.
    # Long Exit: if price falls below the SMA or RSI exceeds rsi_exit,
    #            or if stop-loss or take-profit levels (based on entry) are reached.
    for i in range(len(df_copy)):
        sma_val = df_copy['SMA'].iloc[i]
        rsi_val = df_copy['RSI'].iloc[i]
        current_price = df_copy['Close'].iloc[i]

        # Skip if indicators are not available.
        if pd.isna(sma_val) or pd.isna(rsi_val):
            positions.append(1 if in_position else 0)
            signals.append(0)
            balances.append(balance)
            continue

        signal = 0

        if not in_position:
            if (current_price > sma_val) and (rsi_val < rsi_entry):
                in_position = True
                entry_price = current_price
                signal = 1  # Buy signal
        else:
            stop_loss = entry_price * (1 - stop_loss_pct)
            take_profit = entry_price * (1 + take_profit_pct)
            if (current_price < sma_val) or (rsi_val > rsi_exit) or (current_price <= stop_loss) or (
                    current_price >= take_profit):
                in_position = False
                profit = (current_price - entry_price) * leverage
                balance += profit
                signal = -1  # Sell signal

        # Update running max and check drawdown
        running_max = max(running_max, balance)
        current_drawdown = (running_max - balance) / running_max  # positive percentage

        # If drawdown exceeds 25%, immediately penalize the trial.
        if current_drawdown > 0.25:
            return df_copy, -10000, current_drawdown

        positions.append(1 if in_position else 0)
        signals.append(signal)
        balances.append(balance)

    df_copy['Position'] = positions
    df_copy['Signal'] = signals
    df_copy['Balance'] = balances

    # Compute maximum drawdown for reporting (as a positive percentage)
    balance_series = pd.Series(balances)
    running_max_series = balance_series.cummax()
    drawdowns = (running_max_series - balance_series) / running_max_series
    max_drawdown = drawdowns.max()  # Value between 0 and 1

    return df_copy, balance, max_drawdown


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

    _, final_balance, _ = simulate_trading(
        filtered_df, sma_period, rsi_period,
        rsi_entry, rsi_exit,
        stop_loss_pct, take_profit_pct, leverage
    )
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

    result_df, final_bal, max_dd = simulate_trading(
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
    print(result_df.tail(10))
