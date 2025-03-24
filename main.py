import pandas as pd
import numpy as np
import optuna


def calculate_indicators(df, sma_period, rsi_period, bb_period, bb_std):
    """
    Calculates SMA, RSI, and Bollinger Bands and adds them as columns to df.
    """
    # Simple Moving Average
    df['SMA'] = df['Close'].rolling(window=sma_period).mean()

    # RSI Calculation
    delta = df['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    roll_up = pd.Series(gain).rolling(window=rsi_period).mean()
    roll_down = pd.Series(loss).rolling(window=rsi_period).mean()
    rs = roll_up / roll_down
    df['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands Calculation
    df['BB_MID'] = df['Close'].rolling(window=bb_period).mean()
    df['BB_STD'] = df['Close'].rolling(window=bb_period).std()
    df['BB_UPPER'] = df['BB_MID'] + bb_std * df['BB_STD']
    df['BB_LOWER'] = df['BB_MID'] - bb_std * df['BB_STD']

    return df


def simulate_trading(df, sma_period, rsi_period, bb_period, bb_std, rsi_entry, rsi_exit,
                     stop_loss_pct, take_profit_pct, leverage):
    """
    Simulate trading over the historical data using the given parameters.

    Entry condition for a long position:
      - Price is above the SMA,
      - RSI is below the entry threshold,
      - Price is at or below the lower Bollinger band.

    Exit conditions (if in a long position):
      - Price falls to or below the stop-loss level,
      - Price rises to or above the take-profit level,
      - RSI exceeds the exit threshold.

    The strategy profit is adjusted by the provided leverage.
    """
    # Work on a copy of the DataFrame so as not to modify the original.
    df_copy = df.copy()
    df_copy = calculate_indicators(df_copy, sma_period, rsi_period, bb_period, bb_std)

    balance = 10000.0  # Starting capital
    in_position = False
    entry_price = 0.0

    positions = []  # 1 if in a position, 0 otherwise
    signals = []  # 1 for entry, -1 for exit, 0 for none
    balances = []  # Updated balance over time

    for i in range(len(df_copy)):
        # If any indicator is NaN, append current status and continue.
        if pd.isna(df_copy['SMA'].iloc[i]) or pd.isna(df_copy['RSI'].iloc[i]) \
                or pd.isna(df_copy['BB_LOWER'].iloc[i]) or pd.isna(df_copy['BB_UPPER'].iloc[i]):
            positions.append(1 if in_position else 0)
            signals.append(0)
            balances.append(balance)
            continue

        current_price = df_copy['Close'].iloc[i]
        signal = 0

        if not in_position:
            # Entry condition for long position:
            if (current_price > df_copy['SMA'].iloc[i]) and \
                    (df_copy['RSI'].iloc[i] < rsi_entry) and \
                    (current_price <= df_copy['BB_LOWER'].iloc[i]):
                in_position = True
                entry_price = current_price
                signal = 1  # Buy signal
        else:
            # Define stop-loss and take-profit based on the entry price.
            stop_loss = entry_price * (1 - stop_loss_pct)
            take_profit = entry_price * (1 + take_profit_pct)
            # Exit conditions:
            if (current_price <= stop_loss) or (current_price >= take_profit) or (df_copy['RSI'].iloc[i] > rsi_exit):
                in_position = False
                profit = (current_price - entry_price) * leverage
                balance += profit
                signal = -1  # Sell signal

        positions.append(1 if in_position else 0)
        signals.append(signal)
        balances.append(balance)

    df_copy['Position'] = positions
    df_copy['Signal'] = signals
    df_copy['Balance'] = balances

    return df_copy, balance


def objective(trial):
    """
    Optuna objective function that suggests parameter values, simulates trading,
    and returns the final balance (to be maximized).
    """
    # Strategy and indicator parameters.
    sma_period = trial.suggest_int("sma_period", 10, 100)
    rsi_period = trial.suggest_int("rsi_period", 5, 30)
    bb_period = trial.suggest_int("bb_period", 10, 40)
    bb_std = trial.suggest_uniform("bb_std", 1.5, 3.0)
    rsi_entry = trial.suggest_int("rsi_entry", 20, 40)
    rsi_exit = trial.suggest_int("rsi_exit", 60, 80)

    # Risk management parameters.
    stop_loss_pct = trial.suggest_uniform("stop_loss_pct", 0.005, 0.05)  # e.g., 0.5% to 5%
    risk_reward_ratio = trial.suggest_uniform("risk_reward_ratio", 1.0, 5.0)
    take_profit_pct = stop_loss_pct * risk_reward_ratio
    leverage = trial.suggest_int("leverage", 1, 10)

    # Use the global DataFrame loaded in main.
    global df
    backtest_df, final_balance = simulate_trading(
        df, sma_period, rsi_period, bb_period, bb_std,
        rsi_entry, rsi_exit, stop_loss_pct, take_profit_pct, leverage
    )
    return final_balance


if __name__ == "__main__":
    # 1. Load data from CSV (ensure the CSV has the header as provided)
    data_file = "btc_15min_data.csv"
    df = pd.read_csv(data_file, parse_dates=["Open Time"])
    df.sort_values(by="Open Time", inplace=True)
    df.set_index("Open Time", inplace=True)

    # 2. Create an Optuna study to maximize final balance.
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)  # Increase n_trials for a more thorough search.

    print("Best trial:")
    trial = study.best_trial
    print("Final balance:", trial.value)
    print("Best parameters:", trial.params)

    # 3. Run simulation with the best parameters.
    best_params = trial.params
    best_take_profit_pct = best_params["stop_loss_pct"] * best_params["risk_reward_ratio"]
    best_df, best_balance = simulate_trading(
        df,
        best_params["sma_period"],
        best_params["rsi_period"],
        best_params["bb_period"],
        best_params["bb_std"],
        best_params["rsi_entry"],
        best_params["rsi_exit"],
        best_params["stop_loss_pct"],
        best_take_profit_pct,
        best_params["leverage"]
    )

    print(f"Final balance with best parameters: {best_balance:.2f}")
    print(best_df.tail(20))
