# fetch.py

import pandas as pd
from binance.client import Client
import datetime

# If you keep your API keys in a config.py, do something like this:
# from config import BINANCE_API_KEY, BINANCE_API_SECRET

# Or hardcode them here (not recommended):
BINANCE_API_KEY = "YOUR_API_KEY"
BINANCE_API_SECRET = "YOUR_API_SECRET"


def fetch_binance_data(
        symbol="BTCUSDT",
        interval=Client.KLINE_INTERVAL_15MINUTE,
        start_date="1 Jan 2024",
        end_date=None,
        save_csv=True,
        csv_filename="btc_15min_data.csv"
):
    """
    Fetches historical klines for a specific symbol and interval from Binance.
    Saves data to a CSV if save_csv=True.

    Parameters:
    -----------
    symbol : str
        e.g., "BTCUSDT", "ETHUSDT", etc.
    interval : str
        One of Client.KLINE_INTERVAL_* constants, e.g. Client.KLINE_INTERVAL_15MINUTE
    start_date : str
        Starting date (Binance accepts date strings like "1 Jan 2021")
    end_date : str or None
        Ending date. If None, fetches up to the most recent data.
    save_csv : bool
        If True, saves the fetched data as a CSV file.
    csv_filename : str
        The name/path for the CSV file output.

    Returns:
    --------
    df : pd.DataFrame
        DataFrame containing the historical klines.
    """

    # Initialize the Binance client
    client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)

    # If end_date is not provided, let's fetch up to the current date
    if end_date is None:
        end_date = datetime.datetime.utcnow().strftime("%d %b %Y")

    # Fetch the kline (candlestick) data
    klines = client.get_historical_klines(symbol, interval, start_date, end_date)

    # Create a DataFrame from the klines data
    # The kline data comes in this order:
    # [
    #   [
    #     1499040000000,      // Open time
    #     "0.01634790",       // Open
    #     "0.80000000",       // High
    #     "0.01575800",       // Low
    #     "0.01577100",       // Close
    #     "148976.11427815",  // Volume
    #     1499644799999,      // Close time
    #     "2434.19055334",    // Quote asset volume
    #     308,                // Number of trades
    #     "175.19023394",     // Taker buy base asset volume
    #     "28.46694368",      // Taker buy quote asset volume
    #     "17928899.62484339" // Ignore (deprecated)
    #   ]
    # ]
    columns = [
        "Open Time", "Open", "High", "Low", "Close", "Volume",
        "Close Time", "Quote Asset Volume", "Number of Trades",
        "Taker Buy Base Volume", "Taker Buy Quote Volume", "Ignore"
    ]

    df = pd.DataFrame(klines, columns=columns)

    # Convert numeric columns from strings to floats
    numeric_cols = ["Open", "High", "Low", "Close", "Volume", "Quote Asset Volume",
                    "Taker Buy Base Volume", "Taker Buy Quote Volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convert timestamps
    df["Open Time"] = pd.to_datetime(df["Open Time"], unit='ms')
    df["Close Time"] = pd.to_datetime(df["Close Time"], unit='ms')
    df.set_index("Open Time", inplace=True)

    # Optionally save to CSV
    if save_csv:
        df.to_csv(csv_filename)

    return df


if __name__ == "__main__":
    # Example usage:
    df_data = fetch_binance_data(
        symbol="BTCUSDT",
        interval=Client.KLINE_INTERVAL_15MINUTE,
        start_date="1 Jan 2024",  # start of data
        end_date=None,  # fetch up to latest
        save_csv=True,
        csv_filename="btc_15min_data.csv"
    )
    print("Data fetched and saved to btc_15min_data.csv")
    print(df_data.head())
