#!/usr/bin/env python
"""
Binance Testnet Futures Trading Bot Example (Continuous Loop)

This example demonstrates:
  • Loading API credentials from a .env file
  • Retrieving historical candlestick data from Binance testnet
  • Calculating SMA and RSI technical indicators
  • Making a trading decision based on a simple strategy:
      - If the current close > SMA and RSI < rsi_entry → BUY signal
      - If the current close < SMA and RSI > rsi_exit → SELL signal
      - Otherwise, HOLD.
  • Placing an entry order along with stop‑loss and take‑profit orders
  • Running continuously, waiting for a new closed candle on the 15-minute timeframe.

Before you run:
  1. Create a `.env` file in the same directory with the following:
       API_KEY=YOUR_BINANCE_TESTNET_API_KEY
       API_SECRET=YOUR_BINANCE_TESTNET_API_SECRET
  2. Install required libraries:
       pip install python-dotenv requests pandas numpy

For Binance Futures testnet the base URL is:
  https://testnet.binancefuture.com
"""

import os
import time
import hmac
import hashlib
import requests
import urllib.parse
import json
from datetime import datetime, timezone

import pandas as pd
import numpy as np
from dotenv import load_dotenv
import logging

# -----------------------------------------------------------------------------
# Logging configuration: log only essential information to file and console.
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# -----------------------------------------------------------------------------
# Load environment variables from .env file
# -----------------------------------------------------------------------------
load_dotenv()
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

if not API_KEY or not API_SECRET:
    raise Exception("Please set API_KEY and API_SECRET in your .env file")

# Binance Futures Testnet base URL
BASE_URL = "https://testnet.binancefuture.com"

# -----------------------------------------------------------------------------
# Best trial parameters (from your backtest/optimization)
# -----------------------------------------------------------------------------
BEST_PARAMS = {
    "sma_period": 87,
    "rsi_period": 17,
    "rsi_entry": 22,  # BUY when RSI is below 22
    "rsi_exit": 64,  # SELL when RSI is above 64
    "stop_loss_pct": 0.0395,  # 3.95% stop loss (for long; for short, reversed)
    "risk_reward_ratio": 1.78,
    "leverage": 10,
    "take_profit_pct": 0.0705  # 7.05% take profit
}


# -----------------------------------------------------------------------------
# Utility Functions: Timestamp and Signature
# -----------------------------------------------------------------------------
def get_timestamp():
    """Return current timestamp in milliseconds."""
    return int(time.time() * 1000)


def sign_query(query_string: str, secret: str) -> str:
    """Create an HMAC SHA256 signature for Binance."""
    return hmac.new(secret.encode('utf-8'),
                    query_string.encode('utf-8'),
                    hashlib.sha256).hexdigest()


# -----------------------------------------------------------------------------
# Binance API Request Function
# -----------------------------------------------------------------------------
def binance_request(method: str, endpoint: str, params: dict = None) -> dict:
    """
    Sends a request to Binance Futures testnet.

    :param method: "GET" or "POST"
    :param endpoint: API endpoint (e.g., "/fapi/v1/order")
    :param params: Dictionary of parameters
    :return: Parsed JSON response.
    """
    if params is None:
        params = {}
    params["timestamp"] = get_timestamp()
    params["recvWindow"] = 5000

    query_string = urllib.parse.urlencode(params)
    signature = sign_query(query_string, API_SECRET)
    url = f"{BASE_URL}{endpoint}?{query_string}&signature={signature}"
    headers = {
        "X-MBX-APIKEY": API_KEY
    }
    if method.upper() == "GET":
        response = requests.get(url, headers=headers)
    elif method.upper() == "POST":
        response = requests.post(url, headers=headers)
    else:
        raise ValueError("Unsupported method")
    return response.json()


# -----------------------------------------------------------------------------
# Functions for fetching historical candlestick data
# -----------------------------------------------------------------------------
def get_candles(symbol: str, interval: str, limit: int = 500) -> list:
    """
    Retrieve historical candlestick data.

    :param symbol: Trading pair symbol (e.g., "BTCUSDT")
    :param interval: Candle interval (e.g., "15m" for 15-minute candles)
    :param limit: Number of candles to retrieve
    :return: List of candlestick data.
    """
    endpoint = "/fapi/v1/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    return binance_request("GET", endpoint, params)


# -----------------------------------------------------------------------------
# Technical Indicator Calculations using Pandas
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
# Strategy Decision Function
# -----------------------------------------------------------------------------
def strategy_decision(df: pd.DataFrame, params: dict) -> (str, float):
    """
    Decide whether to enter a long, short, or no trade based on SMA and RSI.

    :param df: DataFrame containing candlestick data.
               Assumes the 5th column (index 4) is the close price and the 7th (index 6) is the candle close time.
    :param params: Dictionary of strategy parameters.
    :return: Tuple (decision, current price) where decision is "BUY", "SELL", or "HOLD".
    """
    # Convert close price column to float
    df["close"] = df[4].astype(float)
    sma = compute_sma(df["close"], params["sma_period"])
    rsi = compute_rsi(df["close"], params["rsi_period"])
    df["sma"] = sma
    df["rsi"] = rsi

    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    last_candle_close = datetime.fromtimestamp(df.iloc[-1][6] / 1000, tz=timezone.utc)

    if now < last_candle_close:
        # Last candle is still open; use previous closed candle.
        last = df.iloc[-2]
    else:
        last = df.iloc[-1]

    current_price = last["close"]

    if current_price > last["sma"] and last["rsi"] < params["rsi_entry"]:
        return "BUY", current_price
    elif current_price < last["sma"] and last["rsi"] > params["rsi_exit"]:
        return "SELL", current_price
    else:
        return "HOLD", current_price


# -----------------------------------------------------------------------------
# Order Placement Functions
# -----------------------------------------------------------------------------
def place_entry_order(symbol: str, side: str, quantity: str, price: str,
                      order_type: str = "LIMIT", timeInForce: str = "GTC") -> dict:
    """Place an entry order."""
    endpoint = "/fapi/v1/order"
    params = {
        "symbol": symbol,
        "side": side,
        "type": order_type,
        "quantity": quantity,
        "price": price,
        "timeInForce": timeInForce
    }
    response = binance_request("POST", endpoint, params)
    logger.info(f"Entry order response: {json.dumps(response, indent=2)}")
    return response


def place_stop_market_order(symbol: str, side: str, quantity: str, stopPrice: str,
                            reduceOnly: bool = True, order_type: str = "STOP_MARKET") -> dict:
    """Place a stop loss market order."""
    endpoint = "/fapi/v1/order"
    params = {
        "symbol": symbol,
        "side": side,
        "type": order_type,
        "quantity": quantity,
        "stopPrice": stopPrice,
        "reduceOnly": str(reduceOnly).lower()
    }
    response = binance_request("POST", endpoint, params)
    logger.info(f"Stop loss order response: {json.dumps(response, indent=2)}")
    return response


def place_take_profit_market_order(symbol: str, side: str, quantity: str, stopPrice: str,
                                   reduceOnly: bool = True, order_type: str = "TAKE_PROFIT_MARKET") -> dict:
    """Place a take profit market order."""
    endpoint = "/fapi/v1/order"
    params = {
        "symbol": symbol,
        "side": side,
        "type": order_type,
        "quantity": quantity,
        "stopPrice": stopPrice,
        "reduceOnly": str(reduceOnly).lower()
    }
    response = binance_request("POST", endpoint, params)
    logger.info(f"Take profit order response: {json.dumps(response, indent=2)}")
    return response


# -----------------------------------------------------------------------------
# Main Trading Loop
# -----------------------------------------------------------------------------
def run_trading_bot():
    symbol = "BTCUSDT"  # Binance Futures symbol
    candle_interval = "15m"  # 15-minute candles
    extra_delay_sec = 5  # Additional seconds to wait after candle close
    order_quantity = "0.001"  # Example fixed order quantity; adjust as needed

    logger.info("Starting continuous trading bot.")

    last_used_candle_close = None

    while True:
        try:
            candles = get_candles(symbol, candle_interval, limit=500)
            if not candles:
                time.sleep(5)
                continue

            df = pd.DataFrame(candles)
            latest_candle_close = datetime.fromtimestamp(df.iloc[-1][6] / 1000, tz=timezone.utc)
            now = datetime.utcnow().replace(tzinfo=timezone.utc)

            # Wait for the candle to be truly closed (with extra delay)
            if now < latest_candle_close + pd.Timedelta(seconds=extra_delay_sec):
                time.sleep(5)
                continue

            # Only process a candle if it is new
            if last_used_candle_close is not None and latest_candle_close <= last_used_candle_close:
                time.sleep(5)
                continue

            last_used_candle_close = latest_candle_close
            logger.info(f"Candle closed at: {last_used_candle_close}. Running trading cycle...")

            decision, current_price = strategy_decision(df, BEST_PARAMS)
            logger.info(f"Decision: {decision} at price: {current_price:.2f}")

            if decision == "HOLD":
                logger.info("No trade signal for this candle.")
            else:
                if decision == "BUY":
                    entry_side = "BUY"
                    sl_side = "SELL"
                    tp_side = "SELL"
                else:  # SELL
                    entry_side = "SELL"
                    sl_side = "BUY"
                    tp_side = "BUY"

                entry_price = str(round(current_price, 2))
                logger.info("Placing entry order...")
                place_entry_order(symbol, entry_side, order_quantity, entry_price)

                if decision == "BUY":
                    stop_loss_price = current_price * (1 - BEST_PARAMS["stop_loss_pct"])
                    take_profit_price = current_price * (1 + BEST_PARAMS["take_profit_pct"])
                else:
                    stop_loss_price = current_price * (1 + BEST_PARAMS["stop_loss_pct"])
                    take_profit_price = current_price * (1 - BEST_PARAMS["take_profit_pct"])

                sl_price = str(round(stop_loss_price, 2))
                tp_price = str(round(take_profit_price, 2))

                logger.info(f"Placing Stop Loss order at: {sl_price}")
                place_stop_market_order(symbol, sl_side, order_quantity, sl_price)

                logger.info(f"Placing Take Profit order at: {tp_price}")
                place_take_profit_market_order(symbol, tp_side, order_quantity, tp_price)

            # Sleep briefly before checking for a new candle
            time.sleep(5)
        except Exception as e:
            logger.error("Error in trading loop: " + str(e))
            time.sleep(10)


if __name__ == "__main__":
    run_trading_bot()
