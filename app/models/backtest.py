import asyncio
import aiohttp
import httpx
import sys
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import joblib  # Library for model serialization
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
import redis.asyncio as redis
import json
import logging
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

# Add the directory containing your modules to the Python path
sys.path.append('/app')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore")
# Load environment variables from the .env file
load_dotenv(dotenv_path=".env.micro.trading")

# Access the environment variables
FILES_PATH = os.getenv("FILES_PATH")
MICRO_CENTRAL_URL = os.getenv("MICRO_CENTRAL_URL")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost")
# Configuration for DigitalOcean Spaces
OBJECT_STORAGE_URL = os.getenv("OBJECT_STORAGE_URL")  # Your DigitalOcean endpoint URL
ACCESS_KEY = os.getenv("ACCESS_KEY")  # Replace with your DigitalOcean Spaces access key
SECRET_KEY = os.getenv("SECRET_KEY")  # Replace with your DigitalOcean Spaces secret key
BUCKET_NAME = 'glogaldv'  # Your bucket name

# Initialize the S3 client with explicit region configuration
s3_client = boto3.client(
    's3', 
    endpoint_url=OBJECT_STORAGE_URL,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    region_name=REGION_NAME  # Ensure region is set
)

def download_model(bucket_name, key, local_path):
    """Download a file from DigitalOcean Spaces."""
    try:
        s3_client.download_file(bucket_name, key, local_path)
        print(f"Model downloaded to {local_path}")
        return True
    except Exception as e:
        print(f"Model not found or error downloading: {e}")
        return False

# Initialize Redis connection
redis_client = redis.from_url(REDIS_URL, decode_responses=True)


def get_last_non_zero_crypto(data):
    non_zero_crypto = data[data['crypto'] > 0]['crypto']
    if not non_zero_crypto.empty:
        return non_zero_crypto.iloc[-1]
    else:
        return 0  # or handle it as needed, e.g., None    
# Def get all Binance

async def send_bot_message(token, message):
    url = f"{MICRO_CENTRAL_URL}/send_notification"
    payload = {
        "token": token,
        "message": message
    }
    headers = {
        "Token": token
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers) as response:
            if response.status == 200:
                return await response.json()
            else:
                response.raise_for_status()

async def get_all_binance(pair, timeframe, token, save=False):
    cache_key = f"binance:{pair}:{timeframe}:{token}:{save}"
    
    # Check if the data exists in Redis
    cached_data = await redis_client.get(cache_key)
    if cached_data:
        print("cached_data for binance")
        return json.loads(cached_data)
    
    url = f"{MICRO_CENTRAL_URL}/historical-data"
    payload = {
        "symbol": pair,
        "kline_size": timeframe,
        "token": token,
        "save": save
    }
    headers = {
        "Token": token
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                
                # Store the data in Redis for 4 hours (14400 seconds)
                await redis_client.setex(cache_key, 14400, json.dumps(data))
                
                return data
            else:
                response.raise_for_status()

# Fetch historical data from the database
async def get_historical_data(token, pair, timeframe, values):
    cache_key = f"historical_data:{pair}:{timeframe}:{values}:{token}"
    
    # Check if the data exists in Redis
    cached_data = await redis_client.get(cache_key)
    if cached_data:
        print("cached_data for historical_data")
        data = pd.read_json(cached_data)
        return data
    
    url = f"{MICRO_CENTRAL_URL}/query-historical-data"
    payload = {
        "pair": pair,
        "timeframe": timeframe,
        "values": values
    }
    headers = {
        "Token": token
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                df = pd.DataFrame(data)
                
                # Convert columns to numeric types
                df['close'] = pd.to_numeric(df['close'])
                df['high'] = pd.to_numeric(df['high'])
                df['low'] = pd.to_numeric(df['low'])
                df['volume'] = pd.to_numeric(df['volume'])
                
                # Store the data in Redis for 4 hours (14400 seconds)
                await redis_client.setex(cache_key, 14400, df.to_json())
                
                return df
            else:
                logger.error(f"Error fetching historical data: {response.status} {await response.text()}")
                response.raise_for_status()

# Add technical indicators to the data
def add_indicators(data):
    # Ensure the columns are of numeric type
    data['close'] = pd.to_numeric(data['close'])
    data['high'] = pd.to_numeric(data['high'])
    data['low'] = pd.to_numeric(data['low'])
    data['volume'] = pd.to_numeric(data['volume'])

    # Relative Strength Index (RSI)
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))

    # Moving Average Convergence Divergence (MACD)
    data['ema_12'] = data['close'].ewm(span=12, adjust=False).mean()
    data['ema_26'] = data['close'].ewm(span=26, adjust=False).mean()
    data['macd'] = data['ema_12'] - data['ema_26']
    data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
    data['macd_diff'] = data['macd'] - data['macd_signal']

    # Bollinger Bands
    data['bollinger_mavg'] = data['close'].rolling(window=20).mean()
    data['bollinger_std'] = data['close'].rolling(window=20).std()
    data['bollinger_hband'] = data['bollinger_mavg'] + (data['bollinger_std'] * 2)
    data['bollinger_lband'] = data['bollinger_mavg'] - (data['bollinger_std'] * 2)

    # Exponential Moving Average (EMA)
    data['ema'] = data['close'].ewm(span=21, adjust=False).mean()

    # Average True Range (ATR) - Volatility indicator
    data['tr'] = pd.concat([
        data['high'] - data['low'],
        (data['high'] - data['close'].shift()).abs(),
        (data['low'] - data['close'].shift()).abs()
    ], axis=1).max(axis=1)
    data['ATR'] = data['tr'].rolling(window=14).mean()

    return data

# Backtest the strategy with initial investment, stop-loss, and fees
async def backtest(data, model, features, initial_investment=10000, stop_loss_threshold=0.05, maker_fee=0.001, taker_fee=0.001, gain_threshold=0.001):
    data = data.dropna().copy()
    data['predicted'] = model.predict(data[features])
    data['strategy_return'] = 0.0  # Initialize strategy return

    # Initialize portfolio values
    data['strategy_portfolio_value'] = initial_investment
    data['cash'] = initial_investment
    data['crypto'] = 0
    position_open = False
    last_buy_price = 0  # New variable to store the last buy price
    first_trade = True  # Variable to track the first trade

    # Iterate over the data to calculate the portfolio value
    for i in range(1, len(data)):
        if data['predicted'].iloc[i - 1] == 1:  # Buy signal
            expected_return = (data['close'].iloc[i] - last_buy_price) / last_buy_price if last_buy_price != 0 else 0  # Use last buy price for expected return
            if not position_open or (position_open and first_trade):  # Execute first trade unconditionally
                # Buy crypto with all available cash (apply maker fee)
                crypto_bought = (data['cash'].iloc[i - 1] / data['close'].iloc[i]) * (1 - maker_fee)
                data['crypto'].iloc[i] = crypto_bought
                data['cash'].iloc[i] = 0
                position_open = True
                last_buy_price = data['close'].iloc[i]  # Set buy price
                first_trade = False  # Set first_trade to False after the first trade
            else:
                # Maintain the position
                data['crypto'].iloc[i] = data['crypto'].iloc[i - 1]
                data['cash'].iloc[i] = data['cash'].iloc[i - 1]

        elif data['predicted'].iloc[i - 1] == 0:  # Sell signal
            if position_open:
                # Calculate expected return and loss based on last buy price
                expected_return = (data['close'].iloc[i] - last_buy_price) / last_buy_price  # Return if sold now
                expected_loss = (last_buy_price - data['close'].iloc[i]) / last_buy_price  # Loss if sold now

                # Sell condition: If the expected return is above the gain threshold, or expected loss is too much
                if expected_return > gain_threshold or expected_loss > stop_loss_threshold:
                    # Sell all crypto and hold cash (apply taker fee)
                    cash_from_sale = data['crypto'].iloc[i - 1] * data['close'].iloc[i] * (1 - taker_fee)
                    data['cash'].iloc[i] = cash_from_sale
                    data['crypto'].iloc[i] = 0
                    position_open = False
                else:
                    # Maintain the position
                    data['crypto'].iloc[i] = data['crypto'].iloc[i - 1]
                    data['cash'].iloc[i] = data['cash'].iloc[i - 1]
            else:
                # Maintain the position
                data['crypto'].iloc[i] = data['crypto'].iloc[i - 1]
                data['cash'].iloc[i] = data['cash'].iloc[i - 1]
        else:
            # No trade signal, maintain the position
            data['crypto'].iloc[i] = data['crypto'].iloc[i - 1]
            data['cash'].iloc[i] = data['cash'].iloc[i - 1]

        # Calculate portfolio value
        data['strategy_portfolio_value'].iloc[i] = data['cash'].iloc[i] + data['crypto'].iloc[i] * data['close'].iloc[i]

        # Apply stop-loss if the portfolio value drops too much
        if position_open and data['strategy_portfolio_value'].iloc[i] / initial_investment < (1 - stop_loss_threshold):
            data['cash'].iloc[i] = data['crypto'].iloc[i] * data['close'].iloc[i] * (1 - taker_fee)
            data['crypto'].iloc[i] = 0
            position_open = False

        # Calculate strategy return as the relative change in portfolio value
        data['strategy_return'].iloc[i] = (data['strategy_portfolio_value'].iloc[i] / data['strategy_portfolio_value'].iloc[i - 1]) - 1

    # Calculate market portfolio value without strategy (buy and hold)
    data['market_portfolio_value'] = initial_investment * (data['close'] / data['close'].iloc[0])

    return data


def plot_backtest_results(data, pair, output_file):
    # Ensure that the 'timestamp' column is in datetime format and set as index
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)

    plt.figure(figsize=(14, 7))

    # Plot the strategy portfolio value and market portfolio value
    plt.plot(data.index, data['strategy_portfolio_value'], label='Strategy Portfolio Value (with fees)')
    plt.plot(data.index, data['market_portfolio_value'], label='Market Portfolio Value')

    # Format the x-axis to show dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  # Adjust the locator as needed

    plt.title(f'Backtest Results for {pair}')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability

    # Save the plot as a PNG file
    plot_file = output_file.replace('.xlsx', '.png')
    plt.savefig(plot_file)
    plt.close()

# Updated run_backtest function with logging
async def run_backtest(pair, timeframe, token, values, stop_loss_threshold=0.05, initial_investment=10000, maker_fee=0.001, taker_fee=0.001, gain_threshold=0.001):
    start = datetime.now()
    logger.info("Starting backtest")

    # Generate dynamic file names
    MODEL_KEY = f'Mockba/trained_models/trained_model_{pair}_{timeframe}.pkl'
    local_model_path = f'trained_model_{pair}_{timeframe}.pkl'
    output_file = f'files/backtest_results_{pair}_{timeframe}_{token}.xlsx'

    if download_model(BUCKET_NAME, MODEL_KEY, local_model_path):
        # Fetch historical data and add technical indicators
        logger.info("Fetching historical data from Binance")
        await get_all_binance(pair, timeframe, token, save=True)
        data = await get_historical_data(token, pair, timeframe, train_values)
        logger.info("Adding technical indicators")
        data = add_indicators(data)

        # Ensure the return column is calculated and present in the DataFrame
        data['return'] = data['close'].pct_change().shift(-1)
        logger.info("Calculated return column")

        # Calculate initial amount of crypto based on initial investment
        initial_crypto_amount = initial_investment / data['close'].iloc[0]
        logger.info(f"Initial crypto amount: {initial_crypto_amount:.6f}")

        model = joblib.load(local_model_path)
        features = ['rsi', 'macd', 'macd_signal', 'macd_diff', 'bollinger_hband', 'bollinger_mavg', 'bollinger_lband', 'ema', 'ATR']

        # Backtest the strategy with stop-loss, fees, and gain threshold
        logger.info("Starting backtest strategy")
        backtest_result = await backtest(data, model, features, initial_investment, stop_loss_threshold, maker_fee, taker_fee, gain_threshold)

        # Get the last non-zero crypto value
        last_crypto_value = get_last_non_zero_crypto(backtest_result)
        logger.info(f"Last non-zero crypto value: {last_crypto_value:.6f}")

        # Display the final portfolio values
        final_strategy_value = backtest_result['strategy_portfolio_value'].iloc[-1]
        final_market_value = backtest_result['market_portfolio_value'].iloc[-1]
        logger.info(f"Final strategy portfolio value: ${final_strategy_value:.2f}")
        logger.info(f"Final market portfolio value: ${final_market_value:.2f}")

        # Calculate the final gain or loss in percentage
        final_percentage_gain_loss = ((final_strategy_value - initial_investment) / initial_investment) * 100
        logger.info(f"Final percentage gain/loss: {final_percentage_gain_loss:.2f}%")

        # Calculate the number of months in the backtest period
        start_date = datetime.strptime(values.split('|')[0], '%Y-%m-%d')
        end_date = datetime.strptime(values.split('|')[1], '%Y-%m-%d')
        num_months = (end_date.year - start_date.year) * 12 + end_date.month - start_date.month + 1
        logger.info(f"Number of months in backtest period: {num_months}")

        # Calculate the average monthly gain or loss percentage
        avg_monthly_percentage_gain_loss = final_percentage_gain_loss / num_months
        logger.info(f"Average monthly percentage gain/loss: {avg_monthly_percentage_gain_loss:.2f}%")

        # Save results to an Excel file
        logger.info("Saving results to Excel file")
        backtest_result.to_excel(output_file, index=False)

        result_explanation = (
            f"Initial crypto amount (with initial investment): {initial_crypto_amount:.6f} {pair.split('USDT')[0]}\n"
            f"Final strategy portfolio value: ${final_strategy_value:.2f}\n"
            f"Final market portfolio value: ${final_market_value:.2f}\n"
            f"Final amount of {pair}: {last_crypto_value:.6f}\n"
            f"Final percentage gain/loss: {final_percentage_gain_loss:.2f}%\n"
            f"Average monthly percentage gain/loss: {avg_monthly_percentage_gain_loss:.2f}%\n"
            f"You have selected a gain threshold of {gain_threshold * 100:.2f}% and a stop-loss threshold of {stop_loss_threshold * 100:.2f}%.\n"
            f"Execution time: {datetime.now() - start}"
        )
        logger.info("Sending result explanation to bot")
        await send_bot_message(token, result_explanation)
        # Plot the results and save the plot as PNG
        logger.info("Plotting backtest results")
        plot_backtest_results(backtest_result, pair, output_file)

        logger.info("Backtest completed")
        return result_explanation
    else:
        logger.info("No model found it must bbe trained")
        await send_bot_message(token, "No model found. Please train a model first.")
        return   


# Example of how to call run_backtest
# if __name__ == "__main__":
#     pair = 'SOLUSDT'
#     timeframe = '1h'
#     token = '556159355'
#     values = '2025-01-01|2025-01-19'
#     stop_loss_threshold = 0.5
#     initial_investment = 100
#     maker_fee = 0.001
#     taker_fee = 0.001
#     gain_threshold = 0.01

#     print(run_backtest(pair, timeframe, token, values, stop_loss_threshold, initial_investment, maker_fee, taker_fee, gain_threshold))
