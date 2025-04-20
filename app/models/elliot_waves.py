import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import mplfinance as mpf
import time
from dotenv import load_dotenv
import os
import joblib
from app.models.bucket import download_model
from deep_translator import GoogleTranslator
import requests
import aiohttp
import redis.asyncio as redis
import logging
from datetime import datetime, timedelta, timezone
from sklearn.preprocessing import MinMaxScaler
import telebot

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from the .env file
load_dotenv(dotenv_path=".env.micro.trading")

BUCKET_NAME = os.getenv("BUCKET_NAME")  # Your bucket name
MICRO_CENTRAL_URL = os.getenv("MICRO_CENTRAL_URL")  # Your micro central URL
# Initialize the Telegram bot
API_TOKEN = os.getenv("API_TOKEN")
bot = telebot.TeleBot(API_TOKEN)

# Initialize Redis connection
try:
    redis_client = redis.from_url(os.getenv("REDIS_URL"))
    redis_client.ping()
except redis.ConnectionError as e:
    print(f"Redis connection error: {e}")
    redis_client = None

# Translate function to convert text to the target language
def translate(text, token):
    """Translate text to the target language using GoogleTranslator."""
    response = requests.get(f"{MICRO_CENTRAL_URL}/tlogin/{token}")
    if response.status_code == 200:
        user_data = response.json()
        target_lang = user_data.get('language', 'en')
        if target_lang == 'en':
            return text
        return GoogleTranslator(source='auto', target=target_lang).translate(text)
    return text  # Return original text if translation fails

# Send bot message function
async def send_bot_message(token, message, file_path=None):
    url = f"{MICRO_CENTRAL_URL}/send_notification"
    payload = {
        "token": token,
        "message": message,
        "file_path": file_path
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


# Function to fetch cryptocurrency data with retry mechanism
async def fetch_crypto_data(asset, interval, token, max_retries=5, retry_delay=5):
    # Set the limit based on the interval for scalping strategy
    if interval == '1h':
        limit = 500  # 500 candles for 1-hour interval
        days_back = limit // 24  # 24 hours in a day
    elif interval == '4h':
        limit = 200  # 200 candles for 4-hour interval
        days_back = limit // 4  # 4 four-hour intervals in a day
    elif interval == '1d':
        limit = 100  # 100 candles for 1-day interval
        days_back = limit  # 1 day per candle
    else:
        limit = 500  # Default limit if interval is not one of the specified
        days_back = limit // 24  # Assume 1-hour candles as default

    # Calculate the date range for the `values` parameter
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days_back)
    values = f"{start_date.strftime('%Y-%m-%d')}|{end_date.strftime('%Y-%m-%d')}"

    cache_key = f"elliot_waves_data:{asset}:{interval}:{values}"
    
    # Check if the data exists in Redis
    cached_data = await redis_client.get(cache_key)
    if cached_data:
        # print("cached_data for historical_data")
        data = pd.read_json(cached_data.decode('utf-8'))  # Decode bytes to string
        return data
    
    url = f"{MICRO_CENTRAL_URL}/query-historical-data"
    payload = {
        "pair": asset,
        "timeframe": interval,
        "values": values
    }
    headers = {
        "Token": token
    }

    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        df = pd.DataFrame(data)
                        
                        # Convert columns to numeric types
                        df['open'] = pd.to_numeric(df['open'])
                        df['close'] = pd.to_numeric(df['close'])
                        df['high'] = pd.to_numeric(df['high'])
                        df['low'] = pd.to_numeric(df['low'])
                        df['volume'] = pd.to_numeric(df['volume'])
                        
                        # Store the data in Redis for 1 hour (3600 seconds)
                        await redis_client.setex(cache_key, 3600, df.to_json())
                        
                        return df
                    else:
                        logger.error(f"Error fetching historical data: {response.status} {await response.text()}")
                        response.raise_for_status()
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise e

# Function to create dataset suitable for XGBRegressor
def create_rf_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back)])
        Y.append(dataset[i + look_back])
    return np.array(X), np.array(Y)

# Function to analyze market trend based on Elliott Waves
def analyze_market_trend(predicted_labels, data):
    # Ensure the timestamp is in datetime format and set as index
    if 'start_timestamp' in data.columns:
        data['start_timestamp'] = pd.to_datetime(data['start_timestamp'], errors='coerce')
        data.set_index('start_timestamp', inplace=True)

    wave_indices = np.where(predicted_labels.flatten() == 1)[0]
    if len(wave_indices) < 5:
        return "Not enough data to determine market trend.", "Not enough identified wave points to analyze the trend.", []

    wave_points = wave_indices[:5]
    waves_info = []

    for i, idx in enumerate(wave_points):
        try:
            timestamp = data.index[idx]
            wave_time = pd.to_datetime(timestamp, errors='coerce')
        except Exception as e:
            logger.warning(f"Failed to parse timestamp at index {idx}: {e}")
            wave_time = pd.NaT

        wave_price = data['close'].iloc[idx]
        waves_info.append((i + 1, wave_time, wave_price))

    # Validate with Elliott Wave rules
    wave_1, wave_2, wave_3, wave_4, wave_5 = wave_points[:5]

    explanations = []
    critical_violations = 0

    # Rule 1: Wave 2 should not retrace more than 100% of Wave 1
    if data['close'].iloc[wave_2] <= data['close'].iloc[wave_1]:
        explanations.append("Wave 2 retraced more than 100% of Wave 1, weakening the pattern.")
        critical_violations += 1

    # Rule 2: Wave 3 must be longer than Wave 1 and not the shortest
    if not (data['close'].iloc[wave_3] > data['close'].iloc[wave_1] and 
            data['close'].iloc[wave_3] > data['close'].iloc[wave_5]):
        explanations.append("Wave 3 does not extend beyond Wave 1, weakening the pattern.")
        critical_violations += 1

    # Rule 3: Wave 4 should not overlap Wave 1 (non-critical)
    if data['close'].iloc[wave_4] <= data['close'].iloc[wave_1]:
        explanations.append("Wave 4 overlaps with Wave 1, which is less ideal but not disqualifying.")

    # Rule 4: Fibonacci retracement for Wave 2 (warning)
    wave_2_retrace = (data['close'].iloc[wave_2] - data['close'].iloc[wave_1]) / (
        data['close'].iloc[wave_3] - data['close'].iloc[wave_1]
    )
    if wave_2_retrace > 0.618:
        explanations.append("Wave 2 retraces more than the 61.8% Fibonacci level, which weakens the structure.")

    # Compose detailed wave summary
    detailed_explanation = "\n".join([
        f"Wave {num}: Time = {wave_time.strftime('%Y-%m-%d %H:%M')}, Price = {price:.2f}"
        for num, wave_time, price in waves_info if pd.notna(wave_time)
    ])

    # Determine trend
    if critical_violations >= 2:
        trend = "Indeterminate"
        explanation = "The Elliott Wave pattern does not fully comply with critical rules, making the trend indeterminate.\n\n"
    else:
        if wave_points[4] > wave_points[2] and wave_points[3] > wave_points[1]:
            trend = "Bullish"
            explanation = (
                "The market is showing a bullish trend.\n\nThe identified wave patterns indicate upward movements, "
                "suggesting strong buying pressure.\n\n"
            )
        else:
            trend = "Bearish"
            explanation = (
                "The market is showing a bearish trend.\n\nThe identified wave patterns indicate downward movements, "
                "suggesting selling pressure.\n\n"
            )

    explanation += detailed_explanation
    return trend, explanation, waves_info


# Function to plot the data and predicted waves with Japanese candlesticks
def plot_predicted_waves(data, predicted_labels, symbol, interval, look_back, trend, waves_info, token, price_predictions=None, num_candles=50):
    # Ensure the index is a DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        if 'start_timestamp' in data.columns:
            # Automatically parse datetime strings or timestamps
            data['start_timestamp'] = pd.to_datetime(data['start_timestamp'], errors='coerce')
            data.set_index('start_timestamp', inplace=True)
        else:
            raise ValueError("Data does not have a 'start_timestamp' column to convert to DatetimeIndex.")


    # Slice the data to include only the last `num_candles` rows
    plot_data = data.iloc[-(num_candles + look_back):]  # Include the look-back period

    # The predictions and labels corresponding to the reduced plot_data
    plot_prediction_indices = plot_data.index[look_back:]
    plot_predicted_labels = predicted_labels[-len(plot_prediction_indices):]

    # Ensure that we have enough labels for the reduced dataset
    if len(plot_predicted_labels) != len(plot_data) - look_back:
        raise ValueError("Adjusted predicted labels do not match the length of the data.")

    plot_predicted_close_prices = plot_data['close'].iloc[look_back:]

    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot the candlesticks
    # Capitalize column names to match mplfinance expectations
    plot_data = plot_data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })

    # Plot the candlesticks
    mpf.plot(plot_data, type='candle', ax=ax, style='charles', show_nontrading=True)


    # Plot the predicted wave points
    wave_times = plot_prediction_indices[plot_predicted_labels.flatten() == 1]
    wave_prices = plot_predicted_close_prices[plot_predicted_labels.flatten() == 1]
    ax.scatter(wave_times, wave_prices, color='red', label='Predicted Wave')

    # Annotate the wave points with numbers
    for wave_num, wave_time, wave_price in waves_info:
        if wave_time in plot_data.index:
            ax.annotate(f'{wave_num}', (wave_time, wave_price), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, color='green')

    # Plot the price predictions if provided
    if price_predictions is not None:
        future_index = pd.date_range(start=plot_data.index[-1], periods=len(price_predictions) + 1, freq=plot_data.index.freq)[1:]
        ax.plot(future_index, price_predictions, color='blue', linestyle='--', marker='o', label='Price Prediction')

    # Improve the legibility of the chart
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.title(f'Predicted Elliott Waves and Price Prediction for {symbol} on {interval} interval')
    plt.xlabel('Timestamp')
    plt.ylabel('Price')
    plt.legend()

    # Define the media folder name
    media_folder = 'media'

    # Create the directory if it doesn't exist
    os.makedirs(media_folder, exist_ok=True)

    plt.savefig(os.path.join(media_folder, f'{symbol}_{interval}_predicted_waves.png'), bbox_inches='tight')
    plt.close()

# Function to analyze multiple intervals and return explanations using XGBRegressor
async def analyze_intervals(asset, token, intervals=['1h', '4h', '1d']):
    explanations = {}
    look_back = 60
    future_steps = 10

    for interval in intervals:
        MODEL_KEY = f'Mockba/elliot_waves_trained_models/{asset}_{interval}_elliot_waves_model.joblib'
        local_model_path = f'temp/elliot_waves_trained_models/{asset}_{interval}_elliot_waves_model.joblib'

        if not download_model(BUCKET_NAME, MODEL_KEY, local_model_path):
            logger.info(f"No model found for {asset} on {interval}.")
            translated_message = translate(f"No model found for {asset} on {interval}. Contact support.", token)
            await send_bot_message(token, translated_message)
            continue  # Skip to next interval

        try:
            data = await fetch_crypto_data(asset, interval, token)

            # Normalize
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data['close'].values.reshape(-1, 1))

            X, Y = create_rf_dataset(scaled_data, look_back)
            X = X.reshape(X.shape[0], -1)

            model = joblib.load(local_model_path)
            predictions = model.predict(X)

            # Predict future prices
            future_inputs = X[-1].reshape(1, -1)
            future_predictions = []
            for _ in range(future_steps):
                future_price = model.predict(future_inputs)
                future_predictions.append(future_price[0])
                future_inputs = np.roll(future_inputs, -1)
                future_inputs[0, -1] = future_price

            future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()
            predicted_labels = (predictions > np.mean(predictions)).astype(int)

            trend, explanation, waves_info = analyze_market_trend(predicted_labels, data)

            final_prediction = future_predictions[-1]
            current_price = data['close'].iloc[-1]
            price_diff_percentage = ((final_prediction - current_price) / current_price) * 100

            explanation += f"\n\nThe predicted price for the next period is {final_prediction:.6f}."
            explanation += f"\n\nThis is a {price_diff_percentage:.2f}% change from the current price of {current_price:.6f}."

            plot_predicted_waves(data, predicted_labels, asset, interval, look_back, trend, waves_info, token, price_predictions=future_predictions, num_candles=50)
            
            title = f"Analysis of {asset} on {interval} interval"
            translated_title = translate(title, token)
            await send_bot_message(token, translated_title)

            translated_explanation = translate(explanation, token)
            await send_bot_message(token, translated_explanation)
            
            chart_path = f'media/{asset}_{interval}_predicted_waves.png'
            if os.path.exists(chart_path):
                file = open(chart_path,'rb')
                bot.send_document(token,file)
                os.remove(chart_path)   
          

        except Exception as e:
            logger.error(f"Error processing {interval}: {e}")
            await send_bot_message(token, translate(f"An error occurred while analyzing {interval} interval: {e}", token))

        finally:
            if os.path.exists(local_model_path):
                os.remove(local_model_path)


    logger.info("All interval analyses completed.")
    return explanations


# Example usage
# if __name__ == "__main__":
#     symbol = 'BTCUSDT'
#     intervals = ['1h', '4h', '1d']
#     token = "556159355"
#     explanations = analyze_intervals(symbol, intervals, token)
#     for interval, details in explanations.items():
#         print(f"{interval} interval explanation:\n{details['explanation']}")
#         print(f"TradingView Link: {details['tradingview_link']}")
