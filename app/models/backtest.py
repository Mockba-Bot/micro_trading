import aiohttp
import time
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
import logging
from collections import Counter
from app.models.bucket import download_model
from deep_translator import GoogleTranslator

# Add the directory containing your modules to the Python path
sys.path.append('/app')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore")
# Load environment variables from the .env file
load_dotenv(dotenv_path=".env.micro.trading")

BUCKET_NAME = os.getenv("BUCKET_NAME")  # Your bucket name
MICRO_CENTRAL_URL = os.getenv("MICRO_CENTRAL_URL")  # Your micro central URL

# Orderly Mockba fees
MAKER_FEE = 0.0003 # 0.03%
TAKER_FEE = 0.0006 # 0.06%

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


# Initialize Redis connection
try:
    redis_client = redis.from_url(os.getenv("REDIS_URL"))
    redis_client.ping()
except redis.ConnectionError as e:
    print(f"Redis connection error: {e}")
    redis_client = None

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

# Fetch historical data from the database
async def get_historical_data(token, pair, timeframe, values, max_retries=3, retry_delay=5):
    cache_key = f"historical_data:{pair}:{timeframe}:{values}:{token}"
    
    # Check if the data exists in Redis
    cached_data = await redis_client.get(cache_key)
    if cached_data:
        print("cached_data for historical_data")
        data = pd.read_json(cached_data.decode('utf-8'))  # Decode bytes to string
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

    for attempt in range(max_retries):
        try:
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


# Add technical indicators to the data
def add_indicators(data, required_features):
    """
    Add only the necessary indicators to the data based on the requested features.
    """
    # Ensure numeric columns
    data[['close', 'high', 'low', 'volume']] = data[['close', 'high', 'low', 'volume']].apply(pd.to_numeric)

    # --- EMA ---
    for feature in required_features:
        if feature.startswith("ema_"):
            try:
                window = int(feature.split("_")[1])  # Extract window size from feature name
                data[feature] = data['close'].ewm(span=window, adjust=False).mean()
            except (IndexError, ValueError):
                print(f"⚠️ Warning: Could not extract window size from feature: {feature}")

    # --- MACD ---
    if any(x in required_features for x in ['macd', 'macd_signal']):
        data['ema_12'] = data['close'].ewm(span=12, adjust=False).mean()
        data['ema_26'] = data['close'].ewm(span=26, adjust=False).mean()
        data['macd'] = data['ema_12'] - data['ema_26']
        data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()

    # --- ATR ---
    for feature in required_features:
        if feature.startswith("atr_"):
            try:
                window = int(feature.split("_")[1])  # Extract window size from feature name
                data['tr'] = pd.concat([
                    data['high'] - data['low'],
                    (data['high'] - data['close'].shift()).abs(),
                    (data['low'] - data['close'].shift()).abs()
                ], axis=1).max(axis=1)
                data[feature] = data['tr'].rolling(window=window).mean()
            except (IndexError, ValueError):
                print(f"⚠️ Warning: Could not extract window size from feature: {feature}")

    # --- Bollinger Bands ---
    for feature in required_features:
        if feature.startswith("bollinger_"):
            try:
                window = int(feature.split("_")[-1])  # Extract window size from feature name
                data['bollinger_mavg'] = data['close'].rolling(window=window).mean()
                data['bollinger_std'] = data['close'].rolling(window=window).std()
                data['bollinger_hband'] = data['bollinger_mavg'] + (data['bollinger_std'] * 2)
                data['bollinger_lband'] = data['bollinger_mavg'] - (data['bollinger_std'] * 2)
            except (IndexError, ValueError):
                print(f"⚠️ Warning: Could not extract window size from feature: {feature}")

    # --- Standard Deviation ---
    for feature in required_features:
        if feature.startswith("std_"):
            try:
                window = int(feature.split("_")[1])  # Extract window size from feature name
                data[feature] = data['close'].rolling(window=window).std()
            except (IndexError, ValueError):
                print(f"⚠️ Warning: Could not extract window size from feature: {feature}")

    # --- RSI ---
    for feature in required_features:
        if feature.startswith("rsi_"):
            try:
                window = int(feature.split("_")[1])  # Extract window size from feature name
                delta = data['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=window).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
                rs = gain / loss
                data[feature] = 100 - (100 / (1 + rs))
            except (IndexError, ValueError):
                print(f"⚠️ Warning: Could not extract window size from feature: {feature}")

    # --- Stochastic Oscillator ---
    for feature in required_features:
        if feature.startswith("stoch_"):
            try:
                window = int(feature.split("_")[-1])  # Extract window size from feature name
                data['stoch_k'] = ((data['close'] - data['low'].rolling(window).min()) /
                                   (data['high'].rolling(window).max() - data['low'].rolling(window).min())) * 100
                data['stoch_d'] = data['stoch_k'].rolling(3).mean()
            except (IndexError, ValueError):
                print(f"⚠️ Warning: Could not extract window size from feature: {feature}")

    # --- Momentum ---
    for feature in required_features:
        if feature.startswith("momentum_"):
            try:
                window = int(feature.split("_")[1])  # Extract window size from feature name
                data[feature] = data['close'].diff(periods=window)
            except (IndexError, ValueError):
                print(f"⚠️ Warning: Could not extract window size from feature: {feature}")

    # --- Rate of Change (ROC) ---
    for feature in required_features:
        if feature.startswith("roc_"):
            try:
                window = int(feature.split("_")[1])  # Extract window size from feature name
                data[feature] = data['close'].pct_change(periods=window) * 100
            except (IndexError, ValueError):
                print(f"⚠️ Warning: Could not extract window size from feature: {feature}")
    
        # --- ADX ---
    for feature in required_features:
        if feature.startswith("adx"):
            try:
                # If the feature has an underscore, assume the portion after it is the window size
                if "_" in feature:
                    window = int(feature.split("_")[1])
                else:
                    window = 14  # Default window for ADX if no underscore

                data['plus_dm'] = data['high'].diff().where(lambda x: x > 0, 0)
                data['minus_dm'] = -data['low'].diff().where(lambda x: x < 0, 0)

                # Calculate True Range (TR)
                data['tr'] = pd.concat([
                    data['high'] - data['low'],
                    (data['high'] - data['close'].shift()).abs(),
                    (data['low'] - data['close'].shift()).abs()
                ], axis=1).max(axis=1)

                # Calculate +DI and -DI
                data['plus_di'] = 100 * (
                    data['plus_dm'].rolling(window=window).mean()
                    / data['tr'].rolling(window=window).mean()
                )
                data['minus_di'] = 100 * (
                    data['minus_dm'].rolling(window=window).mean()
                    / data['tr'].rolling(window=window).mean()
                )

                # Calculate ADX
                data['dx'] = 100 * abs(data['plus_di'] - data['minus_di']) / (data['plus_di'] + data['minus_di'])
                data[feature] = data['dx'].rolling(window=window).mean()

            except (IndexError, ValueError) as e:
                print(f"⚠️ Warning: Could not extract window size from feature: {feature}. Error: {e}")


    # --- Ichimoku Cloud ---
    for feature in required_features:
        if feature.startswith("tenkan_sen_"):
            try:
                window = int(feature.split("_")[-1])  # Extract window size from feature name
                data[feature] = (data['high'].rolling(window=window).max() + data['low'].rolling(window=window).min()) / 2
            except (IndexError, ValueError):
                print(f"⚠️ Warning: Could not extract window size from feature: {feature}")
        if feature.startswith("kijun_sen_"):
            try:
                window = int(feature.split("_")[-1])  # Extract window size from feature name
                data[feature] = (data['high'].rolling(window=window).max() + data['low'].rolling(window=window).min()) / 2
            except (IndexError, ValueError):
                print(f"⚠️ Warning: Could not extract window size from feature: {feature}")
        if feature.startswith("senkou_span_a"):
            data[feature] = ((data['tenkan_sen_9'] + data['kijun_sen_26']) / 2).shift(26)
        if feature.startswith("senkou_span_b"):
            data[feature] = ((data['high'].rolling(window=52).max() + data['low'].rolling(window=52).min()) / 2).shift(26)

    # --- Parabolic SAR ---
    if 'sar' in required_features:
        data['sar'] = np.nan
        af = 0.02  # Acceleration Factor
        max_af = 0.2
        ep = data['high'][0]  # Extreme point
        sar = data['low'][0]  # Start SAR with first low
        trend = 1  # 1 = uptrend, -1 = downtrend
        for i in range(1, len(data)):
            prev_sar = sar
            sar = prev_sar + af * (ep - prev_sar)
            if trend == 1:
                if data['low'][i] < sar:
                    trend = -1
                    sar = ep
                    ep = data['low'][i]
                    af = 0.02
            else:
                if data['high'][i] > sar:
                    trend = 1
                    sar = ep
                    ep = data['high'][i]
                    af = 0.02
            if af < max_af:
                af += 0.02
            data.loc[data.index[i], 'sar'] = sar

    # --- VWAP ---
    if 'vwap' in required_features:
        data['vwap'] = (data['volume'] * (data['high'] + data['low'] + data['close']) / 3).cumsum() / data['volume'].cumsum()

    # Fill NaN values after calculations
    data.fillna(method='bfill', inplace=True)

    return data

def fetch_margin_ratios(symbol):
    """
    Fetch Base MMR, Base IMR, and IMR Factor for a given symbol from Orderly API.
    """
    url = f"https://api-evm.orderly.org/v1/public/info/{symbol}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json().get("data", {})
        return {
            "base_mmr": data.get("base_mmr", 0.05),  # Default to 0.05 if not found
            "base_imr": data.get("base_imr", 0.1),   # Default to 0.1 if not found
            "imr_factor": data.get("imr_factor", 0.00000208),  # Default to 0.00000208 if not found
        }
    else:
        raise Exception(f"Failed to fetch data for {symbol}. Status code: {response.status_code}")


def get_strategy_name(timeframe, features):
    base_features = ["close", "high", "low", "volume"]
    strategy_features = {
        "5m": {
            "Trend-Following": {"features": base_features + ["ema_12", "ema_26", "macd", "macd_signal", "adx", "vwap"], "force_features": True},
            "Volatility Breakout": {"features": base_features + ["atr_14", "bollinger_hband", "bollinger_lband", "std_20", "vwap"], "force_features": True},
            "Momentum Reversal": {"features": base_features + ["rsi_14", "stoch_k_14", "stoch_d_14", "roc_10", "momentum_10", "vwap"], "force_features": True},
            "Momentum + Volatility": {"features": base_features + ["rsi_14", "atr_14", "bollinger_hband", "bollinger_lband", "roc_10", "momentum_10", "vwap"], "force_features": True},
            "Hybrid": {"features": base_features + ["ema_12", "ema_26", "atr_14", "bollinger_hband", "rsi_14", "macd", "vwap"], "force_features": True},
            "Advanced": {"features": base_features + ["tenkan_sen_9", "kijun_sen_26", "senkou_span_a", "senkou_span_b", "sar", "vwap"], "force_features": True},
            "Router": {"features": ["ema_12", "ema_26", "macd", "macd_signal", "adx", "atr_14", "bollinger_hband", "bollinger_lband", "std_20", "rsi_14", "stoch_k_14", "stoch_d_14", "roc_10", "momentum_10", "tenkan_sen_9", "kijun_sen_26", "senkou_span_a", "senkou_span_b", "sar", "vwap"], "force_features": False}
        },
        "1h": {
            "Trend-Following": {"features": base_features + ["ema_20", "ema_50", "macd", "macd_signal", "adx", "vwap"], "force_features": True},
            "Volatility Breakout": {"features": base_features + ["atr_14", "bollinger_hband", "bollinger_lband", "std_20", "vwap"], "force_features": True},
            "Momentum Reversal": {"features": base_features + ["rsi_14", "stoch_k_14", "stoch_d_14", "roc_10", "momentum_10", "vwap"], "force_features": True},
            "Momentum + Volatility": {"features": base_features + ["rsi_14", "atr_14", "bollinger_hband", "bollinger_lband", "roc_10", "momentum_10", "vwap"], "force_features": True},
            "Hybrid": {"features": base_features + ["ema_20", "ema_50", "atr_14", "bollinger_hband", "rsi_14", "macd", "vwap"], "force_features": True},
            "Advanced": {"features": base_features + ["tenkan_sen_9", "kijun_sen_26", "senkou_span_a", "senkou_span_b", "sar", "vwap"], "force_features": True},
            "Router": {"features": ["ema_12", "ema_26", "macd", "macd_signal", "adx", "atr_14", "bollinger_hband", "bollinger_lband", "std_20", "rsi_14", "stoch_k_14", "stoch_d_14", "roc_10", "momentum_10", "tenkan_sen_9", "kijun_sen_26", "senkou_span_a", "senkou_span_b", "sar", "vwap"], "force_features": False}
        },
        "4h": {
            "Trend-Following": {"features": base_features + ["ema_50", "ema_200", "macd", "macd_signal", "adx", "vwap"], "force_features": True},
            "Volatility Breakout": {"features": base_features + ["atr_14", "bollinger_hband", "bollinger_lband", "std_20", "vwap"], "force_features": True},
            "Momentum Reversal": {"features": base_features + ["rsi_14", "stoch_k_14", "stoch_d_14", "roc_10", "momentum_10", "vwap"], "force_features": True},
            "Momentum + Volatility": {"features": base_features + ["rsi_14", "atr_14", "bollinger_hband", "bollinger_lband", "roc_10", "momentum_10", "vwap"], "force_features": True},
            "Hybrid": {"features": base_features + ["ema_50", "ema_200", "atr_14", "bollinger_hband", "rsi_14", "macd", "vwap"], "force_features": True},
            "Advanced": {"features": base_features + ["tenkan_sen_9", "kijun_sen_26", "senkou_span_a", "senkou_span_b", "sar", "vwap"], "force_features": True},
            "Router": {"features": ["ema_12", "ema_26", "macd", "macd_signal", "adx", "atr_14", "bollinger_hband", "bollinger_lband", "std_20", "rsi_14", "stoch_k_14", "stoch_d_14", "roc_10", "momentum_10", "tenkan_sen_9", "kijun_sen_26", "senkou_span_a", "senkou_span_b", "sar", "vwap"], "force_features": False}
        },
        "1d": {
            "Trend-Following": {"features": base_features + ["ema_50", "ema_200", "macd", "macd_signal", "adx", "vwap"], "force_features": True},
            "Volatility Breakout": {"features": base_features + ["atr_14", "bollinger_hband", "bollinger_lband", "std_20", "vwap"], "force_features": True},
            "Momentum Reversal": {"features": base_features + ["rsi_14", "stoch_k_14", "stoch_d_14", "roc_10", "momentum_10", "vwap"], "force_features": True},
            "Momentum + Volatility": {"features": base_features + ["rsi_14", "atr_14", "bollinger_hband", "bollinger_lband", "roc_10", "momentum_10", "vwap"], "force_features": True},
            "Hybrid": {"features": base_features + ["ema_50", "ema_200", "atr_14", "bollinger_hband", "rsi_14", "macd", "vwap"], "force_features": True},
            "Advanced": {"features": base_features + ["tenkan_sen_9", "kijun_sen_26", "senkou_span_a", "senkou_span_b", "sar", "vwap"], "force_features": True},
            "Router": {"features": ["ema_12", "ema_26", "macd", "macd_signal", "adx", "atr_14", "bollinger_hband", "bollinger_lband", "std_20", "rsi_14", "stoch_k_14", "stoch_d_14", "roc_10", "momentum_10", "tenkan_sen_9", "kijun_sen_26", "senkou_span_a", "senkou_span_b", "sar", "vwap"], "force_features": False}
        }
    }

    # Check if the timeframe exists in the strategy_features
    if timeframe in strategy_features:
        # Iterate through the strategies in the given timeframe
        for strategy_name, strategy_data in strategy_features[timeframe].items():
            # Check if the features match the strategy's feature list
            if sorted(features) == sorted(strategy_data["features"]):
                return strategy_name

    # If no match is found, return None or a default value
    return None


async def backtest(
    data,
    model,
    features,
    initial_investment=100,
    stop_loss_threshold=0.02,
    take_profit_threshold=0.005,
    leverage=5,
    withdraw_percentage=0.7,
    compound_percentage=0.3,
    num_trades=None,
    asset="PERP_APT_USDC",
    timeframe="1h",
):
    """
    Backtests a perpetual futures trading strategy using a trained model.
    """

    # Fetch margin ratios for the asset
    symbol = asset  # Construct the symbol (e.g., PERP_BTC_USDC)
    margin_ratios = fetch_margin_ratios(symbol)
    base_mmr = margin_ratios["base_mmr"]
    base_imr = margin_ratios["base_imr"]
    imr_factor = margin_ratios["imr_factor"]

    # If 'start_timestamp' is not in columns, create it from the index
    if 'start_timestamp' not in data.columns:
        data['start_timestamp'] = data.index  # Or use any other date/time source here

    # Make sure we have a column-based index for manipulations
    data = data.reset_index(drop=True)

    # Convert 'start_timestamp' to datetime if needed
    data['start_timestamp'] = pd.to_datetime(data['start_timestamp'], errors='coerce')

    # Drop rows with invalid or missing timestamps
    data.dropna(subset=['start_timestamp'], inplace=True)

    # --- 1️⃣ Prepare Data ---
    data = data.dropna().copy()

    y_proba = model.predict_proba(data[features])
    proba_df = pd.DataFrame(y_proba, columns=model.classes_)

    # 🔧 Confidence threshold to avoid weak signals
    # Step 1: Get the class-wise probabilities
    class_probs = proba_df.mean()  # Average confidence for each class

    # 🛠 Step 2: Dynamically calculate thresholds for each class using percentiles
    # For each class (e.g., -1, 0, 1), compute the Nth percentile (e.g., 50th percentile = median)
    # This acts as a confidence threshold — only predictions stronger than the typical value are used
    thresholds = {}
    for cls in model.classes_:
        cls_probs = proba_df[cls]  # Get all predicted probabilities for this class
        thresholds[cls] = np.percentile(cls_probs, 50)  # 👈 Tune this percentile (e.g., 50, 60, 70)

    # 🧠 Step 3: Apply the thresholds to generate confident predictions
    y_custom = []
    for _, row in proba_df.iterrows():
        best_class = row.idxmax()  # Get class with highest probability for this row
        if row[best_class] >= thresholds[best_class]:
            y_custom.append(best_class)  # ✅ Confident prediction, accept the class
        else:
            y_custom.append(0)  # ⚖️ Not confident enough, fallback to neutral (hold)



    data['predicted'] = y_custom


    # Print signal distribution
    signal_distribution = dict(Counter(y_custom))
    long_conf = proba_df[1].mean()
    short_conf = proba_df[-1].mean()

    # Add a new column for percentage change in 'close'
    data['close_pct_change'] = data['close'].pct_change()  # Percentage change between current and previous 'close'
    
    # Initialize strategy-specific columns
    data['strategy_return'] = 0.0
    data['strategy_portfolio_value'] = initial_investment
    data['cash'] = initial_investment
    data['position'] = 0  # 0 = no position, 1 = long, -1 = short
    data['position_size'] = 0.0
    data['margin_used'] = 0.0
    data['funding_payments'] = 0.0
    data['liquidation'] = 0
    data['profit_withdrawn'] = 0.0  # Track withdrawn profits
    data['compounded_profit'] = 0.0  # Track compounded reinvestment
    data['stop_loss_amount'] = 0.0  # Track stop loss amount
    data['trading_fees'] = 0.0  # Track trading fees
    data['taker_fee_amount'] = 0.0  # Track taker fees
    data['liquidation_price'] = 0.0  # Track liquidation price
    data['liquidation_amount'] = 0.0  # Track liquidation amount
    data['realized_profit'] = 0.0  # Track realized profit

    total_liquidation_amount = 0  
    position_open = False
    last_position_price = 0
    last_funding_time = 0  
    trade_count = 0  # Track the number of trades

    # --- 2️⃣ Get Fees and Rates ---
    funding_rate_map = {
        "5m": 0.0001,  # 0.01% per 5 minutes
        "1h": 0.0005,  # 0.05% per hour
        "4h": 0.001,   # 0.1% per 4 hours
        "1d": 0.002,   # 0.2% per day
    }
    funding_rate = funding_rate_map[timeframe]

    # --- 3️⃣ Calculate Bars Between Funding Fees ---
    timeframe_to_bars = {
        "5m": 96,  # 8 hours = 96 bars (480 minutes / 5 minutes)
        "1h": 8,   # 8 hours = 8 bars
        "4h": 2,   # 8 hours = 2 bars
        "1d": 1/3, # 8 hours = 1/3 of a bar
    }
    bars_between_funding = timeframe_to_bars[timeframe]

    # --- 4️⃣ Define Liquidation Fees ---
    def get_liquidation_fees(asset, leverage):
        """
        Calculate liquidation fees based on the asset and leverage.
        Returns:
            user_liquidation_fee (float): Fee charged to the user for liquidation.
            liquidator_fee (float): Fee paid to the liquidator.
        """
        # High Tier Liquidation Fees
        if asset.split("_")[1] in ["BTC", "ETH"]:
            user_liquidation_fee = 0.008  # 0.80%
            liquidator_fee = 0.004  # 0.40%
        else:
            # Low Tier Liquidation Fees for Altcoins
            if leverage == 10:
                user_liquidation_fee = 0.015  # 1.50%
                liquidator_fee = 0.0075  # 0.75%
            elif leverage == 20:
                user_liquidation_fee = 0.024  # 2.40%
                liquidator_fee = 0.012  # 1.20%
            else:
                user_liquidation_fee = 0.015  # Default for other leverage levels
                liquidator_fee = 0.0075  # Default for other leverage levels

        return user_liquidation_fee, liquidator_fee

    # --- 5️⃣ Define Margin Ratios ---
    def get_margin_ratios(position_notional):
        """
        Calculate Initial Margin Ratio (IMR) and Maintenance Margin Ratio (MMR).
        """
        # Calculate IMR and MMR using fetched values
        imr = max(1 / leverage, base_imr, imr_factor * abs(position_notional) ** (4 / 5))
        mmr = max(base_mmr, base_mmr / base_imr * imr_factor * abs(position_notional) ** (4 / 5))
        return imr, mmr
    
    executed_longs = 0
    executed_shorts = 0
    winning_trades = 0 

    # --- 6️⃣ Iterate Over Data ---
    for i in range(1, len(data)):
        if num_trades is not None and trade_count >= num_trades:
            print(f"Trade limit reached. Stopping at trade {trade_count}")
            break  # Stop if the number of trades reaches the limit

        funding_payment = 0

        # --- Apply Funding Fees Every 8 Hours (Dynamic Bars) ---
        if position_open and (i - last_funding_time) >= bars_between_funding:
            funding_payment = data['position_size'].iloc[i - 1] * data['close'].iloc[i] * funding_rate
            data['funding_payments'].iloc[i] = funding_payment
            data['cash'].iloc[i] = data['cash'].iloc[i - 1] - funding_payment
            last_funding_time = i  # Update last funding time

        # --- Trade Execution Logic ---
        if data['predicted'].iloc[i - 1] == 1:  # Long signal
            if not position_open:
                executed_longs += 1
                # Calculate position size and trading fees (maker fee for opening)
                position_size = (data['cash'].iloc[i - 1] * leverage / data['close'].iloc[i]) * (1 - MAKER_FEE * leverage)
                trading_fee = position_size * data['close'].iloc[i] * MAKER_FEE  # Maker fee for opening the position
                
                # Update position and fee columns
                data['position'].iloc[i] = 1
                data['position_size'].iloc[i] = position_size
                data['margin_used'].iloc[i] = data['cash'].iloc[i - 1]  # Use current cash balance for margin
                data['trading_fees'].iloc[i] = trading_fee  # Store the trading fee
                
                # Update cash balance (subtract trading fee)
                data['cash'].iloc[i] = data['cash'].iloc[i - 1] - trading_fee
                
                position_open = True
                last_position_price = data['close'].iloc[i]
                trade_count += 1  # Increment trade count
            else:
                # If position is already open, carry forward the previous values
                data['position'].iloc[i] = data['position'].iloc[i - 1]
                data['position_size'].iloc[i] = data['position_size'].iloc[i - 1]
                data['margin_used'].iloc[i] = data['margin_used'].iloc[i - 1]
                data['trading_fees'].iloc[i] = 0  # No new fee for holding the position
                data['cash'].iloc[i] = data['cash'].iloc[i - 1]  # Carry forward cash balance

        elif data['predicted'].iloc[i - 1] == -1:  # Short signal
            if not position_open:
                executed_shorts += 1
                # Calculate position size and trading fees (maker fee for opening)
                position_size = (data['cash'].iloc[i - 1] * leverage / data['close'].iloc[i]) * (1 - MAKER_FEE * leverage)
                trading_fee = position_size * data['close'].iloc[i] * MAKER_FEE  # Maker fee for opening the position
                
                # Update position and fee columns
                data['position'].iloc[i] = -1
                data['position_size'].iloc[i] = position_size
                data['margin_used'].iloc[i] = data['cash'].iloc[i - 1]  # Use current cash balance for margin
                data['trading_fees'].iloc[i] = trading_fee  # Store the trading fee
                
                # Update cash balance (subtract trading fee)
                data['cash'].iloc[i] = data['cash'].iloc[i - 1] - trading_fee
                
                position_open = True
                last_position_price = data['close'].iloc[i]
                trade_count += 1  # Increment trade count
            else:
                # If position is already open, carry forward the previous values
                data['position'].iloc[i] = data['position'].iloc[i - 1]
                data['position_size'].iloc[i] = data['position_size'].iloc[i - 1]
                data['margin_used'].iloc[i] = data['margin_used'].iloc[i - 1]
                data['trading_fees'].iloc[i] = 0  # No new fee for holding the position
                data['cash'].iloc[i] = data['cash'].iloc[i - 1]  # Carry forward cash balance

        elif data['predicted'].iloc[i - 1] == 0 and position_open:  # Close position
                # 1) Compute PnL
                pnl = (data['close'].iloc[i] - last_position_price) * data['position_size'].iloc[i - 1] \
                    if data['position'].iloc[i - 1] == 1 \
                    else (last_position_price - data['close'].iloc[i]) * data['position_size'].iloc[i - 1]

                # 2) Calculate taker fee for closing the position
                taker_fee = abs(pnl) * TAKER_FEE * leverage
                data['taker_fee_amount'].iloc[i] = taker_fee  # Store the taker fee amount

                # 3) Apply taker fee to realized profit (this can be negative if pnl < 0)
                realized_profit = pnl * (1 - TAKER_FEE * leverage)
                data['realized_profit'].iloc[i] = realized_profit
               

                # ───────────────────────────────────────────────────────────────────
                # 4) If realized PnL is positive, then split between withdrawal & compounding
                #    Otherwise, no withdrawal or compounding occurs on a losing trade.
                # ───────────────────────────────────────────────────────────────────
                if realized_profit > 0:
                    winning_trades += 1
                    withdrawn_profit = realized_profit * withdraw_percentage if withdraw_percentage > 0 else 0
                    compounded_profit = realized_profit * compound_percentage if compound_percentage > 0 else 0

                    # Update cash with the net effect of compounding minus withdrawals
                    data['cash'].iloc[i] += (compounded_profit - withdrawn_profit)

                    data['profit_withdrawn'].iloc[i] = withdrawn_profit
                    data['compounded_profit'].iloc[i] = compounded_profit
                else:
                    data['cash'].iloc[i] = data['cash'].iloc[i - 1] + realized_profit
                    data['profit_withdrawn'].iloc[i] = 0.0
                    data['compounded_profit'].iloc[i] = 0.0

                # 5) Close the position
                data['position'].iloc[i] = 0
                data['position_size'].iloc[i] = 0
                data['margin_used'].iloc[i] = 0
                position_open = False
                trade_count += 1  # Increment trade count

        else:
            # If no trade is executed, carry forward the previous cash balance
            data['cash'].iloc[i] = data['cash'].iloc[i - 1]    

        # --- Portfolio Value Calculation ---
        if position_open:
            pnl = (data['close'].iloc[i] - last_position_price) * data['position_size'].iloc[i] if data['position'].iloc[i] == 1 else \
                  (last_position_price - data['close'].iloc[i]) * data['position_size'].iloc[i]
            data['strategy_portfolio_value'].iloc[i] = data['cash'].iloc[i] + pnl
        else:
            data['strategy_portfolio_value'].iloc[i] = data['cash'].iloc[i]

        # --- Stop-Loss & Take-Profit ---
        entry_price = last_position_price
        current_price = data['close'].iloc[i]

        if position_open:
            if data['position'].iloc[i] == 1:  # Long
                # Stop-loss check
                if current_price <= entry_price * (1 - stop_loss_threshold):
                    data['stop_loss_amount'].iloc[i] = (entry_price - current_price) * data['position_size'].iloc[i]
                    data['predicted'].iloc[i] = 0  # Close position
                
                # Take-profit check
                elif current_price >= entry_price * (1 + take_profit_threshold):
                    data['predicted'].iloc[i] = 0  # Close position (take profit)

            elif data['position'].iloc[i] == -1:  # Short
                # Stop-loss check
                if current_price >= entry_price * (1 + stop_loss_threshold):
                    data['stop_loss_amount'].iloc[i] = (current_price - entry_price) * data['position_size'].iloc[i]
                    data['predicted'].iloc[i] = 0  # Close position

                # Take-profit check
                elif current_price <= entry_price * (1 - take_profit_threshold):
                    data['predicted'].iloc[i] = 0  # Close position (take profit)


                    
        # --- Liquidation Check ---
        if position_open:
            # Calculate notional value
            notional = data['position_size'].iloc[i] * data['close'].iloc[i]

            # Calculate Account Margin Ratio (AMR)
            total_collateral_value = data['cash'].iloc[i] + pnl
            account_margin_ratio = total_collateral_value / abs(notional)

            # Calculate Maintenance Margin Ratio (MMR)
            _, mmr = get_margin_ratios(notional)

            # Calculate liquidation threshold
            liquidation_threshold = total_collateral_value - (mmr * abs(notional))

            # Calculate liquidation price
            if data['position'].iloc[i] == 1:  # Long position
                liquidation_price = last_position_price * (1 - mmr)
            else:  # Short position
                liquidation_price = last_position_price * (1 + mmr)

            # Update liquidation price and amount columns
            data['liquidation_price'].iloc[i] = liquidation_price
            data['liquidation_amount'].iloc[i] = liquidation_threshold

            # Check if liquidation is triggered
            if account_margin_ratio < mmr:
                # Determine liquidation type (Low Tier or High Tier)
                liquidation_fee, liquidator_fee = get_liquidation_fees(asset, leverage)
                user_liquidation_fee = liquidation_fee * abs(notional)
                liquidator_fee_total = liquidator_fee * abs(notional)

                # Apply liquidation
                data['liquidation'].iloc[i] = 1
                total_liquidation_amount += data['strategy_portfolio_value'].iloc[i - 1] - liquidation_threshold
                data['cash'].iloc[i] = data['strategy_portfolio_value'].iloc[i - 1] - liquidation_threshold - user_liquidation_fee - liquidator_fee_total
                data['position'].iloc[i] = 0
                data['position_size'].iloc[i] = 0
                data['margin_used'].iloc[i] = 0
                position_open = False

    trade_accuracy = winning_trades / trade_count if trade_count else 0

    return data, total_liquidation_amount, trade_count, signal_distribution, trade_accuracy, long_conf, short_conf


# Plot data
def plot_backtest_results(data, pair, output_file):
    """
    Plots the backtest results, including strategy and market portfolio values,
    ensuring start_timestamp is correctly converted to datetime.
    """
    # 1) Make sure we do have a 'start_timestamp'
    if 'start_timestamp' not in data.columns:
        raise KeyError("The 'start_timestamp' column is missing in the data.")
        
    # 2) If it's numeric (Unix epochs), figure out if it's in seconds or milliseconds
    if np.issubdtype(data['start_timestamp'].dtype, np.number):
        max_ts = data['start_timestamp'].max()
        # Rough rule of thumb:
        #   if timestamps are near 1.7e9, it’s probably seconds-since-epoch
        #   if near 1.7e12 or 1.7e13, it’s probably milliseconds
        if max_ts > 1e11:  
            # Likely milliseconds
            data['start_timestamp'] = pd.to_datetime(data['start_timestamp'], unit='ms', errors='coerce')
        else:
            # Likely seconds
            data['start_timestamp'] = pd.to_datetime(data['start_timestamp'], unit='s', errors='coerce')
    else:
        # Otherwise assume it’s a standard date string
        data['start_timestamp'] = pd.to_datetime(data['start_timestamp'], errors='coerce')
    
    # 3) Use start_timestamp as the DataFrame index
    data.set_index('start_timestamp', drop=True, inplace=True)
    
    # 4) Create the figure
    plt.figure(figsize=(14, 7))

    # Normalize the portfolio values to start at 100
    if 'strategy_portfolio_value' in data.columns:
        first_val = data['strategy_portfolio_value'].iloc[0]
        strategy_normalized = (data['strategy_portfolio_value'] / first_val) * 100
        plt.plot(
            data.index, 
            strategy_normalized, 
            label='Strategy Portfolio Value (with fees)', 
            color='blue', 
            linewidth=2
        )

    if 'market_portfolio_value' in data.columns:
        first_val_mkt = data['market_portfolio_value'].iloc[0]
        market_normalized = (data['market_portfolio_value'] / first_val_mkt) * 100
        plt.plot(
            data.index, 
            market_normalized, 
            label='Market Portfolio Value', 
            color='orange', 
            linewidth=2
        )

    # Format the x-axis ticks/labels
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())  
    plt.xticks(rotation=45)

    # Labels, title, legend, grid
    plt.title(f'Backtest Results for {pair}', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Normalized Portfolio Value (%)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save as PNG (adjust as needed)
    plot_file = output_file.replace('.xlsx', '.png')
    plt.savefig(plot_file)
    plt.close()

# Updated run_backtest function with logging
async def run_backtest(pair, timeframe, token, values, stop_loss_threshold=0.05, initial_investment=10000, take_profit_threshold=0.001, leverage=1, features=None, withdraw_percentage=0.7, compound_percentage=0.3, num_trades=None):
    start = datetime.now()
    logger.info("Starting backtest")

    # Generate dynamic file names
    model_name = "_".join(features).replace("[", "").replace("]", "").replace("'", "_").replace(" ", "")
    MODEL_KEY = f'Mockba/trained_models/trained_model_{pair}_{timeframe}_{model_name}.joblib'
    local_model_path = f'temp/trained_model_{pair}_{timeframe}_{model_name}.joblib'
    output_file = f'files/backtest_results_{pair}_{timeframe}_{token}_{model_name}.xlsx'

    if download_model(BUCKET_NAME, MODEL_KEY, local_model_path):
        # Fetch historical data and add technical indicators
        logger.info("Fetching historical data from Orderly")
        data = await get_historical_data(token, pair, timeframe, values)
        logger.info("Adding technical indicators")

        # Ensure the 'close' column exists
        if 'close' not in data.columns:
            raise KeyError("The 'close' column is missing in the historical data.")

        # Add start_timestamp if not already present
        if 'start_timestamp' not in data.columns:
            # Assuming the data has a 'timestamp' column or similar
            if 'timestamp' in data.columns:
                data['start_timestamp'] = data['timestamp']
            else:
                # If no timestamp column is available, create one based on the index
                data['start_timestamp'] = data.index

        # Set start_timestamp as the index
        data.set_index('start_timestamp', inplace=True)

        # Calculate the 'return' column
        data['return'] = data['close'].pct_change().shift(-1)
        logger.info("Calculated return column")

        # Load the model
        model_metadata = joblib.load(local_model_path)
        model = model_metadata["model"]
        used_features = model_metadata.get("used_features", [])

        # Add missing features to the dataset
        missing_features = [f for f in used_features if f not in data.columns]
        if missing_features:
            data = add_indicators(data, missing_features)

        # Ensure dataset contains only trained features and in correct order
        data = data[used_features]

        # Calculate market portfolio value (buy-and-hold strategy)
        if 'return' in data.columns:
            data['market_portfolio_value'] = initial_investment * (1 + data['return'].cumsum())
        else:
            data['market_portfolio_value'] = initial_investment  # Default to initial investment

        # Proceed with backtest using `final_features`
        backtest_result, total_liquidation_amount, trade_count, signal_distribution, trade_accuracy, long_conf, short_conf = await backtest(
              data
            , model
            , used_features
            , initial_investment
            , stop_loss_threshold
            , take_profit_threshold
            , leverage
            , withdraw_percentage
            , compound_percentage
            , num_trades
            , pair
            , timeframe
        )

        # Calculate key metrics
        total_funding_payments = backtest_result['funding_payments'].sum()
        total_trading_fees = backtest_result['trading_fees'].sum()
        total_taker_fees = backtest_result['taker_fee_amount'].sum()
        total_stop_loss_amount = backtest_result['stop_loss_amount'].sum()


        # Use the last row of the 'cash' column as the final strategy value
        final_strategy_value = backtest_result['cash'].iloc[-1]
        total_compounded_profit = final_strategy_value - initial_investment
        final_percentage_gain_loss = ((final_strategy_value - initial_investment) / initial_investment) * 100

        # Identify executed trades and their outcomes
        executed_trades = backtest_result[backtest_result['realized_profit'] != 0]
        winning_trades = executed_trades[executed_trades['realized_profit'] > 0]
        trade_count = len(executed_trades)
        trade_accuracy = len(winning_trades) / trade_count if trade_count > 0 else 0

        # Generate result explanation
        result_explanation = (
            f"**Asset Traded:** {pair}\n\n"
            f"**Timeframe:** {timeframe}\n"
            f"**Trading dates:** {values}\n\n"
            f"**Initial investment:** ${initial_investment:.2f}\n"
            f"**Final percentage gain/loss:** {final_percentage_gain_loss:.2f}% 💹\n"
            f"**Total strategy value (capital + compounded):** ${final_strategy_value:.2f}\n"
            f"**Profit withdrawn:** ${backtest_result['profit_withdrawn'].sum():.2f}  🚀\n"
            f"**Profit withdrawn percentage:** {withdraw_percentage * 100:.2f}%\n"
            f"**Compounded profit:** ${total_compounded_profit:.2f}  🚀\n"
            f"**Compounded profit percentage:** {compound_percentage * 100:.2f}%\n\n"
            f"**Total funding payments:** ${total_funding_payments:.2f}\n"
            f"**Total liquidation amount:** ${total_liquidation_amount:.2f}\n"
            f"**Total stop-loss amount:** ${total_stop_loss_amount:.2f}\n"
            f"**Total maker fees:** ${total_trading_fees:.2f}\n"
            f"**Total taker fees:** ${total_taker_fees:.2f}\n"
            f"**Take Profit threshold:** {take_profit_threshold * 100:.2f}%\n"
            f"**Stop-loss threshold:** {stop_loss_threshold * 100:.2f}%\n"
            f"**Leverage used:** {leverage}x\n"
            f"**Number of trades executed:** {trade_count}\n"
            f"**Strategy name:** {get_strategy_name(timeframe, features)}\n\n"
            f"**Used Features:** {used_features}\n"
            f"🔍 **Avg Prediction Probabilities:** Long: {long_conf:.2f}, Short: {short_conf:.2f}\n"
            f"🎯 **Trade Accuracy:** {trade_accuracy:.2%} "
            f"({len(winning_trades)} winning trades out of {trade_count})\n\n"
            f"**Execution time:** {datetime.now() - start}"
        )
        transtaled_result_explanation = translate(result_explanation, token)
        logger.info(transtaled_result_explanation)
        # await send_bot_message(token, result_explanation)

        # Save results to an Excel file with only relevant columns
        logger.info("Saving results to Excel file")

        # Add new columns for open/close position, taker/maker fees, and liquidation price
        backtest_result['open_close_position'] = backtest_result['position'].apply(
            lambda x: 'open' if x != 0 else 'close'
        )

        # Map position values to descriptive strings
        position_map = {
            1: 'long',
            -1: 'short',
            0: 'hold'
        }
        backtest_result['position'] = backtest_result['position'].map(position_map)
        
       
        # Define the exact column order you want
        final_columns = [
            "start_timestamp",
            "position",
            "close",
            "close_pct_change",
            "liquidation_price",
            "liquidation_amount",
            "cash",
            "position_size",
            "open_close_position",
            "stop_loss_amount",
            "margin_used",
            "funding_payments",  
            "liquidation",
            "profit_withdrawn",
            "strategy_portfolio_value",
            "realized_profit",
            "compounded_profit",
            "taker_fee_amount",
            "trading_fees",
        ]

        # Create a copy of the DataFrame for Excel formatting
        formatted_result = backtest_result.copy()
        
        # Format columns for Excel output
        formatted_result["close_pct_change"] = formatted_result["close_pct_change"].map('{:.2%}'.format)
        formatted_result["close"] = formatted_result["close"].map('{:.6f}'.format)
        formatted_result["liquidation_price"] = formatted_result["liquidation_price"].map('{:.6f}'.format)
        formatted_result["liquidation_amount"] = formatted_result["liquidation_amount"].map('{:.2f}'.format)
        formatted_result["cash"] = formatted_result["cash"].map('{:.2f}'.format)
        formatted_result["position_size"] = formatted_result["position_size"].map('{:.2f}'.format)
        formatted_result["stop_loss_amount"] = formatted_result["stop_loss_amount"].map('{:.2f}'.format)
        formatted_result["margin_used"] = formatted_result["margin_used"].map('{:.2f}'.format)
        formatted_result["funding_payments"] = formatted_result["funding_payments"].map('{:.2f}'.format)
        formatted_result["strategy_portfolio_value"] = formatted_result["strategy_portfolio_value"].map('{:.2f}'.format)
        formatted_result["liquidation"] = formatted_result["liquidation"].map('{:.0f}'.format)
        formatted_result["profit_withdrawn"] = formatted_result["profit_withdrawn"].map('{:.2f}'.format)
        formatted_result["realized_profit"] = formatted_result["realized_profit"].map('{:.2f}'.format)
        formatted_result["compounded_profit"] = formatted_result["compounded_profit"].map('{:.2f}'.format)
        formatted_result["taker_fee_amount"] = formatted_result["taker_fee_amount"].map('{:.2f}'.format)
        formatted_result["trading_fees"] = formatted_result["trading_fees"].map('{:.2f}'.format)

        # Save to Excel with the specified column order using formatted data
        formatted_result[final_columns].to_excel(output_file, index=False)
        
        # Plot using the original numeric data
        plot_backtest_results(backtest_result, pair, output_file)

        # Delete local model file after upload
        if os.path.exists(local_model_path):
           os.remove(local_model_path)
        else:
           print(f"Local file {local_model_path} does not exist.")

        logger.info("Backtest completed")

    else:
        logger.info(f"No model found for {pair}_{timeframe} it must be trained, contact support")
        await send_bot_message(token, "No model found it must be trained, contact support.")

# Example of how to call run_backtest
# if __name__ == "__main__":
#     pair = 'PERP_APT_USDC'
#     timeframe = '1h'
#     token = '556159355'
#     values = '2025-01-01|2025-01-19'
#     stop_loss_threshold = 0.5
#     initial_investment = 100
#     maker_fee = 0.001
#     taker_fee = 0.001
#     take_profit_threshold = 0.01
#     leverage = 1

#     print(run_backtest(pair, timeframe, token, values, stop_loss_threshold, initial_investment, maker_fee, taker_fee, take_profit_threshold, leverage))
