import aiohttp
import time
import sys
import os
import json
import pandas as pd
import numpy as np
import urllib.parse
from dotenv import load_dotenv
import joblib  # Library for model serialization
from datetime import datetime, timedelta
import requests
import redis.asyncio as redis
import logging
from app.models.bucket import download_model
import json
from datetime import datetime, timedelta, timezone
from base58 import b58decode
from base64 import urlsafe_b64encode
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import threading
from deep_translator import GoogleTranslator

# Add the directory containing your modules to the Python path
sys.path.append('/app')

# ✅ Orderly API Config
BASE_URL = os.getenv("ORDERLY_BASE_URL")
ORDERLY_ACCOUNT_ID = os.getenv("ORDERLY_ACCOUNT_ID")
ORDERLY_SECRET = os.getenv("ORDERLY_SECRET")
ORDERLY_PUBLIC_KEY = os.getenv("ORDERLY_PUBLIC_KEY")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", 10))
DEEP_SEEK_API_KEY = os.getenv("DEEP_SEEK_API_KEY")

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

if not ORDERLY_SECRET or not ORDERLY_PUBLIC_KEY:
    raise ValueError("❌ ORDERLY_SECRET or ORDERLY_PUBLIC_KEY environment variables are not set!")

# ✅ Remove "ed25519:" prefix if present in private key
if ORDERLY_SECRET.startswith("ed25519:"):
    ORDERLY_SECRET = ORDERLY_SECRET.replace("ed25519:", "")

# ✅ Decode Base58 Private Key
private_key = Ed25519PrivateKey.from_private_bytes(b58decode(ORDERLY_SECRET))

# ✅ Rate limiter (Ensures max 8 API requests per second globally)
class RateLimiter:
    def __init__(self, max_calls, period):
        self.max_calls = max_calls
        self.period = period
        self.calls = []
        self.lock = threading.Lock()

    def __call__(self):
        with self.lock:
            now = time.time()
            self.calls = [call for call in self.calls if call > now - self.period]
            if len(self.calls) >= self.max_calls:
                sleep_time = self.period - (now - self.calls[0])
                print(f"⏳ Rate limit reached! Sleeping for {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            self.calls.append(time.time())

# ✅ Initialize Global Rate Limiter
rate_limiter = RateLimiter(max_calls=10, period=1)
def fetch_order_book_snapshot(symbol):
    rate_limiter()  # ✅ Apply global rate limit

    timestamp = str(int(time.time() * 1000))
    path = f"/v1/orderbook/{symbol}"  # Include query string
    message = f"{timestamp}GET{path}"

    signature = urlsafe_b64encode(private_key.sign(message.encode())).decode()

    headers = {
        "orderly-timestamp": timestamp,
        "orderly-account-id": ORDERLY_ACCOUNT_ID,
        "orderly-key": ORDERLY_PUBLIC_KEY,
        "orderly-signature": signature,
    }

    url = f"{BASE_URL}{path}"
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"❌ Error fetching data for {symbol}: {response.json()}")
        return None

    data = response.json().get("data", {})
    if not data:
        return None

    asks = data.get("asks", [])
    bids = data.get("bids", [])

    # Create DataFrames
    df_asks = pd.DataFrame(asks)
    df_bids = pd.DataFrame(bids)

    # Add a side column
    if not df_asks.empty:
        df_asks["side"] = "ask"
    if not df_bids.empty:
        df_bids["side"] = "bid"

    # Combine both DataFrames
    df_orderbook = pd.concat([df_bids, df_asks], ignore_index=True)

    # Optional: Sort by price descending if you want
    df_orderbook = df_orderbook.sort_values(by="price", ascending=False).reset_index(drop=True)

    return df_orderbook

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore")
# Load environment variables from the .env file
load_dotenv(dotenv_path=".env.micro.trading")


BUCKET_NAME = os.getenv("BUCKET_NAME")  # Your bucket name
MICRO_CENTRAL_URL = os.getenv("MICRO_CENTRAL_URL")  # Your micro central URL

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


# ✅ Fetch historical Orderly data with global rate limiting
def fetch_historical_orderly(symbol, interval):
    rate_limiter()  # ✅ Apply global rate limit

    timestamp = str(int(time.time() * 1000))
    params = {"symbol": symbol, "type": interval, "limit": 1000}
    path = "/v1/kline"
    query = f"?{urllib.parse.urlencode(params)}"
    message = f"{timestamp}GET{path}{query}"
    signature = urlsafe_b64encode(private_key.sign(message.encode())).decode()

    headers = {
        "orderly-timestamp": timestamp,
        "orderly-account-id": ORDERLY_ACCOUNT_ID,
        "orderly-key": ORDERLY_PUBLIC_KEY,
        "orderly-signature": signature,
    }

    url = f"{BASE_URL}{path}{query}"
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"❌ Error fetching data for {symbol} {interval}: {response.json()}")
        return None

    data = response.json().get("data", {})
    if not data or "rows" not in data:
        return None

    df = pd.DataFrame(data["rows"])
    required_columns = ["start_timestamp", "open", "high", "low", "close", "volume"]
    if set(required_columns).issubset(df.columns):
        df['start_timestamp'] = pd.to_datetime(df['start_timestamp'], unit='ms')
        df.set_index('start_timestamp', inplace=True)
        df = df[["open", "high", "low", "close", "volume"]]
        df = df[~df.index.duplicated(keep='first')]  # Remove duplicates
        return df
    return None


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


async def fetch_asset_info(symbol):
    """
    Fetch asset info from Orderly API including margin and liquidation parameters.
    Cache the result in Redis for 30 days.
    """
    cache_key = f"asset_info:{symbol}"
    ttl_30_days = 30 * 24 * 60 * 60  # 30 days in seconds

    # Check if the data exists in Redis
    if redis_client:
        cached_data = await redis_client.get(cache_key)
        if cached_data:
            print(f"Cache hit for {symbol}")
            return json.loads(cached_data)  # Return cached data

    # Fetch data from the API
    url = f"https://api-evm.orderly.org/v1/public/info/{symbol}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json().get("data", {})
        asset_info = {
            "base_mmr": data.get("base_mmr", 0.05),
            "base_imr": data.get("base_imr", 0.1),
            "imr_factor": data.get("imr_factor", 0.00000208),
            "funding_period": data.get("funding_period", 8),
            "cap_funding": data.get("cap_funding", 0.0075),
            "std_liquidation_fee": data.get("std_liquidation_fee", 0.024),
            "liquidator_fee": data.get("liquidator_fee", 0.012),
            "min_notional": data.get("min_notional", 10),
            "quote_max": data.get("quote_max", 100000),
        }

        # Store the data in Redis for 30 days
        if redis_client:
            await redis_client.setex(cache_key, ttl_30_days, json.dumps(asset_info))

        return asset_info
    else:
        raise Exception(f"Failed to fetch asset info for {symbol} - Status code: {response.status_code}")
    

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


def format_analysis_for_telegram(cached_data):
    """
    Safely decodes Redis-cached data and fixes unicode emojis for Telegram.
    """
    if not cached_data:
        return None

    try:
        # Decode Redis bytes to UTF-8 string (don't touch encoding further!)
        decoded_str = cached_data.decode('utf-8')

        # Parse JSON if necessary
        try:
            parsed = json.loads(decoded_str)
        except json.JSONDecodeError:
            parsed = decoded_str

        # If parsed is still escaped Unicode (e.g., \\ud83d), decode it
        if isinstance(parsed, str) and '\\u' in parsed:
            parsed = bytes(parsed, 'utf-8').decode('unicode_escape')

        return parsed

    except Exception as e:
        print(f"Error formatting cached data: {e}")
        return None

    except Exception as e:
        print(f"Error formatting cached data: {e}")
        return None



async def analize_asset(token, asset, interval, features, market_bias='neutral'):
    print('Getting data for analysis') 
    analysis_translated = None
    # Generate dynamic file names
    model_name = "_".join(features).replace("[", "").replace("]", "").replace("'", "_").replace(" ", "")
    MODEL_KEY = f'Mockba/trained_models/trained_model_{asset}_{interval}_{model_name}.joblib'
    local_model_path = f'temp/trained_model_{asset}_{interval}_{model_name}.joblib'
    
    
    cache_key = f"analisys:{asset}:{interval}:{features}"
    
    cached_data = await redis_client.get(cache_key)
    if cached_data:
        print("Returning cached analysis")
        formatted_text = format_analysis_for_telegram(cached_data)
        translated_text = translate(formatted_text, token)
        await send_bot_message(token, translated_text)
        return
        

    if download_model(BUCKET_NAME, MODEL_KEY, local_model_path):
        data = fetch_historical_orderly(asset, interval)
        #######################################################################################
        #######################################################################################
 
        # --- Step 1 Analize data, add indicators and prepare for prediction ---
        # Load the model
        model_metadata = joblib.load(local_model_path)
        model = model_metadata["model"]
        used_features = model_metadata.get("used_features", [])

        # Add missing features to the dataset
        missing_features = [f for f in used_features if f not in data.columns]
        if missing_features:
            data = add_indicators(data, missing_features)

        # 
        # Set start_timestamp as the index
        # data.set_index('start_timestamp', inplace=True)    

        # Ensure dataset contains only trained features and in correct order
        data = data[used_features]

        # --- 1️⃣ Prepare Data ---
        data = data.dropna().copy()

        # --- Predict class probabilities
        y_proba = model.predict_proba(data[features])
        proba_df = pd.DataFrame(y_proba, columns=model.classes_)

        # --- 🔍 Step 1: Get average class-wise confidence
        class_probs = proba_df.mean()
        # print("📊 Average Confidence per Class:", class_probs.to_dict())

        # --- ⚖️ Step 2: Dynamically adjust thresholds based on class imbalance
        # --- ⚖️ Adjust thresholds based on class imbalance and market bias ---
        base_percentile = 50
        thresholds = {}

        for cls in model.classes_:
            adjustment = 0
            if class_probs[cls] < 0.3:
                adjustment = -10  # Boost underrepresented classes
            elif class_probs[cls] > 0.5:
                adjustment = +10  # Penalize dominant classes

            # Extra adjustment based on market bias
            if market_bias == 'bullish':
                if cls == 1:
                    adjustment -= 5  # Make long signals easier to trigger
                elif cls == -1:
                    adjustment += 5  # Make short signals harder
            elif market_bias == 'bearish':
                if cls == -1:
                    adjustment -= 5  # Make short signals easier
                elif cls == 1:
                    adjustment += 5  # Make long signals harder

            percentile = max(10, min(90, base_percentile + adjustment))
            thresholds[cls] = np.percentile(proba_df[cls], percentile)

        # --- 🧠 Apply class distribution rules to generate predictions ---
        # Define default signal distribution
        target_fraction = {-1: 0.35, 1: 0.15, 0: 0.5}

        # Modify class distribution based on market bias
        if market_bias == 'bullish':
            target_fraction = {-1: 0.2, 1: 0.3, 0: 0.5}  # More longs
        elif market_bias == 'bearish':
            target_fraction = {-1: 0.4, 1: 0.1, 0: 0.5}  # More shorts

        n_samples = len(proba_df)
        target_counts = {cls: int(n_samples * frac) for cls, frac in target_fraction.items()}

        # Start with all hold
        y_custom = np.zeros(n_samples, dtype=int)

        # Assign top-N short and long predictions
        for cls in [-1, 1]:
            top_indices = proba_df[cls].nlargest(target_counts[cls]).index
            y_custom[top_indices] = cls

        # Assign predictions
        data['predicted'] = y_custom

        # Add a new column for percentage change in 'close'
        data['close_pct_change'] = data['close'].pct_change()  # Percentage change between current and previous 'close'
        data = data.tail(100)  # Keep only the last 100 rows
        
        # Export data to CSV
        # data.to_csv(f'temp/data_{asset}_{interval}.csv')
        data_json = json.dumps(data.to_dict(orient='records'))

        #######################################################################################
        #######################################################################################

        # --- Step 2: Get asset info ---
        # Get Asset Info
        asset_info = await fetch_asset_info(asset)
        asset_info_df = pd.DataFrame([asset_info])
        # asset_info_df.to_csv(f'temp/asset_info_{asset}.csv')
        asset_info_json = json.dumps(asset_info_df.to_dict(orient='records'))

        #######################################################################################
        #######################################################################################


        # --- Step 3: Get Order Book Snapshot ---
        # Get Orfer Book Snapshot
        order_book_snapshot = fetch_order_book_snapshot(asset)
        order_book_snapshot_df = pd.DataFrame(order_book_snapshot)
        # order_book_snapshot_df.to_csv(f'temp/order_book_snapshot_{asset}.csv')
        order_book_snapshot_json = json.dumps(order_book_snapshot_df.to_dict(orient='records'))

        #######################################################################################
        #######################################################################################
        # Step 4: Send to DeepSeek API
        prompt = f"""
        **Task:** Generate a professional trading analysis for {asset} {interval} with clear verdict and recommendations, also Generate exactly 2 trade setups (1 long, 1 short) with clear triggers and fixed risk parameters, optimized for Telegram.

        ###Raw Data to Analyze:
        1. PRICE/INDICATORS (Last 100 rows):
        {data_json} 

        2. ASSET CONTEXT:
        {asset_info_json} 

        3. INDICATORS PRESENT:  
        - f"{features}, `Predicted` (ML signals: -1=Short, 0=Hold, 1=Long), close_pct_change" 

        4 ASSET INFO:
        {asset_info_json}

        5. ORDER BOOK SNAPSHOT:
        {order_book_snapshot_json}

        Strict Output Rules:
        📊 {asset} {interval}, {get_strategy_name(interval, features)} TA Report

        Explanation
        - [2-3 sentences explaining the analysis]
        📉 Price Action

        📈 [Trend]
        ➖/➕ [EMA20 vs EMA50] + [ADX <25?>]
        🔸 [VWAP relation] + [MACD direction]

        📊 [Volume]
        ➖/➕ [Volume trend] + [Volume spikes]

        🎯 [Key Zones]
        ▪️ Support: [strongest 2 levels]
        ▪️ Resistance: [strongest 2 levels]

        🤖 [ML Signals]
        🔹 Recent: [last 3 predictions] 
        🔸 Now: [current prediction] @ [price]

        💧 [Liquidity]
        🛡️ Asks: [size] @ [price] 
        🛡️ Bids: [size] @ [price]

        💰 [Asset Info]
        ▪️ Std Liquidation Fee: [value]
        ▪️ Liquidator Fee: [value]
        ▪️ Min Notional: [value]
        ▪️ Quote Max: [value]

        ⚡ [Setups]
        2) [Trigger] → TP:[target] SL:[stop]
        3) [Trigger] → TP:[target] SL:[stop]

        📌 [Verdict and Summary]
        ▪️ [1-sentence bias]
        ⚠️ [Key risk] 
        ⌚ Next check: [time/condition] 

        Rules:
        1. No markdown (**, `, ###, etc)
        2. Use simple dashes (-) for bullets
        3. Keep decimals consistent (2 places)
        """


        # Send to DeepSeek API
        # For your use case (temperature=0.3):
        # Optimal for reliable, repeatable technical analysis.
        # Sacrifices creativity for accuracy and consistency.
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",  # Verify the correct endpoint
            json={
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a trading analyst. Generate concise Elliott Wave reports in PLAIN TEXT only (no markdown). Use emojis but no formatting (** or `). Keep numbers to 2 decimals."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3  # Lower = more deterministic
            },
            headers={"Authorization": f"Bearer {DEEP_SEEK_API_KEY}"}
        )

        if response.status_code == 200:
            analysis = response.json()["choices"][0]["message"]["content"]
            analysis_translated = translate(analysis, token)

            # Store result in Redis with 20-minute expiration
            await redis_client.setex(cache_key,
                timedelta(minutes=20),
                json.dumps(analysis))

            await send_bot_message(token, analysis_translated)
            # print(analysis_translated)
        else:
            print(f"Error: {response.status_code}, {response.text}")
      
        # Delete local model file after upload
        if os.path.exists(local_model_path):
           os.remove(local_model_path)
        else:
           print(f"Local file {local_model_path} does not exist.")

    return analysis_translated       
    