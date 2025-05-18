import aiohttp
import os
import time
import sys
import requests
import pandas as pd
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
import redis
import json
from base58 import b58decode
from base64 import urlsafe_b64encode
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from deep_translator import GoogleTranslator
import threading
import logging
from dotenv import load_dotenv

# Dictionary to store gainers lists for each token
session_gainers = {}

# Connect to Redis
# Initialize Redis connection
try:
    redis_client = redis.from_url(os.getenv("REDIS_URL"))
    redis_client.ping()
except redis.ConnectionError as e:
    print(f"Redis connection error: {e}")
    redis_client = None


# Add the directory containing your modules to the Python path
sys.path.append('/app')

# ✅ Orderly API Config
BASE_URL = os.getenv("ORDERLY_BASE_URL")
ORDERLY_ACCOUNT_ID = os.getenv("ORDERLY_ACCOUNT_ID")
ORDERLY_SECRET = os.getenv("ORDERLY_SECRET")
ORDERLY_PUBLIC_KEY = os.getenv("ORDERLY_PUBLIC_KEY")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", 10))
DEEP_SEEK_API_KEY = os.getenv("DEEP_SEEK_API_KEY")

def translate(text, target_lang):
        return GoogleTranslator(source='auto', target=target_lang).translate(text)

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


def get_strategy_features(timeframe, strategy_name):
    """
    Returns the combined base features + indicators for a given strategy and timeframe
    
    Parameters:
    timeframe (str): The timeframe ('1h', '4h', '1d')
    strategy_name (str): The strategy name ('Momentum + Volatility', 'Hybrid', 'Trend-Following')
    
    Returns:
    list: Combined list of base features and indicators, or None if not found
    """
    strategy_definitions = {
        "1h": {
            "Momentum + Volatility": ["close", "high", "low", "volume", 
                                    "rsi_14", "atr_14", "bollinger_hband_20", 
                                    "bollinger_lband_20", "roc_10", "momentum_10", "vwap"]
        },
        "4h": {
            "Hybrid": ["close", "high", "low", "volume",
                      "ema_50", "ema_200", "atr_14", 
                      "bollinger_hband_20", "rsi_14", "macd", "vwap"]
        },
        "1d": {
            "Trend-Following": ["close", "high", "low", "volume",
                              "ema_50", "ema_200", "macd", 
                              "macd_signal", "adx", "vwap"]
        }
    }
    
    try:
        return strategy_definitions.get(timeframe, {}).get(strategy_name)
    except Exception as e:
        print(f"Error getting strategy features: {e}")
        return None
    

# Calculate the percentage change for the specified interval
def get_percentage_change(symbol, interval='1h'):
    df = fetch_historical_orderly(symbol, interval=interval)
    # get the last two rows of the DataFrame
    df = df.tail(2)
    if len(df) < 2:
        return None
    current_price = df['close'].iloc[-1]
    previous_price = df['close'].iloc[0]
    percentage_change = ((current_price - previous_price) / previous_price) * 100
    return percentage_change


# Process each symbol:
def process_symbol(symbol, interval, change_threshold=0):
    # print(f"Processing symbol: {symbol}")
    #get strategy features, depending on th einterval data
    if interval == '1h':
        strategy = "Momentum + Volatility"
    elif interval == '4h':
        strategy = "Hybrid"
    elif interval == '1d':
        strategy = "Trend-Following"
    try:
        df = fetch_historical_orderly(symbol, interval)
        df = add_indicators(df, get_strategy_features(interval, strategy))
        df['is_gainer'] = ((df['close'] - df['close'].shift(1)) / df['close'].shift(1)) > change_threshold  # >1% gain
        # Check if the last row is a gainer
        if df['is_gainer'].iloc[-1]:
            percentage_change = get_percentage_change(symbol, interval)
            if percentage_change is not None:
                return symbol, df['close'].iloc[-1], df['close'].iloc[0], percentage_change
    except Exception as e:
        print(f"Error processing {symbol}: {e}")
    return None


def get_early_stage_gainers_concurrently(interval='1h', change_threshold=0):
    orderly_symbols = "https://api-evm.orderly.org/v1/public/info"
    response = requests.get(orderly_symbols)
    if response.status_code != 200:
        print(f"❌ Error fetching symbols: {response.json()}")
        return []
    rows = response.json().get("data", {}).get("rows", [])
    symbols = [row["symbol"] for row in rows if "symbol" in row]

    gainers = []

    max_workers = os.cpu_count()  # Get the number of CPU cores on your PC

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {executor.submit(process_symbol, symbol, interval, change_threshold): symbol for symbol in symbols}
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                result = future.result()
                if result:
                    gainers.append(result)
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
    # logger.info(f"Found {len(gainers)} gainers")
    return gainers

# Calculate percentage increase:
async def calculate_percentage_increase(gainers, change_threshold=0):
    gainers_with_increase = []
    for symbol, current_price, initial_price, price_change_percent in gainers:
        percentage_increase = ((current_price - initial_price) / initial_price) * 100
        if price_change_percent > change_threshold:  # Filter based on the change threshold
            gainers_with_increase.append((symbol, percentage_increase, price_change_percent))
    return gainers_with_increase

# Main function to fetch and analyze data based on the interval
async def fetch_and_analyze_gainers(token, interval='1h', change_threshold=0):
    global session_gainers
    session_gainers[token] = []

    # Define Redis key
    redis_key = f"gainers:{token}:{interval}"

    # Check if data exists in Redis
    cached_data = redis_client.get(redis_key)
    if cached_data:
        # Load data from Redis
        filtered_gainers = json.loads(cached_data)
        session_gainers[token] = filtered_gainers
        return filtered_gainers

    # Get all early stage gainers
    early_stage_gainers = get_early_stage_gainers_concurrently(interval, change_threshold)
    gainers_with_increase = await calculate_percentage_increase(early_stage_gainers, change_threshold)

    # Filter gainers with positive percentage increase
    gainers_with_positive_increase = [g for g in gainers_with_increase if g[1] > 0]

    # Sort gainers by percentage increase
    gainers_with_positive_increase.sort(key=lambda x: x[1], reverse=True)

    # Filter gainers with positive interval change
    filtered_gainers = [g for g in gainers_with_positive_increase if g[2] > 0]

    # Store the result in the session_gainers variable for the given token
    session_gainers[token] = filtered_gainers

    # Store the result in Redis for 15 minutes
    redis_client.setex(redis_key, 900, json.dumps(filtered_gainers))  # 900 seconds = 15 minutes

    # Return the filtered gainers
    return filtered_gainers

# Function to retrieve the stored gainers list for a specific token
def get_stored_gainers(token):
    global session_gainers
    return session_gainers.get(token, [])

# Function to clear the stored gainers list for a specific token
def clear_stored_gainers(token):
    global session_gainers
    if token in session_gainers:
        del session_gainers[token]

async def analyze_movers(token, target_lang, interval='1h', change_threshold=0, type='gainers', top_n=10):
    """
    Analyze top gainers or losers and send a readable Telegram message.
    
    Args:
        token (str): Telegram token to identify user/session.
        interval (str): Time interval for analysis ('1h', '4h', '1d').
        change_threshold (float): Minimum absolute % change to include.
        type (str): 'gainers' or 'losers'.
        top_n (int): How many entries to return.
    """
    logger.info(f"Fetching and analyzing {type} for token: {token}, interval: {interval}, threshold: {change_threshold}")

    movers = await fetch_and_analyze_gainers(token, interval, change_threshold)

    if type == 'gainers':
        filtered = [m for m in movers if m[1] > 0]
        filtered.sort(key=lambda x: x[1], reverse=False)  # low to high
        title_icon = "📈"
        title_text = "Early Gainers"
    elif type == 'losers':
        filtered = [m for m in movers if m[1] < 0]
        filtered.sort(key=lambda x: x[1])  # most negative first
        title_icon = "📉"
        title_text = "Early Losers"
    else:
        raise ValueError("Invalid type: must be 'gainers' or 'losers'")

    top_filtered = filtered[:top_n]
    message_lines = [f"{title_icon} *{title_text}* — Interval: `{interval}`\n"]

    if not top_filtered:
        message_lines.append(f"No {type} found with change beyond threshold.")
    else:
        for symbol, increase, price_change_percent in top_filtered:
            message_lines.append(f"• `{symbol}`: {increase:.2f}% (Δ {price_change_percent:.2f}%)")

    message = "\n".join(message_lines)
    message_translated = translate(message, target_lang)
    await send_bot_message(token, message_translated)

    return top_filtered

