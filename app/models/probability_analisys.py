import time
import sys
import os
import json
import pandas as pd
import numpy as np
import urllib.parse
from dotenv import load_dotenv
import joblib  # Library for model serialization
import requests
import redis.asyncio as redis
import logging
from app.models.bucket import download_model
import json
from app.models.sendBotMessage import send_bot_message
from app.models.features import get_features_by_indicator, get_language
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore")
# Load environment variables from the .env file
load_dotenv(dotenv_path=".env.micro.trading")


BUCKET_NAME = os.getenv("BUCKET_NAME")  # Your bucket name
MICRO_CENTRAL_URL = os.getenv("MICRO_CENTRAL_URL")  # Your micro central URL

def translate(text, target_lang):
    if target_lang == 'en':
        return text  # Return original text if translation fails
    else:
        return GoogleTranslator(source='auto', target=target_lang).translate(text)  


# Initialize Redis connection
try:
    redis_client = redis.from_url(os.getenv("REDIS_URL"))
    redis_client.ping()
except redis.ConnectionError as e:
    print(f"Redis connection error: {e}")
    redis_client = None
 

# ✅ Fetch historical Orderly data with global rate limiting
rate_limiter = RateLimiter(max_calls=8, period=1)  # 8 calls per second
def fetch_historical_orderly(symbol, interval):
    rate_limiter()
    timestamp = str(int(time.time() * 1000))
    params = {"symbol": symbol, "type": interval, "limit": 1000}  # Get exactly 100 candles
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
        
        return df.sort_index(ascending=True).head(1000)
    
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
    # If 'bollinger_hband' or 'bollinger_lband' is in required_features, ensure they are added with default window=20 if not already present
    bollinger_features = [f for f in required_features if f.startswith("bollinger_")]
    if ("bollinger_hband" in required_features or "bollinger_lband" in required_features) and not any(f.startswith("bollinger_hband_") or f.startswith("bollinger_lband_") for f in required_features):
        window = 20
        data['bollinger_mavg'] = data['close'].rolling(window=window).mean()
        data['bollinger_std'] = data['close'].rolling(window=window).std()
        data['bollinger_hband'] = data['bollinger_mavg'] + (data['bollinger_std'] * 2)
        data['bollinger_lband'] = data['bollinger_mavg'] - (data['bollinger_std'] * 2)
    for feature in bollinger_features:
        try:
            # If the feature has a window, extract it, else skip (already handled above)
            parts = feature.split("_")
            if len(parts) > 2:
                window = int(parts[-1])
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

class ProbabilityEngine:
    def __init__(self, asset_info):
        self.base_imr = asset_info['base_imr']  # Initial Margin (e.g., 10%)
        self.base_mmr = asset_info['base_mmr']  # Maintenance Margin (e.g., 5%)
        self.liq_fee = asset_info['std_liquidation_fee']  # Liquidation penalty
        
    def calculate_scenarios(self, data, lookahead=5):
        """Bidirectional scenario probabilities (0.5% moves)"""
        # Long scenario: 0.5% up before 0.25% down
        data['long_success'] = (
            (data['close'].rolling(lookahead).max() / data['close'] >= 1.005) & 
            (data['close'].rolling(lookahead).min() / data['close'] > 0.9975
        ).astype(int))
        
        # Short scenario: 0.5% down before 0.25% up
        data['short_success'] = (
            (data['close'].rolling(lookahead).min() / data['close'] <= 0.995) & 
            (data['close'].rolling(lookahead).max() / data['close'] < 1.0025
        ).astype(int))
        return data

    def dynamic_kelly(self, prob_win, reward_risk_ratio, leverage, funding_rate=0):
        """Perpetual-optimized Kelly sizing with leverage constraints
        Args:
            prob_win: Probability of winning the trade (0-1)
            reward_risk_ratio: Expected reward/risk ratio (e.g. 2.0 for 2:1)
            leverage: User's maximum allowed leverage
            funding_rate: Current funding rate (optional)
        Returns:
            Optimal position size as fraction of capital (0-1)
        """
        # Base Kelly formula
        raw_kelly = (prob_win * (reward_risk_ratio + 1) - 1) / reward_risk_ratio
        
        # Funding rate adjustment
        if funding_rate < 0:
            raw_kelly *= (1 + funding_rate/0.02)  # Reduce size in negative funding
        
        # Exchange maximum leverage based on IMR
        exchange_max_leverage = 1 / self.base_imr
        
        # Apply three constraints:
        return min(
            max(raw_kelly, 0.01),  # Minimum 1% position
            leverage,               # User's selected max leverage
            exchange_max_leverage * 0.8  # 80% of exchange max
        )

    def monte_carlo_liquidation(self, price, atr, position_size, leverage, n_sims=2000):
        """Simulate liquidation risk under volatility shocks with position size impact
        
        Args:
            price: Current asset price
            atr: Average True Range (volatility measure)
            position_size: Fraction of capital being risked (0-1)
            leverage: Account leverage multiplier
            n_sims: Number of Monte Carlo simulations
            
        Returns:
            Probability of liquidation (0-1)
        """
        # Convert position size to effective exposure
        exposure_multiplier = 1 + (position_size * 2)  # [1-3] range
        
        # Adjusted volatility based on position size
        daily_vol = (atr / price) * exposure_multiplier
        
        # Liquidation price calculation
        liq_price = price * (1 - (self.base_imr - self.base_mmr)/leverage)
        
        # Monte Carlo simulation with volatility clustering
        shocks = np.random.normal(0, daily_vol, n_sims)
        price_paths = price * (1 + shocks)
        
        # Count liquidation events
        liq_events = np.sum(price_paths <= liq_price)
        
        return min(liq_events / n_sims, 0.99)  # Cap at 99% for numerical stability

    def funding_adjustment(self, prob, funding_rate, cap=0.02):
        """Adjust probability based on funding regime"""
        impact = abs(funding_rate) / cap  # Normalized 0-1
        return prob * (1 - impact) if funding_rate < 0 else prob * (1 + 0.3*impact)
    
    def calculate_advanced_scenarios(self, data, lookahead=5):
        """Calculate 6 key trading scenarios with varying targets"""
        scenarios = {
            # Long scenarios
            'long_03': (data['close'].rolling(lookahead).max() / data['close'] >= 1.003),
            'long_05': (data['close'].rolling(lookahead).max() / data['close'] >= 1.005),
            'long_10': (data['close'].rolling(lookahead).max() / data['close'] >= 1.01),
            'long_15': (data['close'].rolling(lookahead).max() / data['close'] >= 1.015),
            'long_20': (data['close'].rolling(lookahead).max() / data['close'] >= 1.02),
            
            # Short scenarios
            'short_03': (data['close'].rolling(lookahead).min() / data['close'] <= 0.997),
            'short_05': (data['close'].rolling(lookahead).min() / data['close'] <= 0.995),
            'short_10': (data['close'].rolling(lookahead).min() / data['close'] <= 0.99),
            'short_15': (data['close'].rolling(lookahead).min() / data['close'] <= 0.985),
            'short_20': (data['close'].rolling(lookahead).min() / data['close'] <= 0.98),
            
            # Stop-hit scenarios
            'long_stop': (data['close'].rolling(lookahead).min() / data['close'] <= 0.9925),
            'short_stop': (data['close'].rolling(lookahead).max() / data['close'] >= 1.0075)
        }
        
        # Calculate conditional probabilities
        data['long_03_prob'] = scenarios['long_03'].astype(int)
        data['long_05_prob'] = scenarios['long_05'].astype(int)
        data['long_10_prob'] = (scenarios['long_10'] & ~scenarios['long_stop']).astype(int)
        data['long_15_prob'] = (scenarios['long_15'] & ~scenarios['long_stop']).astype(int)
        data['long_20_prob'] = (scenarios['long_20'] & ~scenarios['long_stop']).astype(int)
        
        data['short_03_prob'] = scenarios['short_03'].astype(int)
        data['short_05_prob'] = scenarios['short_05'].astype(int)
        data['short_10_prob'] = (scenarios['short_10'] & ~scenarios['short_stop']).astype(int)
        data['short_15_prob'] = (scenarios['short_15'] & ~scenarios['short_stop']).astype(int)
        data['short_20_prob'] = (scenarios['short_20'] & ~scenarios['short_stop']).astype(int)
        
        return data
    
    def calculate_confidence(self, proba_df, current_funding):
        """5-tier confidence system with funding adjustment"""
        raw_confidence = proba_df.max(axis=1).iloc[-1]
        
        # Funding rate impact (reduce confidence in adverse funding)
        funding_impact = 1 - min(abs(current_funding)/0.02, 0.3)  # Max 30% reduction
        adjusted_confidence = raw_confidence * funding_impact
        
        # Classification
        if adjusted_confidence > 0.8:
            return "🔥 Very High", int(adjusted_confidence*100)
        elif adjusted_confidence > 0.7:
            return "✅ High", int(adjusted_confidence*100)
        elif adjusted_confidence > 0.6:
            return "⚠️ Moderate", int(adjusted_confidence*100)
        elif adjusted_confidence > 0.5:
            return "🛑 Low", int(adjusted_confidence*100)
        else:
            return "❌ Very Low", int(adjusted_confidence*100)


async def get_funding_rate(symbol):
    """Fetch current funding rate for perpetual contract"""
    cache_key = f"funding_rate:{symbol}"
    cache_ttl = 300  # 5 minutes
    
    # Check Redis cache first
    if redis_client:
        cached_rate = await redis_client.get(cache_key)
        if cached_rate:
            return float(cached_rate)
    
    # API request with rate limiting
    rate_limiter()
    url = f"{BASE_URL}/v1/public/funding_rate/{symbol}"
    response = requests.get(url)
    
    if response.status_code == 200:
        funding_rate = float(response.json()['data']['est_funding_rate'])
        
        # Cache the result
        if redis_client:
            await redis_client.setex(cache_key, cache_ttl, str(funding_rate))
            
        return funding_rate
    else:
        print(f"⚠️ Failed to fetch funding rate: {response.text}")
        return 0.0  # Default neutral rate

        
async def analize_probability_asset(token, asset, interval, feature, leverage, target_lang, free_colateral, market_bias='neutral'):
    print('Getting data for analysis') 
    features = get_features_by_indicator(interval, feature)
    analysis_translated = None
    model_name = "_".join(features).replace("[", "").replace("]", "").replace("'", "_").replace(" ", "")
    MODEL_KEY = f'Mockba/trained_models/trained_model_{asset}_{interval}_{model_name}.joblib'
    local_model_path = f'temp/trained_model_{asset}_{interval}_{model_name}.joblib'

    cache_key = f"analize_probability_asset_analysis:{asset}:{interval}"
    
    # --- Try retrieving from Redis ---
    cached = await redis_client.get(cache_key)
    if cached:
        analysis_translated = translate(cached.decode(), target_lang)
        await send_bot_message(token, analysis_translated)
        return cached.decode()

    if download_model(BUCKET_NAME, MODEL_KEY, local_model_path):
        try:
            # --- Data Preparation ---
            data = fetch_historical_orderly(asset, interval)
            if data is None or data.empty:
               print(f"❌ No historical data returned for {asset} {interval}")
               return None
                
            asset_info = await fetch_asset_info(asset)
            prob_engine = ProbabilityEngine(asset_info)
            current_funding = await get_funding_rate(asset)
            
            # --- Model Predictions ---
            model_metadata = joblib.load(local_model_path)
            model = model_metadata["model"]
            used_features = model_metadata.get("used_features", [])

            missing_features = [f for f in used_features if f not in data.columns]
            if missing_features:
                data = add_indicators(data, missing_features)

            # Dynamically extract indicators only if available
            rsi_value = data['rsi_14'].iloc[-1] if 'rsi_14' in features and 'rsi_14' in data.columns else 'N/A'
            macd_latest = data['macd'].iloc[-1] if 'macd' in features and 'macd' in data.columns else None
            macd_signal_latest = data['macd_signal'].iloc[-1] if 'macd_signal' in features and 'macd_signal' in data.columns else None
            macd_status = (
                'Bullish' if macd_latest is not None and macd_signal_latest is not None and macd_latest > macd_signal_latest
                else 'Bearish' if macd_latest is not None and macd_signal_latest is not None
                else 'N/A'
            )

            # Build indicator snapshot for prompt
            indicator_snapshot = ""
            if 'rsi_14' in features:
                indicator_snapshot += f"• RSI(14): {rsi_value}\n"
            if 'macd' in features and 'macd_signal' in features:
                indicator_snapshot += f"• MACD: {macd_latest:.2f} vs Signal: {macd_signal_latest:.2f} → {macd_status}\n"
            indicator_snapshot += f"• Volume (latest): {data['volume'].iloc[-1]:,.0f}"

            # Build guidance string
            confirmation_guidance = ""
            if 'rsi_14' in features:
                confirmation_guidance += "- If RSI is below 30, do NOT recommend 'RSI > 60/70' as confirmation.\n"
            if 'macd' in features and 'macd_signal' in features:
                confirmation_guidance += "- If MACD is bearish, do not suggest bullish divergence.\n"
            confirmation_guidance += "- Reference only actual values shown in the snapshot."

            data = data[used_features].dropna().copy()

            y_proba = model.predict_proba(data[features])
            proba_df = pd.DataFrame(y_proba, columns=model.classes_)

            data = prob_engine.calculate_advanced_scenarios(data)
            scenario_probs = {
                'long': {k: data[f'long_{k.replace(".", "").replace("%", "")}_prob'].mean() for k in ['0.3%', '0.5%', '1.0%', '1.5%', '2.0%']},
                'short': {k: data[f'short_{k.replace(".", "").replace("%", "")}_prob'].mean() for k in ['0.3%', '0.5%', '1.0%', '1.5%', '2.0%']}
            }


            kelly_sizes = {
                side: {
                    k: prob_engine.dynamic_kelly(prob, float(k.strip('%')) * 5, leverage, current_funding)
                    for k, prob in probs.items()
                } for side, probs in scenario_probs.items()
            }

            confidence_level, confidence_score = prob_engine.calculate_confidence(proba_df, current_funding)
            liq_risk = prob_engine.monte_carlo_liquidation(
                price=data['close'].iloc[-1],
                atr=data['atr_14'].iloc[-1],
                position_size=max(max(kelly_sizes['long'].values()), max(kelly_sizes['short'].values())),
                leverage=leverage
            )

            target_fraction = {-1: 0.35, 1: 0.15, 0: 0.5}
            if market_bias == 'bullish':
                target_fraction = {-1: 0.2, 1: 0.3, 0: 0.5}
            elif market_bias == 'bearish':
                target_fraction = {-1: 0.4, 1: 0.1, 0: 0.5}

            n_samples = len(proba_df)
            target_counts = {cls: int(n_samples * frac) for cls, frac in target_fraction.items()}
            y_custom = np.zeros(n_samples, dtype=int)
            for cls in [-1, 1]:
                top_indices = proba_df[cls].nlargest(target_counts[cls]).index
                y_custom[top_indices] = cls
            data['predicted'] = y_custom
            current_prediction = data['predicted'].iloc[-1]

            # get the language for the bot
            language = get_language(target_lang)

            prompt = f"""
            **Language:** Respond in {language} language.
            🌟 *Advanced Trading Signal - {asset} {interval}* 🌟
            💰 Free Collateral: ${free_colateral:,.2f} | ⚖️ Leverage: {leverage}x

            🔷 *Price Action*
            📈 Last Price: ${data['close'].iloc[-1]:,.2f}
            🛡️ Support: ${data['low'].tail(10).min():,.2f}
            🚀 Resistance: ${data['high'].tail(10).max():,.2f}

            🎯 *Probability Matrix*
            👉 *LONG Targets*:
            {chr(10).join([f"{k} → {scenario_probs['long'][k]:.1%} | 🎯 Size: {kelly_sizes['long'][k]:.2f}x" for k in scenario_probs['long']])}

            👉 *SHORT Targets*:
            {chr(10).join([f"{k} → {scenario_probs['short'][k]:.1%} | 🎯 Size: {kelly_sizes['short'][k]:.2f}x" for k in scenario_probs['short']])}

            ⚡ *Signal*: {'🟢 STRONG LONG' if current_prediction == 1 else '🔴 STRONG SHORT' if current_prediction == -1 else '🟡 NEUTRAL'} 
            📊 Confidence: {confidence_score}% {'✅ High' if confidence_score > 70 else '⚠️ Medium' if confidence_score > 50 else '❌ Low'}
            🔄 Market Bias: {market_bias.upper()} {'🐂' if market_bias == 'bullish' else '🐻' if market_bias == 'bearish' else '🦉'}

            💎 *Strategic Recommendation*
            {"🚀 AGGRESSIVE LONG" if scenario_probs['long']['1.0%'] > 0.7 else "🎯 PREFER SHORTS" if scenario_probs['short']['0.5%'] > 0.7 else "🛑 WAIT FOR BETTER ENTRY"}

            📌 *Indicator Snapshot*
            {indicator_snapshot}

            💡 *Model Instruction*  
            Use the indicator snapshot above when suggesting confirmation conditions.  
            {confirmation_guidance}

            ⚠️ *Risk Assessment*
            💀 Liquidation Risk: {liq_risk:.1%} {'(High)' if liq_risk > 0.3 else '(Medium)' if liq_risk > 0.15 else '(Low)'}
            💸 Funding Rate: {current_funding*100:+.2f}% {'(Costly)' if current_funding > 0.0005 else '(Favorable)' if current_funding < -0.0002 else '(Neutral)'}
            🛡️ Max Safe Leverage: {min(1/asset_info['base_imr'], leverage):.1f}x

            🔔 *Final Notes*
            • Next update in: {interval}
            • Max position: {min(max(kelly_sizes['long'].values()) + max(kelly_sizes['short'].values()), leverage):.1f}x
            • 📅 Data points: {len(data)} samples
            • ⚠️ Always use stop-loss!
            """

            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {
                            "role": "system",
                            "content": "Respond in this exact structure:\n\n📊 DIRECTIONAL EDGE ANALYSIS\n[LONG/SHORT] | Best Target: X.X% Move\n┌─────────┬────────────┬────────────┬────────────┐\n│ Target  │ Win Prob   │ Kelly Size │ Edge       │\n├─────────┼────────────┼────────────┼────────────┤\n│ 0.3%    │ XX.X%      │ X.XXx      │ [✅/⚠️/❌] │\n│ 0.5%    │ XX.X%      │ X.XXx      │ [✅/⚠️/❌] │\n└─────────┴────────────┴────────────┴────────────┘\n\n🎯 EXECUTION SUMMARY\n• Preferred Direction: [STRONG LONG/PREFER SHORTS/NEUTRAL]\n• Entry: $XXX-$XXX (Confirm with: [Indicator1+Indicator2])\n• Exit: Take-Profit at X.X% (X.X% position), Stop-Loss at $XXX\n\n⚠️ RISK PROFILE\n• Max Position: X.X% of capital\n• Liquidation Risk: X.X%\n• Funding Impact: ±X.XX%"
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.2
                },
                headers={"Authorization": f"Bearer {DEEP_SEEK_API_KEY}"}
            )

            if response.status_code == 200:
                analysis = response.json()["choices"][0]["message"]["content"]

                # Store in Redis for 20 minutes (1200 seconds)
                await redis_client.set(cache_key, analysis, ex=1200)

                await send_bot_message(token, analysis)
                print("✅ Analysis sent successfully!")
            else:
                print(f"❌ Error in API response: {response.status_code} - {response.text}")
                analysis_translated = translate(f"❌ Error API response: {response.status_code} - {response.text}", target_lang)
                await send_bot_message(token, analysis_translated)   

        except Exception as e:
            logger.error(f"Error processing {interval}: {e}")
            await send_bot_message(token, translate(f"An error occurred while analyzing {interval} interval: {e}", token))         
          
        finally:
            if os.path.exists(local_model_path):
                os.remove(local_model_path)

        return analysis_translated
    else:
        message = f"❌ Model not found for {asset} {interval} with features {features}"
        translated_message = translate(message, target_lang)
        await send_bot_message(token, translated_message)
        return translate(f"❌ Model not found for {asset} {interval} with features {features}", target_lang)
