import numpy as np
import pandas as pd
import time
import urllib.parse
from dotenv import load_dotenv
from base58 import b58decode
from base64 import urlsafe_b64encode
import os
import joblib
from app.models.bucket import download_model, is_model_fresh
from app.models.features import get_language
from deep_translator import GoogleTranslator
import requests
import redis.asyncio as redis
import logging
from app.models.sendBotMessage import send_bot_message
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from sklearn.preprocessing import MinMaxScaler
import threading
import telebot
import json
from scipy.signal import find_peaks


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

# ✅ Orderly API Config
BASE_URL = os.getenv("ORDERLY_BASE_URL")
ORDERLY_ACCOUNT_ID = os.getenv("ORDERLY_ACCOUNT_ID")
ORDERLY_SECRET = os.getenv("ORDERLY_SECRET")
ORDERLY_PUBLIC_KEY = os.getenv("ORDERLY_PUBLIC_KEY")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", 10))
DEEP_SEEK_API_KEY = os.getenv("DEEP_SEEK_API_KEY")

# Initialize Redis connection
try:
    redis_client = redis.from_url(os.getenv("REDIS_URL"))
    redis_client.ping()
except redis.ConnectionError as e:
    print(f"Redis connection error: {e}")
    redis_client = None


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

# Translate function to convert text to the target language
def translate(text, target_lang):
    if target_lang == 'en':
        return text  # Return original text if translation fails
    else:
        return GoogleTranslator(source='auto', target=target_lang).translate(text)  


# ✅ Fetch historical Orderly data with global rate limiting
rate_limiter = RateLimiter(max_calls=10, period=1)
def fetch_historical_orderly(symbol, interval):
    rate_limiter()

    timestamp = str(int(time.time() * 1000))
    params = {"symbol": symbol, "type": interval, "limit": 100}  # Get exactly 100 candles
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
        
        # Sort by index (timestamp) in ascending order
        return df.sort_index(ascending=True).tail(100)

    
    return None


# Function to create dataset suitable for XGBRegressor
def create_rf_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back)])
        Y.append(dataset[i + look_back])
    return np.array(X), np.array(Y)

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


def detect_zigzag_pivots(close_prices, distance=5, prominence=1.0):
    """
    Detect zigzag turning points from close prices using peak detection.

    Returns:
        List of tuples (index, price)
    """
    peaks, _ = find_peaks(close_prices, distance=distance, prominence=prominence)
    troughs, _ = find_peaks(-close_prices, distance=distance, prominence=prominence)

    pivots = [(i, close_prices[i]) for i in peaks]
    pivots += [(i, close_prices[i]) for i in troughs]
    pivots = sorted(pivots, key=lambda x: x[0])  # sort by time

    return pivots

def is_valid_elliott_structure_from_pivots(data, verbose=False):
    """
    Scan recent pivots to identify the best-fitting Elliott Wave pattern (1–5).
    Returns:
        dict with:
            - valid: bool
            - score: float
            - direction: 'bullish' | 'bearish'
            - pivots: list of (index, price)
    """
    closes = data['close'].values
    all_pivots = detect_zigzag_pivots(closes, distance=3, prominence=0.5)

    if len(all_pivots) < 6:
        if verbose:
            print("❌ Not enough pivot points to identify a structure.")
        return {"valid": False, "score": 0.0, "direction": None, "pivots": all_pivots}

    best_score = 0.0
    best_result = None

    # Slide over the last N pivots (max 30) in windows of 6
    max_pivots = min(len(all_pivots), 30)
    for i in range(max_pivots - 5):
        p = all_pivots[i:i+6]
        prices = [pt[1] for pt in p]
        direction = "bullish" if prices[1] > prices[0] else "bearish"
        score = 1.0

        if direction == "bullish":
            wave1 = prices[1] - prices[0]
            wave2 = prices[1] - prices[2]
            wave3 = prices[3] - prices[2]
            wave4 = prices[3] - prices[4]
            wave5 = prices[5] - prices[4]

            if prices[2] < prices[0]:
                score -= 0.2
            if wave3 < wave1 * 0.8:
                score -= 0.3
            if prices[4] < prices[1] * 0.98:
                score -= 0.2
            if wave5 < wave3 * 0.5:
                score -= 0.1

        else:  # bearish
            wave1 = prices[0] - prices[1]
            wave2 = prices[2] - prices[1]
            wave3 = prices[2] - prices[3]
            wave4 = prices[3] - prices[4]
            wave5 = prices[4] - prices[5]

            if prices[2] > prices[0]:
                score -= 0.2
            if wave3 < wave1 * 0.8:
                score -= 0.3
            if prices[4] > prices[1] * 1.02:
                score -= 0.2
            if wave5 < wave3 * 0.5:
                score -= 0.1

        score = max(min(score, 1.0), 0.0)

        if verbose:
            print(f"Tested structure at pivots {i}-{i+5}: Score = {score:.2f}")

        if score > best_score:
            best_score = score
            best_result = {
                "valid": score >= 0.6,
                "score": score,
                "direction": direction,
                "pivots": p
            }

    return best_result if best_result else {
        "valid": False, "score": 0.0, "direction": None, "pivots": all_pivots
    }




# Function to analyze multiple intervals and return explanations using XGBRegressor
async def analyze_intervals(asset, token, interval, target_lang):
    look_back = 60
    future_steps = 10
    analysis_translated = None

    cache_key = f"analyze_intervals_analysis:{asset}:{interval}"
    
    # --- Try retrieving from Redis ---
    cached = await redis_client.get(cache_key)
    if cached:
        await send_bot_message(token, cached.decode())
        return cached.decode()

    MODEL_KEY = f'Mockba/elliot_waves_trained_models/{asset}_{interval}_elliot_waves_model.joblib'
    local_model_path = f'temp/elliot_waves_trained_models/{asset}_{interval}_elliot_waves_model.joblib'
    model_downloaded = False

    # ✅ Check local model freshness or download
    if not is_model_fresh(local_model_path):
        if download_model(BUCKET_NAME, MODEL_KEY, local_model_path):
            model_downloaded = True
        else:
            message = f"❌ Model not found for {asset} {interval}"
            translated_message = translate(message, target_lang)
            await send_bot_message(token, translated_message)
            return translated_message
    else:
        model_downloaded = True

    # ✅ Now always run analysis regardless of fresh or just-downloaded model
    try:
        data = fetch_historical_orderly(asset, interval)

        # 🔍 Wave structure sanity check
        wave_result = is_valid_elliott_structure_from_pivots(data, verbose=True)
        if not wave_result["valid"]:
            message = (
                f"⚠️ Weak or invalid Elliott structure for {asset} ({interval})\n"
                f"Score: {wave_result['score']:.2f} | Direction: {wave_result['direction'] or 'unknown'}"
            )
            await send_bot_message(token, translate(message, target_lang))
            # Optional: return here to skip ML prediction if invalid

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

        data_json = json.dumps(data.to_dict(orient='records'))
        future_predictions_json = json.dumps(future_predictions.tolist())
        predicted_labels_json = json.dumps(predicted_labels.tolist())
        language = get_language(target_lang)

        prompt = f"""
        **Language:** Respond in {language} language.
        **Task:** Generate a professional Elliott Wave analysis for {asset} ({interval}) with ML confirmation, optimized for Telegram traders.

        ###Input Data:
        1. Price Action:
        {data_json}

        2. ML Signals:
        - Trend: {future_predictions_json}
        - Confidence: {predicted_labels_json}

        ###Analysis Rules:
        Explanation
        1 [2-3 sentences explaining the analysis]

        2. Wave Validation:
        - ✅ Valid if:
            - Wave 2 stays above Wave 1 start
            - Wave 3 is longest impulse wave
            - Wave 4 doesn't enter Wave 1 territory
        - ❌ Invalid if any rule breaks

        3. ML Integration:
        - Highlight confidence-weighted conflicts
        - Flag high-probability reversals

        Output Format:
        🌊 {asset} {interval} | Elliot Waves Analysis

        🔍 Pattern Status: 🟢 Valid | 🔴 Invalid  
        - Wave 1: [Start] → [End]  
        - Wave 2: Held at [Price] (X% pullback)  
        - Wave 3: [Current] → [Target]  
        - Next Phase: [Wave 4/5 or A-B-C]  

        📊 Key Levels:  
        - Buy Zone: [Level]  
        - Take Profit: [Level]  
        - Stop Loss: [Level]  

        🤖 ML Cross-Check:  
        - Trend: [Bullish/Bearish] (X% confidence)  
        - Alert: [None/"Warning: ML contradicts Wave 5"]  

        💬 Insight:  
        "The current wave structure suggests [continuation/reversal] is likely, with ML providing [strong/weak] confirmation. The critical level to watch is [Price], where we expect [description of expected price action]. This creates a [high/medium] probability trading opportunity."

        🚀 Final Verdict:  
        "Based on the wave pattern and ML alignment, the recommended action is to [specific instruction] with defined risk management at [Level]. The next confirmation signal would be [price/condition], expected within [timeframe]."

        Rules:
        1. No markdown (**, `, ###, etc)
        2. Use simple dashes (-) for bullets
        3. Keep decimals consistent (2 places)
        """

        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
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
                "temperature": 0.3
            },
            headers={"Authorization": f"Bearer {DEEP_SEEK_API_KEY}"}
        )

        if response.status_code == 200:
            analysis = response.json()["choices"][0]["message"]["content"]
            await redis_client.set(cache_key, analysis, ex=1200)
            await send_bot_message(token, analysis)
        else:
            print(f"Error: {response.status_code}, {response.text}")

    except Exception as e:
        logger.error(f"Error processing {interval}: {e}")
        await send_bot_message(token, translate(f"An error occurred while analyzing {interval} interval: {e}", token))

    finally:
        if model_downloaded and not is_model_fresh(local_model_path):
            os.remove(local_model_path)

    return analysis

    
               


# Example usage
# if __name__ == "__main__":
#     symbol = 'BTCUSDT'
#     intervals = ['1h', '4h', '1d']
#     token = "556159355"
#     explanations = analyze_intervals(symbol, intervals, token)
#     for interval, details in explanations.items():
#         print(f"{interval} interval explanation:\n{details['explanation']}")
#         print(f"TradingView Link: {details['tradingview_link']}")
