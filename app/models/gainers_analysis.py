import os
import time
import sys
import requests
import pandas as pd
import urllib.parse
from app.models.sendBotMessage import send_bot_message  # Assuming this is the correct import path
import redis
import json
from base58 import b58decode
from base64 import urlsafe_b64encode
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from deep_translator import GoogleTranslator
import threading
import logging
from dotenv import load_dotenv
import numpy as np
from typing import List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(dotenv_path=".env.micro.trading")

# Redis Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.from_url(REDIS_URL)

# Orderly API Config
BASE_URL = os.getenv("ORDERLY_BASE_URL")
ORDERLY_ACCOUNT_ID = os.getenv("ORDERLY_ACCOUNT_ID")
ORDERLY_SECRET = os.getenv("ORDERLY_SECRET")
ORDERLY_PUBLIC_KEY = os.getenv("ORDERLY_PUBLIC_KEY")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", 10))
MICRO_CENTRAL_URL = os.getenv("MICRO_CENTRAL_URL")

# Initialize private key
if ORDERLY_SECRET.startswith("ed25519:"):
    ORDERLY_SECRET = ORDERLY_SECRET.replace("ed25519:", "")
private_key = Ed25519PrivateKey.from_private_bytes(b58decode(ORDERLY_SECRET))

# Rate Limiter
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
                time.sleep(sleep_time)
            self.calls.append(time.time())

rate_limiter = RateLimiter(max_calls=10, period=1)

# Core Functions
def fetch_order_book_snapshot(symbol: str) -> Optional[pd.DataFrame]:
    rate_limiter()
    timestamp = str(int(time.time() * 1000))
    path = f"/v1/orderbook/{symbol}"
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
        logger.error(f"Error fetching order book for {symbol}: {response.json()}")
        return None

    data = response.json().get("data", {})
    asks = data.get("asks", [])
    bids = data.get("bids", [])

    df_asks = pd.DataFrame(asks)
    df_bids = pd.DataFrame(bids)

    if not df_asks.empty:
        df_asks["side"] = "ask"
    if not df_bids.empty:
        df_bids["side"] = "bid"

    return pd.concat([df_bids, df_asks], ignore_index=True)

def fetch_historical_orderly(symbol: str, interval: str) -> Optional[pd.DataFrame]:
    rate_limiter()
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
        logger.error(f"Error fetching data for {symbol} {interval}: {response.json()}")
        return None

    data = response.json().get("data", {})
    if not data or "rows" not in data:
        return None

    df = pd.DataFrame(data["rows"])
    required_columns = ["start_timestamp", "open", "high", "low", "close", "volume"]
    if set(required_columns).issubset(df.columns):
        df['start_timestamp'] = pd.to_datetime(df['start_timestamp'], unit='ms')
        df.set_index('start_timestamp', inplace=True)
        return df[["open", "high", "low", "close", "volume"]]
    return None

# Movement Analysis
def calculate_price_change(df: pd.DataFrame) -> float:
    if len(df) < 2:
        return 0.0
    current = df['close'].iloc[-1]
    initial = df['close'].iloc[0]
    return ((current - initial) / initial) * 100

def get_all_symbols() -> List[str]:
    orderly_symbols = "https://api-evm.orderly.org/v1/public/info"
    response = requests.get(orderly_symbols)
    if response.status_code != 200:
        logger.error(f"Error fetching symbols: {response.json()}")
        return []
    return [row["symbol"] for row in response.json().get("data", {}).get("rows", []) if "symbol" in row]

def process_symbol_movement(symbol: str, interval: str, threshold: float) -> Optional[Tuple[str, float, float, float]]:
    try:
        df = fetch_historical_orderly(symbol, interval)
        if df is None or len(df) < 2:
            return None
            
        price_change = calculate_price_change(df)
        if abs(price_change) >= abs(threshold):
            return (
                symbol,
                df['close'].iloc[-1],
                df['close'].iloc[0],
                price_change
            )
    except Exception as e:
        logger.error(f"Error processing {symbol}: {e}")
    return None

def get_movements_concurrently(interval: str = '1h', change_threshold: float = 0) -> List:
    symbols = get_all_symbols()
    if not symbols:
        return []

    results = []
    for symbol in symbols:
        result = process_symbol_movement(symbol, interval, change_threshold)
        if result:
            results.append(result)

    return results

async def calculate_movements(raw_data: List, threshold: float) -> List[Tuple[str, float, float]]:
    movements = []
    for symbol, current_price, initial_price, price_change in raw_data:
        if price_change is None:
            continue
            
        percentage_change = ((current_price - initial_price) / initial_price) * 100
        
        if abs(percentage_change) >= abs(threshold):
            movements.append((symbol, percentage_change, price_change))
    
    return sorted(movements, key=lambda x: abs(x[1]), reverse=True)

async def fetch_and_analyze_movements(token: str, interval: str = '1h', change_threshold: float = 0) -> List[Tuple[str, float, float]]:
    redis_key = f"price_movements:{token}:{interval}"
    
    try:
        if cached := redis_client.get(redis_key):
            return json.loads(cached)
    except redis.RedisError as e:
        logger.error(f"Redis error: {e}")

    raw_movements = get_movements_concurrently(interval, change_threshold)
    processed_movements = await calculate_movements(raw_movements, change_threshold)
    
    try:
        redis_client.setex(redis_key, 900, json.dumps(processed_movements))
    except redis.RedisError as e:
        logger.error(f"Redis cache error: {e}")
    
    return processed_movements

async def analyze_movements(
    token: str,
    target_lang: str,
    interval: str = '1h',
    change_threshold: float = 0,
    movement_type: str = 'gainers',
    top_n: int = 10
) -> List[Tuple[str, float, float]]:
    
    cache_key = f"gainers_losers_analysis:{movement_type}"
    
    # --- Try retrieving from Redis ---
    cached = await redis_client.get(cache_key)
    if cached:
        translated = GoogleTranslator(source='auto', target=target_lang).translate(cached.decode())
        await send_bot_message(token, translated)
        return cached.decode()
    
    all_movements = await fetch_and_analyze_movements(token, interval, change_threshold)
    
    if movement_type == 'gainers':
        filtered = [m for m in all_movements if m[1] >= change_threshold]
        filtered.sort(key=lambda x: x[1], reverse=True)
        title_icon = "📈"
        title_text = "Top Gainers"
    elif movement_type == 'losers':
        filtered = [m for m in all_movements if m[1] <= -change_threshold]
        filtered.sort(key=lambda x: x[1])  # Most negative first
        title_icon = "📉"
        title_text = "Top Losers"
    else:
        raise ValueError("Movement type must be 'gainers' or 'losers'")

    top_movements = filtered[:top_n]
    
    if not top_movements:
        message = f"No significant {title_text.lower()} found (threshold: {change_threshold}%)"
    else:
        message_lines = [f"{title_icon} *{title_text}* — {interval} interval"]
        for symbol, change, interval_change in top_movements:
            direction = "+" if change > 0 else ""
            message_lines.append(f"• `{symbol.upper()}`: {direction}{change:.2f}%")
        message = "\n".join(message_lines)

    translated = GoogleTranslator(source='auto', target=target_lang).translate(message)
    # Store in Redis for 20 minutes (1200 seconds)
    await redis_client.set(cache_key, message, ex=1200)
    
    await send_bot_message(token, translated)
    
    return top_movements