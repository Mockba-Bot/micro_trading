from celery import shared_task, group
import sys
import os
import pandas as pd
import joblib
from datetime import datetime
import requests
from binance.client import Client
from binance.enums import *
import math
import time
from dotenv import load_dotenv
import telebot
import redis
import json
# Add the directory containing your modules to the Python path
sys.path.append('/app')

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost")
API_TOKEN = os.getenv("API_TOKEN")
bot = telebot.TeleBot(API_TOKEN)

from log_config import trader_logger, gainers_logger  # Import the two loggers

# Load environment variables from the .env file
load_dotenv(dotenv_path=".env.micro.trading")
client = Client(api_key=os.getenv("BINANCE_API_KEY"), api_secret=os.getenv("BINANCE_API_SECRET"))


@shared_task
def get_model_path(pair, timeframe):
    return f'trained_models/trained_model_{pair}_{timeframe}.pkl'
    

# Calculate the maximum investment amount based on the latest data
@shared_task(queue="trading")
def calculate_max_investment(df, investment_ratio=1.0, max_investment=350000):
    latest = df.iloc[-1]
    volume = latest['volume']
    close_price = latest['close']
    investment_amount = volume * close_price * investment_ratio
    investment_amount = min(investment_amount, max_investment)
    return investment_amount

# Add technical indicators to the data
from celery import shared_task
import pandas as pd

@shared_task(queue="trading")
def get_historical_data(token, pair, timeframe, limit=500):
    url = f"{MICRO_CENTRAL_URL}/query-historical-data-for-trade"
    payload = {
        "pair": pair,
        "timeframe": timeframe,
        "limit": limit
    }
    headers = {
        "Token": token
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        data = response.json()
        return pd.DataFrame.from_records(data).to_dict()
    else:
        error_text = response.text
        trader_logger.error(f"Error retrieving historical data: {response.status_code} {error_text}")
        return None       


@shared_task(queue="trading")
def add_indicators(data_dict):
    """
    Celery task to compute technical indicators for market data.

    Args:
        data_dict (dict): A dictionary representation of a Pandas DataFrame with OHLCV data.

    Returns:
        dict: The processed DataFrame converted to a dictionary format.
    """
    # Convert dictionary back to DataFrame
    data = pd.DataFrame.from_dict(data_dict)

    # Ensure the columns are of numeric type
    for col in ['close', 'high', 'low', 'volume']:
        data[col] = pd.to_numeric(data[col])

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

    # Convert DataFrame back to dictionary before returning (Celery-safe)
    return data.to_dict()


@shared_task(queue="trading")
def update_model(existing_model, new_data, features):
    new_data['return'] = new_data['close'].pct_change().shift(-1)
    new_data['target'] = (new_data['return'] > 0).astype(int)
    
    X_new = new_data[features].dropna()
    y_new = new_data['target'].dropna().loc[X_new.index]
    
    existing_model.fit(X_new, y_new)
    
    return existing_model

@shared_task(queue="trading")
def control_slippage(expected_price, execution_price, max_slippage_percent):
    slippage = abs(execution_price - expected_price) / expected_price
    if slippage > max_slippage_percent / 100:
        trader_logger.warning(f"Slippage too high: {slippage:.2%}")
        return False
    return True

@shared_task(queue="trading")
def place_market_order(client, symbol, side, quantity, max_slippage_percent):
    try:
        current_price = float(client.get_symbol_ticker(symbol=symbol)['price'])

        step_size = get_step_size(client, symbol)
        min_notional = get_min_notional(client, symbol)

        quantity = math.floor(quantity / step_size) * step_size
        quantity = round(quantity, int(-math.log10(step_size)))

        if control_slippage(current_price, current_price, max_slippage_percent):
            total_value = quantity * current_price
            if total_value < min_notional:
                trader_logger.warning(f"Order not placed: total value {total_value} is below the minimum notional {min_notional}.")
                return None

            if side == 'BUY':
                order = client.order_market_buy(
                    symbol=symbol,
                    quantity=quantity
                )
            elif side == 'SELL':
                order = client.order_market_sell(
                    symbol=symbol,
                    quantity=quantity
                )
            else:
                trader_logger.error(f"Invalid order side: {side}")
                return None

            trader_logger.info(f"Market order placed: {order}")
            return order
        else:
            trader_logger.warning("Order not placed due to high slippage before execution.")
            return None
    except Exception as e:
        trader_logger.error(f"Error placing market order: {e}")
        return None

@shared_task(queue="trading")
def get_min_notional(client, symbol):
    info = client.get_symbol_info(symbol)
    for f in info['filters']:
        if f['filterType'] == 'MIN_NOTIONAL':
            return float(f['minNotional'])
    return 0.0

@shared_task(queue="trading")
def get_step_size(client, symbol):
    info = client.get_symbol_info(symbol)
    for f in info['filters']:
        if f['filterType'] == 'LOT_SIZE':
            return float(f['stepSize'])
    return None

# Connect to Redis
# redis_client = redis.StrictRedis(host='localhost', port=6390, db=0)
try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    redis_client.ping()  # Check if the connection is successful
except redis.ConnectionError as e:
    print(f"Redis connection error: {e}")
    redis_client = None

@shared_task(queue="trading")
def get_base_and_quote_assets(symbol):
    cache_key = f"base_quote_assets_{symbol}"
    cache_expiration = 604800  # 7 days in seconds

    # Check if the result is already cached
    cached_result = redis_client.get(cache_key)
    if cached_result:
        # Return the cached result
        return json.loads(cached_result)

    # If not cached, make the API request
    url = f"https://api.binance.com/api/v3/exchangeInfo?symbol={symbol}"
    response = requests.get(url)

    if response.status_code == 200:
        symbol_info = response.json()
        if 'symbols' in symbol_info:
            symbol_data = symbol_info['symbols']
            for symbol_obj in symbol_data:
                if symbol_obj['symbol'] == symbol:
                    base_asset = symbol_obj['baseAsset']
                    quote_asset = symbol_obj['quoteAsset']
                    filters = symbol_obj['filters']
                    min_lot_size_filter = next((f for f in filters if f['filterType'] == 'LOT_SIZE'), None)
                    if min_lot_size_filter:
                        stepSize = float(min_lot_size_filter['stepSize'])
                    min_notional_filter = next((f for f in filters if f['filterType'] == 'NOTIONAL'), None)
                    if min_notional_filter:
                        min_notional = float(min_notional_filter['minNotional'])
                        max_notional = float(min_notional_filter.get('maxNotional', float('inf')))
                        result = (base_asset, quote_asset, stepSize, min_notional, max_notional)
                        
                        # Cache the result in Redis
                        redis_client.setex(cache_key, cache_expiration, json.dumps(result))
                        
                        return result
            trader_logger.error(f'Symbol not found: {symbol}')
            return None, None, None, None, None
    else:
        trader_logger.error(f'Failed to retrieve symbol info: {response.status_code}')
        return None, None, None, None, None


@shared_task(queue="trading")
def get_capital_accumulated(token, pair, timeframe):
    url = f"{MICRO_CENTRAL_URL}/capital/accumulated"
    payload = {
        "token": token,
        "pair": pair,
        "timeframe": timeframe
    }
    headers = {
        "Token": token
    }

    response = requests.get(url, json=payload, headers=headers)
    if response.status_code == 200:
        data = response.json()
        return data.get('capital_accumulated')
    else:
        error_text = response.text
        trader_logger.error(f"Error retrieving capital accumulated: {response.status_code} {error_text}")
        return None

@shared_task(queue="trading")    
def update_capital_accumulated(token, pair, timeframe, capital_accumulated):
    url = f"{MICRO_CENTRAL_URL}/capital/accumulated"
    payload = {
        "token": token,
        "pair": pair,
        "timeframe": timeframe,
        "capital_accumulated": capital_accumulated
    }
    headers = {
        "Token": token
    }

    response = requests.put(url, json=payload, headers=headers)
    if response.status_code == 200:
        trader_logger.info(f"Capital accumulated updated for token {token}, pair {pair}, and timeframe {timeframe}")
    else:
        error_text = response.text
        trader_logger.error(f"Error updating capital accumulated: {response.status} {error_text}")

@shared_task(queue="trading") 
def store_capital(token, pair, timeframe, capital, crypto_amount, timestamp, cumulative_strategy_return, cumulative_market_return, first_trade, last_price=0.0):
    url = f"{MICRO_CENTRAL_URL}/capital"
    payload = {
        "token": token,
        "pair": pair,
        "timeframe": timeframe,
        "capital": capital,
        "crypto_amount": crypto_amount,
        "timestamp": timestamp,
        "cumulative_strategy_return": cumulative_strategy_return,
        "cumulative_market_return": cumulative_market_return,
        "first_trade": first_trade,
        "last_price": last_price
    }
    headers = {
        "Token": token
    }

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        trader_logger.info(f"Capital and crypto amount stored for token {token}, pair {pair}, and timeframe {timeframe}")
    else:
        error_text = response.text
        trader_logger.error(f"Error storing capital and crypto amount: {response.status} {error_text}")

@shared_task(queue="trading") 
def updateCapitalTimestamp(token, pair, timeframe, timestamp):
    url = f"{MICRO_CENTRAL_URL}/capital/timestamp"
    payload = {
        "token": token,
        "pair": pair,
        "timeframe": timeframe,
        "timestamp": timestamp
    }
    headers = {
        "Token": token
    }

    response = requests.put(url, json=payload, headers=headers)
    if response.status_code == 200:
        trader_logger.info(f"Timestamp updated for token {token}, pair {pair}, and timeframe {timeframe}")
    else:
        error_text = response.text
        trader_logger.error(f"Error updating timestamp: {response.status} {error_text}")

@shared_task(queue="trading") 
def updateCapitalCrypto(token, pair, timeframe, crypto_amount):
    url = f"{MICRO_CENTRAL_URL}/capital/crypto"
    payload = {
        "token": token,
        "pair": pair,
        "timeframe": timeframe,
        "crypto_amount": crypto_amount
    }
    headers = {
        "Token": token
    }

    response = requests.put(url, json=payload, headers=headers)
    if response.status_code == 200:
        trader_logger.info(f"Crypto amount updated for token {token}, pair {pair}, and timeframe {timeframe}")
    else:
        error_text = response.text
        trader_logger.error(f"Error updating crypto amount: {response.status} {error_text}")

@shared_task(queue="trading") 
def get_capital(token, pair, timeframe):
    url = f"{MICRO_CENTRAL_URL}/capital"
    payload = {
        "token": token,
        "pair": pair,
        "timeframe": timeframe
    }
    headers = {
        "Token": token
    }

    response = requests.get(url, json=payload, headers=headers)
    if response.status_code == 200:
        data = response.json()
        return (
            data.get('capital'),
            data.get('crypto_amount'),
            data.get('timestamp'),
            data.get('cumulative_strategy_return'),
            data.get('cumulative_market_return'),
            data.get('first_trade'),
            data.get('last_price')
        )
    else:
        error_text = response.text
        trader_logger.error(f"Error retrieving capital: {response.status} {error_text}")
        return None, None, None, None, None, None, None

class NoNewDataException(Exception):
    pass

@shared_task(queue="trading") 
def dynamic_order_sizing(total_quantity, num_trades):
    orders = []
    split_quantity = total_quantity / num_trades
    for _ in range(num_trades):
        orders.append(split_quantity)
    return orders

@shared_task(queue="trading") 
def calculate_num_trades(total_quantity):
    if total_quantity <= 500:
        return 1
    elif total_quantity <= 1000:
        return 1
    elif total_quantity <= 2000:
        return 1
    elif total_quantity <= 5000:
        return 1
    elif total_quantity <= 8000:
        return 2    
    else:
        return 4

@shared_task(queue="trading") 
def place_twap_order(client, symbol, side, total_quantity, duration, interval, step_size, max_slippage_percent, latest_price=1.0):
    amount = total_quantity * latest_price
    num_trades = calculate_num_trades(amount)
    orders = dynamic_order_sizing(total_quantity, num_trades)
    success = True

    # Initialize shares and fiat
    shares = 0
    fiat = 0
    fee = 0

    for quantity in orders:
        quantity = math.floor(quantity / step_size) * step_size
        order_success = place_market_order(client, symbol, side, quantity, max_slippage_percent)
        
        if not order_success:
            success = False
            shares = 0
            fiat = 0
            break

        # Extract the fills field
        fills = order_success.get('fills', [])

        # Initialize total quantity
        total_quantity = 0.0

        # Iterate over the fills list and sum the qty and commission values
        for fill in fills:
            qty = float(fill.get('qty', 0))
            commission = float(fill.get('commission', 0))
            total_quantity += qty + commission 
        
        # Update shares with the total quantity
        shares += total_quantity
        
        fiat += float(order_success.get('cummulativeQuoteQty'))
        fee += commission

        time.sleep(interval)
    
    return success, shares, fiat, fee

# live trade function
@shared_task(queue="trading") 
def live_trade(pair, token, timeframe, stop_loss_threshold=0.05, gain_threshold=0.001, taker_fee=0.001, max_slippage_percent=1, live=False):
    trade_executed = False

    # Fetch the latest data
    getHistorical.get_all_binance(pair, timeframe, token, save=True)

    # Retrieve API keys
    url = f"{MICRO_CENTRAL_URL}/tlogin/{token}"
    response = requests.get(url)
    if response.status_code == 200:
        api_key = response.json()['api_key']
        api_secret = response.json()['api_secret']
        client = Client(api_key, api_secret)
    else:
        error_text = response.text
        trader_logger.error(f"Error retrieving API keys: {response.status_code} {error_text}")
        return    

    # Load the model and historical data
    model_path = get_model_path(pair, timeframe)
    historical_data = get_historical_data(token, pair, timeframe)
    historical_data = add_indicators(historical_data.todict())

    if not os.path.exists(model_path):
        trader_logger.error(f"Model not found for {pair} {timeframe}")
        raise FileNotFoundError(f"Model not found for {pair} {timeframe}")
    else:
        model = joblib.load(model_path)
        features = ['rsi', 'macd', 'macd_signal', 'macd_diff', 'bollinger_hband', 'bollinger_mavg', 'bollinger_lband', 'ema', 'ATR']
        model = update_model(model, historical_data, features)

    # Retrieve stored capital, crypto amount, and other relevant data
    capital, crypto_amount, latest_timestamp, cumulative_strategy_return, cumulative_market_return, first_trade, last_price = get_capital(token, pair, timeframe)
    capital_accumulated = get_capital_accumulated(token, pair, timeframe)

    base_asset, quote_asset, stepSize, min_notional, max_notional = get_base_and_quote_assets(pair)

    # Adjust the available crypto capital based on actual balance
    # available_crypto = float(client.get_asset_balance(asset=base_asset)['free'])
    # if crypto_amount != available_crypto:
    #     crypto_amount = available_crypto
    #     updateCapitalCrypto(token, pair, timeframe, crypto_amount)

    if first_trade:
        cumulative_strategy_return = 1.0
        cumulative_market_return = 1.0
        capital_accumulated = capital  # Initialize capital_accumulated to the initial capital
        initial_close = historical_data['close'].iloc[-1]  # Set the initial close price for market return

    # Ensure timestamp columns are in datetime format
    historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'])

    # Get the latest data
    latest_data = historical_data.tail(1)
    latest_timestamp_new = latest_data.iloc[-1]['timestamp']
    latest_close = latest_data.iloc[-1]['close']


    # Convert latest_timestamp to datetime if it's not already
    if isinstance(latest_timestamp, str):
        latest_timestamp = datetime.strptime(latest_timestamp, '%Y-%m-%d %H:%M:%S')

    # Skip if no new data
    if latest_timestamp == latest_timestamp_new and not first_trade:
        trader_logger.info(f"No new data available for token {token} and symbol {pair}")
        return

    # Update the latest_timestamp
    latest_timestamp = latest_timestamp_new
    updateCapitalTimestamp(token, pair, timeframe, latest_timestamp)

    latest_features = latest_data[features].dropna()
    if latest_features.empty:
        trader_logger.info(f"No valid data for prediction for token {token}")
        return

    latest_signal = model.predict(latest_features)[0]
    latest_price = float(client.get_symbol_ticker(symbol=pair)['price'])

    # Calculate expected return
    expected_return = (latest_close - last_price) / last_price

    # Calculate potential return and stop-loss trigger
    current_portfolio_value = capital + (crypto_amount * latest_price)
    stop_loss_triggered = (current_portfolio_value < capital_accumulated * (1 - stop_loss_threshold))

    # Update cumulative market return
    if first_trade:
        cumulative_market_return = 1.0
    else:
        initial_close = historical_data['close'].iloc[-1]  # Set the initial close price for market return
        cumulative_market_return = latest_close / initial_close

    # Debugging block
    trader_logger.info(f"latest_signal: {latest_signal}, expected_return: {expected_return}, gain_threshold: {gain_threshold}, capital: {capital}, crypto_amount: {crypto_amount}, stop_loss_triggered: {stop_loss_triggered}, token: {token}, pair: {pair}, timeframe: {timeframe}")     

    # Buy logic
    # if first_trade or (latest_signal == 1 and expected_return > gain_threshold) and capital > 0: # to test will remove the
    # option expected_return > gain_threshold in order to buy with any signal, according to the backtest logic
    if first_trade or (latest_signal == 1 and capital > 0):
    # Execute buy logic    
        max_investment = calculate_max_investment(latest_data)
        capital = min(capital, max_investment) # Adjust capital to the maximum investment amount
        amount_to_buy = capital / latest_price
        amount_to_buy = amount_to_buy * (1 - taker_fee)  # Adjust for taker fee
        # amount_to_buy = math.floor(amount_to_buy / stepSize) * stepSize
        # amount_to_buy = round(amount_to_buy, int(-math.log10(stepSize)))
        if amount_to_buy * latest_price >= min_notional and capital >= amount_to_buy * latest_price:
            try:
                success, shares, fiat, commision = place_twap_order(client, pair, 'BUY', amount_to_buy, duration=60, interval=5, step_size=stepSize, max_slippage_percent=max_slippage_percent, latest_price=latest_price)
                if success:
                    executedQty = shares
                    capital_spent = amount_to_buy * latest_price  # The actual capital spent
                    capital = 0
                    crypto_amount += amount_to_buy  # The actual crypto amount bought

                    # Update cumulative strategy return after buy operation
                    cumulative_strategy_return = current_portfolio_value / capital_accumulated

                    store_capital(token, pair, timeframe, capital, executedQty, latest_timestamp, cumulative_strategy_return, cumulative_market_return, False, latest_price)
                    trader_logger.info(f"BUY ORDER for token {token}: {amount_to_buy} {pair}")
                    message = message = (
                        f"TRADER :\n\n"
                        f"BUY ORDER for token {pair}:\n\n"
                        f"Timeframe: {timeframe}\n"
                        f"Amount: {executedQty:.2f}\n"
                        f"Price: {latest_price:.8f}\n"
                        f"Fee: {commision:.8f}\n"
                        f"Total: {fiat:.2f}\n"
                        f"Strategy Return: {cumulative_strategy_return:.8f}\n"
                        f"Timestamp: {latest_timestamp}\n"
                    )
                    send_trade_notification(token, message)
                    trade_executed = True
                else:
                    trader_logger.error("Failed to execute BUY order")
            except Exception as e:
                trader_logger.error(f"Failed to place BUY order for token {token}: {str(e)}")
        else:
            trader_logger.info(f"Insufficient funds or capital to buy for token {token}")

    # Sell logic
    elif latest_signal == 0 and (expected_return > gain_threshold or stop_loss_triggered) and crypto_amount > 0:
        amount_to_sell = crypto_amount 
        amount_to_sell = amount_to_sell * (1 - taker_fee)  # Adjust for taker fee
        # amount_to_sell = math.floor(amount_to_sell / stepSize) * stepSize
        # amount_to_sell = round(amount_to_sell, int(-math.log10(stepSize)))

        if amount_to_sell * latest_price >= min_notional:
            try:
                # Calculate the total capital to be gained before placing the sell order
                capital_gained = amount_to_sell * latest_price  # Adjusted for taker fee

                # Execute the TWAP sell order
                success, shares, fiat, commision = place_twap_order(client, pair, 'SELL', amount_to_sell, duration=60, interval=5, step_size=stepSize, max_slippage_percent=max_slippage_percent, latest_price=latest_price)

                if success:
                    cummulative_quote_qty = fiat
                    executedQty = 0
                    # Update cumulative strategy return after sell operation
                    cumulative_strategy_return = current_portfolio_value / capital_accumulated

                    # Update cumulative market return after sell operation
                    cumulative_market_return = latest_close / initial_close

                    # Update capital and crypto amount after the sell order
                    capital = cummulative_quote_qty
                    crypto_amount = 0

                    # Update accumulated capital to reflect the highest value reached
                    capital_accumulated = max(capital_accumulated, cummulative_quote_qty)

                    update_capital_accumulated(token, pair, timeframe, capital_accumulated)

                    # Store updated values in the database
                    store_capital(token, pair, timeframe, cummulative_quote_qty, crypto_amount, latest_timestamp, cumulative_strategy_return, cumulative_market_return, False, latest_price)
                    trader_logger.info(f"SELL ORDER for token {token}: {amount_to_sell} {pair}")
                    message = message = (
                        f"TRADER :\n\n"
                        f"SELL ORDER for token {pair}:\n\n"
                        f"Timeframe: {timeframe}\n"
                        f"Amount: {amount_to_sell:.2f}\n"
                        f"Price: {latest_price:.8f}\n"
                        f"Fee: {commision:.8f}\n"
                        f"Total: {fiat:.2f}\n"
                        f"Strategy Return: {cumulative_strategy_return:.8f}\n"
                        f"Timestamp: {latest_timestamp}\n"
                    )
                    send_trade_notification(token, message)
                    
                    trade_executed = True
                else:
                    trader_logger.error("Failed to execute SELL order")
            except Exception as e:
                trader_logger.error(f"Failed to place SELL order for token {token}: {str(e)}")
        else:
            trader_logger.info(f"Insufficient crypto to sell for token {token} pair {pair}")
    else:
        trader_logger.info(f"No action taken for token {token} pair {pair}")

    # Update the model with the latest historical data
    historical_data = get_historical_data(token, pair, timeframe)
    historical_data = add_indicators(historical_data)
    model = update_model(model, historical_data, features)
    joblib.dump(model, model_path)

    if trade_executed:
        first_trade = False
        # Update first_trade in the database to ensure it's not executed again
        store_capital(token, pair, timeframe, capital, crypto_amount, latest_timestamp, cumulative_strategy_return, cumulative_market_return, first_trade, latest_price)

@shared_task(queue="trading") 
def send_trade_notification(token, message):
    url = f"{MICRO_CENTRAL_URL}/send_notification"
    payload = {
        "token": token, "message": message
    }
    headers = {
        "Token": token
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        trader_logger.info(f"Notification sent for token {token}")
    else:
        error_text = response.text
        trader_logger.error(f"Error sending notification: {response.status} {error_text}")    

@shared_task(queue="trading")
def get_trader_info(page: int = 1, page_size: int = 1000):
    """
    Fetch trader information in paginated chunks.
    """
    url = f"{MICRO_CENTRAL_URL}/trader-info"
    payload = {
        "page": page,
        "page_size": page_size
    }
    headers = {
        "Token": token
    }
    

    response = requests.get(url, headers=headers, params=payload)
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    else:
        error_text = response.text
        trader_logger.error(f"Error retrieving trader info: {response.status} {error_text}")
        return pd.DataFrame()


@shared_task(queue="trading")
def process_trader_chunk(chunk):
    """
    Process a chunk of trader data.
    """
    for row in chunk.itertuples():
        live_trade.delay(row.pair, row.token, row.timeframe, row.stop_loss_threshold, row.gain_threshold)


@shared_task(queue="trading")
def trader():
    """
    Executes live trades based on trader information.
    """
    page = 1
    page_size = 1000
    while True:
        trader_info = get_trader_info(page=page, page_size=page_size)
        if trader_info.empty:
            break  # No more data to process

        # Process the chunk asynchronously
        process_trader_chunk.delay(trader_info)

        page += 1

    trader_logger.info("Started live trading tasks for all chunks.")

