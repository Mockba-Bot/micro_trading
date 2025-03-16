import asyncio
from celery import shared_task
from app.tasks.celery_app import celery_app
from app.models.backtest import run_backtest
from app.utils.live_trade import trader
import logging
import requests
import pandas as pd

logger = logging.getLogger(__name__)

@shared_task(queue="trading")
def run_backtest_task(pair, timeframe, token, values, stop_loss_threshold=0.05, initial_investment=10000
    , maker_fee=0.001, taker_fee=0.001, gain_threshold=0.001, leverage=1
    , features=None, withdraw_percentage=0.7, compound_percentage=0.3):
    logger.info(f"Running backtest for {pair} with timeframe {timeframe}")
    
    # ✅ Fix: Use `asyncio.get_event_loop()` instead of `asyncio.run()`
    loop = asyncio.get_event_loop()
    if loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    result = loop.run_until_complete(
        run_backtest(
            pair, 
            timeframe, 
            token, 
            values, 
            stop_loss_threshold, 
            initial_investment, 
            maker_fee, 
            taker_fee, 
            gain_threshold,
            leverage,
            features,
            withdraw_percentage,
            compound_percentage
        )
    )
    
    return result


@shared_task(queue="trading")
def run_trader():
    """
    Celery task to execute the trader function.
    """
    return trader()