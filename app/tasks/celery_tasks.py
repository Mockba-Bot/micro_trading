from app.tasks.celery_app import celery_app
from app.models.backtest import run_backtest
from app.utils.live_trade import trader
import logging
import requests
import pandas as pd

logger = logging.getLogger(__name__)

@celery_app.task(queue="trading")
def run_backtest_task(pair, timeframe, token, values, stop_loss_threshold=0.05, initial_investment=10000, maker_fee=0.001, taker_fee=0.001, gain_threshold=0.001):
    logger.info(f"Running backtest for {pair} with timeframe {timeframe}")
    
    # Use asyncio.run to execute the async function
    result = asyncio.run(
        run_backtest(
            pair, 
            timeframe, 
            token, 
            values, 
            stop_loss_threshold, 
            initial_investment, 
            maker_fee, 
            taker_fee, 
            gain_threshold
        )
    )
    
    return result


@shared_task(queue="trading")
def run_trader():
    """
    Celery task to execute the trader function.
    """
    return trader()