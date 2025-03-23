import asyncio
from celery import shared_task
from app.models.backtest import run_backtest
from app.utils.live_trade import trader
import logging

logger = logging.getLogger(__name__)

@shared_task(queue="trading")
def run_backtest_task(
      pair
    , timeframe
    , token
    , values
    , stop_loss_threshold=0.05
    , initial_investment=10000
    , gain_threshold=0.001
    , leverage=1
    , features=None
    , withdraw_percentage=0.7
    , compound_percentage=0.3
    , num_trades=None):
    
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
            gain_threshold,
            leverage,
            features,
            withdraw_percentage,
            compound_percentage,
            num_trades
        )
    )
    
    return result


@shared_task(queue="trading")
def run_trader():
    """
    Celery task to execute the trader function.
    """
    return trader()