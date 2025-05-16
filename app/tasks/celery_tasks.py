import asyncio
from celery import shared_task
from app.models.backtest import run_backtest
from app.models.elliot_waves import analyze_intervals
from app.models.technical_analisys import analize_asset
from app.models.gainers_analysis import analyze_movers
import logging

logger = logging.getLogger(__name__)

@shared_task(queue="trading")
def run_backtest_task(
      asset
    , timeframe
    , token
    , values
    , free_collateral=100
    , position_size=10000
    , stop_loss_threshold=0.05
    , take_profit_threshold=0.001
    , features=None
    , withdraw_percentage=0.7
    , compound_percentage=0.3
    , num_trades_daily=None
    , market_bias="neutral"
    ):
    
    logger.info(f"Running backtest for {asset} with timeframe {timeframe}")
    
    # ✅ Fix: Use `asyncio.get_event_loop()` instead of `asyncio.run()`
    loop = asyncio.get_event_loop()
    if loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    result = loop.run_until_complete(
        run_backtest(
            asset, 
            timeframe, 
            token, 
            values, 
            free_collateral, 
            position_size,
            stop_loss_threshold, 
            take_profit_threshold,
            features,
            withdraw_percentage,
            compound_percentage,
            num_trades_daily,
            market_bias
        )
    )
    
    return result

@shared_task(queue="trading")
def analyze_intervals_task(
    asset,
    token,
    interval
):
    
    logger.info(f"Analyzing intervals for {asset} with token {token} and interval {interval}")
    
    # ✅ Fix: Use `asyncio.get_event_loop()` instead of `asyncio.run()`
    loop = asyncio.get_event_loop()
    if loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    result = loop.run_until_complete(
        analyze_intervals(
            asset,
            token,
            interval
        )
    )
    return result

@shared_task(queue="trading")
def analyze_asset_task(
    token,
    asset,
    interval,
    features=None,
    leverage=10,
):
    
    logger.info(f"Analyzing asset {asset} with token {token} and timeframe {interval}")
    
    # ✅ Fix: Use `asyncio.get_event_loop()` instead of `asyncio.run()`
    loop = asyncio.get_event_loop()
    if loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    result = loop.run_until_complete(
        analize_asset(
            token,
            asset,
            interval,
            features,
            leverage
        )
    )
    
    return result

@shared_task(queue="trading")
def analyze_movers_task(
    token,
    interval,
    change_threshold=0.05,
    type="gainers",
    top_n=10
):
    
    logger.info(f"Analyzing gainers with token {token} and timeframe {interval}")
    
    # ✅ Fix: Use `asyncio.get_event_loop()` instead of `asyncio.run()`
    loop = asyncio.get_event_loop()
    if loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    result = loop.run_until_complete(
        analyze_movers(
            token,
            interval,
            change_threshold,
            type,
            top_n
        )
    )
    
    return result


# @shared_task(queue="trading")
# def run_trader():
#     """
#     Celery task to execute the trader function.
#     """
#     return trader()