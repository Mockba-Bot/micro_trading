import asyncio
from celery import shared_task
from celery.result import AsyncResult
from app.models.elliot_waves import analyze_intervals
from app.models.technical_analisys import analize_asset
from app.models.gainers_analysis import analyze_movements
from app.models.probability_analisys import analize_probability_asset
import logging

logger = logging.getLogger(__name__)

@shared_task(queue="trading")
def analyze_intervals_task(
    asset,
    token,
    interval,
    target_lang
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
            interval,
            target_lang
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
    target_lang="en"
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
            leverage,
            target_lang
        )
    )
    
    return result

@shared_task(queue="trading")
def analyze_movers_task(
    token,
    target_lang,
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
        analyze_movements(
            token,
            target_lang,
            interval,
            change_threshold,
            type,
            top_n
        )
    )
    
    return result


@shared_task(queue="trading")
def analyze_asset_probability_task(
    token,
    asset,
    interval,
    features=None,
    leverage=10,
    target_lang="en",
    free_collateral=1000
):
    
    logger.info(f"Analyzing probability asset {asset} with token {token} and timeframe {interval}")
    
    # ✅ Fix: Use `asyncio.get_event_loop()` instead of `asyncio.run()`
    loop = asyncio.get_event_loop()
    if loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    result = loop.run_until_complete(
        analize_probability_asset(
            token,
            asset,
            interval,
            features,
            leverage,
            target_lang,
            free_collateral
        )
    )
    
    return result

# @shared_task(queue="trading")
# def run_trader():
#     """
#     Celery task to execute the trader function.
#     """
#     return trader()