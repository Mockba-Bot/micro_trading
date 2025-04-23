from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from celery.result import AsyncResult
from app.tasks.celery_app import celery_app
from app.tasks.celery_tasks import run_backtest_task, analyze_intervals_task
from typing import List, Optional

class BacktestRequest(BaseModel):
    asset: str
    timeframe: str
    token: str
    values: str
    free_collateral: float = 100
    position_size: float = 10000
    stop_loss_threshold: float = 0.05
    take_profit_threshold: float = 0.001
    features: Optional[List[str]] = None
    withdraw_percentage: float = 0.7
    compound_percentage: float = 0.3
    num_trades_daily: Optional[int] = None
    market_bias: str

class AnalyzeIntervalsRequest(BaseModel):
    asset: str
    token: str 

backtest_router = APIRouter()
status_router = APIRouter()
analyze_router = APIRouter()

@backtest_router.post("/backtest")
async def run_backtest_api(request: Request, backtest_request: BacktestRequest):
    """
    Run the backtest with the provided parameters.
    """
    try:
        task = run_backtest_task.delay(
            backtest_request.asset,
            backtest_request.timeframe,
            backtest_request.token,
            backtest_request.values,
            backtest_request.free_collateral,
            backtest_request.position_size,
            backtest_request.stop_loss_threshold,
            backtest_request.take_profit_threshold,
            backtest_request.features,
            backtest_request.withdraw_percentage,
            backtest_request.compound_percentage,
            backtest_request.num_trades_daily,
            backtest_request.market_bias
        )
        return {"task_id": task.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@status_router.get("/status/{task_id}")
async def get_task_status(task_id: str):
    task_result = AsyncResult(task_id, app=celery_app)
    if task_result.state == 'PENDING':
        return {"status": "Pending"}
    elif task_result.state == 'SUCCESS':
        return {"status": "Success", "result": task_result.result}
    elif task_result.state == 'FAILURE':
        return {"status": "Failure", "result": str(task_result.result)}
    else:
        return {"status": task_result.state}
    
@analyze_router.post("/elliot_waves/analyze_intervals")
async def analyze_intervals_api(request: Request, analyze_request: AnalyzeIntervalsRequest):
    """
    Analyze intervals for the given asset and token.
    """
    try:
        task = analyze_intervals_task.delay(
            analyze_request.asset,
            analyze_request.token
        )
        return {"task_id": task.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))    