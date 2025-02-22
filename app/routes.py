from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from celery.result import AsyncResult
from app.tasks.celery_app import celery_app
from app.tasks.celery_tasks import run_backtest_task
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse

class BacktestRequest(BaseModel):
    pair: str
    timeframe: str
    token: str
    values: str
    stop_loss_threshold: float = 0.05
    initial_investment: float = 10000
    maker_fee: float = 0.001
    taker_fee: float = 0.001
    gain_threshold: float = 0.001

limiter = Limiter(key_func=get_remote_address)

backtest_router = APIRouter()
status_router = APIRouter()

@backtest_router.post("/backtest")
@limiter.limit("10/second")
async def run_backtest_api(request: Request, backtest_request: BacktestRequest):
    """
    Run the backtest with the provided parameters.
    """
    try:
        task = run_backtest_task.delay(
            backtest_request.pair,
            backtest_request.timeframe,
            backtest_request.token,
            backtest_request.values,
            backtest_request.stop_loss_threshold,
            backtest_request.initial_investment,
            backtest_request.maker_fee,
            backtest_request.taker_fee,
            backtest_request.gain_threshold
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

# Add exception handler for rate limit exceeded
@backtest_router.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please try again later."}
    )
