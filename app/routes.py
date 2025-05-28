from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from celery.result import AsyncResult
from app.tasks.celery_app import celery_app
from app.tasks.celery_tasks import  analyze_intervals_task, analyze_asset_task, analyze_movers_task, analyze_asset_probability_task
from typing import List, Optional

class AnalyzeIntervalsRequest(BaseModel):
    asset: str
    token: str 
    interval : str
    target_lang: str

class AnalyzeAssetRequest(BaseModel):
    token: str
    asset: str
    timeframe: str
    features: Optional[List[str]] = None    
    leverage: int = 10
    target_lang: str = "en"

class GainersAnalysisRequest(BaseModel):
    token: str
    target_lang: str
    interval: str 
    change_threshold: float   
    type: str = "gainers"  # Default to "gainers"
    top_n: int = 10  # Optional parameter for top N movers

class AnalyzeProbabilityAssetRequest(BaseModel):
    token: str
    asset: str
    timeframe: str
    features: Optional[List[str]] = None    
    leverage: int = 10
    target_lang: str = "en"
    free_collateral: float = 100

status_router = APIRouter()
analyze_router = APIRouter()
analyze_asset_router = APIRouter()
gainers_analysis_router = APIRouter()
analyze_asset_probability_router = APIRouter()

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
            analyze_request.token,
            analyze_request.interval,
            analyze_request.target_lang
        )
        return {"task_id": task.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@analyze_asset_router.post("/analyze_asset")
async def analyze_asset_api(request: Request, analyze_asset_request: AnalyzeAssetRequest):
    """
    Analyze the asset with the given parameters.
    """
    try:
        task = analyze_asset_task.delay(
            analyze_asset_request.token,
            analyze_asset_request.asset,
            analyze_asset_request.timeframe,
            analyze_asset_request.features,
            analyze_asset_request.leverage,
            analyze_asset_request.target_lang
        )
        return {"task_id": task.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))  


@status_router.post("/gainers_analysis")
async def gainers_analysis_api(request: Request, gainers_analysis_request: GainersAnalysisRequest):
    """
    Perform gainers analysis with the given parameters.
    """
    try:
        task = analyze_movers_task.delay(
            gainers_analysis_request.token,
            gainers_analysis_request.target_lang,
            gainers_analysis_request.interval,
            gainers_analysis_request.change_threshold,
            gainers_analysis_request.type,
            gainers_analysis_request.top_n
        )
        return {"task_id": task.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))      


@analyze_asset_probability_router.post("/analyze_probability_asset")
async def analyze_asset_probability_api(request: Request, analyze_asset_probability_request: AnalyzeProbabilityAssetRequest):
    """
    Analyze the asset with the given parameters.
    """
    try:
        task = analyze_asset_probability_task.delay(
            analyze_asset_probability_request.token,
            analyze_asset_probability_request.asset,
            analyze_asset_probability_request.timeframe,
            analyze_asset_probability_request.features,
            analyze_asset_probability_request.leverage,
            analyze_asset_probability_request.target_lang,
            analyze_asset_probability_request.free_collateral
        )
        return {"task_id": task.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))       