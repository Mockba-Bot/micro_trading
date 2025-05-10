import os
import sys
from fastapi import FastAPI
from app.routes import backtest_router, status_router, analyze_router, analyze_asset_router

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(
    title="Crypto Trading API",
    version="1.0.0",
    description="API for running cryptocurrency Trading"
)

# Include the trading routes
app.include_router(backtest_router, prefix="/api/v1/trading")
app.include_router(status_router, prefix="/api/v1/trading")
app.include_router(analyze_router, prefix="/api/v1/trading")
app.include_router(analyze_asset_router, prefix="/api/v1/trading")

# run update of tables
# alembic revision --autogenerate -m "initial tables"
# commit
# alembic upgrade head
# Run project
# uvicorn app.main:app --port 8001 --reload 
# redis-cli -p 6390 FLUSHDB
# redis-cli -p 6390 flushall
# celery -A app.tasks.celery_app.celery_app worker --loglevel=info --concurrency=2 --queues=trading
# celery -A app.tasks.celery_app.celery_app beat --loglevel=info