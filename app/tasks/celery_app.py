import os
import logging
from celery import Celery
from celery.schedules import crontab
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env.micro.central")

broker_url = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
backend_url = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

celery_app = Celery(
    'micro_backtest',
    broker=broker_url,
    backend=backend_url,
    include=['app.tasks.celery_tasks']
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    broker_connection_retry_on_startup=True,
)

# Configure periodic tasks
celery_app.conf.beat_schedule = {
    'run-trader-every-5-minutes': {
        'task': 'app.tasks.celery_tasks.run_trader',  # Path to the run_trader task
        'schedule': 600.0,  # Run every 600 seconds (10 minutes)
        'options': {'queue': 'trading'},  # Ensure it runs in the "trading" queue
    },
}

# Configure logging
logging.basicConfig(level=logging.INFO)

# if __name__ == '__main__':
#     celery_app.start()