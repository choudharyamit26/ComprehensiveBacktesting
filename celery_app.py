from celery import Celery

app = Celery(
    "stock_tasks", broker="redis://localhost:6379/0", backend="redis://localhost:6379/0"
)

app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Kolkata",
    enable_utc=True,
)


# celery -A tasks worker --loglevel=info --pool=solo
# python trigger_task.py
# celery -A tasks flower --port=5555
