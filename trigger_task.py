# trigger_task.py
from tasks import run_intraday_stock_filter

# Launch asynchronously
task = run_intraday_stock_filter.delay("ind_nifty50list.csv")

# (Optional) Get result later
# result = task.get(timeout=300)
# print(result)
