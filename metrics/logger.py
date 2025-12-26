import csv
import os
import time
from threading import Lock

_METRICS_DIR = "metrics/data"
os.makedirs(_METRICS_DIR, exist_ok=True)

_lock = Lock()

def log_metric(name: str, value, session_id: str = "default"):
    """Простая запись метрик в CSV"""
    path = os.path.join(_METRICS_DIR, f"{name}.csv")
    ts = time.time()

    with _lock:
        new_file = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(["timestamp", "session_id", "value"])
            writer.writerow([ts, session_id, value])
