import csv
import datetime
from pathlib import Path
from threading import Lock

log_file_path = ''
_LOG_LOCK = Lock()

def initialize_token_logger():
    global log_file_path
    logs_dir = Path(__file__).resolve().parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_path = str(logs_dir / f'mas_token_usage_{timestamp}.csv')
    
    with open(log_file_path, 'w', newline='') as csvfile:
        fieldnames = ['timestamp', 'model', 'prompt_tokens', 'completion_tokens', 'total_tokens', 'cost']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

def log_token_usage(model, prompt_tokens, completion_tokens, total_tokens, cost):
    if not log_file_path:
        initialize_token_logger()

    with _LOG_LOCK:
        with open(log_file_path, 'a', newline='') as csvfile:
            fieldnames = ['timestamp', 'model', 'prompt_tokens', 'completion_tokens', 'total_tokens', 'cost']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model': model,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': total_tokens,
                'cost': cost
            })
