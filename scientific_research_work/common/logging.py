import logging
import sys

try:
    from concurrent_log_handler import ConcurrentRotatingFileHandler
except ModuleNotFoundError:  # pragma: no cover - fallback for environments without the package
    ConcurrentRotatingFileHandler = None


def setup_logging(log_file_name: str) -> None:
    if ConcurrentRotatingFileHandler is None:
        file_handler = logging.FileHandler(log_file_name, mode="a", encoding="utf-8")
    else:
        file_handler = ConcurrentRotatingFileHandler(
            log_file_name,
            mode="a",
            encoding="utf-8",
            maxBytes=10 * 1024 * 1024,
            backupCount=1,
        )
    handlers = [logging.StreamHandler(sys.stdout), file_handler]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(filename)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )
