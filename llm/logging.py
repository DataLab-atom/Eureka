import logging
from pathlib import Path

try:
    from concurrent_log_handler import ConcurrentRotatingFileHandler
except ModuleNotFoundError:  # pragma: no cover - fallback for environments without the package
    ConcurrentRotatingFileHandler = None


def _get_root_path() -> Path:
    return Path(__file__).resolve().parent


def _build_token_logger(name: str, filename: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        log_path = _get_root_path() / filename
        if ConcurrentRotatingFileHandler is None:
            handler = logging.FileHandler(filename=str(log_path), mode="a", encoding="utf-8")
        else:
            handler = ConcurrentRotatingFileHandler(
                filename=str(log_path),
                mode="a",
                encoding="utf-8",
                maxBytes=20 * 1024 * 1024,
                backupCount=5,
            )
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)

    return logger


input_logger = _build_token_logger("input_token", "input_token.log")
output_logger = _build_token_logger("output_token", "output_token.log")
