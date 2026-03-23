import logging
from typing import Optional


def get_logger(name: str = "asrs-task1", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    lvl = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(lvl)

    handler = logging.StreamHandler()
    handler.setLevel(lvl)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    return logger