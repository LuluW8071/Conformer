"""Centralized logger configuration using Loguru."""

import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"


def setup_logging():
    """
    Configure logging for the entire application using Loguru.
    Prevents duplicate handlers from being added.
    """

    LOGS_DIR.mkdir(exist_ok=True)

    # Remove default logger to avoid duplicates
    logger.remove()

    log_level = "INFO"

    # Colorized format for console
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan> | <blue>{function}</blue>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    # Console logging (colored)
    logger.add(
        sys.stdout,
        level=log_level,
        format=console_format,
        colorize=True, 
    )

    # File logging (no colors)
    file_format = (
        "[{time:YYYY-MM-DD HH:mm:ss}] "
        "[{level}] "
        "[{name}] - {message}"
    )

    logger.add(
        LOGS_DIR / "app.log",
        level=log_level,
        format=file_format,
        rotation="5 MB",
        retention=5,
        compression="zip",
    )

    return logger


# Singlular logger instance
logger = setup_logging()