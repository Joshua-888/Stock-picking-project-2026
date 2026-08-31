"""
src/utils/logging.py

Configures project-wide logging with rotating file and console handlers.

Usage:
    from src.utils.logging import get_logger
    log = get_logger(__name__)
    log.info("Starting monthly update")
    log.warning("Missing data for AAPL")
    log.error("API call failed: %s", error_message)
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path


def setup_logging(
    level: str = "INFO",
    log_file: str = "logs/stock_analysis.log",
    max_bytes: int = 5_242_880,
    backup_count: int = 3,
) -> None:
    """
    Configure the root logger once at application startup.

    Call this once in run_dashboard.py and run_monthly_update.py.
    After calling this, every module that calls get_logger() will
    automatically use the configured handlers.

    Args:
        level:        Log level string ("DEBUG", "INFO", "WARNING", "ERROR").
        log_file:     Path to the rotating log file.
        max_bytes:    Rotate the log file after this many bytes (default 5 MB).
        backup_count: Number of rotated log files to keep.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Ensure the logs directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Remove any handlers that may have been added by previous calls or imports
    root.handlers.clear()

    # Console handler – always active
    console = logging.StreamHandler()
    console.setLevel(numeric_level)
    console.setFormatter(formatter)
    root.addHandler(console)

    # Rotating file handler – keeps history without growing forever
    file_handler = logging.handlers.RotatingFileHandler(
        filename     = log_file,
        maxBytes     = max_bytes,
        backupCount  = backup_count,
        encoding     = "utf-8",
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger.

    Every module in the project should use this instead of
    calling logging.getLogger() directly, so the naming
    convention stays consistent.

    Args:
        name: Typically __name__ of the calling module.

    Returns:
        A configured Logger instance.
    """
    return logging.getLogger(name)
