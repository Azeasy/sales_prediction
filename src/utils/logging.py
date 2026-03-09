"""
Structured logging setup.

Configures a consistent logger for the entire project.
Use `get_logger(__name__)` in each module.
"""

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with a human-readable format."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Call setup_logging() once at process start."""
    return logging.getLogger(name)
