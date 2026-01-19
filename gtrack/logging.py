"""
Logging configuration for gtrack.

This module provides centralized logging for the gtrack package.
Log level can be controlled via the GTRACK_LOGLEVEL environment variable.

Environment Variables
---------------------
GTRACK_LOGLEVEL : str
    Set the logging level. Options: DEBUG, INFO, WARNING, ERROR, CRITICAL.
    Default is WARNING (minimal output).

Usage
-----
>>> from gtrack.logging import get_logger
>>> logger = get_logger(__name__)
>>> logger.info("Processing started")
>>> logger.debug("Detailed debug information")

To enable verbose output, set the environment variable before importing gtrack:

    export GTRACK_LOGLEVEL=INFO   # Show progress messages
    export GTRACK_LOGLEVEL=DEBUG  # Show detailed debug info
"""

import logging
import os
import sys
from typing import Optional


# Package-level logger name
LOGGER_NAME = "gtrack"

# Default log level (quiet by default)
DEFAULT_LOG_LEVEL = "WARNING"

# Format for log messages
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FORMAT_SIMPLE = "%(levelname)s: %(message)s"

# Track if logging has been configured
_logging_configured = False


def _get_log_level_from_env() -> int:
    """Get log level from GTRACK_LOGLEVEL environment variable."""
    level_name = os.environ.get("GTRACK_LOGLEVEL", DEFAULT_LOG_LEVEL).upper()

    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "WARN": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    return level_map.get(level_name, logging.WARNING)


def configure_logging(
    level: Optional[int] = None,
    format_string: Optional[str] = None,
    stream: Optional[object] = None,
) -> None:
    """
    Configure gtrack logging.

    This function sets up the logging configuration for the entire package.
    It is called automatically when the package is imported, but can be
    called again to reconfigure.

    Parameters
    ----------
    level : int, optional
        Logging level (e.g., logging.DEBUG, logging.INFO).
        If None, reads from GTRACK_LOGLEVEL environment variable.
    format_string : str, optional
        Format string for log messages.
        If None, uses a simple format for INFO and above, detailed for DEBUG.
    stream : file-like, optional
        Stream to write logs to. Default is sys.stderr.
    """
    global _logging_configured

    if level is None:
        level = _get_log_level_from_env()

    if format_string is None:
        format_string = LOG_FORMAT_SIMPLE if level >= logging.INFO else LOG_FORMAT

    if stream is None:
        stream = sys.stderr

    # Get or create the root gtrack logger
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create console handler
    handler = logging.StreamHandler(stream)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(format_string))

    logger.addHandler(handler)

    # Prevent propagation to root logger
    logger.propagate = False

    _logging_configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a gtrack module.

    Parameters
    ----------
    name : str
        Name of the module (typically __name__).

    Returns
    -------
    logging.Logger
        Logger instance for the module.

    Examples
    --------
    >>> logger = get_logger(__name__)
    >>> logger.info("Starting computation")
    """
    global _logging_configured

    # Auto-configure if not already done
    if not _logging_configured:
        configure_logging()

    # Ensure name is under gtrack namespace
    if not name.startswith(LOGGER_NAME):
        name = f"{LOGGER_NAME}.{name}"

    return logging.getLogger(name)


def set_log_level(level: int) -> None:
    """
    Set the log level for all gtrack loggers.

    Parameters
    ----------
    level : int
        Logging level (e.g., logging.DEBUG, logging.INFO).

    Examples
    --------
    >>> import logging
    >>> from gtrack.logging import set_log_level
    >>> set_log_level(logging.DEBUG)  # Enable debug output
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


def enable_verbose() -> None:
    """
    Enable verbose output (INFO level).

    Convenience function equivalent to set_log_level(logging.INFO).
    """
    set_log_level(logging.INFO)


def enable_debug() -> None:
    """
    Enable debug output (DEBUG level).

    Convenience function equivalent to set_log_level(logging.DEBUG).
    """
    set_log_level(logging.DEBUG)


def disable_logging() -> None:
    """
    Disable all gtrack logging output.

    Convenience function equivalent to set_log_level(logging.CRITICAL + 1).
    """
    set_log_level(logging.CRITICAL + 1)
