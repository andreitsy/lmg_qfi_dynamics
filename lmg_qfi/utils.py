"""Logging and utility functions for the LMG QFI simulation."""

import logging

_logger_initialized = False


def log_message(message: str, log_level: int = logging.INFO):
    """
    This function writes a message to stdout and logs the operation.
    """
    global _logger_initialized
    if not _logger_initialized:
        _logger_initialized = True
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],
        )
    logging.log(log_level, message)


def setup_logging(log_handler: logging.Handler):
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[log_handler],
    )
