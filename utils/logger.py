import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional, Union


def _setup_logger(
    name: str = "locket",
    level: Union[str, int] = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Create logger
    logger_instance: logging.Logger = logging.getLogger(name)

    # Handle both string and int level types
    if isinstance(level, str):
        log_level: int = getattr(logging, level.upper())
    else:
        log_level = level

    logger_instance.setLevel(log_level)

    # Clear existing handlers to avoid duplicates
    logger_instance.handlers.clear()

    # Create formatter
    formatter: logging.Formatter = logging.Formatter(format_string)

    # Console handler
    console_handler: logging.StreamHandler[sys.TextIOWrapper] = logging.StreamHandler(
        sys.stdout
    )
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger_instance.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path: Path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler: logging.FileHandler = logging.FileHandler(str(log_path))
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger_instance.addHandler(file_handler)

    return logger_instance


class Logger:
    """
    A wrapper around the standard logging module with robust typing.
    """

    def __init__(self) -> None:
        self._logger = _setup_logger()

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log a debug message."""
        self._logger.debug(message, *args, **kwargs)

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log an info message."""
        self._logger.info(message, *args, **kwargs)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log a warning message."""
        self._logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log an error message."""
        self._logger.error(message, *args, **kwargs)

    def critical(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log a critical message."""
        self._logger.critical(message, *args, **kwargs)

    def exception(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log an exception with traceback."""
        self._logger.exception(message, *args, **kwargs)

    def save(self, data: Any, file_path: str, indent: int = 4, **kwargs: Any) -> None:
        """Save data to a JSON file."""
        with open(f"./logs/{file_path}", "w") as f:
            json.dump(data, f, indent=indent, **kwargs)


# Global logger instance
logger = Logger()
