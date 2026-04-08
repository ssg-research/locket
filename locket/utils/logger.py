# Authors: Tony He, Vasisht Duddu, N Asokan
# Copyright 2026 Secure Systems Group, University of Waterloo & Aalto University, https://crysp.uwaterloo.ca/research/SSG/
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Global Logger Module

This module provides a singleton logger that can be safely imported and used
across multiple modules without causing double logging.

Usage in other modules:
    from locket.utils.logger import logger

    logger.info("This message will only appear once")
    logger.error("Error message")
    logger.debug("Debug message")
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional, Union

from locket.config import PROJECT_DIR


class ColoredFormatter(logging.Formatter):
    """A logging formatter that adds ANSI color codes to log messages."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"  # Reset to default color

    def format(self, record: logging.LogRecord) -> str:
        # Add color to the level name
        if record.levelname in self.COLORS:
            colored_levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
            )
            # Temporarily replace the levelname for formatting
            original_levelname = record.levelname
            record.levelname = colored_levelname

        # Format the message
        formatted_message = super().format(record)

        # Restore original levelname
        if record.levelname in self.COLORS:
            record.levelname = original_levelname

        return formatted_message


# Global flag to track if logger has been configured
_logger_configured = False


def _setup_logger(
    name: str = "locket",
    level: Union[str, int] = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    global _logger_configured

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

    # Disable propagation to root logger to prevent double logging
    logger_instance.propagate = False

    # Only configure handlers once globally
    if not _logger_configured:
        # Clear any existing handlers to prevent duplicates
        logger_instance.handlers.clear()

        # Create formatters
        formatter: logging.Formatter = logging.Formatter(format_string)
        colored_formatter: ColoredFormatter = ColoredFormatter(format_string)

        # Console handler with colored output
        console_handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(colored_formatter)
        logger_instance.addHandler(console_handler)

        # File handler (optional)
        if log_file:
            log_path: Path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler: logging.FileHandler = logging.FileHandler(str(log_path))
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger_instance.addHandler(file_handler)

        # Mark as configured
        _logger_configured = True

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
        log_path = Path(f"{PROJECT_DIR}/logs") / file_path
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Truncate filename from the left if it's too long (max 255 chars for most filesystems)
        max_filename_length = 255
        filename = log_path.name
        if len(filename) > max_filename_length:
            # Keep the extension and truncate from the left
            extension = log_path.suffix
            base_name = log_path.stem
            truncated_base = base_name[-(max_filename_length - len(extension)) :]
            log_path = log_path.parent / f"{truncated_base}{extension}"

        with open(str(log_path), "w") as f:
            json.dump(data, f, indent=indent, **kwargs)


# Global logger instance - use this throughout the application
# Import this in other modules: from locket.utils.logger import logger
logger = Logger()
