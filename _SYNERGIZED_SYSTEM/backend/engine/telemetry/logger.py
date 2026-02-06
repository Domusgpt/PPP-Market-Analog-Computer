"""
Structured Logging System
=========================

Provides structured, contextual logging for the optical encoding system.

Features:
- JSON-structured log output
- Context propagation
- Log levels with filtering
- File and console handlers
- Correlation IDs for tracing
"""

import logging
import json
import sys
import threading
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import contextmanager
import functools


class LogLevel:
    """Log levels."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


@dataclass
class LogContext:
    """Logging context that propagates through operations."""
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    operation: str = ""
    component: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add context if available
        if hasattr(record, 'context') and record.context:
            log_entry["correlation_id"] = record.context.correlation_id
            if record.context.operation:
                log_entry["operation"] = record.context.operation
            if record.context.component:
                log_entry["component"] = record.context.component
            if record.context.metadata:
                log_entry["metadata"] = record.context.metadata

        # Add extra fields
        if hasattr(record, 'extra_data'):
            log_entry["data"] = record.extra_data

        # Add exception info
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


class ConsoleFormatter(logging.Formatter):
    """Human-readable console formatter."""

    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, '')
        reset = self.RESET

        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        # Build prefix
        prefix_parts = [f"{color}[{record.levelname[0]}]{reset}"]
        prefix_parts.append(f"[{timestamp}]")

        if hasattr(record, 'context') and record.context:
            if record.context.correlation_id:
                prefix_parts.append(f"[{record.context.correlation_id}]")
            if record.context.operation:
                prefix_parts.append(f"[{record.context.operation}]")

        prefix = " ".join(prefix_parts)
        message = record.getMessage()

        # Add extra data
        if hasattr(record, 'extra_data') and record.extra_data:
            data_str = " | " + " ".join(f"{k}={v}" for k, v in record.extra_data.items())
            message += data_str

        return f"{prefix} {message}"


class StructuredLogger:
    """
    Structured logger with context support.

    Usage:
        logger = StructuredLogger("optical_encoder")

        # Basic logging
        logger.info("Starting encoding")
        logger.error("Encoding failed", error=str(e))

        # With context
        with logger.context(operation="encode", frame_id=123):
            logger.info("Processing frame")
            # All logs in this block have the context

        # Log with data
        logger.info("Encoding complete", duration_ms=45.2, features=12)
    """

    _context_stack = threading.local()

    def __init__(
        self,
        name: str = "optical_encoder",
        level: int = logging.INFO,
        log_file: Optional[Path] = None,
        json_output: bool = False
    ):
        self.name = name
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        self._logger.handlers = []  # Clear existing handlers

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        if json_output:
            console_handler.setFormatter(StructuredFormatter())
        else:
            console_handler.setFormatter(ConsoleFormatter())

        self._logger.addHandler(console_handler)

        # File handler (always JSON)
        if log_file:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(StructuredFormatter())
            self._logger.addHandler(file_handler)

    def _get_context(self) -> Optional[LogContext]:
        """Get current context from thread-local stack."""
        if not hasattr(self._context_stack, 'stack'):
            self._context_stack.stack = []

        if self._context_stack.stack:
            return self._context_stack.stack[-1]
        return None

    def _log(self, level: int, message: str, **kwargs):
        """Internal log method with context and extra data."""
        record = self._logger.makeRecord(
            self.name, level, "", 0, message, (), None
        )
        record.context = self._get_context()
        record.extra_data = kwargs if kwargs else None

        self._logger.handle(record)

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message."""
        self._log(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self._log(logging.CRITICAL, message, **kwargs)

    def exception(self, message: str, **kwargs):
        """Log exception with traceback."""
        self._logger.exception(message, extra={'extra_data': kwargs})

    @contextmanager
    def context(
        self,
        operation: str = "",
        component: str = "",
        correlation_id: Optional[str] = None,
        **metadata
    ):
        """
        Context manager for scoped logging context.

        Usage:
            with logger.context(operation="encode", frame_id=123):
                logger.info("Processing")  # Includes context
        """
        if not hasattr(self._context_stack, 'stack'):
            self._context_stack.stack = []

        # Inherit correlation_id from parent context
        parent = self._get_context()
        if correlation_id is None and parent:
            correlation_id = parent.correlation_id

        ctx = LogContext(
            correlation_id=correlation_id or str(uuid.uuid4())[:8],
            operation=operation,
            component=component,
            metadata=metadata
        )

        self._context_stack.stack.append(ctx)
        try:
            yield ctx
        finally:
            self._context_stack.stack.pop()

    def logged(self, level: int = logging.INFO):
        """
        Decorator for logging function entry/exit.

        Usage:
            @logger.logged()
            def my_function(x, y):
                ...
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                func_name = func.__name__

                self._log(level, f"Entering {func_name}",
                         args_count=len(args), kwargs_keys=list(kwargs.keys()))

                try:
                    result = func(*args, **kwargs)
                    self._log(level, f"Exiting {func_name}", success=True)
                    return result
                except Exception as e:
                    self._log(logging.ERROR, f"Exception in {func_name}",
                             error=str(e), error_type=type(e).__name__)
                    raise

            return wrapper
        return decorator

    def set_level(self, level: int):
        """Set logging level."""
        self._logger.setLevel(level)
        for handler in self._logger.handlers:
            handler.setLevel(level)


# Global logger instance
logger = StructuredLogger()
