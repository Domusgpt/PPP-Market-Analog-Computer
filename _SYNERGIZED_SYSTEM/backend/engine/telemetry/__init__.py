"""
Telemetry Module - Observability Infrastructure
===============================================

Provides comprehensive monitoring, profiling, and debugging capabilities:

- Metrics: Performance counters, timing, histograms
- Profiler: CPU/memory profiling with flame graphs
- Logger: Structured logging with context
- Tracer: Distributed tracing for encoding pipelines

Usage:
    from src.telemetry import metrics, logger, profiler

    @metrics.timed("encode_operation")
    def my_function():
        with logger.context(operation="encode"):
            ...

    # Get metrics report
    report = metrics.get_report()
"""

from .metrics import MetricsCollector, metrics
from .profiler import Profiler, profile
from .logger import StructuredLogger, logger
from .tracer import Tracer, trace

__all__ = [
    "MetricsCollector",
    "metrics",
    "Profiler",
    "profile",
    "StructuredLogger",
    "logger",
    "Tracer",
    "trace",
]
