"""
Metrics Collection System
=========================

Collects performance metrics for the optical encoding system:
- Timing measurements (encode time, cascade time, etc.)
- Counters (number of encodings, state transitions)
- Histograms (feature distributions, intensity ranges)
- Gauges (current state, memory usage)
"""

import time
import functools
import threading
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from collections import defaultdict
import json
from pathlib import Path


@dataclass
class TimingSample:
    """A single timing measurement."""
    name: str
    duration_ms: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricsSummary:
    """Summary statistics for a metric."""
    count: int
    total: float
    mean: float
    std: float
    min: float
    max: float
    p50: float
    p95: float
    p99: float


class MetricsCollector:
    """
    Collects and aggregates performance metrics.

    Thread-safe metrics collection with support for:
    - Timers: Measure operation durations
    - Counters: Track event counts
    - Histograms: Distribution of values
    - Gauges: Point-in-time values

    Usage:
        metrics = MetricsCollector()

        # Timer
        with metrics.timer("encode_time"):
            result = encoder.encode(data)

        # Or as decorator
        @metrics.timed("encode_operation")
        def encode(data):
            ...

        # Counter
        metrics.increment("encodings_total")

        # Histogram
        metrics.record("intensity_mean", value)

        # Gauge
        metrics.set_gauge("active_cells", count)

        # Get report
        report = metrics.get_report()
    """

    def __init__(self, name: str = "optical_encoder"):
        self.name = name
        self._lock = threading.Lock()

        # Storage
        self._timings: Dict[str, List[float]] = defaultdict(list)
        self._counters: Dict[str, int] = defaultdict(int)
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._gauges: Dict[str, float] = {}

        # Timing samples for detailed analysis
        self._samples: List[TimingSample] = []
        self._max_samples = 10000

        # Start time
        self._start_time = time.time()

    def timer(self, name: str, metadata: Optional[Dict] = None):
        """
        Context manager for timing operations.

        Usage:
            with metrics.timer("encode_frame"):
                process_frame()
        """
        return _TimerContext(self, name, metadata or {})

    def timed(self, name: str):
        """
        Decorator for timing functions.

        Usage:
            @metrics.timed("encode_data")
            def encode_data(self, data):
                ...
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.timer(name):
                    return func(*args, **kwargs)
            return wrapper
        return decorator

    def record_timing(
        self,
        name: str,
        duration_ms: float,
        metadata: Optional[Dict] = None
    ):
        """Record a timing measurement."""
        with self._lock:
            self._timings[name].append(duration_ms)

            sample = TimingSample(
                name=name,
                duration_ms=duration_ms,
                timestamp=time.time(),
                metadata=metadata or {}
            )
            self._samples.append(sample)

            # Limit sample storage
            if len(self._samples) > self._max_samples:
                self._samples = self._samples[-self._max_samples:]

    def increment(self, name: str, value: int = 1):
        """Increment a counter."""
        with self._lock:
            self._counters[name] += value

    def decrement(self, name: str, value: int = 1):
        """Decrement a counter."""
        with self._lock:
            self._counters[name] -= value

    def record(self, name: str, value: float):
        """Record a value to a histogram."""
        with self._lock:
            self._histograms[name].append(value)

    def set_gauge(self, name: str, value: float):
        """Set a gauge value."""
        with self._lock:
            self._gauges[name] = value

    def get_gauge(self, name: str) -> Optional[float]:
        """Get current gauge value."""
        with self._lock:
            return self._gauges.get(name)

    def get_counter(self, name: str) -> int:
        """Get current counter value."""
        with self._lock:
            return self._counters.get(name, 0)

    def _compute_summary(self, values: List[float]) -> MetricsSummary:
        """Compute summary statistics for a list of values."""
        if not values:
            return MetricsSummary(0, 0, 0, 0, 0, 0, 0, 0, 0)

        sorted_values = sorted(values)
        n = len(sorted_values)

        return MetricsSummary(
            count=n,
            total=sum(values),
            mean=statistics.mean(values),
            std=statistics.stdev(values) if n > 1 else 0,
            min=min(values),
            max=max(values),
            p50=sorted_values[int(n * 0.50)],
            p95=sorted_values[min(int(n * 0.95), n - 1)],
            p99=sorted_values[min(int(n * 0.99), n - 1)]
        )

    def get_timing_summary(self, name: str) -> Optional[MetricsSummary]:
        """Get summary for a timing metric."""
        with self._lock:
            if name not in self._timings:
                return None
            return self._compute_summary(self._timings[name])

    def get_histogram_summary(self, name: str) -> Optional[MetricsSummary]:
        """Get summary for a histogram."""
        with self._lock:
            if name not in self._histograms:
                return None
            return self._compute_summary(self._histograms[name])

    def get_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive metrics report.

        Returns dict with:
        - timings: Summary stats for each timed operation
        - counters: Current counter values
        - histograms: Summary stats for each histogram
        - gauges: Current gauge values
        - meta: Collection metadata
        """
        with self._lock:
            report = {
                "name": self.name,
                "meta": {
                    "collection_start": self._start_time,
                    "collection_duration_s": time.time() - self._start_time,
                    "total_samples": len(self._samples)
                },
                "timings": {},
                "counters": dict(self._counters),
                "histograms": {},
                "gauges": dict(self._gauges)
            }

            for name, values in self._timings.items():
                summary = self._compute_summary(values)
                report["timings"][name] = {
                    "count": summary.count,
                    "total_ms": summary.total,
                    "mean_ms": summary.mean,
                    "std_ms": summary.std,
                    "min_ms": summary.min,
                    "max_ms": summary.max,
                    "p50_ms": summary.p50,
                    "p95_ms": summary.p95,
                    "p99_ms": summary.p99
                }

            for name, values in self._histograms.items():
                summary = self._compute_summary(values)
                report["histograms"][name] = {
                    "count": summary.count,
                    "mean": summary.mean,
                    "std": summary.std,
                    "min": summary.min,
                    "max": summary.max,
                    "p50": summary.p50,
                    "p95": summary.p95,
                    "p99": summary.p99
                }

            return report

    def print_report(self):
        """Print formatted metrics report to console."""
        report = self.get_report()

        print("\n" + "=" * 60)
        print(f"METRICS REPORT: {report['name']}")
        print("=" * 60)

        print(f"\nCollection Duration: {report['meta']['collection_duration_s']:.2f}s")
        print(f"Total Samples: {report['meta']['total_samples']}")

        if report['timings']:
            print("\n--- TIMINGS ---")
            for name, stats in report['timings'].items():
                print(f"\n  {name}:")
                print(f"    count: {stats['count']}")
                print(f"    mean:  {stats['mean_ms']:.3f} ms")
                print(f"    p95:   {stats['p95_ms']:.3f} ms")
                print(f"    p99:   {stats['p99_ms']:.3f} ms")

        if report['counters']:
            print("\n--- COUNTERS ---")
            for name, value in report['counters'].items():
                print(f"  {name}: {value}")

        if report['gauges']:
            print("\n--- GAUGES ---")
            for name, value in report['gauges'].items():
                print(f"  {name}: {value}")

        print("\n" + "=" * 60)

    def save_report(self, path: Path):
        """Save metrics report to JSON file."""
        report = self.get_report()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(report, f, indent=2)

    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self._timings.clear()
            self._counters.clear()
            self._histograms.clear()
            self._gauges.clear()
            self._samples.clear()
            self._start_time = time.time()


class _TimerContext:
    """Context manager for timing operations."""

    def __init__(self, collector: MetricsCollector, name: str, metadata: Dict):
        self.collector = collector
        self.name = name
        self.metadata = metadata
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.perf_counter() - self.start_time) * 1000
        self.metadata['success'] = exc_type is None
        self.collector.record_timing(self.name, duration_ms, self.metadata)
        return False


# Global metrics instance
metrics = MetricsCollector()
