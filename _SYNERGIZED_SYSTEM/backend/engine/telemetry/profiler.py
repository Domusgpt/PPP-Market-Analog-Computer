"""
Performance Profiler
====================

CPU and memory profiling for the optical encoding system.

Provides:
- Function-level timing profiles
- Memory allocation tracking
- Call graph generation
- Hotspot identification
"""

import time
import functools
import cProfile
import pstats
import io
import tracemalloc
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
from collections import defaultdict
from pathlib import Path
import json


@dataclass
class ProfileFrame:
    """A single profiling frame."""
    function_name: str
    file_path: str
    line_number: int
    duration_ms: float
    memory_delta_kb: float
    call_count: int


@dataclass
class ProfileResult:
    """Complete profiling result."""
    name: str
    total_time_ms: float
    total_memory_kb: float
    frames: List[ProfileFrame]
    call_graph: Dict[str, List[str]]
    hotspots: List[Tuple[str, float]]


class Profiler:
    """
    Performance profiler for encoding operations.

    Usage:
        profiler = Profiler()

        # Profile a block
        with profiler.profile("encode_operation"):
            result = encoder.encode(data)

        # Or as decorator
        @profiler.profiled("encode_data")
        def encode_data(data):
            ...

        # Get results
        result = profiler.get_result("encode_operation")

        # Memory profiling
        with profiler.memory_profile("memory_test"):
            large_array = np.zeros((1000, 1000))
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._lock = threading.Lock()

        # Storage
        self._profiles: Dict[str, cProfile.Profile] = {}
        self._timings: Dict[str, List[float]] = defaultdict(list)
        self._memory_snapshots: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        self._call_counts: Dict[str, int] = defaultdict(int)

        # Memory tracking state
        self._memory_tracking = False

    def profile(self, name: str):
        """
        Context manager for profiling a code block.

        Usage:
            with profiler.profile("my_operation"):
                do_work()
        """
        return _ProfileContext(self, name)

    def profiled(self, name: str):
        """
        Decorator for profiling functions.

        Usage:
            @profiler.profiled("my_function")
            def my_function():
                ...
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.profile(name):
                    return func(*args, **kwargs)
            return wrapper
        return decorator

    def memory_profile(self, name: str):
        """
        Context manager for memory profiling.

        Usage:
            with profiler.memory_profile("memory_test"):
                data = load_large_data()
        """
        return _MemoryProfileContext(self, name)

    def record_profile(
        self,
        name: str,
        duration_ms: float,
        profile: Optional[cProfile.Profile] = None
    ):
        """Record a profiling measurement."""
        with self._lock:
            self._timings[name].append(duration_ms)
            self._call_counts[name] += 1

            if profile is not None:
                self._profiles[name] = profile

    def record_memory(self, name: str, start_kb: float, end_kb: float):
        """Record memory measurement."""
        with self._lock:
            self._memory_snapshots[name].append((start_kb, end_kb))

    def get_result(self, name: str) -> Optional[ProfileResult]:
        """Get profiling result for a named operation."""
        with self._lock:
            if name not in self._timings:
                return None

            timings = self._timings[name]
            total_time = sum(timings)

            # Memory stats
            memory_data = self._memory_snapshots.get(name, [])
            total_memory = sum(end - start for start, end in memory_data)

            # Parse cProfile data if available
            frames = []
            hotspots = []
            call_graph = {}

            if name in self._profiles:
                profile = self._profiles[name]
                stats = pstats.Stats(profile)

                # Get top functions by cumulative time
                stats.sort_stats('cumulative')
                for func_info, (cc, nc, tt, ct, callers) in list(stats.stats.items())[:20]:
                    file_path, line_no, func_name = func_info

                    frame = ProfileFrame(
                        function_name=func_name,
                        file_path=file_path,
                        line_number=line_no,
                        duration_ms=ct * 1000,
                        memory_delta_kb=0,
                        call_count=nc
                    )
                    frames.append(frame)
                    hotspots.append((f"{func_name} ({file_path}:{line_no})", ct * 1000))

                    # Build call graph
                    if callers:
                        caller_names = []
                        for caller_info in callers:
                            if isinstance(caller_info, tuple) and len(caller_info) >= 3:
                                caller_names.append(caller_info[2])
                        call_graph[func_name] = caller_names

            return ProfileResult(
                name=name,
                total_time_ms=total_time,
                total_memory_kb=total_memory,
                frames=frames,
                call_graph=call_graph,
                hotspots=sorted(hotspots, key=lambda x: x[1], reverse=True)[:10]
            )

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all profiles."""
        with self._lock:
            summary = {}

            for name in self._timings:
                timings = self._timings[name]
                memory = self._memory_snapshots.get(name, [])

                summary[name] = {
                    "call_count": self._call_counts[name],
                    "total_time_ms": sum(timings),
                    "mean_time_ms": sum(timings) / len(timings) if timings else 0,
                    "total_memory_kb": sum(e - s for s, e in memory),
                }

            return summary

    def print_summary(self):
        """Print formatted profiling summary."""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("PROFILING SUMMARY")
        print("=" * 60)

        for name, stats in summary.items():
            print(f"\n{name}:")
            print(f"  Calls: {stats['call_count']}")
            print(f"  Total Time: {stats['total_time_ms']:.2f} ms")
            print(f"  Mean Time: {stats['mean_time_ms']:.2f} ms")
            if stats['total_memory_kb'] > 0:
                print(f"  Memory Delta: {stats['total_memory_kb']:.2f} KB")

        print("\n" + "=" * 60)

    def print_hotspots(self, name: str, top_n: int = 10):
        """Print hotspots for a specific profile."""
        result = self.get_result(name)
        if not result:
            print(f"No profile found for '{name}'")
            return

        print(f"\n--- HOTSPOTS: {name} ---")
        for i, (func, time_ms) in enumerate(result.hotspots[:top_n], 1):
            print(f"  {i}. {func}: {time_ms:.2f} ms")

    def save_profile(self, name: str, path: Path):
        """Save profile to file."""
        result = self.get_result(name)
        if not result:
            return

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "name": result.name,
            "total_time_ms": result.total_time_ms,
            "total_memory_kb": result.total_memory_kb,
            "hotspots": result.hotspots,
            "frames": [
                {
                    "function": f.function_name,
                    "file": f.file_path,
                    "line": f.line_number,
                    "time_ms": f.duration_ms,
                    "calls": f.call_count
                }
                for f in result.frames
            ]
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def reset(self):
        """Reset all profiling data."""
        with self._lock:
            self._profiles.clear()
            self._timings.clear()
            self._memory_snapshots.clear()
            self._call_counts.clear()


class _ProfileContext:
    """Context manager for profiling."""

    def __init__(self, profiler: Profiler, name: str):
        self.profiler = profiler
        self.name = name
        self.profile = None
        self.start_time = None

    def __enter__(self):
        if self.profiler.enabled:
            self.profile = cProfile.Profile()
            self.profile.enable()
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.perf_counter() - self.start_time) * 1000

        if self.profile:
            self.profile.disable()

        self.profiler.record_profile(self.name, duration_ms, self.profile)
        return False


class _MemoryProfileContext:
    """Context manager for memory profiling."""

    def __init__(self, profiler: Profiler, name: str):
        self.profiler = profiler
        self.name = name
        self.start_memory = 0

    def __enter__(self):
        if not tracemalloc.is_tracing():
            tracemalloc.start()

        snapshot = tracemalloc.take_snapshot()
        stats = snapshot.statistics('lineno')
        self.start_memory = sum(stat.size for stat in stats) / 1024  # KB

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        snapshot = tracemalloc.take_snapshot()
        stats = snapshot.statistics('lineno')
        end_memory = sum(stat.size for stat in stats) / 1024  # KB

        self.profiler.record_memory(self.name, self.start_memory, end_memory)
        return False


# Global profiler instance
profile = Profiler()
