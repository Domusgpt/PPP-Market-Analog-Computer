"""
Distributed Tracing System
==========================

Provides end-to-end tracing for encoding pipelines.

Features:
- Span-based tracing
- Parent-child relationships
- Timing and metadata
- Export to various formats
"""

import time
import threading
import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
from pathlib import Path
import json


@dataclass
class Span:
    """A single trace span representing an operation."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    status: str = "OK"
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    children: List['Span'] = field(default_factory=list)

    def finish(self):
        """Mark span as finished."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000

    def set_tag(self, key: str, value: Any):
        """Add a tag to the span."""
        self.tags[key] = value

    def log(self, message: str, **fields):
        """Add a log entry to the span."""
        self.logs.append({
            "timestamp": time.time(),
            "message": message,
            **fields
        })

    def set_error(self, error: Exception):
        """Mark span as error."""
        self.status = "ERROR"
        self.tags["error"] = True
        self.tags["error.type"] = type(error).__name__
        self.tags["error.message"] = str(error)

    def to_dict(self) -> Dict:
        """Convert span to dictionary."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "operation_name": self.operation_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "tags": self.tags,
            "logs": self.logs,
            "children": [c.to_dict() for c in self.children]
        }


class Tracer:
    """
    Distributed tracer for encoding pipelines.

    Usage:
        tracer = Tracer()

        # Start a trace
        with tracer.start_trace("encode_audio") as trace:
            # Create child spans
            with tracer.start_span("load_audio") as span:
                span.set_tag("file_size", 1024)
                audio = load_audio()

            with tracer.start_span("process_frames") as span:
                for frame in frames:
                    with tracer.start_span("encode_frame") as frame_span:
                        frame_span.set_tag("frame_id", frame.id)
                        encode(frame)

        # Get trace data
        trace_data = tracer.get_trace(trace.trace_id)

        # Export
        tracer.export_json("traces/encode_trace.json")
    """

    _context = threading.local()

    def __init__(self, service_name: str = "optical_encoder"):
        self.service_name = service_name
        self._lock = threading.Lock()
        self._traces: Dict[str, Span] = {}
        self._active_spans: Dict[str, List[Span]] = {}

    def _get_current_span(self) -> Optional[Span]:
        """Get current active span from thread context."""
        if not hasattr(self._context, 'span_stack'):
            self._context.span_stack = []

        if self._context.span_stack:
            return self._context.span_stack[-1]
        return None

    def _push_span(self, span: Span):
        """Push span onto thread-local stack."""
        if not hasattr(self._context, 'span_stack'):
            self._context.span_stack = []
        self._context.span_stack.append(span)

    def _pop_span(self) -> Optional[Span]:
        """Pop span from thread-local stack."""
        if hasattr(self._context, 'span_stack') and self._context.span_stack:
            return self._context.span_stack.pop()
        return None

    @contextmanager
    def start_trace(self, operation_name: str, **tags):
        """
        Start a new trace (root span).

        Usage:
            with tracer.start_trace("encode_audio") as trace:
                # All spans created here are children of this trace
                ...
        """
        trace_id = str(uuid.uuid4())
        span_id = str(uuid.uuid4())[:16]

        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=None,
            operation_name=operation_name,
            start_time=time.time(),
            tags={
                "service": self.service_name,
                **tags
            }
        )

        with self._lock:
            self._traces[trace_id] = span
            self._active_spans[trace_id] = [span]

        self._push_span(span)

        try:
            yield span
        except Exception as e:
            span.set_error(e)
            raise
        finally:
            span.finish()
            self._pop_span()

    @contextmanager
    def start_span(self, operation_name: str, **tags):
        """
        Start a child span.

        Usage:
            with tracer.start_span("load_data") as span:
                span.set_tag("size", 1024)
                data = load()
        """
        parent = self._get_current_span()

        if parent is None:
            # No active trace, create standalone span
            trace_id = str(uuid.uuid4())
            parent_span_id = None
        else:
            trace_id = parent.trace_id
            parent_span_id = parent.span_id

        span_id = str(uuid.uuid4())[:16]

        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            start_time=time.time(),
            tags=tags
        )

        # Add as child of parent
        if parent:
            parent.children.append(span)

        with self._lock:
            if trace_id in self._active_spans:
                self._active_spans[trace_id].append(span)

        self._push_span(span)

        try:
            yield span
        except Exception as e:
            span.set_error(e)
            raise
        finally:
            span.finish()
            self._pop_span()

    def get_trace(self, trace_id: str) -> Optional[Dict]:
        """Get trace data by ID."""
        with self._lock:
            if trace_id in self._traces:
                return self._traces[trace_id].to_dict()
        return None

    def get_all_traces(self) -> List[Dict]:
        """Get all recorded traces."""
        with self._lock:
            return [span.to_dict() for span in self._traces.values()]

    def get_trace_summary(self, trace_id: str) -> Optional[Dict]:
        """Get summary of a trace."""
        trace = self.get_trace(trace_id)
        if not trace:
            return None

        def count_spans(span_dict):
            count = 1
            for child in span_dict.get('children', []):
                count += count_spans(child)
            return count

        def get_error_spans(span_dict):
            errors = []
            if span_dict.get('status') == 'ERROR':
                errors.append(span_dict['operation_name'])
            for child in span_dict.get('children', []):
                errors.extend(get_error_spans(child))
            return errors

        return {
            "trace_id": trace_id,
            "root_operation": trace['operation_name'],
            "total_duration_ms": trace['duration_ms'],
            "total_spans": count_spans(trace),
            "status": trace['status'],
            "errors": get_error_spans(trace)
        }

    def print_trace(self, trace_id: str, indent: int = 0):
        """Print trace tree to console."""
        trace = self.get_trace(trace_id)
        if not trace:
            print(f"Trace not found: {trace_id}")
            return

        def print_span(span_dict, level=0):
            prefix = "  " * level
            status = "✓" if span_dict['status'] == 'OK' else "✗"
            duration = span_dict.get('duration_ms', 0)
            print(f"{prefix}{status} {span_dict['operation_name']} ({duration:.2f}ms)")

            for child in span_dict.get('children', []):
                print_span(child, level + 1)

        print(f"\n--- TRACE: {trace_id} ---")
        print_span(trace)
        print()

    def export_json(self, path: Path):
        """Export all traces to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        traces = self.get_all_traces()

        with open(path, 'w') as f:
            json.dump({
                "service": self.service_name,
                "traces": traces
            }, f, indent=2)

    def export_jaeger(self, trace_id: str) -> Dict:
        """
        Export trace in Jaeger-compatible format.

        Can be imported into Jaeger UI for visualization.
        """
        trace = self.get_trace(trace_id)
        if not trace:
            return {}

        def to_jaeger_span(span_dict) -> Dict:
            return {
                "traceID": span_dict['trace_id'].replace('-', ''),
                "spanID": span_dict['span_id'],
                "operationName": span_dict['operation_name'],
                "references": [
                    {
                        "refType": "CHILD_OF",
                        "traceID": span_dict['trace_id'].replace('-', ''),
                        "spanID": span_dict['parent_span_id']
                    }
                ] if span_dict['parent_span_id'] else [],
                "startTime": int(span_dict['start_time'] * 1000000),  # microseconds
                "duration": int((span_dict.get('duration_ms', 0)) * 1000),
                "tags": [
                    {"key": k, "type": "string", "value": str(v)}
                    for k, v in span_dict.get('tags', {}).items()
                ],
                "logs": [
                    {
                        "timestamp": int(log['timestamp'] * 1000000),
                        "fields": [{"key": "message", "value": log['message']}]
                    }
                    for log in span_dict.get('logs', [])
                ],
                "processID": "p1",
                "warnings": None
            }

        def collect_spans(span_dict) -> List[Dict]:
            spans = [to_jaeger_span(span_dict)]
            for child in span_dict.get('children', []):
                spans.extend(collect_spans(child))
            return spans

        return {
            "data": [{
                "traceID": trace['trace_id'].replace('-', ''),
                "spans": collect_spans(trace),
                "processes": {
                    "p1": {
                        "serviceName": self.service_name,
                        "tags": []
                    }
                }
            }]
        }

    def clear(self):
        """Clear all traces."""
        with self._lock:
            self._traces.clear()
            self._active_spans.clear()


# Global tracer instance
trace = Tracer()
