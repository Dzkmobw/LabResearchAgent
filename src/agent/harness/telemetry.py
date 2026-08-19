"""Small local telemetry helpers.

LangSmith traces are enabled by decorating graph nodes. These local trace
records make the same run understandable even without LangSmith credentials.
"""

from __future__ import annotations

from typing import Any

from agent.schemas import TraceEvent


def node_event(node: str, status: str, message: str, **metadata: Any) -> TraceEvent:
    """Create a node-level trace event."""
    return TraceEvent(node=node, status=status, message=message, metadata=metadata)
