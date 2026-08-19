"""Unified tool gateway for budgets, retries, cache, and logs."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from agent.schemas import ToolCallLog, ToolResult

ToolHandler = Callable[..., Awaitable[ToolResult]]

_CACHE: dict[str, ToolResult] = {}


class GatewayCallResult(BaseModel):
    """Tool result plus trace accounting."""

    result: ToolResult
    log: ToolCallLog
    next_tool_call_count: int


@dataclass
class ToolGateway:
    """Control layer between the graph and external tools."""

    max_tool_calls: int
    tool_timeout_seconds: int
    max_retries: int
    tool_call_count: int = 0
    tools: dict[str, ToolHandler] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.tools, dict):
            self.tools = {}

    def register(self, name: str, handler: ToolHandler) -> None:
        """Register one allowed tool."""
        self.tools[name] = handler

    async def call(self, name: str, arguments: dict[str, Any]) -> GatewayCallResult:
        """Call a registered tool with cache, retry, timeout, and budget checks."""
        started = time.perf_counter()
        summary = _summarize_arguments(arguments)

        if name not in self.tools:
            result = ToolResult(
                tool_name=name,
                status="error",
                error_type="tool_not_registered",
                message=f"Tool is not registered: {name}",
            )
            return self._finish(result, started, summary, attempts=0)

        cache_key = _cache_key(name, arguments)
        if cache_key in _CACHE:
            result = _CACHE[cache_key]
            log = ToolCallLog(
                tool_name=name,
                arguments_summary=summary,
                status=result.status,
                attempts=0,
                duration_ms=(time.perf_counter() - started) * 1000,
                cache_hit=True,
                error_type=result.error_type,
                message=result.message,
            )
            return GatewayCallResult(
                result=result,
                log=log,
                next_tool_call_count=self.tool_call_count,
            )

        if self.tool_call_count >= self.max_tool_calls:
            result = ToolResult(
                tool_name=name,
                status="skipped",
                error_type="budget_exhausted",
                message="Tool call budget exhausted.",
            )
            return self._finish(result, started, summary, attempts=0, increment=False)

        attempts = 0
        last_error: Exception | None = None
        max_attempts = self.max_retries + 1
        handler = self.tools[name]

        for attempts in range(1, max_attempts + 1):
            try:
                self.tool_call_count += 1
                result = await asyncio.wait_for(
                    handler(**arguments),
                    timeout=self.tool_timeout_seconds,
                )
                _CACHE[cache_key] = result
                return self._finish(result, started, summary, attempts=attempts)
            except TimeoutError as exc:
                last_error = exc
                if attempts >= max_attempts:
                    break
            except Exception as exc:  # noqa: BLE001 - gateway must classify tool failures.
                last_error = exc
                if attempts >= max_attempts:
                    break

        error_type = "timeout" if isinstance(last_error, TimeoutError) else "tool_error"
        result = ToolResult(
            tool_name=name,
            status="error",
            error_type=error_type,
            message=str(last_error) if last_error else "Unknown tool error.",
        )
        return self._finish(result, started, summary, attempts=attempts)

    def _finish(
        self,
        result: ToolResult,
        started: float,
        summary: dict[str, Any],
        attempts: int,
        increment: bool = True,
    ) -> GatewayCallResult:
        if increment is False:
            next_count = self.tool_call_count
        else:
            next_count = self.tool_call_count
        log = ToolCallLog(
            tool_name=result.tool_name,
            arguments_summary=summary,
            status=result.status,
            attempts=attempts,
            duration_ms=(time.perf_counter() - started) * 1000,
            error_type=result.error_type,
            message=result.message,
        )
        return GatewayCallResult(result=result, log=log, next_tool_call_count=next_count)


def _cache_key(name: str, arguments: dict[str, Any]) -> str:
    payload = json.dumps(arguments, sort_keys=True, default=str)
    return hashlib.sha256(f"{name}:{payload}".encode()).hexdigest()


def _summarize_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in arguments.items():
        if isinstance(value, str) and len(value) > 120:
            summary[key] = f"{value[:117]}..."
        elif isinstance(value, list):
            summary[key] = value[:5]
        else:
            summary[key] = value
    return summary
