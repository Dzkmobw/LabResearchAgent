"""Optional MCP-backed web search tool.

The project keeps MCP behind the ToolGateway. If an MCP search server is not
configured, this tool returns a structured skipped result instead of pretending
that web search happened.
"""

from __future__ import annotations

import json
import os
from typing import Any

from agent.schemas import Evidence, SourceType, ToolResult


class MCPWebSearchTool:
    """Adapter for a web search MCP tool exposed through langchain-mcp-adapters."""

    async def search(self, query: str, limit: int = 5) -> ToolResult:
        """Run web search through a configured MCP server."""
        servers_json = os.getenv("LAB_AGENT_MCP_SERVERS")
        preferred_tool = os.getenv("LAB_AGENT_MCP_SEARCH_TOOL")

        if not servers_json:
            return ToolResult(
                tool_name="mcp.web_search",
                status="skipped",
                error_type="mcp_not_configured",
                message=(
                    "Set LAB_AGENT_MCP_SERVERS and LAB_AGENT_MCP_SEARCH_TOOL to enable "
                    "MCP web search."
                ),
            )

        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
        except ImportError:
            return ToolResult(
                tool_name="mcp.web_search",
                status="skipped",
                error_type="mcp_adapter_missing",
                message="Install langchain-mcp-adapters to enable MCP web search.",
            )

        servers = json.loads(servers_json)
        client = MultiServerMCPClient(servers)
        tools = await client.get_tools()
        selected = _select_search_tool(tools, preferred_tool)
        if selected is None:
            return ToolResult(
                tool_name="mcp.web_search",
                status="error",
                error_type="mcp_search_tool_missing",
                message="No MCP tool with a search-like name was found.",
            )

        raw = await _invoke_search_tool(selected, query, limit)
        evidence = _normalize_search_result(raw, query)
        return ToolResult(
            tool_name="mcp.web_search",
            status="ok",
            items=[raw],
            evidence=evidence,
            metadata={"query": query, "limit": limit},
        )


def _select_search_tool(tools: list[Any], preferred_tool: str | None) -> Any | None:
    if preferred_tool:
        for tool in tools:
            if getattr(tool, "name", None) == preferred_tool:
                return tool
    for tool in tools:
        name = (getattr(tool, "name", "") or "").lower()
        if "search" in name or "web" in name:
            return tool
    return None


async def _invoke_search_tool(tool: Any, query: str, limit: int) -> Any:
    try:
        return await tool.ainvoke({"query": query, "max_results": limit})
    except TypeError:
        return await tool.ainvoke({"query": query})


def _normalize_search_result(raw: Any, query: str) -> list[Evidence]:
    rows = raw if isinstance(raw, list) else [raw]
    evidence: list[Evidence] = []
    for row in rows[:8]:
        if isinstance(row, dict):
            url = row.get("url") or row.get("link") or row.get("source_url")
            title = row.get("title") or row.get("name") or "MCP web search result"
            snippet = row.get("snippet") or row.get("content") or row.get("text") or ""
        else:
            url = None
            title = "MCP web search result"
            snippet = str(row)[:240]
        evidence.append(
            Evidence(
                claim=f"MCP web search returned a candidate web source for: {query}. {snippet[:180]}",
                source_url=url,
                source_title=title,
                source_type=SourceType.WEB_SEARCH,
                confidence=0.55,
                is_inference=False,
                metadata={"query": query},
            )
        )
    return evidence
