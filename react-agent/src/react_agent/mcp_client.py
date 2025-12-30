"""MCP (Model Context Protocol) client for integrating external services."""

import json
import os
from typing import Any, Dict, Optional

import aiohttp
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClient:
    """Client for managing MCP server connections."""

    @staticmethod
    async def call_firecrawl(
        action: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call Firecrawl MCP server.

        Args:
            action: Action to perform ("scrape", "crawl", "search")
            params: Parameters for the action

        Returns:
            Result from Firecrawl

        Example:
            result = await MCPClient.call_firecrawl("scrape", {
                "url": "https://me.go.kr/...",
                "formats": ["markdown", "html"]
            })
        """
        try:
            # MCP server configuration
            server_params = StdioServerParameters(
                command="npx",
                args=["-y", "firecrawl-mcp"],
                env={
                    "FIRECRAWL_API_KEY": os.getenv("FIRECRAWL_API_KEY", "fc-1f87ff40c3104275bb6a949320ed9cf0"),
                    "FIRECRAWL_RETRY_MAX_ATTEMPTS": "5",
                    "FIRECRAWL_RETRY_INITIAL_DELAY": "2000",
                    "FIRECRAWL_RETRY_MAX_DELAY": "30000",
                    "FIRECRAWL_RETRY_BACKOFF_FACTOR": "3",
                }
            )

            # Connect to MCP server and call tool
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    # Initialize the session
                    await session.initialize()

                    # Call the appropriate tool based on action
                    tool_name = f"firecrawl_{action}"
                    result = await session.call_tool(tool_name, params)

                    return result

        except Exception as e:
            # Fallback to mock data if MCP fails
            return {
                "success": False,
                "error": str(e),
                "fallback": "MCP connection failed - using mock data"
            }


async def scrape_url(url: str, formats: list[str] = None) -> Dict[str, Any]:
    """Scrape a single URL using Firecrawl.

    Args:
        url: URL to scrape
        formats: List of formats to return (["markdown", "html"])

    Returns:
        Scraped content
    """
    if formats is None:
        formats = ["markdown"]

    return await MCPClient.call_firecrawl("scrape", {
        "url": url,
        "formats": formats
    })


async def search_web(query: str, limit: int = 5) -> Dict[str, Any]:
    """Search the web using Firecrawl.

    Args:
        query: Search query
        limit: Maximum number of results

    Returns:
        Search results
    """
    return await MCPClient.call_firecrawl("search", {
        "query": query,
        "limit": limit
    })


async def crawl_site(url: str, max_depth: int = 2) -> Dict[str, Any]:
    """Crawl a website using Firecrawl.

    Args:
        url: Starting URL
        max_depth: Maximum crawl depth

    Returns:
        Crawled content
    """
    return await MCPClient.call_firecrawl("crawl", {
        "url": url,
        "maxDepth": max_depth
    })
