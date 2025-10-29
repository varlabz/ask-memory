"""
MCP Server for Memory History.

Provides tools to add and search conversation memory history.
"""

import argparse

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from ask.core.config import EmbedderConfig, LLMConfig, load_config
from ask_memory.memory_history.memory_history import MemoryChunk, MemoryHistory


class MemoryHistoryResult(BaseModel):
    """Memory history search result for MCP tools."""
    
    query: str = Field(description="The original user query or request")
    content: str = Field(description="The LLM response or content")
    context: str = Field(description="One sentence summary of the interaction")
    tags: str = Field(description="Broad categories or themes")
    keywords: str = Field(description="Key concepts and terminology")
    timestamp: str = Field(description="ISO format timestamp of when the memory was created")


mcp = FastMCP("Memory History", instructions="Store and search conversation memory")
_memory: MemoryHistory


@mcp.tool()
async def add_memory_history(
    query: str = Field(description="The user query or prompt"),
    response: str = Field(description="The LLM response or content"),
) -> str:
    """Add a new memory history entry with automatic analysis."""
    await _memory.add(query, response)
    return f"Memory entry added: {query[:50]}..."


@mcp.tool()
async def search_memory_history(
    query: str = Field(description="The search query to find related memories"),
) -> list[MemoryHistoryResult]:
    """Search memory history for entries related to the query."""
    results = await _memory.search(query)
    return [
        MemoryHistoryResult(
            query=chunk.metadata.query,
            content=chunk.metadata.content,
            context=chunk.metadata.context,
            tags=chunk.metadata.tags,
            keywords=chunk.metadata.keywords,
            timestamp=chunk.metadata.timestamp,
        )
        for chunk in results
    ]


def main():
    """Main entry point for the MCP server."""
    parser = argparse.ArgumentParser(description="Memory History MCP Server")
    parser.add_argument("-c", "--config", default=".ask.yaml", help="Config file path")
    parser.add_argument("--transport", choices=["stdio", "sse", "http"], default="stdio")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO")
    args = parser.parse_args()

    global _memory
    llm = load_config([args.config], LLMConfig, "llm")
    embedder = load_config([args.config], EmbedderConfig, "embedder")
    _memory = MemoryHistory(llm, embedder)

    mcp.settings.log_level = args.log_level
    transport = "streamable-http" if args.transport == "http" else args.transport
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()

