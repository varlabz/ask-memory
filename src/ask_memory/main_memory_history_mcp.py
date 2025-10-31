"""
MCP Server for Memory History.

Provides tools to add and search conversation memory history.
"""

import argparse
from datetime import datetime, timedelta

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
    timestamp: int = Field(description="Unix timestamp of when the memory was created")


mcp = FastMCP("Memory History", instructions="Store and search conversation memory")
_memory: MemoryHistory


@mcp.tool()
async def get_history_time(
    # seconds: int = Field(description="Number of seconds to go back in time from now", default=0),
    # minutes: int = Field(description="Number of minutes to go back in time from now", default=0),
    hours: int = Field(description="Number of hours to go back in time from now", default=0),
    days: int = Field(description="Number of days to go back in time from now", default=0),
    months: int = Field(description="Number of months (30 days each) to go back in time from now", default=0),
) -> int:
    """
    Get a Unix timestamp for the current time or a past time.
    
    Use this tool to:
    - Get current Unix timestamp (call with no arguments)
    - Calculate a timestamp for a past time (e.g., 7 days ago, 2 hours ago)
    - Generate 'after' parameter values for query_memory_history
    
    All time parameters go backwards from now. For example:
    - get_history_time() returns current timestamp
    - get_history_time(days=7) returns timestamp from 7 days ago
    - get_history_time(hours=2) returns timestamp from 2 hours ago
    - get_history_time(months=1) returns timestamp from 30 days ago

    Returns: Unix timestamp as integer (seconds since Jan 1, 1970 UTC)
    """
    now = datetime.now()
    offset = timedelta(
        # seconds=seconds,
        # minutes=minutes,
        hours=hours,
        days=days + (months * 30)
    )
    result_time = now - offset
    return int(result_time.timestamp())


@mcp.tool()
async def add_memory_history(
    query: str = Field(description="The user's original query or prompt text"),
    response: str = Field(description="The assistant's response or content that was generated"),
) -> MemoryHistoryResult:
    """
    Store a conversation exchange (query and response) in memory history.
    
    Use this tool to:
    - Save important conversation interactions for future reference
    - Build a searchable history of Q&A exchanges
    - Enable contextual retrieval of past conversations
    
    The system automatically analyzes and indexes the entry with:
    - One-sentence context summary
    - Relevant keywords and tags
    - Searchable embeddings
    - Timestamp for temporal filtering
    
    Example usage:
    - After generating a useful response, save it for future reference
    - Store user preferences or decisions made during conversations
    - Keep track of factual information provided to the user
    
    Returns: The created memory entry with all analyzed metadata
    """
    chunk = await _memory.add(query, response)
    return MemoryHistoryResult(
        query=chunk.metadata.query,
        content=chunk.metadata.content,
        context=chunk.metadata.context,
        tags=chunk.metadata.tags,
        keywords=chunk.metadata.keywords,
        timestamp=chunk.metadata.timestamp,
    )


async def get_memory_history_page(
    page: int = Field(description="Page number to retrieve (1-based index)", default=1),
    page_size: int = Field(description="Maximum number of results to return per page (1-100)", default=50),
    after: int | None = Field(description="Unix timestamp - only return entries created at or before this time. Use get_time() to generate this value.", default=None),
) -> list[MemoryHistoryResult]:
    """
    Retrieve a paginated list of memory history entries sorted by timestamp.
    
    Use this tool to:
    - Browse through all stored memories chronologically
    - Get recent conversation history
    - Retrieve memories within a specific time range
    
    Filtering by time:
    - Use 'after' parameter to get only older memories
    - First use get_history_time() to generate the timestamp, e.g., get_history_time(days=7) for memories older than 7 days
    - If after is not provided, returns all entries up to now
    
    Pagination:
    - Results are sorted by timestamp (oldest to newest)
    - Use 'page' to navigate through results
    - Adjust 'page_size' to control how many results per page
    
    Returns: List of memory entries with full analyzed metadata
    """
    if after is None:
        after = int(datetime.now().timestamp())
    chunks = _memory.get_page(page=page, page_size=page_size, after=after)
    return [
        MemoryHistoryResult(
            query=chunk.metadata.query,
            content=chunk.metadata.content,
            context=chunk.metadata.context,
            tags=chunk.metadata.tags,
            keywords=chunk.metadata.keywords,
            timestamp=chunk.metadata.timestamp,
        )
        for chunk in chunks
    ]

@mcp.tool()
async def query_memory_history(
    query: str = Field(description="Natural language search query to find semantically related memories"),
    after: int | None = Field(description="Unix timestamp - only search memories created at or before this time. Use get_time() to generate this value.", default=None),
) -> list[MemoryHistoryResult]:
    """
    Search memory history using semantic similarity and intelligent ranking.
    
    Use this tool to:
    - Find relevant past conversations based on meaning, not just keywords
    - Retrieve context from previous interactions on similar topics
    - Answer questions using information from past conversations
    - Discover related memories even with different wording
    
    How it works:
    1. Performs semantic search using embeddings (finds conceptually similar content)
    2. Ranks results by relevance using AI reranking
    3. Returns top 7 most relevant results sorted by timestamp
    
    Time filtering:
    - Use 'after' to search only older memories (e.g., "what did we discuss last week?")
    - First call get_history_time() to generate the timestamp: get_history_time(days=7)
    - If after is not provided, searches all memories up to now
    
    Search tips:
    - Use natural language queries (e.g., "discussions about Python", "what was the user's preference")
    - The search understands concepts, not just exact word matches
    - Results include the original query, response, context summary, tags, and keywords
    
    Returns: List of up to 7 most relevant memory entries with full details, sorted by timestamp
    """
    if after is None:
        after = int(datetime.now().timestamp())
    results = await _memory.query(query, after=after)
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

