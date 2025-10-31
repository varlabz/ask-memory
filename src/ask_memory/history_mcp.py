"""
MCP Server for History Management.

Provides tools to add, retrieve, count, and clear conversation history entries
stored in SQLite database.
"""

import argparse
from datetime import datetime, timedelta

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from ask_memory.core.history import History, HistoryEntry


class HistoryEntryResult(BaseModel):
    """History entry result for MCP tools."""
    
    id: int = Field(description="Unique identifier for the history entry")
    query: str = Field(description="The original user query or request")
    content: str = Field(description="The LLM response or content")
    timestamp: int = Field(description="Unix timestamp of when the entry was created")


# Initialize MCP server
mcp = FastMCP(
    "History Manager",
    instructions="Store and retrieve conversation history entries with timestamp-based filtering"
)

# Global history instance
_history: History


@mcp.tool()
async def get_timestamp(
    hours: int = Field(description="Number of hours to go back in time from now", default=0),
    days: int = Field(description="Number of days to go back in time from now", default=0),
    months: int = Field(description="Number of months (30 days each) to go back in time from now", default=0),
) -> int:
    """
    Get a Unix timestamp for the current time or a past time.
    
    Use this tool to:
    - Get current Unix timestamp (call with no arguments)
    - Calculate a timestamp for a past time (e.g., 7 days ago, 2 hours ago)
    - Generate 'after' parameter values for get_history_page
    
    All time parameters go backwards from now. For example:
    - get_timestamp() returns current timestamp
    - get_timestamp(days=7) returns timestamp from 7 days ago
    - get_timestamp(hours=2) returns timestamp from 2 hours ago
    - get_timestamp(months=1) returns timestamp from 30 days ago

    Returns:
        Unix timestamp as integer (seconds since Jan 1, 1970 UTC)
    """
    now = datetime.now()
    offset = timedelta(
        hours=hours,
        days=days + (months * 30)
    )
    result_time = now - offset
    return int(result_time.timestamp())


@mcp.tool()
async def add_history(
    query: str = Field(description="The user's original query or prompt text"),
    content: str = Field(description="The assistant's response or content that was generated"),
    timestamp: int | None = Field(
        description="Unix timestamp for the entry (optional, defaults to current time)",
        default=None
    ),
) -> HistoryEntryResult:
    """
    Add a new history entry to the database.
    
    Use this tool to:
    - Save conversation exchanges (query and response)
    - Store important interactions for future reference
    - Build a searchable history of conversations
    
    The entry is automatically timestamped if no timestamp is provided.
    
    Example usage:
    - After generating a response, save it with the original query
    - Store user preferences or decisions made during conversations
    - Keep track of all interactions chronologically
    
    Args:
        query: The user's original query or prompt
        content: The assistant's response or generated content
        timestamp: Optional Unix timestamp (defaults to current time)
    
    Returns:
        The created history entry with ID and timestamp
    """
    entry_id = _history.add_history(query=query, content=content, timestamp=timestamp)
    
    # Retrieve the entry to get the actual timestamp
    entries = _history.get_page(page=1, page_size=1)
    if entries and entries[0].id == entry_id:
        entry = entries[0]
        if entry.id is None:
            raise RuntimeError("Entry ID should not be None after insertion")
        return HistoryEntryResult(
            id=entry.id,
            query=entry.query,
            content=entry.content,
            timestamp=entry.timestamp
        )
    
    # Fallback if entry not found
    ts = timestamp if timestamp else int(datetime.now().timestamp())
    return HistoryEntryResult(
        id=entry_id,
        query=query,
        content=content,
        timestamp=ts
    )


@mcp.tool()
async def get_history_page(
    page: int = Field(description="Page number to retrieve (1-based index)", default=1, ge=1),
    page_size: int = Field(
        description="Maximum number of results to return per page (1-100)",
        default=10,
        ge=1,
        le=100
    ),
    after: int | None = Field(
        description="Unix timestamp - only return entries created after this time. Use get_timestamp() to generate this value.",
        default=None
    ),
) -> list[HistoryEntryResult]:
    """
    Retrieve a paginated list of history entries sorted by timestamp (most recent first).
    
    Use this tool to:
    - Browse through all stored history entries chronologically
    - Get recent conversation history
    - Retrieve entries within a specific time range
    
    Filtering by time:
    - Use 'after' parameter to get only entries created after a specific time
    - First use get_timestamp() to generate the timestamp, e.g., get_timestamp(days=7) for entries from the last 7 days
    - If after is not provided, returns all entries
    
    Pagination:
    - Results are sorted by timestamp (most recent first)
    - Use 'page' to navigate through results
    - Adjust 'page_size' to control how many results per page (max 100)
    
    Args:
        page: Page number (1-based, minimum 1)
        page_size: Number of entries per page (1-100)
        after: Optional Unix timestamp filter
    
    Returns:
        List of history entries with ID, query, content, and timestamp
    """
    entries = _history.get_page(page=page, page_size=page_size, after=after)
    return [
        HistoryEntryResult(
            id=entry.id if entry.id is not None else 0,
            query=entry.query,
            content=entry.content,
            timestamp=entry.timestamp
        )
        for entry in entries
    ]


# @mcp.tool()
# async def count_history(
#     after: int | None = Field(
#         description="Unix timestamp - count only entries created after this time. Use get_timestamp() to generate this value.",
#         default=None
#     ),
# ) -> int:
#     """
#     Get the total count of history entries.
    
#     Use this tool to:
#     - Check how many total entries exist
#     - Count entries within a specific time range
#     - Determine total pages for pagination
    
#     Time filtering:
#     - Use 'after' to count only entries created after a specific time
#     - First call get_timestamp() to generate the timestamp: get_timestamp(days=7)
#     - If after is not provided, counts all entries
    
#     Example usage:
#     - count_history() - Get total number of all entries
#     - count_history(after=get_timestamp(days=7)) - Count entries from last 7 days
#     - count_history(after=get_timestamp(months=1)) - Count entries from last month
    
#     Args:
#         after: Optional Unix timestamp filter
    
#     Returns:
#         Total number of entries matching the criteria
#     """
#     return _history.count(after=after)


@mcp.tool()
async def clear_history() -> int:
    """
    Delete all history entries from the database.
    
    Use this tool to:
    - Clear all stored conversation history
    - Reset the history database
    - Remove all entries from the current collection
    
    WARNING: This operation cannot be undone. All history entries will be permanently deleted.
    
    Returns:
        Number of entries that were deleted
    """
    return _history.clear()


def main() -> None:
    """Main entry point for the MCP server."""
    parser = argparse.ArgumentParser(
        description="History Management MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (stdio transport)
  uv run -m ask_memory.history_mcp

  # Run with custom database and collection
  uv run -m ask_memory.history_mcp --db-path ./my_history.db --collection chat_history

  # Run with HTTP transport
  uv run -m ask_memory.history_mcp --transport http --log-level DEBUG
        """
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="./history.db",
        help="Path to SQLite database file (default: ./history.db)"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="history",
        help="Collection name (table name) (default: history)"
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "http"],
        default="stdio",
        help="MCP transport protocol (default: stdio)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()

    # Initialize global history instance
    global _history
    _history = History(collection_name=args.collection, db_path=args.db_path)

    # Configure and run MCP server
    mcp.settings.log_level = args.log_level
    transport = "streamable-http" if args.transport == "http" else args.transport
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
