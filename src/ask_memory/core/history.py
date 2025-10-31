import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass  
class HistoryEntry:
    query: str              # user query
    content: str            # llm response    
    timestamp: int          # Unix timestamp of when the memory was created
    id: int | None = None  # Database row ID


class History:
    """
    SQLite-based history management for storing query-content pairs.
    
    Provides methods to add history entries and retrieve paginated results
    with optional timestamp filtering.
    """

    def __init__(self, collection_name: str = "history", db_path: str = "./history.db"):
        """
        Initialize History with SQLite database.
        
        Args:
            db_path: Path to the SQLite database file
            collection_name: Name of the table to store history entries
        """
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self._init_db()
    
    def _init_db(self) -> None:
        """Create the history table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.collection_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp INTEGER NOT NULL
                )
            """)
            # Create index on timestamp for efficient filtering
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.collection_name}_timestamp 
                ON {self.collection_name}(timestamp)
            """)
            conn.commit()
    
    def add_history(self, query: str, content: str, timestamp: int | None = None) -> int:
        """
        Add a new history entry.
        
        Args:
            query: User query string
            content: LLM response content
            timestamp: Unix timestamp (defaults to current time if not provided)
            
        Returns:
            ID of the inserted row
            
        Example:
            history = History()
            entry_id = history.add_history("What is Paris?", "Paris is the capital of France")
        """
        if timestamp is None:
            timestamp = int(datetime.now().timestamp())
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"INSERT INTO {self.collection_name} (query, content, timestamp) VALUES (?, ?, ?)",
                (query, content, timestamp)
            )
            conn.commit()
            row_id = cursor.lastrowid
            if row_id is None:
                raise RuntimeError("Failed to get last row ID from database")
            return row_id
    
    def get_page(
        self, 
        page: int = 1, 
        page_size: int = 10, 
        after: int | None = None
    ) -> list[HistoryEntry]:
        """
        Get paginated history entries with optional timestamp filtering.
        
        Args:
            page: Page number (1-indexed)
            page_size: Number of entries per page
            after: Optional Unix timestamp to filter entries created after this time
            
        Returns:
            List of HistoryEntry objects for the requested page
            
        Raises:
            ValueError: If page or page_size is less than 1
            
        Example:
            history = History()
            # Get first page with 20 items
            entries = history.get_page(page=1, page_size=20)
            
            # Get entries created after a specific timestamp
            import time
            week_ago = int(time.time()) - (7 * 24 * 60 * 60)
            recent_entries = history.get_page(page=1, page_size=10, after=week_ago)
        """
        if page < 1:
            raise ValueError("Page must be >= 1")
        if page_size < 1:
            raise ValueError("Page size must be >= 1")
        
        offset = (page - 1) * page_size
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if after is not None:
                cursor.execute(
                    f"""
                    SELECT id, query, content, timestamp 
                    FROM {self.collection_name} 
                    WHERE timestamp > ?
                    ORDER BY timestamp DESC 
                    LIMIT ? OFFSET ?
                    """,
                    (after, page_size, offset)
                )
            else:
                cursor.execute(
                    f"""
                    SELECT id, query, content, timestamp 
                    FROM {self.collection_name} 
                    ORDER BY timestamp DESC 
                    LIMIT ? OFFSET ?
                    """,
                    (page_size, offset)
                )
            
            rows = cursor.fetchall()
            return [
                HistoryEntry(
                    query=row[1],
                    content=row[2],
                    timestamp=row[3],
                    id=row[0]
                )
                for row in rows
            ]
    
    def count(self, after: int | None = None) -> int:
        """
        Get total count of history entries.
        
        Args:
            after: Optional Unix timestamp to count entries created after this time
            
        Returns:
            Total number of entries
            
        Example:
            history = History()
            total = history.count()
            print(f"Total entries: {total}")
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if after is not None:
                cursor.execute(
                    f"SELECT COUNT(*) FROM {self.collection_name} WHERE timestamp > ?",
                    (after,)
                )
            else:
                cursor.execute(f"SELECT COUNT(*) FROM {self.collection_name}")
            
            return cursor.fetchone()[0]

    def clear(self) -> int:
        """
        Delete all entries from the collection.
        
        Returns:
            Number of rows deleted
            
        Example:
            history = History()
            deleted_count = history.clear()
            print(f"Deleted {deleted_count} entries")
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"DELETE FROM {self.collection_name}")
            conn.commit()
            return cursor.rowcount


if __name__ == "__main__":
    """CLI entry point for managing history."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Manage history entries in SQLite database"
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
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Add command
    add_parser = subparsers.add_parser("add", help="Add a new history entry")
    add_parser.add_argument("--query", type=str, required=True, help="User query")
    add_parser.add_argument("--content", type=str, required=True, help="LLM response content")
    add_parser.add_argument("--timestamp", type=int, help="Unix timestamp (optional, defaults to now)")
    
    # Get page command
    page_parser = subparsers.add_parser("get", help="Get paginated history entries")
    page_parser.add_argument("--page", type=int, default=1, help="Page number (default: 1)")
    page_parser.add_argument("--page-size", type=int, default=10, help="Number of entries per page (default: 10)")
    page_parser.add_argument("--after", type=int, help="Filter entries after this Unix timestamp")
    
    # Clear command
    subparsers.add_parser("clear", help="Delete all history entries")
    
    # Count command
    count_parser = subparsers.add_parser("count", help="Get total count of entries")
    count_parser.add_argument("--after", type=int, help="Count entries after this Unix timestamp")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        exit(1)
    
    history = History(collection_name=args.collection, db_path=args.db_path)
    
    if args.command == "add":
        entry_id = history.add_history(
            query=args.query,
            content=args.content,
            timestamp=args.timestamp
        )
        print(f"Added entry with ID: {entry_id}")
    
    elif args.command == "get":
        entries = history.get_page(
            page=args.page,
            page_size=args.page_size,
            after=args.after
        )
        total = history.count(after=args.after)
        
        print(f"Page {args.page} (Total entries: {total})")
        print("-" * 80)
        
        if not entries:
            print("No entries found.")
        else:
            for entry in entries:
                dt = datetime.fromtimestamp(entry.timestamp)
                print(f"\nID: {entry.id}")
                print(f"Timestamp: {dt.strftime('%Y-%m-%d %H:%M:%S')} ({entry.timestamp})")
                print(f"Query: {entry.query}")
                print(f"Content: {entry.content[:100]}{'...' if len(entry.content) > 100 else ''}")
                print("-" * 80)
    
    elif args.command == "clear":
        deleted = history.clear()
        print(f"Deleted {deleted} entries from collection '{args.collection}'")
    
    elif args.command == "count":
        total = history.count(after=args.after)
        if args.after:
            dt = datetime.fromtimestamp(args.after)
            print(f"Entries after {dt.strftime('%Y-%m-%d %H:%M:%S')}: {total}")
        else:
            print(f"Total entries: {total}")

