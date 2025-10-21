from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import io
import os
from typing import Dict, Any, Generator, TypedDict, TYPE_CHECKING
from markitdown import MarkItDown
from unstructured.partition.auto import partition

if TYPE_CHECKING:
    from ask_memory.retriever import Retriever

def _timestamp(): return datetime.now().isoformat(sep='@')

@dataclass
class Chunk:
    text: str
    id: str

    @dataclass
    class Metadata:
        source: str
        chunk_index: int
        timestamp: str = _timestamp()
    metadata: Metadata

if __name__ == "__main__":
    # get file path from command line argument
    import sys
    if len(sys.argv) < 2:
        print("Usage: python document.py <file_path>")
        sys.exit(1)     
        
    for i, chunk in enumerate(
        partition(
            filename=sys.argv[1],
            chunking_strategy="by_title",
            max_characters=1000,
        )
    ):
        print("-"*20)
        print(f"{chunk}")
        
        
