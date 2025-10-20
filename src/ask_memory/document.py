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
class DocumentASK:
    text: str
    id: str

    @dataclass
    class Metadata:
        source: str
        chunk_index: int
        timestamp: str = _timestamp()
    metadata: Metadata

def _chunk_file_to_documents(file_path: str, ) -> Generator[DocumentASK, None, None]:
    # buff = MarkItDown(enable_plugins=True).convert(file_path)
    # save file into new file
    # with open("qq.md", 'w') as f:
    #     f.write(buff.text_content)
    # file = io.BytesIO(buff.text_content.encode('utf-8'))    
    for i, chunk in enumerate(
        partition(
            filename=file_path,
            chunking_strategy="by_title",
            max_characters=1000,
        )
    ):
        print(f">>{chunk}<<")
        doc = DocumentASK(
            text=chunk.text,
            id=f"{file_path}:{chunk}",
            metadata=DocumentASK.Metadata(
                source=os.path.abspath(file_path),
                chunk_index=i,
                timestamp=_timestamp()
            )
        )
        yield doc

# def load_document_from_file(retriever: Retriever, file_path: str):
#     for document in _chunk_file_to_documents(file_path, ):
#         retriever.add(document)

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
        
        
