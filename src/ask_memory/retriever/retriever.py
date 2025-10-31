from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

@dataclass
class Chunk[MetaType]:
    text: str
    metadata: MetaType


class Retriever[ChunkType = Chunk](ABC):
    @abstractmethod
    def add(self, chunk: ChunkType) -> ChunkType: ...

    @abstractmethod
    def get_page(self, page: int, page_size: int, after: int) -> list[ChunkType]: ...

    @abstractmethod
    def query(self, query: str, results: int, after: int) -> list[ChunkType]: ...
      
    @abstractmethod
    def clear(self) -> None: ... 