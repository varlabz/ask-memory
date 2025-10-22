from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

@dataclass
class Chunk[MetaType]:
    text: str
    metadata: MetaType


class Retriever[ChunkType = Chunk](ABC):
    @abstractmethod
    def add(self, chunk: ChunkType) -> None:
        ...

    @abstractmethod
    def get(self, query: str, n_results: int = 5) -> list[ChunkType]:
        ...