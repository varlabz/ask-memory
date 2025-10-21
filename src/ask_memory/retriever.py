from abc import ABC, abstractmethod
from .chunk import Chunk

class Retriever(ABC):
    @abstractmethod
    def add(self, document: Chunk):
        pass

    @abstractmethod
    def search(self, query: str, n_results: int = 5) -> list[Chunk]:
        pass