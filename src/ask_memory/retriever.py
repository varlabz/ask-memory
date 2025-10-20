from abc import ABC, abstractmethod
from .document import DocumentASK

class Retriever(ABC):
    @abstractmethod
    def add(self, document: DocumentASK):
        pass

    @abstractmethod
    def search(self, query: str, n_results: int = 5) -> list[DocumentASK]:
        pass