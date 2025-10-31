
import os
import dataclasses
from typing import get_type_hints, get_args
from uuid import uuid4
from chromadb import Client, Documents, EmbeddingFunction, PersistentClient, QueryResult
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction,OpenAIEmbeddingFunction,GoogleGenerativeAiEmbeddingFunction
from chromadb.api.types import Embeddable
from ask.core.config import EmbedderConfig, ProviderEnum

from .retriever import Chunk, Retriever

def get_embedding_function(config: EmbedderConfig) -> EmbeddingFunction:
    # model format for embedding function -  provider:model_name
    provider, model_name = config.model.split(":")
    if not model_name:
        raise ValueError("Model name must be in the format 'provider:model_name'")

    if provider == ProviderEnum.LMSTUDIO:
        return OpenAIEmbeddingFunction(model_name=model_name, api_base=config.base_url)

    if provider == ProviderEnum.OLLAMA:
        return OllamaEmbeddingFunction(model_name=model_name, url=config.base_url if config.base_url else "http://localhost:11434")

    os.environ["PROVIDER_API_KEY"] = config.api_key or ""
    return {
        ProviderEnum.OPENAI: OpenAIEmbeddingFunction,
        ProviderEnum.GOOGLE: GoogleGenerativeAiEmbeddingFunction,
    }[ProviderEnum(provider)](model_name=model_name, api_key_env_var="PROVIDER_API_KEY")

class RetrieverChroma[ChunkType: Chunk](Retriever[ChunkType]):

    def __init__(self, cls: type[ChunkType], collection_name: str, embedding_function: EmbeddingFunction, path: str = "./chroma-db"):
        self._cls = cls
        self._metadata_type = get_args(cls)[0]
        self._client = PersistentClient(path=path, settings=Settings(anonymized_telemetry=False))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )

    def add(self, chunk: ChunkType) -> ChunkType:
        self._collection.add(
            ids=uuid4().hex,
            documents=[chunk.text],
            metadatas=[dataclasses.asdict(chunk.metadata)]
        )
        return chunk

    def query(self, query: str, results: int, after: int) -> list[ChunkType]:
        res: QueryResult = self._collection.query(
            query_texts=[query],
            n_results=results,
            where={"timestamp": {"$lte": after}}
        )
        docs = res.get('documents')
        metadatas = res.get('metadatas')
        if not docs or not metadatas:
            return []
        ret = []
        for text, meta in zip(docs[0], metadatas[0]):
            ret.append(self._cls(
                text=text,
                metadata=self._metadata_type(**meta),
            ))
        return ret

    def get_page(self, page: int, page_size: int, after: int) -> list[ChunkType]:
        offset = (page - 1) * page_size
        results = self._collection.get(
            offset=offset,
            limit=page_size,
            where={"timestamp": {"$lte": after}}
        )
        docs = results.get('documents')
        metadatas = results.get('metadatas')
        if not docs or not metadatas:
            return []
        ret = []
        for text, meta in zip(docs, metadatas):
            ret.append(self._cls(
                text=text,
                metadata=self._metadata_type(**meta),
            ))
        return ret

    def clear(self) -> None:
        self._client.delete_collection(self._collection.name)