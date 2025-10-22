
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

class RetrieverChroma[ChunkType: Chunk = Chunk](Retriever[ChunkType]):
    
    def __init__(self, cls: type[ChunkType], collection_name: str, embedding_function: EmbeddingFunction, ):
        self._cls = cls
        self._metadata_type = get_args(cls)[0]
        self._client = PersistentClient(path="./chroma-db")
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )

    def add(self, chunk: ChunkType) -> None:
        self._collection.add(
            ids=uuid4().hex,
            documents=[chunk.text],
            metadatas=[dataclasses.asdict(chunk.metadata)]
        )

    def get(self, query: str, n_results: int = 5) -> list[ChunkType]:
        results: QueryResult = self._collection.query(
            query_texts=[query],
            n_results=n_results
        )
        docs = results.get('documents')
        metadatas = results.get('metadatas')
        if not docs or not metadatas:
            return []
        ret = []
        for text, meta in zip(docs[0], metadatas[0]):
            ret.append(self._cls(
                text=text,
                metadata=self._metadata_type(**meta),
            ))
        return ret