
import os
import dataclasses
from chromadb import Client, Documents, EmbeddingFunction, PersistentClient, QueryResult
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction,OpenAIEmbeddingFunction,GoogleGenerativeAiEmbeddingFunction
from chromadb.api.types import Embeddable
from ask.core.config import EmbedderConfig, ProviderEnum

from ..retriever import Retriever
from ..chunk import Chunk



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

class RetrieverChroma(Retriever):
    def __init__(self, collection_name: str, embedding_function: EmbeddingFunction):

        self.client = PersistentClient(path="./chroma-db")
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )

    def add(self, document: Chunk):
        self.collection.add(
            documents=[document.text],
            ids=[document.id],
            metadatas=[dataclasses.asdict(document.metadata)]
        )

    def search(self, query: str, n_results: int = 5) -> list[Chunk]:
        results: QueryResult = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        docs = results.get('documents')
        ids = results.get('ids')
        metadatas = results.get('metadatas')
        if not docs or not ids or not metadatas:
            return []
        documents = []
        for doc_text, doc_id, meta in zip(docs[0], ids[0], metadatas[0]):
            documents.append(Chunk(
                text=doc_text,
                id=doc_id,
                metadata=Chunk.Metadata(
                    source=str(meta['source']),
                    chunk_index=int(str(meta['chunk_index'])),  
                    timestamp=str(meta['timestamp'])  
                )
            ))
        return documents