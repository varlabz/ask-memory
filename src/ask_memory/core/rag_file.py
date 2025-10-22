# agentic search

from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from pydantic import Field

from ask import AgentASK, ContextASK
from ask.core.config import load_config_dict
from ask.core.memory import Memory
from ask.core.config import EmbedderConfig, Config, LLMConfig

from ask_memory.chunker.markdown_blocks import NodeType, blocks_to_markdown, markdown_to_blocks
from ask_memory.chunker.markdown_blocks_chunk import blocks_chunk
from ask_memory.retrievers.retriever import Chunk, Retriever
from ask_memory.retrievers.chroma import RetrieverChroma, get_embedding_function
from ask_memory.core.utils import file_to_markdown

class AgentChunk(ContextASK):
    data: str = Field(description="The original data of the chunk.")
    url: str = Field(description="The original URL of the chunk.")

class AgentChunkInput(ContextASK):
    request: str = Field(description="The original query.")
    data: list[AgentChunk] = Field(description="The original data of the chunks with URLs.")

class NoMemory(Memory):
    def get(self) -> list: return []
    def set(self, messages: list): pass

def _create_search_agent(llm: LLMConfig) -> AgentASK[AgentChunkInput, str]:
    return AgentASK[AgentChunkInput, str].create_from_config(load_config_dict({
        "agent": {    
            "name": "Chunker",
            "instructions": dedent(f"""
                You are a Retrieval-Augmented Generation (RAG) agent specialized in analyzing text chunks to answer user queries.

                Your task is to:
                1. Analyze the provided text chunks in the context of the user's request
                2. Synthesize relevant information from the chunks
                3. Provide a comprehensive and accurate final answer based on the analysis
                4. Include the URLs of the chunks that were most relevant to answering the request
                5. If the chunks do not contain sufficient information to answer the request, state this clearly and list any partially relevant URLs
                6. Collect only unique URLs to avoid repetition

                Input:
                {AgentChunkInput.to_input()}
                
                Output: 
                A clear, concise, and well-structured response that directly addresses the request using the provided chunks, with matched URLs.
            """),
            "input_type": str,
            "output_type": str,
        },
        "llm": llm,
    }), NoMemory())

@dataclass
class FileChunkMeta:
    source: str
    data: str

FileChunk = Chunk[FileChunkMeta]

embedder = EmbedderConfig(
    model="ollama:nomic-embed-text", 
    base_url="http://bacook.local:11434", 
)


class FileRAG:
    _retriever: Retriever[FileChunk]
    _search_agent: AgentASK[AgentChunkInput, str]
    
    def __init__(self, llm: LLMConfig, collection_name: str = "ask_memory"):
        self._retriever = RetrieverChroma[FileChunk](FileChunk, collection_name, get_embedding_function(embedder))
        self._search_agent = _create_search_agent(llm)

    def add_file(self, filename: str,) -> None:
        blocks = markdown_to_blocks(file_to_markdown(filename))
        for i, chunk in enumerate(blocks_chunk(blocks)):
            self._retriever.add(FileChunk(
                text=blocks_to_markdown([chunk.block], filter_func=lambda b: b.type != NodeType.BLOCK_CODE),
                metadata=FileChunkMeta(
                    source=Path(filename).absolute().as_posix(),
                    data=blocks_to_markdown([chunk.block]),
                ),
            ))

    async def search(self, request: str) -> str:
        results = self._retriever.get(request)
        response = await self._search_agent.run(AgentChunkInput(
            request=request,
            data=[AgentChunk(data=res.metadata.data, url=res.metadata.source) for res in results],
        ))
        return response
