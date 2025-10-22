# agentic search

from dataclasses import dataclass
import json
from pathlib import Path
import sys
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

class AgentChunkInput(ContextASK):
    request: str = Field(description="The query.")
    data: list[str] = Field(description="The data of the chunks.")
    error: str = Field(default="No errors", description="Error message, if any.")

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
                1. Analyze the provided text chunks in the context of the user's request.
                2. Synthesize relevant information from the chunks.
                3. Provide a comprehensive and accurate final answer based on the analysis.
                4. If the chunks do not contain sufficient information to answer the request, state this clearly.

                Input:
                {AgentChunkInput.to_input()}
                
                Output: 
                A clear, concise, and well-structured response that directly addresses the request using the provided chunks.
                
                Review your response carefully to ensure it accurately reflects the information in the chunks.
            """),
            "input_type": str,
            "output_type": str,
        },
        "llm": llm,
    }), NoMemory())

def _create_rank_agent(llm: LLMConfig) -> AgentASK[AgentChunkInput, str]:
    return AgentASK[AgentChunkInput, str].create_from_config(load_config_dict({
        "agent": {    
            "name": "Ranker",
            "instructions": dedent(f"""
                You are a ranking agent specialized in evaluating the relevance of text chunks to a given request.

                Your task is to:
                1. Analyze each text chunk provided in the data list.
                2. Determine how relevant each chunk is to the user's request.
                3. Assign a relevance score from 1 to 10 for each chunk, where:
                   - 1 means the chunk is not relevant at all to the request
                   - 10 means the chunk is highly relevant and directly addresses the request

                Input format:
                {AgentChunkInput.to_input()}
                
                Output format: 
                A comma-separated list of integers from 1 to 10, one score for each chunk in the data list, 
                in the same order as the chunks appear in the data.
                Number of scores must match number of chunks.
                
                Examples:
                If there are 3 chunks, output something like: [8,3,9]
                or
                If there are 2 chunks, output something like: [18,5]
                or
                If there are 1 chunk, output something like: [2]
                
                Review your scores carefully to ensure they accurately reflect the relevance of each chunk to the request.
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
    _rank_agent: AgentASK[AgentChunkInput, str]
    
    def __init__(self, llm: LLMConfig, collection_name: str = "ask_memory"):
        self._retriever = RetrieverChroma[FileChunk](FileChunk, collection_name, get_embedding_function(embedder))
        self._search_agent = _create_search_agent(llm)
        self._rank_agent = _create_rank_agent(llm)

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

    async def request(self, request: str) -> str:
        results = self._retriever.get(request, n_results=6)
        response = await self._search_agent.run(AgentChunkInput(
            request=request,
            data=[res.metadata.data for res in results],
        ))
        return response

    class RanksValueError(Exception):
        def __init__(self, message):
            super().__init__(message)

    async def _rank(self, request: str, results: list[FileChunk], error: str|None = None) -> list[int]:
        ranks = await self._rank_agent.run(AgentChunkInput(
            request=request,
            data=[res.metadata.data for res in results],
            error=error or "No errors",
        ))
        try:
            ranks = json.loads(ranks)
            if len(ranks) <= len(results):
                # after ranking chunks, agent takes first n results, so we can pad with 1s the rest
                print("Rank count does not match chunk count, padding with 1s.", file=sys.stderr)
                ranks += [1]*(len(results) - len(ranks))
            else:
                raise FileRAG.RanksValueError("Previous rank count does not match chunk count. Try again.")
            
            return ranks
        except Exception as e:
            print("Failed to parse numbers: ", ranks, file=sys.stderr)
            # if not started with [ and ended with ], add them and try again
            if not ranks.startswith("["): ranks = "[" + ranks
            if not ranks.endswith("]"): ranks = ranks + "]"
            ranks = json.loads(ranks)
            if len(ranks) <= len(results):
                ranks += [1]*(len(results) - len(ranks))
            else:
                raise FileRAG.RanksValueError("Previous rank count does not match chunk count. Try again.")
            return ranks

    async def request_with_rank(self, request: str) -> str:
        async def _run_rank(results: list[FileChunk]) -> list[int]:
            error = None
            for i in range(3):  # try up to 3 times
                try:
                    return await self._rank(request, results, error=error)
                except FileRAG.RanksValueError as e:
                    print(f"Rank attempt {i+1} failed: {e}", file=sys.stderr)
                    error = str(e)
                    
            raise ValueError("Failed to get ranks after 3 attempts.")

        results = self._retriever.get(request, n_results=8)
        ranks = await _run_rank(results)
        # Reorder results based on ranks
        ranked_results = [res for _, res in sorted(zip(ranks, results), key=lambda  x: x[0], reverse=True)]
        # take top 3/4 of results based on ranks
        results = ranked_results[:int(len(ranked_results) * 0.75)]
        response = await self._search_agent.run(AgentChunkInput(
            request=request,
            data=[res.metadata.data for res in results],
        ))
        return response
