
from dataclasses import dataclass
from datetime import datetime
import json
import sys
from textwrap import dedent
from pydantic import Field, BaseModel

from ask.core.context import load_string_json
from ask.core.agent import AgentASK
from ask.core.memory import Memory
from ask.core.config import EmbedderConfig, Config, LLMConfig

from ask_memory.memory_history.agent import AnalysisInput, AnalysisOutput, RerankInput, RerankOutput, create_analysis_agent, create_rerank_agent
from ask_memory.retriever.chroma import RetrieverChroma, get_embedding_function
from ask_memory.retriever.retriever import Chunk, Retriever


@dataclass  
class MemoryMeta:
    query: str              # user query
    content: str            # llm response    
    context: str            # one sentence summary
    tags: str               # broad categories/themes
    keywords: str           # key concepts/terminology
    timestamp: int = int(datetime.now().timestamp()) 
   
MemoryChunk = Chunk[MemoryMeta]

class MemoryHistory:
    _retriever: Retriever[MemoryChunk]
    _agent: AgentASK[AnalysisInput, AnalysisOutput]

    def __init__(self, llm: LLMConfig, embedder: EmbedderConfig, collection_name: str = "ask_memory_history"):
        self._retriever = RetrieverChroma[MemoryChunk](MemoryChunk, collection_name, get_embedding_function(embedder))
        self._agent = create_analysis_agent(llm)
        self._reranker = create_rerank_agent(llm)

    async def add(self, query: str, response: str) -> MemoryChunk:
        res = await self._agent.run(
            AnalysisInput(
                query=query,
                response=response
            )
        )
        text = dedent(f"""
            Query: {query}
            Response: {response}
            Context: {res.context}
            Keywords: {', '.join(res.keywords)}
            Tags: {', '.join(res.tags)}
        """)
        chunk = MemoryChunk(
            text=text,
            metadata=MemoryMeta(
                query=query,
                content=response,
                context=res.context,
                tags=', '.join(res.tags),
                keywords=', '.join(res.keywords),
            ),
        )
        return self._retriever.add(chunk)

    def get_page(self, page: int = 1, page_size: int = 50, after: int = int(datetime.now().timestamp())) -> list[MemoryChunk]:
        """
        Get a page of memory history entries.
        Args:
            page: Page number (1-based)
            page_size: Number of results per page
            after: Unix timestamp to filter entries created before this time
        """
        res = self._retriever.get_page(page=page, page_size=page_size, after=after)
        res.sort(key=lambda c: c.metadata.timestamp, reverse=False)
        return res

    async def query(self, query: str, after: int) -> list[MemoryChunk]:
        """
        Search memory history with ranking.
        
        Args:
            query: Search query string
            after: Unix timestamp to filter entries created before this time
            
        Returns:
            List of MemoryHistoryOutput sorted by timestamp, limited to first 7 results with highest rank
        """
        # Get retriever results
        chunks = self._retriever.query(query, results=7*3, after=after)
        print(f"Retrieved chunks: {len(chunks)}", file=sys.stderr)
        rerank_result = await self._reranker.run(RerankInput(query=query, responses=[chunk.text for chunk in chunks]))
        print("Rerank result:", rerank_result, file=sys.stderr)
        if len(rerank_result.ranks) != len(chunks):
            raise ValueError("Reranker output length does not match number of retrieved chunks")
        ranked_results = list(zip(chunks, rerank_result.ranks))
        # filter out low rank scores, check reranker output for scores
        ranked_results = [item for item in ranked_results if item[1] > 16]
        # Sort by rank score (descending) and take top 7
        ranked_results.sort(key=lambda x: x[1], reverse=True)
        chunks = [chunk for chunk, _ in ranked_results[:7]]
        # Sort by timestamp in increasing order
        chunks.sort(key=lambda c: c.metadata.timestamp, reverse=False)
        return chunks

    def clear(self):
        self._retriever.clear()


if __name__ == "__main__":
    import argparse
    import asyncio
    
    parser = argparse.ArgumentParser(description="Manage memory history")
    parser.add_argument('--add', nargs=2, metavar=('query', 'response'), help='Add a query and response to memory')
    parser.add_argument('--clear', action='store_true', help='Clear all memory history')
    parser.add_argument('--query', metavar='query', help='Query the memory history for related responses')
    parser.add_argument('--after', metavar='timestamp', help='ISO timestamp or Unix timestamp to filter entries created before this time (used with --query)', default=None)
    
    args = parser.parse_args()
    
    # Convert after timestamp if provided
    if args.after:
        try:
            # Try to parse as int (Unix timestamp)
            after_timestamp = int(args.after)
        except ValueError:
            # Try to parse as ISO format string
            try:
                after_timestamp = int(datetime.fromisoformat(args.after).timestamp())
            except ValueError:
                print(f"Invalid timestamp format: {args.after}", file=sys.stderr)
                sys.exit(1)
    else:
        after_timestamp = int(datetime.now().timestamp())
    
    from ask.core.config import TraceConfig, load_config
    from ask.core.instrumentation import setup_instrumentation_config
    
    setup_instrumentation_config(
        load_config(["~/.config/ask/trace.yaml"], type=TraceConfig, key="trace"),
    )
    
    # llm = LLMConfig(
    #     model="ollama:gemma3:4b-it-q4_K_M", #qwen3:1.7b-q4_K_M", #
    #     base_url="http://bacook.local:11434/v1/",
    #     temperature=0.0,
    #     use_tools=False,
    # )
    # embedder = EmbedderConfig(
    #     model="ollama:nomic-embed-text", 
    #     base_url="http://bacook.local:11434/", 
    # )
    config: str = "~/.config/ask/llm-memory-ollama.yaml"
    llm: LLMConfig = load_config([config], LLMConfig, "llm")
    embedder: EmbedderConfig = load_config([config], EmbedderConfig, "embedder")
    memory_history = MemoryHistory(llm, embedder)
    
    if args.add:
        query, response = args.add
        asyncio.run(memory_history.add(query, response))
        print("Memory entry added successfully.")
    elif args.clear:
        memory_history.clear()
        print("Memory history cleared.")
    elif args.query:
        results = asyncio.run(memory_history.query(args.query, after=after_timestamp))
        if results:
            print("Related responses:")
            for i, response in enumerate(results, 1):
                print(f"{i}. {response}")
        else:
            print("No related responses found.")
    else:
        parser.print_help()