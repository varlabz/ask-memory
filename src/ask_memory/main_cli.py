import argparse
import asyncio
import os
from pathlib import Path
from ask.core.config import EmbedderConfig, ProviderEnum, load_config_dict, LLMConfig

from ask_memory.chunker.markdown_blocks import NodeType, blocks_to_markdown, markdown_to_blocks
from ask_memory.chunker.markdown_blocks_chunk import blocks_chunk
from ask_memory.retriever.retriever import Retriever
from ask_memory.core.rag_file import AgentChunkInput, FileRAG
from ask_memory.core.utils import file_to_markdown
from .retriever.chroma import RetrieverChroma, get_embedding_function
from datetime import datetime

llm = load_config_dict({
        "model": "ollama:gemma3:4b-it-q4_K_M", #qwen3:1.7b-q4_K_M", #gemma3:4b-it-q4_K_M",
        "base_url": "http://bacook.local:11434/v1/",
        "temperature": 0.0,
    }, 
    model_type=LLMConfig,
)

rag = FileRAG(llm)

async def main():
    parser = argparse.ArgumentParser(description="ASK Memory CLI")
    parser.add_argument("-F", "--file", type=str, help="Path to file to add to the retriever")
    parser.add_argument("-S", "--search", type=str, help="Query to search in the retriever")
    parser.add_argument("-A", "--ask", type=str, help="Query to ask the agent")
    args = parser.parse_args()

    if args.file:
        if not os.path.exists(args.file):
            print(f"File {args.file} does not exist.")
            return

        rag.add_file(args.file)
        print(f"Added file {args.file} to retriever.")

    if args.search:
        results = rag._retriever.get(args.search, n_results=10)
        print("Search results:")
        for i, result in enumerate(results):
            print(f"{i+1}. {result}")
            
    if args.ask:
        response = await rag.request(args.ask)
        print("Agent response:")
        print(response)

if __name__ == "__main__":
    from ask.core.config import TraceConfig, load_config
    from ask.core.instrumentation import setup_instrumentation_config
    
    setup_instrumentation_config(
        load_config(["~/.config/ask/trace.yaml"], type=TraceConfig, key="trace"),
    )
    
    asyncio.run(main())
