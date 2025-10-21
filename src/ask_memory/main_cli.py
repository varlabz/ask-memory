import argparse
import os
from ask.core.config import EmbedderConfig, ProviderEnum
from .retrievers.chroma import RetrieverChroma, get_embedding_function
from .chunk import Chunk, _chunk_file_to_documents
from datetime import datetime

embedder = EmbedderConfig(
    model="ollama:nomic-embed-text", 
    base_url="http://bacook.local:11434", 
)

def main():
    parser = argparse.ArgumentParser(description="ASK Memory CLI")
    parser.add_argument("-F", "--file", type=str, help="Path to file to add to the retriever")
    parser.add_argument("-S", "--search", type=str, help="Query to search in the retriever")
    args = parser.parse_args()

    retriever = RetrieverChroma("ask_memory", get_embedding_function(embedder))

    if args.file:
        if not os.path.exists(args.file):
            print(f"File {args.file} does not exist.")
            return
        
        for document in _chunk_file_to_documents(args.file, ):
            retriever.add(document)
        print(f"Added file {args.file} to retriever.")

    if args.search:
        results = retriever.search(args.search)
        print("Search results:")
        for i, result in enumerate(results):
            print(f"{i+1}. {result}")

if __name__ == "__main__":
    main()
