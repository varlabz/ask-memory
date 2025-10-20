from pathlib import Path
from textwrap import dedent
import json
from markitdown import MarkItDown
import tiktoken
from unstructured.partition.text import partition_text
from unstructured.partition.auto import partition
from unstructured.partition.utils.constants import PartitionStrategy

from ask import AgentASK
from ask.core.config import load_config_dict
from ask.core.memory import Memory

llm = {
    "model": "ollama:gemma3:4b-it-q4_K_M",
    "base_url": "http://bacook.local:11434/v1/",
    "temperature": 0.0,
}

# each agent run will start with no history as fresh start
class NoMemory(Memory):
    def get(self) -> list: return []
    def set(self, messages: list): pass

chunk_agent = AgentASK[str, str].create_from_config(load_config_dict({
    "agent": {    
        "name": "Chunker",
        "instructions": dedent("""
            You are an assistant specialized in splitting text into thematically consistent sections. 
            The text has been divided into chunks, each marked with <|start_chunk_X|> and <|end_chunk_X|> tags, where X is the chunk number. 
            Your task is to identify the points where splits should occur, such that consecutive chunks of similar themes stay together. 
            Respond with a list of chunk IDs where you believe a split should be made. 
            For example, if chunks 1 and 2 belong together but chunk 3 starts a new topic, 
            you would suggest a split after chunk 2. 
            THE CHUNKS MUST BE IN ASCENDING ORDER.
            Your response should be in the form of list of integers like this:
            [3, 5, 19, ...]
            or
            [10]
            or
            [2, 8, ...]
        """),
        "input_type": str,
        "output_type": str,
    },
    "llm": llm,
}), NoMemory())

PROMPT = dedent("""
    CHUNKED_TEXT:  
    {chunked_input} 
    Respond only with the IDs of the chunks where you believe a split should occur. 
    YOU MUST RESPOND WITH AT LEAST ONE SPLIT. 
    THESE SPLITS MUST BE IN ASCENDING ORDER AND EQUAL OR LARGER THAN: {current_chunk}. 
    {invalid_response}
""")

PROMPT_INVALID = dedent("""
    The previous response of {invalid_response} was invalid. 
    DO NOT REPEAT THIS ARRAY OF NUMBERS. 
    Please try again.
""")

def token_count_str(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(string, disallowed_special=()))

class LLMSemanticChunker:
    async def split_file(self, file_path: str) -> list[str]:
        return await self.split_text(MarkItDown(enable_plugins=True).convert(file_path).markdown)

    async def split_text(self, text) -> list[str]:
        chunks = partition_text(
            text=text,
            max_characters=50,
            paragraph_grouper=False,
        )
        return await self._do_chunks(chunks)

    async def _numbers(self, chunked_input: str, current_chunk: int) -> list[int]:
        async def _run_iter(prompt: str) -> list[int] | None:
            numbers = (await chunk_agent.run(prompt)).strip()
            print(numbers)
            # parse numbers as json array
            try:
                numbers = json.loads(numbers)
            except json.JSONDecodeError:
                print("Failed to parse numbers: ", numbers, file=sys.stderr)
                # if not started with [ and ended with ], add them and try again
                if not numbers.startswith("["):
                    numbers = "[" + numbers
                if not numbers.endswith("]"):
                    numbers = numbers + "]"
                numbers = json.loads(numbers)
            # Check if the numbers are in ascending order and are equal to or larger than current_chunk
            if not (numbers != sorted(numbers) or any(number < current_chunk for number in numbers)):
                return numbers

            print("Invalid response: ", numbers, file=sys.stderr)
            return None
        
        # hucky way to reset memory
        # need memory to recover from invalid responses but not on fresh start
        chunk_agent._memory = Memory() 
        numbers = await _run_iter(PROMPT.format(
            chunked_input=chunked_input, 
            current_chunk=current_chunk, 
            invalid_response=""),
        )
        if numbers is not None:
            return numbers

        # iterate 3 times to get valid response
        for i in range(3):
            numbers = await _run_iter(
                PROMPT.format(
                    chunked_input=chunked_input,
                    current_chunk=current_chunk,
                    invalid_response=PROMPT_INVALID.format(invalid_response=numbers)),
            )
            if numbers is not None:
                return numbers        
            
        raise ValueError("Failed to get valid response from chunk agent.")
    
    async def _do_chunks(self, chunks) -> list[str]:        
        split_indices = []
        short_cut = len(split_indices) > 0
        current_chunk = 0
        while True and not short_cut:
            if current_chunk >= len(chunks) - 4:
                break

            token_count = 0
            chunked_input = ''
            for i in range(current_chunk, len(chunks)):
                token_count += token_count_str(chunks[i].text)
                chunked_input += f"<|start_chunk_{i+1}|>{chunks[i]}<|end_chunk_{i+1}|>\n"
                if token_count > 1000:  # define max tokens per input for embedding model
                    break

            print(f"current_chunk: {current_chunk}, token_count: {token_count}")
            numbers = await self._numbers(chunked_input, current_chunk + 1)
            split_indices.extend(numbers)
            current_chunk = numbers[-1]

            if len(numbers) == 0:
                break

            
        chunks_to_split_after = [i - 1 for i in split_indices]
        docs = []
        current_chunk = ''
        for i, chunk in enumerate(chunks):
            current_chunk += chunk.text + ' '
            if i in chunks_to_split_after:
                docs.append(current_chunk.strip())
                current_chunk = ''
        if current_chunk:
            docs.append(current_chunk.strip())

        return docs
    
if __name__ == "__main__":
    import sys
    import asyncio
    from ask.core.config import TraceConfig, load_config
    from ask.core.instrumentation import setup_instrumentation_config
    
    setup_instrumentation_config(
        load_config(["~/.config/ask/trace.yaml"], type=TraceConfig, key="trace"),
    )
    if len(sys.argv) < 2:
        raise ValueError("Please provide a file path as an argument.")
    
    chunker = LLMSemanticChunker()
    docs = asyncio.run(chunker.split_file(sys.argv[1]))
    for i, doc in enumerate(docs):
        print(f"--- Document {i} ---")
        print(doc)    