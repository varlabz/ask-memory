
from dataclasses import dataclass
from typing import Generator, Iterable, List, Optional

from ask_memory.chunker.markdown_blocks import BaseBlock, HeadingBlock, NodeType

@dataclass
class ChunkBlock:
    title: List[str]
    block: BaseBlock


def blocks_chunk(
    blocks: Iterable[BaseBlock],
    title: Optional[List[str]] = None,
) -> Generator[ChunkBlock, None, None]:
    current_title = list(title) if title else []
    for block in blocks:
        if isinstance(block, HeadingBlock):
            heading_text = block.text
            next_title = current_title + [heading_text]
            for chunk in blocks_chunk(block.children, next_title):
                yield chunk
        else:
            yield ChunkBlock(title=list(current_title), block=block)
    
if __name__ == "__main__":
    # get file path from command line arguments
    # parse the file and print the chunks
    import sys
    from ask_memory.chunker.markdown_blocks import markdown_to_blocks, blocks_to_markdown    

    if len(sys.argv) < 2:
        print("Usage: python script.py <markdown_file>")
        sys.exit(1)

    markdown_file = sys.argv[1]
    with open(markdown_file, "r") as f:
        markdown_content = f.read()

    blocks = markdown_to_blocks(markdown_content)
    for chunk in blocks_chunk(blocks):
        print(f"{' > '.join(chunk.title)}")
        print(blocks_to_markdown([chunk.block], filter_func=lambda b: b.type != NodeType.BLOCK_CODE))
