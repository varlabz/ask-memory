from textwrap import dedent

import pytest

from ask_memory.chunker.markdown_blocks import (
    BaseBlock,
    HeadingBlock,
    NodeType,
    ParagraphBlock,
    block_from_dict,
)
from ask_memory.chunker.markdown_blocks_chunk import blocks_chunk


def _blocks_from_markdown(markdown: str) -> list[BaseBlock]:
    import mistune
    from ask_memory.chunker.markdown_blocks import convert_block, nest_blocks
    from collections.abc import Mapping

    md = mistune.create_markdown(renderer="ast", plugins=["table"])
    ast = md(markdown)
    blocks: list[BaseBlock] = []
    for raw_node in ast:
        if not isinstance(raw_node, Mapping):
            continue
        converted = convert_block(raw_node)
        if converted is not None:
            blocks.append(converted)
    return nest_blocks(blocks)


def test_blocks_chunk_assigns_heading_titles_to_descendants():
    markdown = dedent(
        """
        # Title

        Paragraph text.
        """
    )

    blocks = _blocks_from_markdown(markdown)
    chunks = list(blocks_chunk(blocks))

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.title == ["Title"]
    assert isinstance(chunk.block, ParagraphBlock)
    assert chunk.block.type == NodeType.PARAGRAPH
    assert chunk.block.text == "Paragraph text."


def test_blocks_chunk_handles_nested_headings_and_top_level_content():
    markdown = dedent(
        """
        Leading paragraph.

        # Section

        Intro under section.

        ## Subsection

        Subsection body.
        """
    )

    blocks = _blocks_from_markdown(markdown)
    chunks = list(blocks_chunk(blocks))

    paragraph_chunks = [
        chunk for chunk in chunks if isinstance(chunk.block, ParagraphBlock)
    ]
    assert [chunk.title for chunk in paragraph_chunks] == [
        [],
        ["Section"],
        ["Section", "Subsection"],
    ]
    assert [getattr(chunk.block, "text", None) for chunk in paragraph_chunks] == [
        "Leading paragraph.",
        "Intro under section.",
        "Subsection body.",
    ]


def test_blocks_chunk_accepts_heading_dataclasses():
    heading = HeadingBlock(
        type=NodeType.HEADING,
        level=1,
        text="Heading",
        children=[ParagraphBlock(type=NodeType.PARAGRAPH, text="Body paragraph")],
    )

    chunks = list(blocks_chunk([heading]))

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.title == ["Heading"]
    assert isinstance(chunk.block, ParagraphBlock)
    assert chunk.block.text == "Body paragraph"


def test_blocks_chunk_returns_empty_iterable_for_no_blocks():
    assert list(blocks_chunk([])) == []


def test_blocks_chunk_does_not_mutate_provided_title_list():
    markdown = "Paragraph only."
    blocks = _blocks_from_markdown(markdown)
    seed_title = ["Seed"]

    chunks = list(blocks_chunk(blocks, title=seed_title))

    assert seed_title == ["Seed"]
    assert chunks[0].title == ["Seed"]


def test_blocks_chunk_handles_heading_without_text():
    blocks = [
        HeadingBlock(
            type=NodeType.HEADING,
            level=1,
            text="",
            children=[
                ParagraphBlock(
                    type=NodeType.PARAGRAPH,
                    text="Content under untitled heading",
                )
            ],
        )
    ]

    chunks = list(blocks_chunk(blocks))

    assert chunks[0].title == [""]
    assert isinstance(chunks[0].block, ParagraphBlock)
    assert chunks[0].block.text == "Content under untitled heading"


def test_blocks_chunk_handles_unsupported_block_type():
    class NotABlock:
        pass

    # The generator should yield the unsupported type without raising an error.
    chunks = list(blocks_chunk([NotABlock()]))  # type: ignore[arg-type]
    assert len(chunks) == 1
    assert isinstance(chunks[0].block, NotABlock)
