import json
from textwrap import dedent

from ask_memory.chunker.markdown_blocks import blocks_to_markdown, markdown_to_blocks, ParagraphBlock


def test_extracts_expected_block_types():
    markdown = dedent(
        """
        # Title

        Intro paragraph with **emphasis**.

        - Item 1
        - Item 2

        1. Step one
        2. Step two

        | Col A | Col B |
        | ----- | ----- |
        | Foo   | Bar   |

        > Quoted text
        """
    )

    blocks = markdown_to_blocks(markdown)

    # Dump to JSON to ensure serialisable output and stable assertion diff.
    assert json.loads(json.dumps([block.to_dict() for block in blocks])) == [
        {
            "type": "heading",
            "level": 1,
            "text": "Title",
            "children": [
                {"type": "paragraph", "text": "Intro paragraph with emphasis."},
                {
                    "type": "list",
                    "ordered": False,
                    "items": [
                        {
                            "type": "list_item",
                            "children": [
                                {"type": "paragraph", "text": "Item 1"},
                            ],
                        },
                        {
                            "type": "list_item",
                            "children": [
                                {"type": "paragraph", "text": "Item 2"},
                            ],
                        },
                    ],
                },
                {
                    "type": "list",
                    "ordered": True,
                    "items": [
                        {
                            "type": "list_item",
                            "children": [
                                {"type": "paragraph", "text": "Step one"},
                            ],
                        },
                        {
                            "type": "list_item",
                            "children": [
                                {"type": "paragraph", "text": "Step two"},
                            ],
                        },
                    ],
                },
                {
                    "type": "table",
                    "header": ["Col A", "Col B"],
                    "rows": [["Foo", "Bar"]],
                },
                {
                    "type": "block_quote",
                    "children": [
                        {"type": "paragraph", "text": "Quoted text"},
                    ],
                },
            ],
        }
    ]


def test_includes_code_blocks():
    markdown = "```python\nprint('hi')\n```\n"

    blocks = markdown_to_blocks(markdown)
    assert [block.to_dict() for block in blocks] == [
        {
            "type": "block_code",
            "language": "python",
            "text": "print('hi')\n",
        }
    ]


def test_ignores_untracked_block_types():
    markdown = dedent(
        """
        Plain paragraph

        ---

        <custom>Raw HTML block</custom>
        """
    )

    blocks = markdown_to_blocks(markdown)

    # Thematic breaks are dropped; inline HTML wrappers are stripped from paragraph text.
    assert [block.to_dict() for block in blocks] == [
        {
            "type": "paragraph",
            "text": "Plain paragraph",
        },
        {
            "type": "paragraph",
            "text": "Raw HTML block",
        },
    ]


def test_heading_levels_up_to_five():
    markdown = dedent(
        """
        # H1
        ## H2
        ### H3
        #### H4
        ##### H5
        """
    )

    blocks = markdown_to_blocks(markdown)

    assert [block.to_dict() for block in blocks] == [
        {
            "type": "heading",
            "level": 1,
            "text": "H1",
            "children": [
                {
                    "type": "heading",
                    "level": 2,
                    "text": "H2",
                    "children": [
                        {
                            "type": "heading",
                            "level": 3,
                            "text": "H3",
                            "children": [
                                {
                                    "type": "heading",
                                    "level": 4,
                                    "text": "H4",
                                    "children": [
                                        {
                                            "type": "heading",
                                            "level": 5,
                                            "text": "H5",
                                            "children": [],
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ],
        }
    ]


def test_mixed_heading_sequence():
    markdown = dedent(
        """
        ### H3
        # H1
        ## H2
        ### H3
        #### H4
        ##### H5
        ##### H5
        ##### H5
        # H1
        ## H2
        """
    )

    blocks = markdown_to_blocks(markdown)

    assert [block.to_dict() for block in blocks] == [
        {
            "type": "heading",
            "level": 3,
            "text": "H3",
            "children": [],
        },
        {
            "type": "heading",
            "level": 1,
            "text": "H1",
            "children": [
                {
                    "type": "heading",
                    "level": 2,
                    "text": "H2",
                    "children": [
                        {
                            "type": "heading",
                            "level": 3,
                            "text": "H3",
                            "children": [
                                {
                                    "type": "heading",
                                    "level": 4,
                                    "text": "H4",
                                    "children": [
                                        {
                                            "type": "heading",
                                            "level": 5,
                                            "text": "H5",
                                            "children": [],
                                        },
                                        {
                                            "type": "heading",
                                            "level": 5,
                                            "text": "H5",
                                            "children": [],
                                        },
                                        {
                                            "type": "heading",
                                            "level": 5,
                                            "text": "H5",
                                            "children": [],
                                        },
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ],
        },
        {
            "type": "heading",
            "level": 1,
            "text": "H1",
            "children": [
                {
                    "type": "heading",
                    "level": 2,
                    "text": "H2",
                    "children": [],
                }
            ],
        },
    ]


def test_roundtrip_generation_matches_structure():
    markdown = dedent(
        """
        # Title

        Intro paragraph with **emphasis**.

        - Item 1
          - Nested bullet

        1. Step one
        2. Step two

        ```python
        print('hi')
        ```

        > Block quote
        """
    )

    original_blocks = markdown_to_blocks(markdown)
    regenerated_markdown = blocks_to_markdown(original_blocks)
    regenerated_blocks = markdown_to_blocks(regenerated_markdown)

    assert regenerated_blocks == original_blocks


def test_table_with_list_items_roundtrip():
    markdown = dedent(
        """
        | Feature | Details |
        | --- | --- |
        | Pros | - Fast\\n- Reliable |
        | Cons | - Expensive |
        """
    )

    blocks = markdown_to_blocks(markdown)
    assert [block.to_dict() for block in blocks] == [
        {
            "type": "table",
            "header": ["Feature", "Details"],
            "rows": [
                ["Pros", "- Fast\\n- Reliable"],
                ["Cons", "- Expensive"],
            ],
        }
    ]

    regenerated = blocks_to_markdown(blocks)
    assert "- Fast" in regenerated


def test_paragraph_with_embedded_escaped_list():
    markdown = dedent(
        """
        Paragraph with inline bullet list:
        \\- Item one
        \\- Item two
        Closing sentence.
        """
    )

    blocks = markdown_to_blocks(markdown)

    assert [block.to_dict() for block in blocks] == [
        {
            "type": "paragraph",
            "text": "Paragraph with inline bullet list:\n- Item one\n- Item two\nClosing sentence.",
        }
    ]

    regenerated = blocks_to_markdown(blocks)
    assert regenerated == (
        "Paragraph with inline bullet list:\n"
        "- Item one\n"
        "- Item two\n"
        "Closing sentence.\n"
    )


def test_inline_code_preserved_in_list_item():
    markdown = "- Variables and functions: `snake_case`\n"

    blocks = markdown_to_blocks(markdown)

    assert [block.to_dict() for block in blocks] == [
        {
            "type": "list",
            "ordered": False,
            "items": [
                {
                    "type": "list_item",
                    "children": [
                        {
                            "type": "paragraph",
                            "text": "Variables and functions: `snake_case`",
                        }
                    ],
                }
            ],
        }
    ]

    regenerated = blocks_to_markdown(blocks)
    assert "`snake_case`" in regenerated


def test_blocks_to_markdown_default_filter():
    """Test that default filter includes all blocks."""
    markdown = dedent(
        """
        # Heading

        Paragraph text.

        - List item
        """
    )

    blocks = markdown_to_blocks(markdown)
    result = blocks_to_markdown(blocks)

    # Should include all blocks
    assert "# Heading" in result
    assert "Paragraph text." in result
    assert "- List item" in result


def test_blocks_to_markdown_filter_exclude_headings():
    """Test filter that excludes heading blocks."""
    markdown = dedent(
        """
        # Heading

        Paragraph text.

        ## Another Heading

        More text.
        """
    )

    blocks = markdown_to_blocks(markdown)
    result = blocks_to_markdown(blocks, filter_func=lambda block: block.type.value != "heading")

    # Since all content is nested under headings, filtering out headings results in empty output
    assert result.strip() == ""


def test_blocks_to_markdown_filter_only_paragraphs():
    """Test filter that includes only paragraph blocks."""
    markdown = dedent(
        """
        First paragraph.

        - List item

        Second paragraph.
        """
    )

    blocks = markdown_to_blocks(markdown)
    result = blocks_to_markdown(blocks, filter_func=lambda block: block.type.value == "paragraph")

    # Should include only top-level paragraphs
    assert "# Heading" not in result
    assert "- List item" not in result
    assert "First paragraph." in result
    assert "Second paragraph." in result


def test_blocks_to_markdown_filter_by_content():
    """Test filter based on block content."""
    from ask_memory.chunker.markdown_blocks import HeadingBlock, ParagraphBlock

    markdown = dedent(
        """
        # Special Heading

        This is special content.

        # Normal Heading

        This is normal content.
        """
    )

    blocks = markdown_to_blocks(markdown)
    result = blocks_to_markdown(
        blocks,
        filter_func=lambda block: (
            isinstance(block, (HeadingBlock, ParagraphBlock)) and 
            "special" in block.text.lower()
        ) if isinstance(block, (HeadingBlock, ParagraphBlock)) else True
    )

    # Should include only the special heading and its children
    assert "# Special Heading" in result
    assert "This is special content." in result
    assert "# Normal Heading" not in result
    assert "This is normal content." not in result


def test_blocks_to_markdown_filter_exclude_code_blocks():
    """Test filter that excludes code blocks."""
    markdown = dedent(
        """
        Here is some code:

        ```python
        print("hello")
        ```

        And some text after.
        """
    )

    blocks = markdown_to_blocks(markdown)
    result = blocks_to_markdown(blocks, filter_func=lambda block: block.type.value != "block_code")

    # Should exclude code block but include other content
    assert "Here is some code:" in result
    assert "```python" not in result
    assert 'print("hello")' not in result
    assert "And some text after." in result
