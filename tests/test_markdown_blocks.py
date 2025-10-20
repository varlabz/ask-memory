import json
from textwrap import dedent

from ask_memory.markdown_blocks import blocks_to_markdown, parse_markdown_blocks


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

    blocks = parse_markdown_blocks(markdown)

    # Dump to JSON to ensure serialisable output and stable assertion diff.
    assert json.loads(json.dumps(blocks)) == [
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
                    "type": "blockquote",
                    "children": [
                        {"type": "paragraph", "text": "Quoted text"},
                    ],
                },
            ],
        }
    ]


def test_includes_code_blocks():
    markdown = "```python\nprint('hi')\n```\n"

    blocks = parse_markdown_blocks(markdown)
    assert blocks == [
        {
            "type": "code",
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

    blocks = parse_markdown_blocks(markdown)

    # Thematic breaks are dropped; inline HTML wrappers are stripped from paragraph text.
    assert blocks == [
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

    blocks = parse_markdown_blocks(markdown)

    assert blocks == [
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

    blocks = parse_markdown_blocks(markdown)

    assert blocks == [
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

    original_blocks = parse_markdown_blocks(markdown)
    regenerated_markdown = blocks_to_markdown(original_blocks)
    regenerated_blocks = parse_markdown_blocks(regenerated_markdown)

    assert regenerated_blocks == original_blocks


def test_blocks_to_markdown_filter():
    markdown = dedent(
        """
        # Title

        First paragraph.

        Second paragraph.
        """
    )

    blocks = parse_markdown_blocks(markdown)

    filtered = blocks_to_markdown(
        blocks,
        include=lambda block: block.get("type") != "paragraph"
        or block.get("text") != "First paragraph.",
    )

    assert "First paragraph." not in filtered
    assert "Second paragraph." in filtered


def test_table_with_list_items_roundtrip():
    markdown = dedent(
        """
        | Feature | Details |
        | --- | --- |
        | Pros | - Fast\\n- Reliable |
        | Cons | - Expensive |
        """
    )

    blocks = parse_markdown_blocks(markdown)
    assert blocks == [
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

    blocks = parse_markdown_blocks(markdown)

    assert blocks == [
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

    blocks = parse_markdown_blocks(markdown)

    assert blocks == [
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
