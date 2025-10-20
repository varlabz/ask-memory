"""Utilities for extracting top-level Markdown blocks into JSON-ready dicts.

Requires ``mistune`` to be installed in the current environment.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional

import mistune

Block = Dict[str, Any]


class NodeType(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    BLOCK_TEXT = "block_text"
    LIST = "list"
    LIST_ITEM = "list_item"
    BLOCK_QUOTE = "block_quote"
    TABLE = "table"
    BLOCK_CODE = "block_code"
    TEXT = "text"

def parse_markdown_blocks(markdown_text: str) -> List[Block]:
    """Parse Markdown text into a list of block-level dictionaries.

    Only headings, paragraphs, lists, tables, and block quotes are kept.
    The resulting structures are safe to serialise with :mod:`json` or
    similar tooling.
    """
    # ``renderer='ast'`` instructs mistune to return a list of node dicts.
    md = mistune.create_markdown(renderer="ast", plugins=["table"])
    ast = md(markdown_text)

    blocks: List[Block] = []

    for raw_node in ast:
        if not isinstance(raw_node, dict):
            continue
        converted = convert_block(raw_node)
        if converted is not None:
            blocks.append(converted)

    return nest_blocks(blocks)


def blocks_to_markdown(
    blocks: List[Block],
    *,
    include: Optional[Callable[[Block], bool]] = None,
) -> str:
    """Render a list of block dictionaries back into Markdown text."""
    if include is not None:
        blocks = [
            filtered
            for block in blocks
            if (filtered := _filter_block(block, include)) is not None
        ]

    lines: List[str] = []
    for block in blocks:
        lines.extend(render_block(block))
    lines = _trim_trailing_blank_lines(lines)
    if not lines:
        return ""
    return "\n".join(lines).strip("\n") + "\n"


def convert_block(node: Block) -> Block | None:
    """Convert a mistune AST node into a JSON-ready block preserving hierarchy."""
    node_type_value = node.get("type")
    if not isinstance(node_type_value, str):
        return None

    try:
        node_type = NodeType(node_type_value)
    except ValueError:
        return None

    if node_type is NodeType.HEADING:
        return {
            "type": "heading",
            "level": node.get("attrs", {}).get("level"),
            "text": extract_text(node),
            "children": [],
        }

    if node_type in {NodeType.PARAGRAPH, NodeType.BLOCK_TEXT}:
        return {
            "type": "paragraph",
            "text": extract_text(node),
        }

    if node_type is NodeType.LIST:
        attrs = node.get("attrs", {})
        items = []
        for child in node.get("children", []):
            if not isinstance(child, dict):
                continue
            child_type = child.get("type")
            child_type_enum = None
            if isinstance(child_type, str):
                try:
                    child_type_enum = NodeType(child_type)
                except ValueError:
                    child_type_enum = None

            if child_type_enum is NodeType.LIST_ITEM:
                items.append(
                    {
                        "type": "list_item",
                        "children": convert_children(child),
                    }
                )
                continue
            converted_child = convert_block(child)
            if converted_child is not None:
                items.append(converted_child)
        return {
            "type": "list",
            "ordered": attrs.get("ordered", False),
            "items": items,
        }

    if node_type is NodeType.LIST_ITEM:
        return {
            "type": "list_item",
            "children": convert_children(node),
        }

    if node_type is NodeType.BLOCK_QUOTE:
        return {
            "type": "blockquote",
            "children": convert_children(node),
        }

    if node_type is NodeType.TABLE:
        return normalise_table(node)

    if node_type is NodeType.BLOCK_CODE:
        attrs = node.get("attrs", {})
        return {
            "type": "code",
            "language": attrs.get("info"),
            "text": node.get("raw", ""),
        }

    return None


def convert_children(node: Block) -> List[Block]:
    """Recursively convert child nodes, filtering unsupported types."""
    children: List[Block] = []
    for child in node.get("children", []):
        if not isinstance(child, dict):
            continue
        converted = convert_block(child)
        if converted is not None:
            children.append(converted)
    return children


def extract_text(node: Block) -> str:
    """Recursively collect raw text from a node and its descendants."""
    node_type_value = node.get("type")
    if isinstance(node_type_value, str) and node_type_value == NodeType.TEXT.value:
        return node.get("raw", "")

    children: Iterable[Block] = node.get("children", [])
    return "".join(extract_text(child) for child in children)


def render_block(block: Block, indent: int = 0) -> List[str]:
    block_type = block.get("type")

    if block_type == NodeType.HEADING.value:
        level = block.get("level") or 1
        heading_line = f"{'#' * int(level)} {block.get('text', '').strip()}".rstrip()
        child_lines: List[str] = []
        for child in block.get("children", []):
            child_lines.extend(render_block(child, indent))
        if child_lines:
            return [heading_line, "", *child_lines]
        return [heading_line]

    if block_type == "paragraph":
        return [" " * indent + block.get("text", "").strip()]

    if block_type == "code":
        language = block.get("language") or ""
        fence = f"```{language}".rstrip()
        code = block.get("text", "").rstrip("\n")
        return [fence, code, "```"]

    if block_type == "blockquote":
        inner_lines: List[str] = []
        for child in block.get("children", []):
            inner_lines.extend(render_block(child, indent))
        quoted = []
        for line in inner_lines or [""]:
            quoted.append(f"> {line}".rstrip())
        return quoted

    if block_type == "list":
        ordered = bool(block.get("ordered"))
        rendered: List[str] = []
        for idx, item in enumerate(block.get("items", []), start=1):
            rendered.extend(render_list_item(item, ordered, indent, idx))
        return rendered

    if block_type == "list_item":
        return render_list_item(block, False, indent)

    if block_type == "table":
        header = block.get("header", [])
        rows = block.get("rows", [])
        header_line = " | ".join(header)
        separator = " | ".join(["---" for _ in header])
        table_lines = [header_line, separator]
        for row in rows:
            table_lines.append(" | ".join(row))
        return table_lines

    return []


def render_list_item(item: Block, ordered: bool, indent: int, index: int = 1) -> List[str]:
    prefix = f"{index}. " if ordered else "- "
    child_blocks = item.get("children", [])
    if not child_blocks:
        return [" " * indent + prefix.rstrip()]

    rendered: List[str] = []
    first_child = child_blocks[0]

    if first_child.get("type") == "paragraph":
        para_lines = render_block(first_child, 0)
        if para_lines:
            rendered.append(" " * indent + prefix + para_lines[0].lstrip())
            for line in para_lines[1:]:
                rendered.append(" " * (indent + len(prefix)) + line.lstrip())
        else:
            rendered.append(" " * indent + prefix.rstrip())
        remaining_children = child_blocks[1:]
    else:
        rendered.append(" " * indent + prefix.rstrip())
        remaining_children = child_blocks

    for child in remaining_children:
        child_lines = render_block(child, indent + len(prefix))
        rendered.extend(child_lines)

    return rendered


def _trim_trailing_blank_lines(lines: List[str]) -> List[str]:
    trimmed = list(lines)
    while trimmed and not trimmed[-1].strip():
        trimmed.pop()
    return trimmed


def _filter_block(block: Block, include: Callable[[Block], bool]) -> Block | None:
    if not include(block):
        return None

    filtered: Block = {k: v for k, v in block.items()}

    if "children" in block and isinstance(block["children"], list):
        child_list: List[Block] = []
        for child in block["children"]:
            if isinstance(child, dict):
                child_filtered = _filter_block(child, include)
                if child_filtered is not None:
                    child_list.append(child_filtered)
        filtered["children"] = child_list

    if filtered.get("type") == "list":
        items: List[Block] = []
        for item in filtered.get("items", []):
            if isinstance(item, dict):
                item_filtered = _filter_block(item, include)
                if item_filtered is not None:
                    items.append(item_filtered)
        filtered["items"] = items

    return filtered


def nest_blocks(blocks: List[Block]) -> List[Block]:
    """Arrange blocks so that headings own subsequent content based on level."""
    nested: List[Block] = []
    heading_stack: List[Block] = []

    for block in blocks:
        block_type = block.get("type")
        if block_type == NodeType.HEADING.value:
            level = block.get("level")
            if level is None:
                level = 0
            if "children" not in block or block["children"] is None:
                block["children"] = []

            while heading_stack and (heading_stack[-1].get("level") or 0) >= level:
                heading_stack.pop()

            if heading_stack:
                heading_stack[-1]["children"].append(block)
            else:
                nested.append(block)

            heading_stack.append(block)
        else:
            if heading_stack:
                heading_stack[-1]["children"].append(block)
            else:
                nested.append(block)

    return nested


def normalise_table(node: Block) -> Block:
    """Convert a mistune table node into a simple dict layout."""
    header: List[str] = []
    rows: List[List[str]] = []

    head_node = next(
        (
            child
            for child in node.get("children", [])
            if child.get("type") == "table_head"
        ),
        None,
    )
    if head_node:
        header = [extract_text(cell) for cell in head_node.get("children", [])]

    body_node = next(
        (
            child
            for child in node.get("children", [])
            if child.get("type") == "table_body"
        ),
        None,
    )
    if body_node:
        for row_node in body_node.get("children", []):
            if row_node.get("type") != "table_row":
                continue
            row_cells = [extract_text(cell) for cell in row_node.get("children", [])]
            rows.append(row_cells)

    return {
        "type": "table",
        "header": header,
        "rows": rows,
    }


__all__ = ["parse_markdown_blocks", "blocks_to_markdown"]


if __name__ == "__main__":
    import json
    import sys
    if len(sys.argv) < 2:
        print("Please provide a file path as an argument.")
        sys.exit(1)
    
    file_path = sys.argv[1]
    with open(file_path, "r", encoding="utf-8") as f:
        markdown_text = f.read()    
    blocks = parse_markdown_blocks(markdown_text)
    print(json.dumps(blocks, indent=2))