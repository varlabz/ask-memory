"""
Utilities for extracting top-level Markdown blocks into JSON-ready dicts.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
import collections.abc

import mistune

class NodeType(str, Enum):
    """Enum for mistune AST node types."""
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    BLOCK_TEXT = "block_text"
    LIST = "list"
    LIST_ITEM = "list_item"
    BLOCK_QUOTE = "block_quote"
    TABLE = "table"
    BLOCK_CODE = "block_code"
    TEXT = "text"

@dataclass
class BaseBlock:
    """Base dataclass for a Markdown block.

    Instances are converted to plain dicts with :meth:`to_dict` before being
    returned from public APIs to preserve backward compatibility.
    """
    type: NodeType

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict

        return asdict(
            self,
            dict_factory=lambda data: {
                k: v.value if isinstance(v, NodeType) else v for k, v in data
            },
        )

    def to_md(self, indent: int = 0) -> str:
        """Render the block to Markdown string."""
        raise NotImplementedError


@dataclass
class HeadingBlock(BaseBlock):
    level: int = 1
    text: str = ""
    children: List["BaseBlock"] = field(default_factory=list)

    def to_md(self, indent: int = 0) -> str:
        level = self.level if isinstance(self.level, int) else 1
        heading_line = f"{'#' * int(level)} {self.text.strip()}".rstrip()
        child_lines: List[str] = [line for child in self.children 
                                       for line in child.to_md(indent).split('\n')]
        if child_lines:
            return "\n".join([heading_line, "", *child_lines])
        return heading_line


@dataclass
class ParagraphBlock(BaseBlock):
    text: str = ""

    def to_md(self, indent: int = 0) -> str:
        return " " * indent + self.text.strip()


@dataclass
class CodeBlock(BaseBlock):
    language: Optional[str] = None
    text: str = ""

    def to_md(self, indent: int = 0) -> str:
        language = self.language or ""
        fence = f"```{language}".rstrip()
        code = self.text.rstrip("\n")
        return "\n".join([fence, code, "```"])


@dataclass
class ListItemBlock(BaseBlock):
    children: List["BaseBlock"] = field(default_factory=list)

    def to_md(self, indent: int = 0, ordered: bool = False, index: int = 1) -> str:
        prefix = f"{index}. " if ordered else "- "
        child_blocks = self.children
        if not child_blocks:
            return " " * indent + prefix.rstrip()

        rendered: List[str] = []
        first_child = child_blocks[0]
        if isinstance(first_child, ParagraphBlock):
            para_lines = first_child.to_md(0).split('\n')
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
            rendered.extend(child.to_md(indent + len(prefix)).split('\n'))

        return "\n".join(rendered)


@dataclass
class ListBlock(BaseBlock):
    ordered: bool = False
    items: List["BaseBlock"] = field(default_factory=list)

    def to_md(self, indent: int = 0) -> str:
        rendered: List[str] = []
        for idx, item in enumerate(self.items, start=1):
            prefix = f"{idx}. " if self.ordered else "- "
            if isinstance(item, ListItemBlock):
                rendered.extend(item.to_md(indent, self.ordered, idx).split('\n'))
            else:
                item_lines = item.to_md(0).split('\n')
                if item_lines:
                    rendered.append(" " * indent + prefix + item_lines[0].lstrip())
                    for line in item_lines[1:]:
                        rendered.append(" " * (indent + len(prefix)) + line.lstrip())
                else:
                    rendered.append(" " * indent + prefix.rstrip())
        return "\n".join(rendered)


@dataclass
class BlockquoteBlock(BaseBlock):
    children: List["BaseBlock"] = field(default_factory=list)

    def to_md(self, indent: int = 0) -> str:
        inner_lines: List[str] = []
        for child in self.children:
            inner_lines.extend(child.to_md(indent).split('\n'))
        quoted: List[str] = []
        for line in inner_lines or [""]:
            quoted.append(f"> {line}".rstrip())
        return "\n".join(quoted)


@dataclass
class TableBlock(BaseBlock):
    header: List[str] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)

    def to_md(self, indent: int = 0) -> str:
        header_line = " | ".join(self.header)
        separator = " | ".join(["---" for _ in self.header])
        table_lines = [header_line, separator]
        for row in self.rows:
            table_lines.append(" | ".join(row))
        return "\n".join(table_lines)



def markdown_to_blocks(markdown_text: str) -> List[BaseBlock]:
    """Parse Markdown text into a list of block-level dictionaries."""

    md = mistune.create_markdown(renderer="ast", plugins=["table"])
    ast = md(markdown_text)

    blocks: List[BaseBlock] = []
    for raw_node in ast:
        if not isinstance(raw_node, dict):
            continue
        converted = convert_block(raw_node)
        if converted is not None:
            blocks.append(converted)

    nested_blocks = nest_blocks(blocks)
    return nested_blocks


def blocks_to_markdown(
    blocks: list[BaseBlock],
    filter_func: Callable[[BaseBlock], bool] = lambda block: True,
) -> str:
    """Render a sequence of block definitions back into Markdown text.

    Args:
        blocks: List of BaseBlock objects to render.
        filter_func: Filter function that takes a BaseBlock and returns bool.
                    Only blocks for which filter_func returns True will be rendered.
                    Defaults to including all blocks.

    Returns:
        Markdown string representation of the blocks.
    """

    lines: List[str] = []
    for block in blocks:
        if filter_func(block):
            lines.extend(block.to_md().split('\n'))
    lines = _trim_trailing_blank_lines(lines)
    if not lines:
        return ""
    return "\n".join(lines).strip("\n") + "\n"


def convert_block(node: dict[str, Any]) -> BaseBlock | None:
    """Convert a mistune AST node into a structured :class:`BaseBlock`."""

    node_type_value = node.get("type")
    if not isinstance(node_type_value, str):
        return None

    try:
        node_type = NodeType(node_type_value)
    except ValueError:
        return None

    if node_type is NodeType.HEADING:
        attrs = node.get("attrs", {})
        level = attrs.get("level") if isinstance(attrs, dict) else 1
        level_value = level if isinstance(level, int) else 1
        return HeadingBlock(
            type=NodeType.HEADING,
            level=level_value,
            text=extract_text(node),
            children=[],
        )

    if node_type in {NodeType.PARAGRAPH, NodeType.BLOCK_TEXT}:
        return ParagraphBlock(type=NodeType.PARAGRAPH, text=extract_text(node))

    if node_type is NodeType.LIST:
        attrs = node.get("attrs", {})
        ordered = False
        if isinstance(attrs, dict):
            ordered = bool(attrs.get("ordered", False))
        items: List[BaseBlock] = []
        children = node.get("children", [])
        if not isinstance(children, list):
            children = []

        for child in children:
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
                li = ListItemBlock(
                    type=NodeType.LIST_ITEM,
                    children=convert_children(child),
                )
                items.append(li)
                continue
            converted_child = convert_block(child)
            if converted_child is not None:
                items.append(converted_child)
        return ListBlock(type=NodeType.LIST, ordered=ordered, items=items)

    if node_type is NodeType.LIST_ITEM:
        return ListItemBlock(
            type=NodeType.LIST_ITEM,
            children=convert_children(node),
        )

    if node_type is NodeType.BLOCK_QUOTE:
        return BlockquoteBlock(
            type=NodeType.BLOCK_QUOTE,
            children=convert_children(node),
        )

    if node_type is NodeType.TABLE:
        return normalise_table(node)

    if node_type is NodeType.BLOCK_CODE:
        attrs = node.get("attrs", {})
        language = None
        if isinstance(attrs, dict):
            lang_value = attrs.get("info")
            language = lang_value if isinstance(lang_value, str) else None

        raw_text = node.get("raw", "")
        text_value = raw_text if isinstance(raw_text, str) else ""
        return CodeBlock(
            type=NodeType.BLOCK_CODE,
            language=language,
            text=text_value,
        )

    return None


def convert_children(node: dict[str, Any]) -> List[BaseBlock]:
    """Recursively convert child nodes, filtering unsupported types."""

    children: List[BaseBlock] = []
    child_nodes = node.get("children", [])
    if not isinstance(child_nodes, list):
        return children

    for child in child_nodes:
        if not isinstance(child, dict):
            continue
        converted = convert_block(child)
        if converted is not None:
            children.append(converted)
    return children


def extract_text(node: dict[str, Any]) -> str:
    """Recursively collect raw text from a node and its descendants."""

    node_type_value = node.get("type")
    if isinstance(node_type_value, str):
        if node_type_value == NodeType.TEXT.value:
            raw = node.get("raw", "")
            return raw if isinstance(raw, str) else ""
        if node_type_value in {"softbreak", "linebreak"}:
            return "\n"
        if node_type_value == "codespan":
            raw = node.get("raw", "")
            return f"`{raw}`" if isinstance(raw, str) else ""

    children = node.get("children", [])
    if not isinstance(children, list):
        return ""
    return "".join(
        extract_text(child) for child in children if isinstance(child, dict)
    )


def block_from_dict(data: dict[str, Any]) -> BaseBlock:
    """Convert a JSON-ready block dictionary into a :class:`BaseBlock`."""

    type_value = data.get("type")
    if isinstance(type_value, NodeType):
        node_type = type_value
    elif isinstance(type_value, str):
        try:
            node_type = NodeType(type_value)
        except ValueError:
            raise ValueError(f"Unsupported block type: {type_value}")
    else:
        raise ValueError("Block type must be a string or NodeType")

    if node_type is NodeType.HEADING:
        level_value = data.get("level")
        level = level_value if isinstance(level_value, int) else 1
        text_value = data.get("text", "")
        text = text_value if isinstance(text_value, str) else str(text_value)
        children_value = data.get("children", [])
        children = [
            block_from_dict(child)
            for child in children_value
            if isinstance(child, dict)
        ]
        return HeadingBlock(
            type=NodeType.HEADING,
            level=level,
            text=text,
            children=children,
        )

    if node_type is NodeType.PARAGRAPH:
        text_value = data.get("text", "")
        text = text_value if isinstance(text_value, str) else str(text_value)
        return ParagraphBlock(type=NodeType.PARAGRAPH, text=text)

    if node_type is NodeType.LIST:
        ordered = bool(data.get("ordered", False))
        items_value = data.get("items", [])
        items: List[BaseBlock] = []
        if isinstance(items_value, collections.abc.Sequence) and not isinstance(
            items_value, (str, bytes)
        ):
            for item in items_value:
                if isinstance(item, dict):
                    try:
                        items.append(block_from_dict(item))
                    except ValueError:
                        continue
        return ListBlock(type=NodeType.LIST, ordered=ordered, items=items)

    if node_type is NodeType.LIST_ITEM:
        children_value = data.get("children", [])
        children = [
            block_from_dict(child)
            for child in children_value
            if isinstance(child, dict)
        ]
        return ListItemBlock(type=NodeType.LIST_ITEM, children=children)

    if node_type is NodeType.BLOCK_QUOTE:
        children_value = data.get("children", [])
        children = [
            block_from_dict(child)
            for child in children_value
            if isinstance(child, dict)
        ]
        return BlockquoteBlock(type=NodeType.BLOCK_QUOTE, children=children)

    if node_type is NodeType.TABLE:
        header_value = data.get("header", [])
        header: List[str] = []
        if isinstance(header_value, collections.abc.Sequence) and not isinstance(
            header_value, (str, bytes)
        ):
            header = [str(cell) for cell in header_value]

        rows_value = data.get("rows", [])
        rows: List[List[str]] = []
        if isinstance(rows_value, collections.abc.Sequence) and not isinstance(
            rows_value, (str, bytes)
        ):
            for row in rows_value:
                if isinstance(row, collections.abc.Sequence) and not isinstance(row, (str, bytes)):
                    rows.append([str(cell) for cell in row])
        return TableBlock(type=NodeType.TABLE, header=header, rows=rows)

    if node_type is NodeType.BLOCK_CODE:
        language_value = data.get("language")
        language = language_value if isinstance(language_value, str) else None
        text_value = data.get("text", "")
        text = text_value if isinstance(text_value, str) else str(text_value)
        return CodeBlock(type=NodeType.BLOCK_CODE, language=language, text=text)

    raise ValueError(f"Unsupported block type: {node_type}")


def _trim_trailing_blank_lines(lines: List[str]) -> List[str]:
    """Remove trailing blank lines from a list of strings."""

    trimmed = list(lines)
    while trimmed and not trimmed[-1].strip():
        trimmed.pop()
    return trimmed


def nest_blocks(blocks: list[BaseBlock]) -> List[BaseBlock]:
    """Arrange blocks so that headings own subsequent content based on level."""

    nested: List[BaseBlock] = []
    heading_stack: List[HeadingBlock] = []

    for block in blocks:
        if isinstance(block, HeadingBlock):
            level = block.level if isinstance(block.level, int) else 0

            while heading_stack and (heading_stack[-1].level or 0) >= level:
                heading_stack.pop()

            if heading_stack:
                heading_stack[-1].children.append(block)
            else:
                nested.append(block)

            heading_stack.append(block)
        else:
            if heading_stack:
                heading_stack[-1].children.append(block)
            else:
                nested.append(block)

    return nested


def normalise_table(node: dict[str, Any]) -> TableBlock:
    """Convert a mistune table node into a :class:`TableBlock` layout."""

    header: List[str] = []
    rows: List[List[str]] = []

    children = node.get("children", [])
    if not isinstance(children, list):
        children = []

    head_node = next(
        (
            child
            for child in children
            if isinstance(child, dict) and child.get("type") == "table_head"
        ),
        None,
    )
    if head_node:
        head_children = head_node.get("children", [])
        if isinstance(head_children, list):
            header = [
                extract_text(cell)
                for cell in head_children
                if isinstance(cell, dict)
            ]

    body_node = next(
        (
            child
            for child in children
            if isinstance(child, dict) and child.get("type") == "table_body"
        ),
        None,
    )
    if body_node:
        body_children = body_node.get("children", [])
        if isinstance(body_children, list):
            for row_node in body_children:
                if not isinstance(row_node, dict) or row_node.get("type") != "table_row":
                    continue
                row_children = row_node.get("children", [])
                if isinstance(row_children, list):
                    row_cells = [
                        extract_text(cell)
                        for cell in row_children
                        if isinstance(cell, dict)
                    ]
                    rows.append(row_cells)

    return TableBlock(type=NodeType.TABLE, header=header, rows=rows)


__all__ = ["markdown_to_blocks", "blocks_to_markdown"]


if __name__ == "__main__":
    import json
    import sys
    if len(sys.argv) < 2:
        print("Please provide a file path as an argument.")
        sys.exit(1)
    
    file_path = sys.argv[1]
    if not file_path.lower().endswith(('.md', '.json')):
        print("Error: File must have .md or .json extension.")
        sys.exit(1)
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        if file_path.lower().endswith('.md'):
            # Generate JSON from Markdown
            blocks = markdown_to_blocks(content)
            print(json.dumps([block.to_dict() for block in blocks], indent=2))
        elif file_path.lower().endswith('.json'):
            # Generate Markdown from JSON
            data = json.loads(content)
            blocks = [block_from_dict(item) for item in data]
            markdown = blocks_to_markdown(blocks)
            print(markdown)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{file_path}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    