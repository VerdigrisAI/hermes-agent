"""Bounded local extraction for documents delivered by gateway adapters.

## Contract

``extract_docx_text(data, max_chars, max_xml_bytes)`` parses only the main
``word/document.xml`` part of an OOXML package. It preserves paragraph/table
order, renders heading styles and tables as Markdown, never performs network
I/O, bounds both XML expansion and returned text, and reports malformed input
through ``DocumentExtractionError`` rather than silently returning an empty
document.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import re
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile


WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": WORD_NS}
DEFAULT_MAX_DOCX_XML_BYTES = 10 * 1024 * 1024
DEFAULT_MAX_DOCUMENT_CHARS = 120_000
MAX_TABLE_ROWS = 1_000
MAX_TABLE_COLUMNS = 64


class DocumentExtractionError(ValueError):
    """Typed, user-safe document extraction failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ExtractedDocument:
    text: str
    truncated: bool = False


def _tag(local_name: str) -> str:
    return f"{{{WORD_NS}}}{local_name}"


def _paragraph_text(paragraph: ElementTree.Element) -> str:
    parts: list[str] = []
    for node in paragraph.iter():
        if node.tag == _tag("t"):
            parts.append(node.text or "")
        elif node.tag == _tag("tab"):
            parts.append("\t")
        elif node.tag in {_tag("br"), _tag("cr")}:
            parts.append("\n")
    return "".join(parts).strip()


def _heading_prefix(paragraph: ElementTree.Element) -> str:
    style = paragraph.find("./w:pPr/w:pStyle", NS)
    if style is None:
        return ""
    value = style.get(_tag("val"), "")
    match = re.fullmatch(r"Heading\s*([1-9])", value, re.IGNORECASE)
    if match:
        return "#" * min(int(match.group(1)), 6) + " "
    if value.lower() in {"title", "subtitle"}:
        return "# " if value.lower() == "title" else "## "
    return ""


def _escape_table_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def _table_markdown(
    table: ElementTree.Element, *, max_chars: int
) -> tuple[str, bool]:
    rows: list[list[str]] = []
    stored_chars = 0
    truncated = False
    for row_index, row in enumerate(table.findall("./w:tr", NS)):
        if row_index >= MAX_TABLE_ROWS or stored_chars >= max_chars:
            truncated = True
            break
        cells: list[str] = []
        for cell_index, cell in enumerate(row.findall("./w:tc", NS)):
            if cell_index >= MAX_TABLE_COLUMNS or stored_chars >= max_chars:
                truncated = True
                break
            paragraphs = [
                text
                for paragraph in cell.findall(".//w:p", NS)
                if (text := _paragraph_text(paragraph))
            ]
            value = _escape_table_cell("\n".join(paragraphs))
            remaining = max_chars - stored_chars
            if len(value) > remaining:
                value = value[:remaining]
                truncated = True
            cells.append(value)
            stored_chars += len(value)
        if cells:
            rows.append(cells)
    if not rows:
        return "", truncated
    width = max(len(row) for row in rows)
    rendered = [
        "| " + " | ".join(row + [""] * (width - len(row))) + " |"
        for row in rows
    ]
    rendered.insert(1, "| " + " | ".join("---" for _ in range(width)) + " |")
    table_text = "\n".join(rendered)
    return table_text[:max_chars], truncated or len(table_text) > max_chars


def extract_docx_text(
    data: bytes,
    *,
    max_chars: int = DEFAULT_MAX_DOCUMENT_CHARS,
    max_xml_bytes: int = DEFAULT_MAX_DOCX_XML_BYTES,
) -> ExtractedDocument:
    """Extract ordered, bounded Markdown-like text from DOCX bytes."""
    if max_chars < 1 or max_xml_bytes < 1:
        raise ValueError("DOCX extraction limits must be positive")
    try:
        with ZipFile(BytesIO(data)) as archive:
            try:
                info = archive.getinfo("word/document.xml")
            except KeyError as exc:
                raise DocumentExtractionError(
                    "malformed_docx", "DOCX package is missing word/document.xml"
                ) from exc
            if info.file_size > max_xml_bytes:
                raise DocumentExtractionError(
                    "docx_too_large",
                    f"DOCX main document XML exceeds the {max_xml_bytes}-byte extraction limit",
                )
            with archive.open(info) as document_member:
                document_xml = document_member.read(max_xml_bytes + 1)
    except DocumentExtractionError:
        raise
    except (BadZipFile, OSError, RuntimeError, ValueError) as exc:
        raise DocumentExtractionError(
            "malformed_docx", "DOCX package is malformed or unreadable"
        ) from exc

    if len(document_xml) > max_xml_bytes:
        raise DocumentExtractionError(
            "docx_too_large",
            f"DOCX main document XML exceeds the {max_xml_bytes}-byte extraction limit",
        )
    if b"<!DOCTYPE" in document_xml.upper():
        raise DocumentExtractionError(
            "malformed_docx", "DOCX main document XML contains a forbidden DTD"
        )
    try:
        root = ElementTree.fromstring(document_xml)
    except ElementTree.ParseError as exc:
        raise DocumentExtractionError(
            "malformed_docx", "DOCX main document XML is malformed"
        ) from exc

    body = root.find("./w:body", NS)
    if body is None:
        raise DocumentExtractionError(
            "malformed_docx", "DOCX main document has no body"
        )

    blocks: list[str] = []
    used_chars = 0
    truncated = False
    for child in body:
        separator_chars = 2 if blocks else 0
        remaining = max_chars - used_chars - separator_chars
        if remaining <= 0:
            truncated = True
            break
        if child.tag == _tag("p"):
            paragraph = _paragraph_text(child)
            if paragraph:
                block = _heading_prefix(child) + paragraph
                if len(block) > remaining:
                    block = block[:remaining]
                    truncated = True
                blocks.append(block)
                used_chars += separator_chars + len(block)
        elif child.tag == _tag("tbl"):
            table, table_truncated = _table_markdown(child, max_chars=remaining)
            if table:
                blocks.append(table)
                used_chars += separator_chars + len(table)
            truncated = truncated or table_truncated
        if truncated:
            break

    text = "\n\n".join(blocks)
    return ExtractedDocument(text=text, truncated=truncated)
