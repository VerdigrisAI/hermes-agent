"""Behavioral tests for bounded OOXML document extraction."""

from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from gateway.document_extract import DocumentExtractionError, extract_docx_text


def _docx(document_xml: str) -> bytes:
    buf = BytesIO()
    with ZipFile(buf, "w", ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "<Types />")
        archive.writestr("word/document.xml", document_xml)
    return buf.getvalue()


def _docx_with_corrupted_document_stream(document_xml: str) -> bytes:
    payload = bytearray(_docx(document_xml))
    with ZipFile(BytesIO(payload)) as archive:
        info = archive.getinfo("word/document.xml")
        header = info.header_offset
        name_length = int.from_bytes(payload[header + 26:header + 28], "little")
        extra_length = int.from_bytes(payload[header + 28:header + 30], "little")
        compressed_start = header + 30 + name_length + extra_length
        payload[compressed_start + info.compress_size // 2] ^= 0xFF
    return bytes(payload)


def test_extract_docx_preserves_heading_paragraph_table_order() -> None:
    data = _docx(
        """<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Overview</w:t></w:r></w:p>
    <w:p><w:r><w:t>Revenue plan</w:t></w:r></w:p>
    <w:tbl>
      <w:tr><w:tc><w:p><w:r><w:t>Region</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Owner</w:t></w:r></w:p></w:tc></w:tr>
      <w:tr><w:tc><w:p><w:r><w:t>West</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Zoë</w:t></w:r></w:p></w:tc></w:tr>
    </w:tbl>
    <w:p><w:r><w:t>Next section</w:t></w:r></w:p>
  </w:body>
</w:document>"""
    )

    result = extract_docx_text(data)

    assert result.truncated is False
    assert result.text == (
        "# Overview\n\nRevenue plan\n\n"
        "| Region | Owner |\n| --- | --- |\n| West | Zoë |\n\nNext section"
    )


def test_extract_docx_preserves_tabs_and_line_breaks() -> None:
    data = _docx(
        """<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>
<w:p><w:r><w:t>A</w:t><w:tab/><w:t>B</w:t><w:br/><w:t>C</w:t></w:r></w:p>
</w:body></w:document>"""
    )

    assert extract_docx_text(data).text == "A\tB\nC"


def test_extract_docx_reports_truncation() -> None:
    data = _docx(
        """<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>
<w:p><w:r><w:t>abcdefghijklmnopqrstuvwxyz</w:t></w:r></w:p>
</w:body></w:document>"""
    )

    result = extract_docx_text(data, max_chars=10)

    assert result.text == "abcdefghij"
    assert result.truncated is True


def test_extract_docx_rejects_main_xml_above_configured_limit() -> None:
    data = _docx(
        """<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>
<w:p><w:r><w:t>too large</w:t></w:r></w:p></w:body></w:document>"""
    )

    with pytest.raises(DocumentExtractionError) as exc_info:
        extract_docx_text(data, max_xml_bytes=20)

    assert exc_info.value.code == "docx_too_large"


def test_extract_docx_rejects_dtd_entity_expansion() -> None:
    data = _docx(
        """<?xml version="1.0"?>
<!DOCTYPE w:document [<!ENTITY payload "expanded">]>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:body><w:p><w:r><w:t>&payload;</w:t></w:r></w:p></w:body></w:document>"""
    )

    with pytest.raises(DocumentExtractionError, match="forbidden DTD"):
        extract_docx_text(data)


def test_extract_docx_caps_irregular_table_columns_and_output() -> None:
    cells = "".join(
        f"<w:tc><w:p><w:r><w:t>column-{index}</w:t></w:r></w:p></w:tc>"
        for index in range(100)
    )
    data = _docx(
        """<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>"""
        f"<w:tbl><w:tr>{cells}</w:tr></w:tbl>"
        "</w:body></w:document>"
    )

    result = extract_docx_text(data, max_chars=500)

    assert result.truncated is True
    assert len(result.text) <= 500
    assert "column-0" in result.text


@pytest.mark.parametrize(
    "data",
    [
        b"not a zip",
        _docx("<Types />"),
        _docx_with_corrupted_document_stream(
            '<w:document xmlns:w="http://schemas.openxmlformats.org/'
            'wordprocessingml/2006/main"><w:body /></w:document>'
        ),
    ],
)
def test_extract_docx_rejects_malformed_packages(data: bytes) -> None:
    with pytest.raises(DocumentExtractionError, match="DOCX") as exc_info:
        extract_docx_text(data)

    assert exc_info.value.code == "malformed_docx"


def test_extract_docx_preserves_blocks_wrapped_in_content_controls() -> None:
    data = _docx(
        """<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>
<w:sdt><w:sdtPr/><w:sdtContent>
<w:p><w:r><w:t>Approval terms</w:t></w:r></w:p>
</w:sdtContent></w:sdt>
<w:p><w:r><w:t>After the control</w:t></w:r></w:p>
</w:body></w:document>"""
    )

    result = extract_docx_text(data)

    assert result.text == "Approval terms\n\nAfter the control"
    assert result.truncated is False


def test_table_column_cap_does_not_suppress_later_document_blocks() -> None:
    cells = "".join(
        f"<w:tc><w:p><w:r><w:t>column-{index}</w:t></w:r></w:p></w:tc>"
        for index in range(65)
    )
    data = _docx(
        """<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>"""
        f"<w:tbl><w:tr>{cells}</w:tr></w:tbl>"
        "<w:p><w:r><w:t>Required closing paragraph</w:t></w:r></w:p>"
        "</w:body></w:document>"
    )

    result = extract_docx_text(data, max_chars=10_000)

    assert "Required closing paragraph" in result.text
    assert result.truncated is True
