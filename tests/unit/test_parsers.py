import io
import pytest
import pandas as pd
import pymupdf as fitz

from app.core.errors import DocumentProcessingError
from app.ingestion.parser import DocumentParser, ParsedDocument


def create_in_memory_pdf(pages_text: list[str], title: str = "Quarterly Financial Report") -> bytes:
    """Helper creating a real valid PDF document in memory using PyMuPDF."""
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((50, 72), text, fontsize=12)
    doc.set_metadata({"title": title, "author": "Enterprise Analytics Team"})
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_parse_real_pdf_multi_page():
    page_1 = "Enterprise Intelligence Platform Overview: Stage 1 Foundation and Stage 2 Database."
    page_2 = "Financial Metrics Q3: Total Revenue $45.2M, Net Profit Margin 18.4%."
    pdf_bytes = create_in_memory_pdf([page_1, page_2], title="Q3 Performance")

    parsed = DocumentParser.parse("performance_report.pdf", pdf_bytes)

    assert isinstance(parsed, ParsedDocument)
    assert parsed.filename == "performance_report.pdf"
    assert parsed.file_type == "pdf"
    assert len(parsed.sections) == 2
    assert parsed.sections[0].metadata["page_number"] == 1
    assert "Overview" in parsed.sections[0].content
    assert parsed.sections[1].metadata["page_number"] == 2
    assert "Financial Metrics" in parsed.sections[1].content
    assert parsed.content_hash == DocumentParser.calculate_sha256(pdf_bytes)


def test_parse_txt_utf8_and_latin1():
    text_content = "This is an enterprise policy document outlining security and access protocols."
    txt_bytes = text_content.encode("utf-8")

    parsed = DocumentParser.parse("policy.txt", txt_bytes)
    assert parsed.file_type == "txt"
    assert len(parsed.sections) == 1
    assert parsed.sections[0].content == text_content

    # Latin-1 encoded with accents
    latin1_text = "Règlement intérieur de sécurité informatique et conformité légale."
    latin1_bytes = latin1_text.encode("latin-1")
    parsed_latin = DocumentParser.parse("reglement.txt", latin1_bytes)
    assert "Règlement" in parsed_latin.sections[0].content


def test_parse_markdown_with_headers():
    md_content = """# Architecture Strategy
The system adopts a modular hexagonal architecture.

## Retrieval Engine
Reciprocal Rank Fusion combines sparse and dense vectors.

### Telemetry Subsystem
Token budgeting tracks costs across providers.
"""
    md_bytes = md_content.encode("utf-8")
    parsed = DocumentParser.parse("architecture.md", md_bytes)

    assert parsed.file_type == "md"
    assert len(parsed.sections) == 3
    headers = [s.metadata.get("header") for s in parsed.sections]
    assert "Architecture Strategy" in headers
    assert "Retrieval Engine" in headers
    assert "Telemetry Subsystem" in headers


def test_parse_csv_tabular_data():
    df = pd.DataFrame({
        "EmployeeID": [101, 102, 103],
        "Name": ["Alice Smith", "Bob Jones", "Charlie Brown"],
        "Department": ["Engineering", "Security", "Operations"],
        "Salary": [135000, 142000, 118000],
    })
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode("utf-8")

    parsed = DocumentParser.parse("employees.csv", csv_bytes)
    assert parsed.file_type == "csv"
    assert len(parsed.sections) == 3
    assert "Row 1" in parsed.sections[0].content
    assert "Alice Smith" in parsed.sections[0].content
    assert "Engineering" in parsed.sections[0].content
    assert parsed.metadata["row_count"] == 3


def test_reject_empty_file():
    with pytest.raises(DocumentProcessingError) as exc:
        DocumentParser.parse("empty.txt", b"")
    assert "empty or contains only whitespace" in str(exc.value)

    with pytest.raises(DocumentProcessingError) as exc:
        DocumentParser.parse("spaces.txt", b"   \n\n\t  ")
    assert "empty or contains only whitespace" in str(exc.value)


def test_reject_unsupported_format():
    with pytest.raises(DocumentProcessingError) as exc:
        DocumentParser.parse("malicious.exe", b"MZ\x90\x00\x03\x00\x00\x00")
    assert "Unsupported file format" in str(exc.value)


def test_reject_corrupted_pdf():
    corrupted_bytes = b"%PDF-1.4\nSome invalid truncated stream without EOF markers"
    with pytest.raises(DocumentProcessingError) as exc:
        DocumentParser.parse("corrupted.pdf", corrupted_bytes)
    assert "corrupted" in str(exc.value).lower() or "text" in str(exc.value).lower()
