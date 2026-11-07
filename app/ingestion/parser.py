import hashlib
import io
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd
import pymupdf as fitz

from app.core.errors import DocumentProcessingError


@dataclass
class ParsedSection:
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedDocument:
    filename: str
    file_type: str
    file_size_bytes: int
    content_hash: str
    sections: List[ParsedSection]
    raw_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentParser:
    """Multi-format document parsing engine supporting PDF, TXT, Markdown, and CSV."""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown", ".csv"}

    @classmethod
    def calculate_sha256(cls, file_bytes: bytes) -> str:
        """Computes SHA-256 hash of file content for cryptographic deduplication."""
        hasher = hashlib.sha256()
        hasher.update(file_bytes)
        return hasher.hexdigest()

    @classmethod
    def parse(cls, filename: str, file_bytes: bytes) -> ParsedDocument:
        """Parses raw document bytes into a structured ParsedDocument.
        
        Raises DocumentProcessingError if file is invalid, empty, or unreadable.
        """
        if not file_bytes or len(file_bytes.strip()) == 0:
            raise DocumentProcessingError(f"File '{filename}' is empty or contains only whitespace.")

        _, ext = os.path.splitext(filename.lower())
        if ext not in cls.SUPPORTED_EXTENSIONS:
            supported = ", ".join(cls.SUPPORTED_EXTENSIONS)
            raise DocumentProcessingError(
                f"Unsupported file format '{ext}'. Supported formats: {supported}."
            )

        file_size = len(file_bytes)
        content_hash = cls.calculate_sha256(file_bytes)

        if ext == ".pdf":
            return cls._parse_pdf(filename, file_bytes, file_size, content_hash)
        elif ext in {".txt", ".md", ".markdown"}:
            return cls._parse_text(filename, file_bytes, ext, file_size, content_hash)
        elif ext == ".csv":
            return cls._parse_csv(filename, file_bytes, file_size, content_hash)
        else:
            raise DocumentProcessingError(f"Unhandled file extension: {ext}")

    @classmethod
    def _parse_pdf(cls, filename: str, file_bytes: bytes, file_size: int, content_hash: str) -> ParsedDocument:
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
        except Exception as e:
            raise DocumentProcessingError(f"Corrupted or invalid PDF file '{filename}': {str(e)}")

        if doc.page_count == 0:
            doc.close()
            raise DocumentProcessingError(f"PDF document '{filename}' has 0 pages.")

        sections: List[ParsedSection] = []
        raw_text_parts: List[str] = []

        for page_idx in range(doc.page_count):
            page = doc[page_idx]
            page_text = page.get_text("text").strip()
            if page_text:
                sections.append(
                    ParsedSection(
                        content=page_text,
                        metadata={
                            "page_number": page_idx + 1,
                            "source_file": filename,
                            "char_count": len(page_text),
                        },
                    )
                )
                raw_text_parts.append(page_text)

        pdf_meta = {
            "page_count": doc.page_count,
            "title": doc.metadata.get("title") or filename,
            "author": doc.metadata.get("author") or "Unknown",
        }
        doc.close()

        full_raw_text = "\n\n".join(raw_text_parts).strip()
        if not full_raw_text:
            raise DocumentProcessingError(
                f"PDF '{filename}' contains no extractable text. Scanned or image-only PDFs require OCR."
            )

        return ParsedDocument(
            filename=filename,
            file_type="pdf",
            file_size_bytes=file_size,
            content_hash=content_hash,
            sections=sections,
            raw_text=full_raw_text,
            metadata=pdf_meta,
        )

    @classmethod
    def _parse_text(
        cls, filename: str, file_bytes: bytes, ext: str, file_size: int, content_hash: str
    ) -> ParsedDocument:
        # Decode text with utf-8, fallback to latin-1
        text = None
        for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
            try:
                text = file_bytes.decode(encoding)
                break
            except UnicodeDecodeError:
                continue

        if text is None:
            raise DocumentProcessingError(f"Unable to decode text file '{filename}' with supported encodings.")

        text = text.strip()
        if not text:
            raise DocumentProcessingError(f"Text file '{filename}' contains no readable content.")

        clean_type = "md" if ext in {".md", ".markdown"} else "txt"
        sections: List[ParsedSection] = []

        if clean_type == "md":
            # Break by markdown headers if present
            lines = text.split("\n")
            current_header = "Introduction"
            current_lines: List[str] = []

            for line in lines:
                if line.startswith(("# ", "## ", "### ", "#### ")):
                    if current_lines:
                        sec_text = "\n".join(current_lines).strip()
                        if sec_text:
                            sections.append(
                                ParsedSection(
                                    content=sec_text,
                                    metadata={"header": current_header, "source_file": filename},
                                )
                            )
                        current_lines = []
                    current_header = line.lstrip("#").strip()
                current_lines.append(line)

            if current_lines:
                sec_text = "\n".join(current_lines).strip()
                if sec_text:
                    sections.append(
                        ParsedSection(
                            content=sec_text,
                            metadata={"header": current_header, "source_file": filename},
                        )
                    )
        else:
            sections.append(
                ParsedSection(
                    content=text,
                    metadata={"source_file": filename, "char_count": len(text)},
                )
            )

        return ParsedDocument(
            filename=filename,
            file_type=clean_type,
            file_size_bytes=file_size,
            content_hash=content_hash,
            sections=sections,
            raw_text=text,
            metadata={"format": clean_type},
        )

    @classmethod
    def _parse_csv(cls, filename: str, file_bytes: bytes, file_size: int, content_hash: str) -> ParsedDocument:
        try:
            df = pd.read_csv(io.BytesIO(file_bytes))
        except Exception as e:
            raise DocumentProcessingError(f"Malformed or unreadable CSV '{filename}': {str(e)}")

        if df.empty:
            raise DocumentProcessingError(f"CSV file '{filename}' contains no data rows.")

        columns = [str(c).strip() for c in df.columns]
        sections: List[ParsedSection] = []
        raw_rows: List[str] = []

        # Convert each row into semantically rich key-value representations
        for idx, row in df.iterrows():
            row_items = []
            for col in columns:
                val = row[col]
                if pd.notna(val):
                    row_items.append(f"{col}: {val}")
            if row_items:
                row_str = f"Row {idx + 1} | " + " | ".join(row_items)
                sections.append(
                    ParsedSection(
                        content=row_str,
                        metadata={
                            "row_number": idx + 1,
                            "source_file": filename,
                            "columns": columns,
                        },
                    )
                )
                raw_rows.append(row_str)

        full_raw_text = "\n".join(raw_rows).strip()
        if not full_raw_text:
            raise DocumentProcessingError(f"CSV file '{filename}' contains only empty or NaN values.")

        return ParsedDocument(
            filename=filename,
            file_type="csv",
            file_size_bytes=file_size,
            content_hash=content_hash,
            sections=sections,
            raw_text=full_raw_text,
            metadata={
                "row_count": len(df),
                "column_count": len(columns),
                "columns": columns,
            },
        )
