import pytest
from app.ingestion.chunker import TextChunker, Chunk
from app.ingestion.parser import ParsedSection


def test_chunker_short_text():
    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    text = "Short text under the chunk threshold."
    chunks = chunker.chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0].content == text
    assert chunks[0].token_count > 0
    assert chunks[0].chunk_index == 0


def test_chunker_recursive_splitting_and_overlap():
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    long_text = (
        "Paragraph one is introducing the core capabilities of the system. "
        "It explains the architectural decisions behind multi-provider routing.\n\n"
        "Paragraph two delves into vector database storage and full-text search indexing. "
        "It details how Reciprocal Rank Fusion generates authoritative citations.\n\n"
        "Paragraph three summarizes the evaluation methodology and metrics tracking."
    )
    chunks = chunker.chunk_text(long_text)
    assert len(chunks) > 1

    # Sequential index checks
    for i, c in enumerate(chunks):
        assert c.chunk_index == i
        assert len(c.content) <= 180  # Reasonable boundary bounds
        assert c.token_count > 0

    # Ensure overlap between consecutive chunks
    overlap_detected = False
    for i in range(len(chunks) - 1):
        words_current = set(chunks[i].content.split()[-4:])
        words_next = set(chunks[i + 1].content.split()[:4])
        if words_current.intersection(words_next):
            overlap_detected = True
            break
    assert overlap_detected, "Consecutive chunks should preserve contextual overlap."


def test_chunk_sections_preserves_metadata():
    chunker = TextChunker(chunk_size=200, chunk_overlap=30)
    sections = [
        ParsedSection(content="Page 1 Content: Executive overview and vision.", metadata={"page_number": 1}),
        ParsedSection(content="Page 2 Content: Financial projections and runway.", metadata={"page_number": 2}),
    ]
    chunks = chunker.chunk_sections(sections)
    assert len(chunks) >= 2
    assert chunks[0].metadata["page_number"] == 1
    assert chunks[-1].metadata["page_number"] == 2


def test_chunker_invalid_overlap():
    with pytest.raises(ValueError):
        TextChunker(chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError):
        TextChunker(chunk_size=100, chunk_overlap=120)
