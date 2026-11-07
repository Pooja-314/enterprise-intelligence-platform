import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from app.config import settings
from app.ingestion.parser import ParsedSection


@dataclass
class Chunk:
    chunk_index: int
    content: str
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class TextChunker:
    """Recursive semantic text chunker with overlap and token budgeting."""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size.")

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimates token count using a calibrated character/word heuristic (avg ~4 chars/token)."""
        if not text:
            return 0
        words = len(text.split())
        chars = len(text)
        return max(1, int((words * 1.3 + chars / 4.0) / 2))

    def chunk_sections(self, sections: List[ParsedSection]) -> List[Chunk]:
        """Chunks a list of parsed sections while preserving sectional metadata (page/header/row)."""
        all_chunks: List[Chunk] = []
        global_index = 0

        for section in sections:
            section_chunks = self.chunk_text(
                text=section.content,
                base_metadata=section.metadata,
                start_index=global_index,
            )
            all_chunks.extend(section_chunks)
            global_index += len(section_chunks)

        return all_chunks

    def chunk_text(
        self,
        text: str,
        base_metadata: Optional[Dict[str, Any]] = None,
        start_index: int = 0,
    ) -> List[Chunk]:
        """Splits a single text into overlapping chunks respecting natural semantic boundaries."""
        text = text.strip()
        if not text:
            return []

        # If text is already smaller than chunk size, return single chunk
        if len(text) <= self.chunk_size:
            return [
                Chunk(
                    chunk_index=start_index,
                    content=text,
                    token_count=self.estimate_tokens(text),
                    metadata=dict(base_metadata or {}),
                )
            ]

        # Recursive splitting strategy: \n\n -> \n -> sentence -> space
        raw_splits = self._split_text(text, self.chunk_size)
        merged_chunks = self._merge_splits(raw_splits, self.chunk_size, self.chunk_overlap)

        chunks: List[Chunk] = []
        for i, chunk_text in enumerate(merged_chunks):
            chunk_meta = dict(base_metadata or {})
            chunk_meta["char_length"] = len(chunk_text)
            chunks.append(
                Chunk(
                    chunk_index=start_index + i,
                    content=chunk_text,
                    token_count=self.estimate_tokens(chunk_text),
                    metadata=chunk_meta,
                )
            )

        return chunks

    def _split_text(self, text: str, max_chars: int) -> List[str]:
        """Splits text on natural boundaries: paragraphs -> lines -> sentences -> words."""
        # Split on double newlines
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        result: List[str] = []

        for p in paragraphs:
            if len(p) <= max_chars:
                result.append(p)
            else:
                # Split paragraph on single newlines
                lines = [line.strip() for line in p.split("\n") if line.strip()]
                for line in lines:
                    if len(line) <= max_chars:
                        result.append(line)
                    else:
                        # Split line on sentence boundaries (. ! ?)
                        sentences = re.split(r"(?<=[.!?])\s+", line)
                        for sent in sentences:
                            sent = sent.strip()
                            if not sent:
                                continue
                            if len(sent) <= max_chars:
                                result.append(sent)
                            else:
                                # Hard split on word boundaries
                                words = sent.split(" ")
                                curr_piece = []
                                curr_len = 0
                                for w in words:
                                    if curr_len + len(w) + 1 > max_chars:
                                        if curr_piece:
                                            result.append(" ".join(curr_piece))
                                        curr_piece = [w]
                                        curr_len = len(w)
                                    else:
                                        curr_piece.append(w)
                                        curr_len += len(w) + 1
                                if curr_piece:
                                    result.append(" ".join(curr_piece))
        return result

    def _merge_splits(self, splits: List[str], max_chars: int, overlap_chars: int) -> List[str]:
        """Merges small text splits into chunks of up to max_chars with overlap."""
        if not splits:
            return []

        chunks: List[str] = []
        current_chunk: List[str] = []
        current_len = 0

        for split in splits:
            split_len = len(split)
            if current_chunk and (current_len + split_len + 1 > max_chars):
                joined = " ".join(current_chunk)
                chunks.append(joined)

                # Keep overlap from the end of the current chunk
                overlap_splits: List[str] = []
                overlap_acc = 0
                for prev in reversed(current_chunk):
                    if overlap_acc + len(prev) <= overlap_chars:
                        overlap_splits.insert(0, prev)
                        overlap_acc += len(prev) + 1
                    else:
                        break

                current_chunk = overlap_splits + [split]
                current_len = sum(len(s) for s in current_chunk) + len(current_chunk) - 1
            else:
                current_chunk.append(split)
                current_len += split_len + (1 if len(current_chunk) > 1 else 0)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
