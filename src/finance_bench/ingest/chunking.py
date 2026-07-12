from __future__ import annotations

import re
import uuid
from typing import Dict, List

from finance_bench.types.schemas import DocumentChunk


# SEC / financial filing aware section boundaries
SECTION_RE = re.compile(
    r"""
    (
        \n\s*ITEM\s+\d+[A-Z]?\.*.*?\n
        |
        \n[A-Z][A-Z\s,&\-]{5,}\n
        |
        \n\d+\.\s+[A-Z].*?\n
        |
        \n\s*MANAGEMENT'S\s+DISCUSSION.*?\n
        |
        \n\s*RISK\s+FACTORS.*?\n
    )
    """,
    re.VERBOSE,
)


class TextChunker:
    """
    Production-grade retrieval chunker.

    Features:
    - deterministic chunk IDs
    - normalized preprocessing
    - section-aware chunking
    - overlap tracking
    - retrieval reproducibility
    - metadata propagation
    """

    def __init__(
        self,
        chunk_size: int = 350,
        chunk_overlap: int = 50,
        chunking_version: str = "v1",
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_version = chunking_version

    # PUBLIC

    def chunk_document(
        self,
        document_id: str,
        text: str,
        metadata: Dict | None = None,
    ) -> List[DocumentChunk]:

        metadata = metadata or {}

        text = self._normalize_text(text)

        sections = self._split_sections(text)

        chunks: List[DocumentChunk] = []

        global_chunk_index = 0

        for section_index, section in enumerate(sections):

            section_chunks = self._chunk_section(
                document_id=document_id,
                text=section,
                metadata=metadata,
                section_index=section_index,
                start_chunk_index=global_chunk_index,
            )

            chunks.extend(section_chunks)

            global_chunk_index += len(section_chunks)

        return chunks

    # internal

    def _normalize_text(
        self,
        text: str,
    ) -> str:

        if not isinstance(text, str):
            text = str(text)

        text = text.replace("\x00", " ")

        # normalize whitespace
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _split_sections(
        self,
        text: str,
    ) -> List[str]:

        splits = SECTION_RE.split(text)

        sections = []

        current = ""

        for piece in splits:

            piece = piece.strip()

            if not piece:
                continue

            # preserve headers with content
            if SECTION_RE.match(f"\n{piece}\n"):

                if current:
                    sections.append(current.strip())

                current = piece

            else:
                current += " " + piece

        if current:
            sections.append(current.strip())

        # fallback
        if not sections:
            sections = [text]

        return sections

    def _chunk_section(
        self,
        document_id: str,
        text: str,
        metadata: Dict,
        section_index: int,
        start_chunk_index: int,
    ) -> List[DocumentChunk]:

        words = text.split()

        if not words:
            return []

        chunks = []

        step = self.chunk_size - self.chunk_overlap

        for start_word in range(0, len(words), step):

            end_word = min(
                start_word + self.chunk_size,
                len(words),
            )

            chunk_words = words[start_word:end_word]

            chunk_text = " ".join(chunk_words)

            chunk_index = (
                start_chunk_index + len(chunks)
            )

            chunk_id = self._stable_chunk_id(
                document_id=document_id,
                section_index=section_index,
                start_word=start_word,
                end_word=end_word,
            )

            chunk_metadata = {
                **metadata,
                "chunk_index": chunk_index,
                "section_index": section_index,
                "start_word": start_word,
                "end_word": end_word,
                "chunk_size": len(chunk_words),
                "chunk_overlap": self.chunk_overlap,
                "chunking_version": self.chunking_version,
            }

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    text=chunk_text,
                    metadata=chunk_metadata,
                )
            )

            if end_word >= len(words):
                break

        return chunks

    @staticmethod
    def _stable_chunk_id(
        document_id: str,
        section_index: int,
        start_word: int,
        end_word: int,
    ) -> str:

        content = (
            f"{document_id}|"
            f"{section_index}|"
            f"{start_word}|"
            f"{end_word}"
        )

        return str(
            uuid.uuid5(
                uuid.NAMESPACE_DNS,
                content,
            )
        )