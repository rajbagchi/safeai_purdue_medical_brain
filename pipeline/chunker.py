"""
Smart chunking: semantic chunks from extracted content with BM25 index.
"""

import json
import re
from typing import Dict, List, Optional, Any

from rank_bm25 import BM25Okapi

from .config import ExtractionConfig


class SmartChunker:
    """
    Creates semantic chunks from extracted content.
    Preserves document structure and medical context.
    """

    def __init__(self, extraction_result: Dict, config: ExtractionConfig):
        self.extraction = extraction_result
        self.config = config
        self.chunks: List[Dict] = []
        self.chunk_index: Dict[str, Any] = {}

    def chunk_by_headings(self) -> List[Dict]:
        """Create chunks based on document headings."""
        print("\n🧩 Creating semantic chunks...")

        for page in self.extraction.get("pages", []):
            sections = self._group_by_headings(page)

            for section in sections:
                chunk = self._build_chunk(section, page["page"])
                if chunk:
                    self.chunks.append(chunk)

        self._add_table_chunks()

        print(f"  Created {len(self.chunks)} semantic chunks")
        return self.chunks

    def _group_by_headings(self, page: Dict) -> List[Dict]:
        """Group page content under headings."""
        sections: List[Dict] = []
        current: Dict[str, Any] = {
            "heading": "Untitled",
            "level": 3,
            "content": [],
            "tables": [],
            "start_y": 0,
        }

        all_elements: List[tuple] = []

        for h in page.get("headings", []):
            all_elements.append(("heading", h))

        for b in page.get("text_blocks", []):
            all_elements.append(("text", b))

        all_elements.sort(key=lambda x: x[1].get("y_pos", 0))

        for elem_type, elem in all_elements:
            if elem_type == "heading":
                if current["content"] or current["tables"]:
                    sections.append(current.copy())

                current = {
                    "heading": elem["text"],
                    "level": elem.get("level", 3),
                    "content": [],
                    "tables": [],
                    "start_y": elem.get("y_pos", 0),
                }
            elif elem_type == "text":
                current["content"].append(elem["text"])

        if current["content"] or current["tables"]:
            sections.append(current)

        return sections

    def _build_chunk(
        self,
        section: Dict,
        page_num: int,
    ) -> Optional[Dict]:
        """Build a chunk from a section."""
        if not section["content"] and not section["tables"]:
            return None

        text_parts = []
        if section["heading"] != "Untitled":
            text_parts.append(
                f"{'#' * section['level']} {section['heading']}"
            )
        text_parts.extend(section["content"])

        section_tables = []
        for table in self.extraction.get("tables", []):
            if table.get("page") == page_num:
                section_tables.append(table)

        chunk_text = "\n\n".join(text_parts)

        if (
            len(chunk_text) < self.config.min_chunk_size
            and not section_tables
        ):
            return None

        return {
            "chunk_id": f"chunk_{len(self.chunks):06d}",
            "page": page_num,
            "heading": section["heading"],
            "level": section["level"],
            "text": chunk_text,
            "tables": section_tables,
            "has_tables": len(section_tables) > 0,
            "char_count": len(chunk_text),
            "word_count": len(chunk_text.split()),
        }

    def _add_table_chunks(self) -> None:
        """Add standalone chunks for important tables."""
        for table in self.extraction.get("tables", []):
            already_included = False
            for chunk in self.chunks:
                if table in chunk.get("tables", []):
                    already_included = True
                    break

            if not already_included and table.get("num_rows", 0) > 1:
                table_chunk = {
                    "chunk_id": f"chunk_table_{len(self.chunks):06d}",
                    "page": table.get("page", 0),
                    "heading": (
                        f"Table: {table.get('headers', ['Unknown'])[0] if table.get('headers') else 'Medical Table'}"
                    ),
                    "level": 3,
                    "text": f"## Dosing Table\n\n{table.get('markdown', '')}",
                    "tables": [table],
                    "has_tables": True,
                    "char_count": len(table.get("markdown", "")),
                    "word_count": len(table.get("markdown", "").split()),
                    "is_table_only": True,
                }
                self.chunks.append(table_chunk)

    def create_search_index(self) -> Dict:
        """Create BM25 search index from chunks."""
        print("\n🔍 Creating search index...")

        tokenized_chunks: List[List[str]] = []
        for chunk in self.chunks:
            text = chunk["text"]
            tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
            tokens = [t for t in tokens if len(t) > 1]
            tokenized_chunks.append(tokens)

            if chunk.get("tables"):
                for table in chunk["tables"]:
                    table_text = json.dumps(table.get("data", "")).lower()
                    table_tokens = re.findall(r"[a-zA-Z0-9]+", table_text)
                    table_tokens = [t for t in table_tokens if len(t) > 1]
                    tokenized_chunks[-1].extend(table_tokens)

        bm25 = BM25Okapi(tokenized_chunks)

        self.chunk_index = {
            "bm25": bm25,
            "chunks": self.chunks,
            "tokenized": tokenized_chunks,
        }

        return self.chunk_index
