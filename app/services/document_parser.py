"""Document parser service extracting text and generating semantic chunks from PDF, TXT, MD, CSV, and JSON."""

import io
import re
import csv
import json
import logging
from typing import Any, Dict, List, Tuple
from pathlib import Path

logger = logging.getLogger("ai_chat_api.document_parser")

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".csv", ".json"}


class DocumentParserService:
    """Service to parse multi-format documents and split text into overlapping RAG chunks."""

    @staticmethod
    def is_supported(filename: str) -> bool:
        """Check if filename has a supported extension."""
        ext = Path(filename).suffix.lower()
        return ext in SUPPORTED_EXTENSIONS

    @classmethod
    def extract_text(cls, file_bytes: bytes, filename: str) -> Tuple[str, Dict[str, Any]]:
        """Extract clean text and document metadata from file bytes based on file extension."""
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type '{ext}'. Supported types are: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")

        metadata: Dict[str, Any] = {
            "filename": filename,
            "extension": ext,
            "file_size_bytes": len(file_bytes),
        }

        if ext == ".pdf":
            text, pdf_meta = cls._extract_pdf(file_bytes)
            metadata.update(pdf_meta)
            return text, metadata

        elif ext in {".txt", ".md"}:
            text = cls._decode_bytes(file_bytes)
            metadata["file_type"] = "markdown" if ext == ".md" else "text"
            return text, metadata

        elif ext == ".csv":
            text, csv_meta = cls._extract_csv(file_bytes)
            metadata.update(csv_meta)
            return text, metadata

        elif ext == ".json":
            text, json_meta = cls._extract_json(file_bytes)
            metadata.update(json_meta)
            return text, metadata

        raise ValueError(f"Unhandled file extension: {ext}")

    @staticmethod
    def _decode_bytes(file_bytes: bytes) -> str:
        """Decode raw bytes into string, trying UTF-8 then latin-1."""
        try:
            return file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return file_bytes.decode("latin-1", errors="replace")

    @classmethod
    def _extract_pdf(cls, file_bytes: bytes) -> Tuple[str, Dict[str, Any]]:
        """Extract text from PDF pages using pypdf."""
        import pypdf

        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        num_pages = len(reader.pages)
        page_texts = []

        for idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            page_text = page_text.strip()
            if page_text:
                page_texts.append(f"[Page {idx + 1}]\n{page_text}")

        full_text = "\n\n".join(page_texts).strip()
        metadata = {
            "file_type": "pdf",
            "page_count": num_pages,
            "has_text": bool(full_text),
        }
        return full_text, metadata

    @classmethod
    def _extract_csv(cls, file_bytes: bytes) -> Tuple[str, Dict[str, Any]]:
        """Parse CSV rows into readable natural language records."""
        raw_text = cls._decode_bytes(file_bytes)
        stream = io.StringIO(raw_text.strip())
        reader = csv.reader(stream)

        rows = list(reader)
        if not rows:
            return "", {"file_type": "csv", "row_count": 0}

        header = rows[0]
        records = []
        for r_idx, row in enumerate(rows[1:], start=1):
            if not any(cell.strip() for cell in row):
                continue
            fields = []
            for col_idx, cell in enumerate(row):
                col_name = header[col_idx].strip() if col_idx < len(header) else f"Column_{col_idx+1}"
                fields.append(f"{col_name}: {cell.strip()}")
            records.append(f"Record {r_idx} -> " + ", ".join(fields))

        csv_text = "\n".join(records)
        metadata = {
            "file_type": "csv",
            "row_count": len(rows) - 1,
            "columns": header,
        }
        return csv_text, metadata

    @classmethod
    def _extract_json(cls, file_bytes: bytes) -> Tuple[str, Dict[str, Any]]:
        """Parse JSON text into formatted string."""
        raw_text = cls._decode_bytes(file_bytes)
        data = json.loads(raw_text)
        formatted = json.dumps(data, indent=2)
        metadata = {
            "file_type": "json",
            "is_array": isinstance(data, list),
        }
        return formatted, metadata

    @staticmethod
    def chunk_text(
        text: str,
        chunk_size: int = 600,
        overlap: int = 80,
    ) -> List[str]:
        """Split text into overlapping semantic chunks respecting paragraph and sentence boundaries."""
        if not text or not text.strip():
            return []

        clean_text = re.sub(r"\r\n|\r", "\n", text).strip()

        # If text is smaller than chunk_size, return it as a single chunk
        if len(clean_text) <= chunk_size:
            return [clean_text]

        # Split into paragraphs first
        paragraphs = [p.strip() for p in clean_text.split("\n\n") if p.strip()]
        chunks: List[str] = []
        current_chunk = ""

        for para in paragraphs:
            # If paragraph itself is too large, split by sentences
            if len(para) > chunk_size:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                        current_chunk = f"{current_chunk} {sentence}".strip()
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                            # Apply overlap from the tail of current_chunk
                            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                            current_chunk = f"{overlap_text} {sentence}".strip()
                        else:
                            # Sentence is alone larger than chunk_size, force slice
                            for i in range(0, len(sentence), chunk_size - overlap):
                                chunks.append(sentence[i : i + chunk_size].strip())
            else:
                if len(current_chunk) + len(para) + 2 <= chunk_size:
                    current_chunk = f"{current_chunk}\n\n{para}".strip()
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                        overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                        current_chunk = f"{overlap_text}\n\n{para}".strip()
                    else:
                        current_chunk = para

        if current_chunk and current_chunk not in chunks:
            chunks.append(current_chunk)

        # Remove duplicate or empty chunks
        filtered_chunks = [c.strip() for c in chunks if len(c.strip()) > 10]
        return filtered_chunks or [clean_text[:chunk_size]]
