"""
loader.py — Docling-based document loader for OTTER

Converts PDFs, DOCX, PPTX, HTML, and URLs to clean markdown text
before passing to OTTERCompressor. Intentionally lives outside src/
so OTTER's core stays format-agnostic.

Usage:
    from loader import DocumentLoader
    loader = DocumentLoader()
    text = loader.load("paper.pdf")          # file path
    text = loader.load(pdf_bytes, "doc.pdf") # raw bytes
    text = loader.load("https://...")        # URL
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm", ".md"}


class DocumentLoader:
    """
    Thin wrapper around Docling's DocumentConverter.
    Lazy-loads the converter on first use — same pattern as SentenceEncoder.
    """

    def __init__(self) -> None:
        self._converter = None

    @property
    def converter(self):
        if self._converter is None:
            from docling.document_converter import DocumentConverter
            self._converter = DocumentConverter()
        return self._converter

    def load(self, source: str | Path | bytes, filename: str = "document.pdf") -> str:
        """
        Convert *source* to clean markdown text.

        Parameters
        ----------
        source : str | Path | bytes
            File path, URL string, or raw file bytes.
        filename : str
            Original filename — used when source is bytes to hint the format.

        Returns
        -------
        str
            Markdown-formatted plain text suitable for OTTER compression.
        """
        from docling.datamodel.base_models import DocumentStream

        if isinstance(source, (bytes, bytearray)):
            stream = DocumentStream(name=filename, stream=BytesIO(source))
            result = self.converter.convert(stream)
        else:
            result = self.converter.convert(str(source))

        return result.document.export_to_markdown()

    @staticmethod
    def needs_extraction(source: str | Path | bytes, filename: str = "") -> bool:
        """
        Return True if source should go through Docling rather than
        being passed directly to OTTER as plain text.
        """
        if isinstance(source, (bytes, bytearray)):
            return True
        s = str(source)
        if s.startswith("http://") or s.startswith("https://"):
            return True
        check = filename or s
        return Path(check).suffix.lower() in SUPPORTED_EXTENSIONS
