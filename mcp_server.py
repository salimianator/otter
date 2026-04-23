"""
mcp_server.py — OTTER MCP server

Exposes three tools to Claude Code / Claude.ai:
  • otter_compress          — compress text; pass compress=False to skip (baseline mode)
  • otter_compare           — side-by-side original vs compressed + token stats
  • otter_compress_document — Docling → OTTER for PDF / DOCX / PPTX / URL inputs
"""

from __future__ import annotations
import sys
from pathlib import Path

# Make otter/src/ and otter/ importable regardless of working directory.
_root = Path(__file__).parent
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root))

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "otter",
    instructions=(
        "OTTER compresses long text before it reaches the LLM, reducing token usage "
        "while preserving task-relevant content. "
        "Call otter_compare to see the compression effect on any passage. "
        "Call otter_compress with compress=False to get the unmodified baseline text "
        "and compare token counts."
    ),
)

# ── Lazy singletons ──────────────────────────────────────────────────────────

_compressor = None
_loader = None


def _get_compressor():
    global _compressor
    if _compressor is None:
        from compress import OTTERCompressor
        _compressor = OTTERCompressor()
    return _compressor


def _get_loader():
    global _loader
    if _loader is None:
        from loader import DocumentLoader
        _loader = DocumentLoader()
    return _loader


# ── Token counting ───────────────────────────────────────────────────────────

def _count_tokens(text: str) -> int:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


# ── Tools ────────────────────────────────────────────────────────────────────

@mcp.tool()
def otter_compress(text: str, query: str, compress: bool = True) -> dict:
    """
    Compress a passage with OTTER before using it in a response.

    Parameters
    ----------
    text : str
        The document or passage to compress.
    query : str
        The question or task the text is meant to answer.
    compress : bool
        Set False to return the original text unchanged — use this to get the
        uncompressed baseline so you can compare responses and token counts.

    Returns
    -------
    dict
        text               — the (possibly compressed) text to use
        compressed         — whether compression was applied
        original_tokens    — token count before compression
        compressed_tokens  — token count after compression
        reduction_pct      — percentage of tokens removed
        kept_sentences     — sentences kept (null when compress=False)
        original_sentences — total sentences (null when compress=False)
        query_class        — dominant query class detected by OTTER
    """
    original_tokens = _count_tokens(text)

    if not compress:
        return {
            "text": text,
            "compressed": False,
            "original_tokens": original_tokens,
            "compressed_tokens": original_tokens,
            "reduction_pct": 0.0,
            "kept_sentences": None,
            "original_sentences": None,
            "query_class": None,
        }

    result = _get_compressor().compress(text, query)
    compressed_tokens = _count_tokens(result["compressed"])
    reduction_pct = (
        round((1 - compressed_tokens / original_tokens) * 100, 1)
        if original_tokens else 0.0
    )

    return {
        "text": result["compressed"],
        "compressed": True,
        "original_tokens": original_tokens,
        "compressed_tokens": compressed_tokens,
        "reduction_pct": reduction_pct,
        "kept_sentences": result["kept_sentences"],
        "original_sentences": result["original_sentences"],
        "query_class": result.get("selection_method"),
    }


@mcp.tool()
def otter_compare(text: str, query: str) -> dict:
    """
    Run OTTER and return both the original and compressed text side-by-side
    with token counts. Use this when you want to show — in a single call —
    what changes between the compressed and uncompressed versions.

    Parameters
    ----------
    text : str
        The document or passage to compress.
    query : str
        The question or task the text is meant to answer.

    Returns
    -------
    dict
        original_text      — the full original text
        compressed_text    — the OTTER-compressed text
        original_tokens    — token count of the original
        compressed_tokens  — token count after compression
        reduction_pct      — percentage of tokens removed
        kept_sentences     — sentences kept
        original_sentences — total sentences
        query_class        — dominant query class detected by OTTER
    """
    original_tokens = _count_tokens(text)
    result = _get_compressor().compress(text, query)
    compressed_tokens = _count_tokens(result["compressed"])
    reduction_pct = (
        round((1 - compressed_tokens / original_tokens) * 100, 1)
        if original_tokens else 0.0
    )

    return {
        "original_text": text,
        "compressed_text": result["compressed"],
        "original_tokens": original_tokens,
        "compressed_tokens": compressed_tokens,
        "reduction_pct": reduction_pct,
        "kept_sentences": result["kept_sentences"],
        "original_sentences": result["original_sentences"],
        "query_class": result.get("selection_method"),
    }


@mcp.tool()
def otter_compress_document(source: str, query: str, compress: bool = True) -> dict:
    """
    Load a document via Docling (PDF, DOCX, PPTX, HTML, or URL), then
    optionally compress it with OTTER.

    Parameters
    ----------
    source : str
        Absolute file path or URL.
    query : str
        The question or task the document is meant to answer.
    compress : bool
        Set False to extract the text via Docling without OTTER compression.

    Returns
    -------
    dict
        Same structure as otter_compress, plus an additional key:
        extracted_chars — character count of the raw extracted text.
    """
    text = _get_loader().load(source)
    result = otter_compress(text, query, compress=compress)
    result["extracted_chars"] = len(text)
    return result


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
