"""
compress.py — OTTER context compressor

Wires the four OTTER components (SentenceSegmenter, SentenceEncoder,
QueryClassifier, QueryPlanner) into a single clean interface.

Usage — one-off:
    from compress import compress
    result = compress(document, query)

Usage — repeated calls (models loaded once):
    from compress import OTTERCompressor
    compressor = OTTERCompressor()
    for doc, q in pairs:
        result = compressor.compress(doc, q)
"""

from __future__ import annotations

from segmenter  import SentenceSegmenter
from encoder    import SentenceEncoder
from classifier import QueryClassifier
from planner    import QueryPlanner


class OTTERCompressor:
    """
    Full OTTER compression pipeline in a single reusable object.

    Initialise once, call compress() many times — models are loaded
    only on first use and cached for subsequent calls.

    Parameters
    ----------
    anchor_sentences : int
        Sentences force-kept at each end of the document (default: 2).
    flow_window : int
        Neighbourhood radius for flow bonus propagation (default: 2).
    flow_decay : float
        Flow bonus at distance-1; distance-2 gets flow_decay*0.5 (default: 0.6).
    min_words : int
        Minimum words per segment before merging (default: 6).
    encoder_model : str
        HuggingFace sentence-transformer model ID.
    """

    def __init__(
        self,
        anchor_sentences: int = 2,
        flow_window: int = 2,
        flow_decay: float = 0.6,
        min_words: int = 6,
        encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.segmenter  = SentenceSegmenter(min_words=min_words)
        self.encoder    = SentenceEncoder(model_name=encoder_model)
        self.classifier = QueryClassifier(encoder=self.encoder)   # shared encoder
        self.planner    = QueryPlanner(
            anchor_sentences=anchor_sentences,
            flow_window=flow_window,
            flow_decay=flow_decay,
        )

    def compress(
        self,
        document: str,
        query:    str,
    ) -> dict:
        """
        Run the full OTTER pipeline on *document* given *query*.

        Parameters
        ----------
        document : str
            Raw long-context document text.
        query : str
            Query string; may be multi-part (sub-questions separated by
            sentence boundaries are handled automatically).

        Returns
        -------
        dict with keys:
            compressed          – str, final compressed text
            original_sentences  – int
            kept_sentences      – int
            compression_ratio   – float (kept / original)
            token_reduction_pct – float ((1 - ratio) * 100)
            weights             – dict from QueryClassifier
            scores              – dict {anchor, flow, flash, combined} arrays
            selection_method    – str, 'kneedle' or 'marginal_return'
        """
        # Guard: empty query (e.g. multi_news has no input field) — substitute a
        # generic abstractive prompt so Flash scoring gets a meaningful query
        # vector rather than near-zero cosine similarities across all sentences.
        if not query or not query.strip():
            query = "Summarize the main points of this document."

        # (a) Segment
        sentences = self.segmenter.segment(document)
        if not sentences:
            return {
                "compressed":          "",
                "original_sentences":  0,
                "kept_sentences":      0,
                "compression_ratio":   0.0,
                "token_reduction_pct": 0.0,
                "weights":             {},
                "scores":              {},
            }

        # (b) Encode
        embeddings = self.encoder.encode(sentences)

        # (c) Classify
        weights = self.classifier.get_weights(query)

        # (d) Score
        score_dict = self.planner.score(
            embeddings=embeddings,
            query=query,
            weights=weights,
        )

        # (e) Select — continuously interpolated via three-class softmax weights
        kept_sentences = self.planner.select(
            sentences=sentences,
            scores=score_dict["combined"],
            weights=weights,
        )

        # (f) Assemble
        compressed_text = " ".join(kept_sentences)

        # (g) Return
        return {
            "compressed":          compressed_text,
            "original_sentences":  len(sentences),
            "kept_sentences":      len(kept_sentences),
            "compression_ratio":   len(kept_sentences) / len(sentences),
            "token_reduction_pct": (1 - len(kept_sentences) / len(sentences)) * 100,
            "weights":             weights,
            "scores":              score_dict,
            "selection_method":    weights["dominant"],
        }


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def compress(
    document:   str,
    query:      str,
    compressor: OTTERCompressor | None = None,
) -> dict:
    """
    Compress *document* given *query* using the OTTER pipeline.

    Parameters
    ----------
    document : str
        Raw long-context document text.
    query : str
        Query string (single or multi-part).
    compressor : OTTERCompressor, optional
        Pre-initialised compressor to reuse across calls.
        If None, a fresh OTTERCompressor is created (models will reload).

    Returns
    -------
    dict
        Same structure as OTTERCompressor.compress().
    """
    if compressor is None:
        compressor = OTTERCompressor()
    return compressor.compress(document, query)


# ---------------------------------------------------------------------------
# Smoke test — three qasper examples end-to-end
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))

    DATA_PATH = Path(__file__).parent.parent / "data" / "longbench_qasper.jsonl"

    print("Initialising OTTERCompressor …")
    compressor = OTTERCompressor()
    print("Ready.\n")

    with open(DATA_PATH, "r", encoding="utf-8") as fh:
        records = [json.loads(fh.readline()) for _ in range(3)]

    for ex_idx, record in enumerate(records, 1):
        context = record["context"]
        query   = record["input"]

        result  = compressor.compress(context, query)
        w       = result["weights"]
        kept    = result["kept_sentences"]
        total   = result["original_sentences"]
        ratio   = result["compression_ratio"]
        reduc   = result["token_reduction_pct"]
        ext     = w.get("extractive_score", 0)

        q_display = (query[:77] + "…") if len(query) > 80 else query

        # First 3 sentences of compressed output
        compressed_sents = result["compressed"].split(". ")[:3]

        print("=" * 70)
        print(f"Example {ex_idx}")
        print(f"  Query            : {q_display}")
        print(f"  Original sents   : {total}")
        print(f"  Kept sents       : {kept}")
        print(f"  Compression ratio: {ratio:.2%}  ({kept}/{total} kept)")
        print(f"  Token reduction  : {reduc:.1f}%")
        print(f"  Extractive score : {ext:.4f}")
        print(f"  First 3 compressed sentences:")
        for i, s in enumerate(compressed_sents[:3]):
            snippet = (s[:110] + "…") if len(s) > 113 else s
            print(f"    [{i}] {snippet}")
    print("=" * 70)
