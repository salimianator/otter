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

Multi-document usage:
    result = compressor.compress(document="", query=q,
                                 documents=["doc1 text", "doc2 text"])
"""

from __future__ import annotations

import numpy as np

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
    cross_doc_weight : float
        Base cross-doc correlation weight (extractive end, default: 0.3).
    cross_doc_weight_abs : float
        Cross-doc weight ceiling for purely abstractive queries (default: 0.6).
    per_doc_cap_ratio : float
        Max fraction of kept-sentence budget any single document may occupy
        (default: 0.6).  Excess sentences evicted lowest-score-first.
    cross_doc_uniqueness_blend : float
        0 = pure shared-content signal, 1 = pure uniqueness signal (default: 0.5).
    multi_doc_min_coverage : float
        Minimum fraction of sentences to keep for no-query multi-doc inputs
        (default: 0.60).  Analogous to the single-doc substituted-query floor.
    """

    def __init__(
        self,
        anchor_sentences:         int   = 2,
        flow_window:              int   = 2,
        flow_decay:               float = 0.6,
        min_words:                int   = 6,
        encoder_model:            str   = "sentence-transformers/all-MiniLM-L6-v2",
        cross_doc_weight:         float = 0.3,
        cross_doc_weight_abs:     float = 0.6,
        per_doc_cap_ratio:        float = 0.6,
        cross_doc_uniqueness_blend: float = 0.5,
        multi_doc_min_coverage:   float = 0.60,
    ) -> None:
        self.multi_doc_min_coverage = multi_doc_min_coverage
        self.segmenter  = SentenceSegmenter(min_words=min_words)
        self.encoder    = SentenceEncoder(model_name=encoder_model)
        self.classifier = QueryClassifier(encoder=self.encoder)   # shared encoder
        self.planner    = QueryPlanner(
            anchor_sentences=anchor_sentences,
            flow_window=flow_window,
            flow_decay=flow_decay,
            cross_doc_weight=cross_doc_weight,
            cross_doc_weight_abs=cross_doc_weight_abs,
            per_doc_cap_ratio=per_doc_cap_ratio,
            cross_doc_uniqueness_blend=cross_doc_uniqueness_blend,
        )

    def compress(
        self,
        document:  str,
        query:     str,
        documents: list[str] | None = None,   # multi-doc: overrides document when len > 1
    ) -> dict:
        """
        Run the full OTTER pipeline on *document* (or *documents*) given *query*.

        Single-document path is identical to the original behaviour when
        ``documents`` is None or contains only one entry.

        Multi-document path (``documents`` with len > 1):
          - Segments each document separately and concatenates sentences.
          - Computes ``doc_boundaries`` (exclusive end index per document).
          - When ``query`` is empty, injects a synthetic centroid anchor
            (L2-normalised mean of all sentence embeddings) as the Flash
            query vector instead of a text substitution.
          - Applies cross-document correlation scoring in the planner.
          - Enforces a per-document coverage floor in selection.

        Parameters
        ----------
        document : str
            Raw long-context document text (used when documents is None/single).
        query : str
            Query string; may be multi-part.
        documents : list[str] or None
            Optional list of raw document strings for multi-document mode.
            When provided and len > 1, ``document`` is ignored.

        Returns
        -------
        dict with keys:
            compressed              – str, final compressed text
            original_sentences      – int
            kept_sentences          – int
            compression_ratio       – float (kept / original)
            token_reduction_pct     – float ((1 - ratio) * 100)
            weights                 – dict from QueryClassifier
            scores                  – dict {anchor, flow, flash, cross_doc, combined}
            selection_method        – str, dominant query class
            query_was_substituted   – bool
            cross_doc_weight        – float or None (None for single-doc)
            used_synthetic_anchor   – bool
            per_doc_floor_applied   – bool
            added_for_floor         – int
        """
        # ── Determine single-doc vs multi-doc ────────────────────────────────
        if documents is not None and len(documents) > 1:
            n_docs = len(documents)
            all_sentences: list[str] = []
            doc_boundaries: list[int] = []
            for doc in documents:
                doc_sents = self.segmenter.segment(doc)
                all_sentences.extend(doc_sents)
                doc_boundaries.append(len(all_sentences))
            sentences = all_sentences
            is_multi_doc = True
        else:
            n_docs = 1
            doc_boundaries = None
            is_multi_doc = False
            # Fall back to the single document argument
            src_doc  = documents[0] if (documents and len(documents) == 1) else document
            sentences = self.segmenter.segment(src_doc)

        # Guard: empty query handling
        SUBSTITUTED_QUERY     = "Summarize the main points of this document."
        query_was_substituted = False

        if not query or not query.strip():
            if not is_multi_doc:
                # Single-doc: substitute generic summary query (existing behaviour)
                query = SUBSTITUTED_QUERY
                query_was_substituted = True
            # Multi-doc with empty query: synthetic anchor used instead (handled below)

        if not sentences:
            return {
                "compressed":            "",
                "original_sentences":    0,
                "kept_sentences":        0,
                "compression_ratio":     0.0,
                "token_reduction_pct":   0.0,
                "weights":               {},
                "scores":                {},
                "query_was_substituted": query_was_substituted,
                "cross_doc_weight":      None,
                "used_synthetic_anchor": False,
                "per_doc_floor_applied": False,
                "added_for_floor":       0,
            }

        # (b) Encode all sentences
        embeddings = self.encoder.encode(sentences)

        # (c) Classify query
        # For empty-query multi-doc use a neutral summarisation query for the
        # classifier so weights are abstractive-leaning; the actual Flash
        # scoring will use the synthetic anchor below.
        classify_query = query if query.strip() else SUBSTITUTED_QUERY
        weights = self.classifier.get_weights(classify_query)

        # (d) Synthetic centroid anchor — Change 3
        # When query is empty AND we have multiple documents, compute the
        # L2-normalised mean of all sentence embeddings as a "query" vector.
        # This replaces the null/zero anchor and pulls Flash scores toward
        # the most representative sentences across all documents.
        anchor_embedding: np.ndarray | None = None
        if is_multi_doc and (not query or not query.strip()):
            centroid      = embeddings.mean(axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm > 0:
                anchor_embedding = (centroid / centroid_norm).astype(np.float32)

        # (e) Score
        score_dict = self.planner.score(
            embeddings=embeddings,
            query=query,
            weights=weights,
            doc_boundaries=doc_boundaries,
            anchor_embedding=anchor_embedding,
        )

        # (f) Select
        kept_sentences, select_meta = self.planner.select(
            sentences=sentences,
            scores=score_dict["combined"],
            weights=weights,
            doc_boundaries=doc_boundaries,
        )

        # (g) Single-doc coverage floor for substituted queries (existing behaviour)
        if query_was_substituted and not is_multi_doc:
            min_coverage = int(len(sentences) * 0.55)
            if len(kept_sentences) < min_coverage:
                ranked_indices = list(np.argsort(score_dict["combined"])[::-1])
                top_indices    = sorted(ranked_indices[:min_coverage])
                kept_sentences = [sentences[i] for i in top_indices]

        # (g2) Multi-doc no-query coverage floor
        # Without a real query, compression is guessing — keep at least
        # multi_doc_min_coverage of all sentences so Qwen has enough
        # material to produce a faithful summary across all articles.
        multi_doc_floor_applied = False
        if is_multi_doc and not query.strip():
            min_coverage = int(len(sentences) * self.multi_doc_min_coverage)
            if len(kept_sentences) < min_coverage:
                ranked_indices = list(np.argsort(score_dict["combined"])[::-1])
                top_indices    = sorted(ranked_indices[:min_coverage])
                kept_sentences = [sentences[i] for i in top_indices]
                multi_doc_floor_applied = True

        # (h) Assemble
        compressed_text = " ".join(kept_sentences)

        # (i) Return
        return {
            "compressed":            compressed_text,
            "original_sentences":    len(sentences),
            "kept_sentences":        len(kept_sentences),
            "compression_ratio":     len(kept_sentences) / len(sentences),
            "token_reduction_pct":   (1 - len(kept_sentences) / len(sentences)) * 100,
            "weights":               weights,
            "scores":                score_dict,
            "selection_method":      weights["dominant"],
            "query_was_substituted": query_was_substituted,
            # Multi-doc metadata (None / False / 0 for single-doc)
            "cross_doc_weight":        score_dict["cross_doc_weight"] if is_multi_doc else None,
            "used_synthetic_anchor":   score_dict["used_synthetic_anchor"],
            "per_doc_cap_applied":     select_meta["per_doc_cap_applied"],
            "removed_for_cap":         select_meta["removed_for_cap"],
            "multi_doc_floor_applied": multi_doc_floor_applied,
        }


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def compress(
    document:   str,
    query:      str,
    compressor: OTTERCompressor | None = None,
    documents:  list[str] | None       = None,
) -> dict:
    """
    Compress *document* (or *documents*) given *query* using the OTTER pipeline.

    Parameters
    ----------
    document : str
        Raw long-context document text.
    query : str
        Query string (single or multi-part).
    compressor : OTTERCompressor, optional
        Pre-initialised compressor to reuse across calls.
        If None, a fresh OTTERCompressor is created (models will reload).
    documents : list[str], optional
        Multi-document mode: list of raw document strings.

    Returns
    -------
    dict
        Same structure as OTTERCompressor.compress().
    """
    if compressor is None:
        compressor = OTTERCompressor()
    return compressor.compress(document, query, documents=documents)


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
        print(f"  cross_doc_weight : {result['cross_doc_weight']}")
        print(f"  used_synth_anchor: {result['used_synthetic_anchor']}")
        print(f"  First 3 compressed sentences:")
        for i, s in enumerate(compressed_sents[:3]):
            snippet = (s[:110] + "…") if len(s) > 113 else s
            print(f"    [{i}] {snippet}")
    print("=" * 70)
