"""
encoder.py — OTTER sentence encoder

Wraps sentence-transformers/all-MiniLM-L6-v2 to produce dense 384 dimensions
sentence embeddings used throughout the OTTER's compression pipeline.
"""

import numpy as np
from sentence_transformers import SentenceTransformer


class SentenceEncoder:
    """
    Thin wrapper around a SentenceTransformer model.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID (default: all-MiniLM-L6-v2).
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load and cache the underlying SentenceTransformer."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, sentences: list[str]) -> np.ndarray:
        """
        Encode a list of sentences into a 2-D embedding matrix.

        Parameters
        ----------
        sentences : list[str]
            Input sentences to encode.

        Returns
        -------
        np.ndarray
            L2-normalised shape [N, 384] float32 array of sentence embeddings.
            Normalisation means cosine similarity reduces to a plain dot product,
            making downstream Flash scoring cheaper.
        """
        embeddings = self.model.encode(
            sentences,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query string into a 1-D embedding vector.

        Parameters
        ----------
        query : str
            The query text.

        Returns
        -------
        np.ndarray
            Shape [384,] float32 vector.
        """
        return self.encode([query])[0]

    def encode_query_multi(self, query: str) -> np.ndarray:
        """
        Encode a query (single or multi-part) into a [K x 384] matrix.

        Splits *query* into sub-queries via SentenceSegmenter and encodes
        each independently.  Always returns a 2-D matrix so the caller
        can apply per-sub-query scoring (e.g. ``embeddings @ Q.T`` then
        ``np.max(..., axis=1)``) rather than collapsing to a single vector
        before scoring.  This preserves the distinct retrieval signal of
        each sub-question.

        Imported inside the method to avoid a circular import
        (segmenter.py does not import encoder.py, but encoder.py is
        imported first at module level in other pipeline files).

        Parameters
        ----------
        query : str
            Raw query string; may contain one or more sentences.

        Returns
        -------
        np.ndarray
            Shape [K, 384] float32, one L2-normalised row per sub-query.
            K=1 for a single question, K>1 for multi-part queries.
        """
        # Local import avoids circular dependency at module level
        from segmenter import SentenceSegmenter   # noqa: PLC0415

        seg = SentenceSegmenter(min_words=3)      # low threshold for short sub-queries
        sub_queries = seg.segment(query)

        # Always return 2-D [K, 384] — encode() handles K=1 fine
        return self.encode(sub_queries)           # already L2-normalised per row


# ---------------------------------------------------------------------------
# End-to-end smoke test: segmenter → encoder → top-3 retrieval
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    # Make sure sibling modules are importable when run directly
    sys.path.insert(0, str(Path(__file__).parent))
    from segmenter import SentenceSegmenter

    DATA_PATH = Path(__file__).parent.parent / "data" / "longbench_qasper.jsonl"

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        record = json.loads(f.readline())

    context = record["context"]
    query   = record["input"]

    # --- segment ---
    segmenter = SentenceSegmenter(min_words=6)
    sentences = segmenter.segment(context)

    # --- encode ---
    encoder      = SentenceEncoder()
    sent_vecs    = encoder.encode(sentences)          # [N, 384], L2-normalised
    query_vec    = encoder.encode_query(query)        # [384,],   L2-normalised

    print(f"Query            : {query!r}")
    print(f"Sentences segmented : {len(sentences)}")
    print(f"Embedding matrix : {sent_vecs.shape}")
    print(f"Query vector     : {query_vec.shape}")
    print(f"L2 norm (sent[0]): {np.linalg.norm(sent_vecs[0]):.6f}  (expect ~1.0)")

    # --- top-3 cosine similarity (= dot product on unit vectors) ---
    scores  = sent_vecs @ query_vec          # [N]
    top3_idx = np.argsort(scores)[::-1][:3]

    print(f"\nTop-3 sentences for query:")
    for rank, idx in enumerate(top3_idx, 1):
        print(f"  Rank {rank} | score {scores[idx]:.2f} | {sentences[idx]}")

    # -----------------------------------------------------------------------
    # Test 2 — encode_query_multi: single vs. multi-part query
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("encode_query_multi — single vs. multi-part query")
    print("=" * 60)

    SINGLE = "What is the main contribution?"
    MULTI  = (
        "What is the main contribution? "
        "Who are the primary authors? "
        "Which dataset was used for evaluation?"
    )

    for label, q in [("Single-part", SINGLE), ("Multi-part ", MULTI)]:
        mat = encoder.encode_query_multi(q)           # now [K, 384]
        row_norms = np.linalg.norm(mat, axis=1)
        print(f"\n  {label}: {q!r}")
        print(f"    output shape         : {mat.shape}  "
              f"(expect [1, 384] or [K, 384])")
        print(f"    L2 norm (each row)   : {np.array2string(row_norms, precision=6)}  "
              f"(expect all ~1.0)")
