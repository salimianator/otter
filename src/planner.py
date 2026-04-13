"""
planner.py — OTTER query planner

Core scoring and selection component of the OTTER pipeline.
Takes sentence embeddings, a query vector, and adaptive classifier
weights, then scores every sentence via three complementary signals
(Anchor, Flash, Flow) before selecting a coherent subset.

Anchor  — force-keeps document boundaries so the LLM always has
          structural context (intro paragraph, conclusion).
Flash   — cosine similarity to the query; direct relevance signal.
Flow    — propagates a bonus to sentences neighbouring high-Flash
          regions, preserving local coherence around key passages.
"""

from __future__ import annotations

import numpy as np


class QueryPlanner:
    """
    Scores and selects sentences using Anchor + Flash + Flow signals.

    Parameters
    ----------
    anchor_sentences : int
        Number of sentences force-kept at the start and end of the
        document regardless of query similarity (default: 2).
    flow_window : int
        How many neighbours around a Flash-selected sentence receive
        a flow bonus (default: 2).
    flow_decay : float
        Base bonus for distance-1 neighbours.  Distance-2 neighbours
        receive ``flow_decay * 0.5``.  Default: 0.6  →  d1=0.6, d2=0.3.
    """

    def __init__(
        self,
        anchor_sentences: int = 2,
        flow_window: int = 2,
        flow_decay: float = 0.6,
    ) -> None:
        self.anchor_sentences = anchor_sentences
        self.flow_window      = flow_window
        self.flow_decay       = flow_decay

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        embeddings: np.ndarray,   # [N, 384] float32, L2-normalised
        query:      str,          # raw query string (may be multi-part)
        weights:    dict,         # output of QueryClassifier.get_weights()
    ) -> dict:                    # keys: anchor, flow, flash, combined
        """
        Compute per-sentence scores for all three signals plus their blend.

        Three sub-scores are computed independently then blended with
        the adaptive weights (alpha, beta, gamma) from the classifier.
        The query string is encoded internally via
        ``SentenceEncoder.encode_query_multi`` so multi-part queries
        (several questions separated by sentence boundaries) are handled
        correctly — a sentence only needs to match *one* sub-query to
        receive a high Flash score.

        Parameters
        ----------
        embeddings : np.ndarray, shape [N, 384]
            L2-normalised sentence embeddings.
        query : str
            Raw query string; may contain one or more sub-questions.
        weights : dict
            Must contain keys ``alpha``, ``beta``, ``gamma`` (floats).

        Returns
        -------
        dict with keys:
            ``anchor``   – [N,] float32 anchor scores
            ``flow``     – [N,] float32 flow scores
            ``flash``    – [N,] float32 flash (cosine-sim) scores
            ``combined`` – [N,] float32 weighted blend
        """
        # Import here to keep the module-level footprint small and avoid
        # any chance of a circular import at load time.
        from encoder import SentenceEncoder   # noqa: PLC0415

        _enc = SentenceEncoder()
        query_vec = _enc.encode_query_multi(query)   # [384,], L2-normalised

        N = len(embeddings)

        # (a) Anchor scores ------------------------------------------------
        anchor = np.zeros(N, dtype=np.float32)
        n_anchor = min(self.anchor_sentences, N)
        anchor[:n_anchor] = 1.0
        if N > n_anchor:                              # avoid double-setting
            anchor[max(N - n_anchor, n_anchor):] = 1.0

        # (b) Flash scores -------------------------------------------------
        flash = (embeddings @ query_vec).astype(np.float32)   # [N,]

        # (c) Flow scores --------------------------------------------------
        flow = np.zeros(N, dtype=np.float32)

        n_flash_seeds = max(1, N // 10)
        flash_top_idx = set(np.argsort(flash)[-n_flash_seeds:].tolist())

        for idx in flash_top_idx:
            for dist in range(1, self.flow_window + 1):
                bonus = self.flow_decay * (1.0 if dist == 1 else 0.5 ** (dist - 1))
                for neighbour in (idx - dist, idx + dist):
                    if 0 <= neighbour < N and neighbour not in flash_top_idx:
                        flow[neighbour] += bonus

        flow = np.clip(flow, 0.0, 1.0)

        # (d) Combined score -----------------------------------------------
        combined = (
            weights["alpha"] * anchor +
            weights["beta"]  * flow   +
            weights["gamma"] * flash
        ).astype(np.float32)

        return {
            "anchor":   anchor,
            "flow":     flow,
            "flash":    flash,
            "combined": combined,
        }

    def select(
        self,
        sentences: list[str],
        scores:    np.ndarray,       # [N,] combined array from score()["combined"]
        threshold: float = 0.85,
    ) -> list[str]:
        """
        Select sentences via cumulative-score thresholding.

        Sentences are ranked by score descending; we keep adding until
        their cumulative score reaches ``threshold`` × total score.
        Kept indices are then sorted back into *document order* so the
        LLM receives a chronologically coherent passage.

        Parameters
        ----------
        sentences : list[str]
            Original ordered sentence list.
        scores : np.ndarray, shape [N,]
            Per-sentence combined scores from :meth:`score`.
        threshold : float
            Fraction of total score to capture (default: 0.85).

        Returns
        -------
        list[str]
            Subset of sentences in their original document order.
        """
        N = len(scores)
        if N == 0:
            return []

        ranked_idx  = np.argsort(scores)[::-1]          # high → low
        cumsum      = np.cumsum(scores[ranked_idx])
        total_score = cumsum[-1]

        # Find the minimum number of sentences that covers the threshold
        cutoff_pos  = int(np.searchsorted(cumsum, threshold * total_score))
        cutoff_pos  = min(cutoff_pos, N - 1)             # clamp

        kept_idx = sorted(ranked_idx[: cutoff_pos + 1].tolist())   # doc order
        return [sentences[i] for i in kept_idx]


# ---------------------------------------------------------------------------
# End-to-end smoke test: segmenter → encoder → classifier → planner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))

    from segmenter   import SentenceSegmenter
    from encoder     import SentenceEncoder
    from classifier  import QueryClassifier

    DATA_PATH = Path(__file__).parent.parent / "data" / "longbench_qasper.jsonl"

    # Load first example
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        record = json.loads(f.readline())

    context = record["context"]
    query   = record["input"]

    # --- Step 1: segment ---
    segmenter  = SentenceSegmenter(min_words=6)
    sentences  = segmenter.segment(context)

    # --- Step 2: encode ---
    encoder    = SentenceEncoder()
    sent_vecs  = encoder.encode(sentences)

    # --- Step 3: classify → weights ---
    classifier = QueryClassifier(encoder)
    weights    = classifier.get_weights(query)

    # --- Step 4: score + select (query string passed directly) ---
    planner   = QueryPlanner()
    score_d   = planner.score(sent_vecs, query, weights)
    kept      = planner.select(sentences, score_d["combined"], threshold=0.85)

    ratio = len(kept) / len(sentences)

    print("=" * 66)
    print("OTTER end-to-end pipeline — first qasper example")
    print("=" * 66)
    print(f"Query              : {query!r}")
    print()
    print(f"Classifier weights :")
    print(f"  extractive_score = {weights['extractive_score']:.4f}")
    print(f"  alpha (Anchor)   = {weights['alpha']:.4f}")
    print(f"  beta  (Flow)     = {weights['beta']:.4f}")
    print(f"  gamma (Flash)    = {weights['gamma']:.4f}")
    print()
    print(f"Sentences before   : {len(sentences)}")
    print(f"Sentences after    : {len(kept)}")
    print(f"Compression ratio  : {ratio:.2%}  ({len(kept)}/{len(sentences)} kept)")

    print("\n--- First 5 kept sentences ---")
    for i, s in enumerate(kept[:5]):
        print(f"  [{i}] {s}")

    print("\n--- Last 5 kept sentences ---")
    for i, s in enumerate(kept[-5:], start=len(kept) - 5):
        print(f"  [{i}] {s}")
