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
        anchor_sentences: int   = 2,
        flow_window:      int   = 2,
        flow_decay:       float = 0.6,
        min_keep_ratio:   float = 0.25,  # never keep fewer than 25% of sentences
        enum_decay:       float = 0.70,  # default running-mean decay (overridden by classifier)
    ) -> None:
        self.anchor_sentences = anchor_sentences
        self.flow_window      = flow_window
        self.flow_decay       = flow_decay
        self.min_keep_ratio   = min_keep_ratio
        self.enum_decay       = enum_decay

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
        ``SentenceEncoder.encode_query_multi`` which returns a [K x 384]
        matrix (K=1 for single queries).  Flash scores are computed as
        ``max over K`` of the per-sub-query cosine similarities, so a
        sentence only needs to match *one* sub-query to score highly.

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

        _enc      = SentenceEncoder()
        query_mat = _enc.encode_query_multi(query)   # [K, 384], L2-normalised

        N = len(embeddings)

        # (a) Anchor scores ------------------------------------------------
        anchor = np.zeros(N, dtype=np.float32)
        n_anchor = min(self.anchor_sentences, N)
        anchor[:n_anchor] = 1.0
        if N > n_anchor:                              # avoid double-setting
            anchor[max(N - n_anchor, n_anchor):] = 1.0

        # (b) Flash scores -------------------------------------------------
        # [N x K] similarity matrix — each column is one sub-query's scores.
        # Taking the row-wise max gives each sentence credit for its best
        # matching sub-query; a sentence only needs to answer *one* question
        # to score highly.
        flash_per_query = (embeddings @ query_mat.T).astype(np.float32)  # [N, K]
        flash = np.max(flash_per_query, axis=1)                           # [N,]

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

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _unified_cutoff(
        self,
        scores_ranked: np.ndarray,
        floor_ratio:   float,
        enum_decay:    float,
    ) -> int:
        """
        Continuously-blended selection cutoff driven by classifier weights.

        Blends two complementary signals:

        Step 1 — Running-mean decay (enumeration/extractive precision):
            Keep adding sentences while each next score stays above
            enum_decay × current running mean.  Stops at the first
            sentence that pulls quality meaningfully below the group mean.

        Step 2 — Marginal floor (abstractive breadth):
            Keep all sentences above floor_ratio × global mean.
            Designed for flat distributions where precision cutoff
            fires too early.

        Step 3 — Blend proportional to floor_ratio:
            Low floor_ratio  → decay_cutoff dominates (extractive)
            High floor_ratio → floor_cutoff dominates (abstractive)
            Middle           → balanced blend            (enumeration)

        Returns
        -------
        int
            Cutoff index into scores_ranked (number of sentences to keep).
        """
        N = len(scores_ranked)

        # Step 1 — running-mean decay
        kept_scores  = [scores_ranked[0]]
        decay_cutoff = N
        for i in range(1, N):
            running_mean = np.mean(kept_scores)
            if scores_ranked[i] >= running_mean * enum_decay:
                kept_scores.append(scores_ranked[i])
            else:
                decay_cutoff = i
                break

        # Step 2 — global marginal floor
        mean_score   = np.mean(scores_ranked)
        floor        = floor_ratio * mean_score
        floor_cutoff = next(
            (i for i, s in enumerate(scores_ranked) if s < floor),
            N,
        )

        # Step 3 — blend
        cutoff = int(round(
            (1 - floor_ratio) * decay_cutoff +
            floor_ratio       * floor_cutoff
        ))
        return max(1, cutoff)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(
        self,
        sentences: list[str],
        scores:    np.ndarray,  # [N,] combined array from score()["combined"]
        weights:   dict,        # full weights dict from QueryClassifier.get_weights()
    ) -> list[str]:
        """
        Select sentences using continuously-interpolated selection.

        The classifier's softmax weights (w_ext / w_enum / w_abs) are used
        to interpolate floor_ratio and enum_decay, which in turn control
        how precisely vs. broadly sentences are kept.  No hard switching —
        every query gets a blend of all three strategies.

        Parameters
        ----------
        sentences : list[str]
            Original ordered sentence list.
        scores : np.ndarray, shape [N,]
            Per-sentence combined scores from :meth:`score`.
        weights : dict
            Output of QueryClassifier.get_weights() — must contain
            'floor_ratio', 'enum_decay', and 'dominant'.

        Returns
        -------
        list[str]
            Kept sentences in their original document order.
        """
        N = len(scores)
        if N == 0:
            return []

        floor_ratio = float(weights["floor_ratio"])
        enum_decay  = float(weights["enum_decay"])
        dominant    = weights["dominant"]

        # (a) Minimum keep count
        min_keep = max(3, int(N * self.min_keep_ratio))

        # (b) Rank descending
        ranked_indices = np.argsort(scores)[::-1]
        scores_ranked  = scores[ranked_indices]

        # (c) Unified blended cutoff
        cutoff = self._unified_cutoff(scores_ranked, floor_ratio, enum_decay)

        # (d) Apply minimum keep floor
        cutoff = max(cutoff, min_keep)

        # (e–f) Collect and restore document order
        kept_ranked         = ranked_indices[:cutoff]
        kept_original_order = sorted(kept_ranked.tolist())

        print(
            f'[planner] dominant={dominant} | '
            f'floor={floor_ratio:.2f} | decay={enum_decay:.2f} | '
            f'cutoff={cutoff}/{N}'
        )

        return [sentences[i] for i in kept_original_order]


# ---------------------------------------------------------------------------
# End-to-end smoke test: segmenter → encoder → classifier → planner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))

    from segmenter  import SentenceSegmenter
    from encoder    import SentenceEncoder
    from classifier import QueryClassifier

    DATA_DIR = Path(__file__).parent.parent / "data"

    segmenter  = SentenceSegmenter(min_words=6)
    encoder    = SentenceEncoder()
    classifier = QueryClassifier(encoder)
    planner    = QueryPlanner()

    # -----------------------------------------------------------------------
    # Datasets: qasper → should trigger Kneedle (extractive)
    #           qmsum  → should trigger marginal_return (abstractive)
    # -----------------------------------------------------------------------
    SUITES = [
        ("qasper", "longbench_qasper.jsonl"),
        ("qmsum",  "longbench_qmsum.jsonl"),
    ]

    for subset_name, filename in SUITES:
        data_path = DATA_DIR / filename
        if not data_path.exists():
            print(f"\n[SKIP] {filename} not found — run benchmark.py to download first.")
            continue

        print("\n" + "=" * 66)
        print(f"Subset: {subset_name.upper()} — 3 examples")
        print("=" * 66)

        with open(data_path, "r", encoding="utf-8") as fh:
            records = [json.loads(fh.readline()) for _ in range(3)]

        for ex_idx, record in enumerate(records, 1):
            context = record["context"]
            query   = record["input"]

            # Apply same fallback as compress.py
            if not query or not query.strip():
                query = "Summarize the main points of this document."

            sentences = segmenter.segment(context)
            sent_vecs = encoder.encode(sentences)
            weights   = classifier.get_weights(query)
            score_d   = planner.score(sent_vecs, query, weights)

            kept = planner.select(
                sentences=sentences,
                scores=score_d["combined"],
                weights=weights,
            )

            ratio  = len(kept) / max(len(sentences), 1)
            q_disp = (query[:57] + "…") if len(query) > 60 else query

            print(
                f"  [{ex_idx}] query={q_disp!r}\n"
                f"       dominant={weights['dominant']:<13}  "
                f"floor={weights['floor_ratio']:.2f}  decay={weights['enum_decay']:.2f}  "
                f"kept={len(kept)}/{len(sentences)}  ratio={ratio:.1%}"
            )
