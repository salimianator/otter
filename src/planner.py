"""
planner.py — OTTER query planner

Core scoring and selection component of the OTTER pipeline.
Takes sentence embeddings, a query vector, and adaptive classifier
weights, then scores every sentence via three complementary signals
(Anchor, Flash, Flow) before selecting a coherent subset.

Anchor    — force-keeps document boundaries so the LLM always has
            structural context (intro paragraph, conclusion).
Flash     — cosine similarity to the query; direct relevance signal.
Flow      — propagates a bonus to sentences neighbouring high-Flash
            regions, preserving local coherence around key passages.
Cross-doc — (multi-doc only) boosts sentences that correlate highly
            with sentences in *other* documents, surfacing shared themes.
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
    cross_doc_weight : float
        Base weight for the cross-document correlation bonus (default: 0.3).
        Interpolated upward toward ``cross_doc_weight_abs`` for abstractive
        queries; only applied when ``doc_boundaries`` contains more than one entry.
    cross_doc_weight_abs : float
        Cross-doc weight ceiling for purely abstractive queries (default: 0.6).
        Linearly interpolated with ``cross_doc_weight`` using the classifier's
        ``w_abs`` score so summarisation tasks get a stronger cross-doc signal.
    per_doc_cap_ratio : float
        Maximum fraction of the total kept-sentence budget that any single
        document may occupy (default: 0.6).  Excess sentences are evicted
        lowest-score-first so under-represented documents get implicit headroom.
        Only applied in multi-doc mode.
    """

    def __init__(
        self,
        anchor_sentences:     int   = 2,
        flow_window:          int   = 2,
        flow_decay:           float = 0.6,
        min_keep_ratio:       float = 0.25,  # never keep fewer than 25% of sentences
        enum_decay:           float = 0.70,  # default running-mean decay (overridden by classifier)
        cross_doc_weight:     float = 0.3,   # cross-doc base weight (extractive end)
        cross_doc_weight_abs: float = 0.6,   # cross-doc ceiling (abstractive end)
        per_doc_cap_ratio:    float = 0.6,   # max fraction of kept budget per document
    ) -> None:
        self.anchor_sentences     = anchor_sentences
        self.flow_window          = flow_window
        self.flow_decay           = flow_decay
        self.min_keep_ratio       = min_keep_ratio
        self.enum_decay           = enum_decay
        self.cross_doc_weight     = cross_doc_weight
        self.cross_doc_weight_abs = cross_doc_weight_abs
        self.per_doc_cap_ratio    = per_doc_cap_ratio

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _cross_doc_scores(
        self,
        embeddings:     np.ndarray,  # [N, D] L2-normalised
        doc_boundaries: list[int],   # exclusive end index per document
    ) -> np.ndarray:                 # [N,] float32 cross-doc scores
        """
        Cross-document correlation score for each sentence.

        For sentence *i* in document *d*, the score is the mean of
        max(cosine_sim(i, d')) over all other documents d'.  Sentences
        that are highly relevant across multiple documents score highest.

        Since embeddings are L2-normalised, dot product == cosine similarity.

        Parameters
        ----------
        embeddings : np.ndarray, shape [N, D]
            L2-normalised sentence embeddings for all documents concatenated.
        doc_boundaries : list[int]
            Exclusive end indices per document.  len(doc_boundaries) == n_docs.

        Returns
        -------
        np.ndarray, shape [N,] float32
        """
        N            = len(embeddings)
        cross_scores = np.zeros(N, dtype=np.float32)
        starts       = [0] + list(doc_boundaries[:-1])
        ends         = list(doc_boundaries)

        for d_idx, (s, e) in enumerate(zip(starts, ends)):
            if e <= s:
                continue
            doc_embs: np.ndarray      = embeddings[s:e]       # [n_d, D]
            per_other: list[np.ndarray] = []

            for d2, (s2, e2) in enumerate(zip(starts, ends)):
                if d2 == d_idx or e2 <= s2:
                    continue
                other_embs = embeddings[s2:e2]               # [n_d2, D]
                sim_mat    = doc_embs @ other_embs.T          # [n_d, n_d2]
                per_other.append(sim_mat.max(axis=1).astype(np.float32))  # [n_d]

            if per_other:
                cross_scores[s:e] = np.mean(per_other, axis=0)

        return cross_scores

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

    def score(
        self,
        embeddings:       np.ndarray,               # [N, D] float32, L2-normalised
        query:            str,                      # raw query string (may be multi-part)
        weights:          dict,                     # output of QueryClassifier.get_weights()
        doc_boundaries:   list[int] | None = None,  # exclusive end index per document
        cross_doc_weight: float | None     = None,  # None → self.cross_doc_weight
        anchor_embedding: np.ndarray | None = None, # synthetic anchor for empty-query multi-doc
    ) -> dict:
        """
        Compute per-sentence scores for all three signals plus their blend.

        Three sub-scores are computed independently then blended with
        the adaptive weights (alpha, beta, gamma) from the classifier.
        The query string is encoded internally via
        ``SentenceEncoder.encode_query_multi`` which returns a [K x D]
        matrix (K=1 for single queries).  Flash scores are computed as
        ``max over K`` of the per-sub-query cosine similarities, so a
        sentence only needs to match *one* sub-query to score highly.

        When ``anchor_embedding`` is provided (multi-doc, empty query),
        it is used directly as the Flash query vector instead of encoding
        ``query``, and ``used_synthetic_anchor`` is set True in the result.

        When ``doc_boundaries`` is provided and n_docs > 1, a cross-document
        correlation score is computed and added to the combined score with
        weight ``cross_doc_weight``.  Single-doc path is unmodified.

        Parameters
        ----------
        embeddings : np.ndarray, shape [N, D]
            L2-normalised sentence embeddings.
        query : str
            Raw query string; may contain one or more sub-questions.
        weights : dict
            Must contain keys ``alpha``, ``beta``, ``gamma`` (floats).
        doc_boundaries : list[int] or None
            Exclusive end indices per document (cumulative sentence counts).
            E.g., two docs with 5 and 8 sentences → [5, 13].
            None or length-1 list → single-document mode, no cross-doc.
        cross_doc_weight : float or None
            Weight for the cross-document term.
            None → use ``self.cross_doc_weight`` (default 0.3).
        anchor_embedding : np.ndarray or None
            Pre-computed L2-normalised vector [D,] used as the Flash query
            instead of encoding ``query``.  Only active when n_docs > 1.

        Returns
        -------
        dict with keys:
            ``anchor``               – [N,] float32 anchor scores
            ``flow``                 – [N,] float32 flow scores
            ``flash``                – [N,] float32 flash (cosine-sim) scores
            ``cross_doc``            – [N,] float32 cross-doc scores (zeros for n_docs==1)
            ``combined``             – [N,] float32 weighted blend
            ``cross_doc_weight``     – float, weight used for cross-doc term
            ``used_synthetic_anchor``– bool, True when anchor_embedding was injected
        """
        if cross_doc_weight is None:
            # Interpolate between base and abstractive ceiling using w_abs
            # from the classifier.  Summarisation tasks (w_abs → 1) get a
            # stronger cross-doc signal; extractive tasks stay near the base.
            w_abs = float(weights.get("w_abs", 0.0))
            cross_doc_weight = (
                self.cross_doc_weight
                + w_abs * (self.cross_doc_weight_abs - self.cross_doc_weight)
            )

        n_docs = len(doc_boundaries) if doc_boundaries is not None else 1

        # ── resolve query vector ──────────────────────────────────────
        used_synthetic_anchor = False
        if anchor_embedding is not None and n_docs > 1:
            # Synthetic centroid anchor replaces query encoding for
            # empty-query multi-doc inputs (Change 3).
            query_mat = anchor_embedding[np.newaxis, :].astype(np.float32)  # [1, D]
            used_synthetic_anchor = True
        else:
            # Import here to keep module-level footprint small and avoid
            # any chance of a circular import at load time.
            from encoder import SentenceEncoder   # noqa: PLC0415
            _enc      = SentenceEncoder()
            query_mat = _enc.encode_query_multi(query)   # [K, D], L2-normalised

        N = len(embeddings)

        # (a) Anchor scores ------------------------------------------------
        anchor = np.zeros(N, dtype=np.float32)
        if doc_boundaries is not None and n_docs > 1:
            # Multi-doc: apply anchor independently per document so every
            # article's opening and closing sentences are force-kept, not
            # just the start/end of the concatenated sentence list.
            _starts = [0] + list(doc_boundaries[:-1])
            _ends   = list(doc_boundaries)
            for _s, _e in zip(_starts, _ends):
                _n  = _e - _s
                _na = min(self.anchor_sentences, _n)
                anchor[_s:_s + _na] = 1.0
                if _n > _na:
                    anchor[max(_e - _na, _s + _na):_e] = 1.0
        else:
            # Single-doc: original behaviour (unchanged)
            n_anchor = min(self.anchor_sentences, N)
            anchor[:n_anchor] = 1.0
            if N > n_anchor:                          # avoid double-setting
                anchor[max(N - n_anchor, n_anchor):] = 1.0

        # (b) Flash scores -------------------------------------------------
        # [N x K] similarity matrix — each column is one sub-query's scores.
        # Row-wise max gives each sentence credit for its best sub-query.
        flash_per_query = (embeddings @ query_mat.T).astype(np.float32)  # [N, K]
        flash           = np.max(flash_per_query, axis=1)                 # [N,]

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

        # (d) Cross-doc scores — multi-doc only (Change 1) ----------------
        if doc_boundaries is not None and n_docs > 1:
            cross_doc = self._cross_doc_scores(embeddings, doc_boundaries)
        else:
            cross_doc = np.zeros(N, dtype=np.float32)

        # (e) Combined score -----------------------------------------------
        combined = (
            weights["alpha"] * anchor +
            weights["beta"]  * flow   +
            weights["gamma"] * flash  +
            cross_doc_weight * cross_doc
        ).astype(np.float32)

        return {
            "anchor":                anchor,
            "flow":                  flow,
            "flash":                 flash,
            "cross_doc":             cross_doc,
            "combined":              combined,
            "cross_doc_weight":      cross_doc_weight,
            "used_synthetic_anchor": used_synthetic_anchor,
        }

    def select(
        self,
        sentences:      list[str],
        scores:         np.ndarray,  # [N,] combined array from score()["combined"]
        weights:        dict,        # full weights dict from QueryClassifier.get_weights()
        doc_boundaries: list[int] | None = None,  # exclusive end index per document
    ) -> tuple[list[str], dict]:
        """
        Select sentences using continuously-interpolated selection.

        The classifier's softmax weights (w_ext / w_enum / w_abs) are used
        to interpolate floor_ratio and enum_decay, which in turn control
        how precisely vs. broadly sentences are kept.  No hard switching —
        every query gets a blend of all three strategies.

        When ``doc_boundaries`` is provided and n_docs > 1, a per-document
        coverage floor is applied after the initial cutoff: every document
        gets at least ``floor(global_kept_fraction * n_sentences_in_doc)``
        sentences selected, with any deficit filled greedily by next-highest-
        scoring sentences from that document (Change 2).

        Parameters
        ----------
        sentences : list[str]
            Original ordered sentence list.
        scores : np.ndarray, shape [N,]
            Per-sentence combined scores from :meth:`score`.
        weights : dict
            Output of QueryClassifier.get_weights() — must contain
            'floor_ratio', 'enum_decay', and 'dominant'.
        doc_boundaries : list[int] or None
            Exclusive end indices per document.  None → single-doc mode.

        Returns
        -------
        tuple[list[str], dict]
            - Kept sentences in their original document order.
            - Metadata dict with keys:
                ``per_doc_cap_applied`` – bool, True when cap was enforced
                ``removed_for_cap``    – int, sentences evicted by coverage cap
        """
        N = len(scores)
        if N == 0:
            return [], {"per_doc_cap_applied": False, "removed_for_cap": 0}

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

        # (e) Build kept set
        kept_set = set(ranked_indices[:cutoff].tolist())

        # (f) Per-doc coverage cap — multi-doc only -----------------------
        # No single document may contribute more than per_doc_cap_ratio of
        # the total kept budget.  Excess sentences are evicted lowest-score-
        # first, giving implicit headroom to under-represented documents.
        removed_for_cap     = 0
        per_doc_cap_applied = False
        n_docs = len(doc_boundaries) if doc_boundaries is not None else 1

        if doc_boundaries is not None and n_docs > 1:
            per_doc_cap_applied = True
            max_per_doc = max(1, int(np.floor(self.per_doc_cap_ratio * len(kept_set))))
            starts = [0] + list(doc_boundaries[:-1])
            ends   = list(doc_boundaries)

            for d_start, d_end in zip(starts, ends):
                doc_kept = sorted(
                    [i for i in range(d_start, d_end) if i in kept_set],
                    key=lambda i: scores[i],   # ascending: lowest score first
                )
                excess = len(doc_kept) - max_per_doc
                if excess > 0:
                    for i in doc_kept[:excess]:
                        kept_set.discard(i)
                        removed_for_cap += 1

        # (g) Restore document order
        kept_original_order = sorted(kept_set)

        print(
            f'[planner] dominant={dominant} | '
            f'floor={floor_ratio:.2f} | decay={enum_decay:.2f} | '
            f'cutoff={cutoff}/{N}'
            + (f' | removed_for_cap={removed_for_cap}' if per_doc_cap_applied else '')
        )

        meta = {
            "per_doc_cap_applied": per_doc_cap_applied,
            "removed_for_cap":     removed_for_cap,
        }
        return [sentences[i] for i in kept_original_order], meta


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
    # Original datasets: qasper → extractive, qmsum → abstractive
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

            kept, _meta = planner.select(
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

    # -----------------------------------------------------------------------
    # Multi-doc smoke test (Change 1, 2, 3)
    # -----------------------------------------------------------------------
    print("\n\n" + "=" * 66)
    print("SMOKE TEST A — TWO-DOCUMENT INPUT")
    print("=" * 66)

    DOC1_SENTS = [
        "Attention mechanisms allow models to focus on relevant parts of the input sequence.",
        "The transformer architecture relies on self-attention in both its encoder and decoder stacks.",
        "Positional encodings are added to input embeddings so the model can retain order information.",
        "Multi-head attention enables the model to jointly attend to information from different subspaces.",
        "Layer normalisation is applied after each sub-layer in the transformer to stabilise training.",
    ]
    DOC2_SENTS = [
        "BERT is a bidirectional transformer pre-trained on masked language modelling and next sentence prediction.",
        "Fine-tuning adds a single output layer on top of the pre-trained BERT model for each downstream task.",
        "BERT achieves state-of-the-art results on eleven natural language processing benchmarks.",
        "The [CLS] token embedding is used as the aggregate sequence representation for classification tasks.",
        "Pre-training on large text corpora allows BERT to capture rich linguistic representations.",
    ]

    all_sents_2doc = DOC1_SENTS + DOC2_SENTS
    doc_boundaries_2 = [len(DOC1_SENTS), len(DOC1_SENTS) + len(DOC2_SENTS)]

    query_2doc  = "How does attention work in transformers?"
    weights_2doc = classifier.get_weights(query_2doc)
    embs_2doc    = encoder.encode(all_sents_2doc)

    score_2doc = planner.score(
        embeddings=embs_2doc,
        query=query_2doc,
        weights=weights_2doc,
        doc_boundaries=doc_boundaries_2,
    )
    kept_2doc, meta_2doc = planner.select(
        sentences=all_sents_2doc,
        scores=score_2doc["combined"],
        weights=weights_2doc,
        doc_boundaries=doc_boundaries_2,
    )

    print(f"\nQuery : {query_2doc!r}")
    print(f"  cross_doc_weight      = {score_2doc['cross_doc_weight']:.3f}")
    print(f"  used_synthetic_anchor = {score_2doc['used_synthetic_anchor']}")
    print(f"  per_doc_cap_applied   = {meta_2doc['per_doc_cap_applied']}")
    print(f"  removed_for_cap       = {meta_2doc['removed_for_cap']}")
    print(f"\n  {'idx':>4}  {'doc':>3}  {'combined':>9}  {'cross_doc':>9}  sentence[:60]")
    print("  " + "-" * 85)
    for i, s in enumerate(all_sents_2doc):
        doc_id  = 0 if i < len(DOC1_SENTS) else 1
        marker  = " <kept>" if s in kept_2doc else ""
        print(
            f"  {i:>4}  {doc_id:>3}  "
            f"{score_2doc['combined'][i]:>9.4f}  "
            f"{score_2doc['cross_doc'][i]:>9.4f}  "
            f"{s[:60]}{marker}"
        )
    print(f"\n  Kept {len(kept_2doc)}/{len(all_sents_2doc)} sentences")

    # -----------------------------------------------------------------------
    # Single-doc smoke test — must be bit-identical to pre-change behaviour
    # -----------------------------------------------------------------------
    print("\n\n" + "=" * 66)
    print("SMOKE TEST B — SINGLE-DOCUMENT INPUT (doc_boundaries=None)")
    print("=" * 66)

    query_1doc   = query_2doc
    weights_1doc = classifier.get_weights(query_1doc)
    embs_1doc    = encoder.encode(DOC1_SENTS)

    score_1doc = planner.score(
        embeddings=embs_1doc,
        query=query_1doc,
        weights=weights_1doc,
        doc_boundaries=None,   # ← single-doc path
    )
    kept_1doc, meta_1doc = planner.select(
        sentences=DOC1_SENTS,
        scores=score_1doc["combined"],
        weights=weights_1doc,
        doc_boundaries=None,
    )

    cross_doc_all_zero = bool(np.all(score_1doc["cross_doc"] == 0.0))
    print(f"\n  cross_doc all zeros   = {cross_doc_all_zero}  (expect True)")
    print(f"  used_synthetic_anchor = {score_1doc['used_synthetic_anchor']}  (expect False)")
    print(f"  per_doc_cap_applied   = {meta_1doc['per_doc_cap_applied']}  (expect False)")
    print(f"\n  {'idx':>4}  {'combined':>9}  {'cross_doc':>9}  sentence[:60]")
    print("  " + "-" * 75)
    for i, s in enumerate(DOC1_SENTS):
        marker = " <kept>" if s in kept_1doc else ""
        print(
            f"  {i:>4}  "
            f"{score_1doc['combined'][i]:>9.4f}  "
            f"{score_1doc['cross_doc'][i]:>9.4f}  "
            f"{s[:60]}{marker}"
        )
    print(f"\n  Kept {len(kept_1doc)}/{len(DOC1_SENTS)} sentences")

    if cross_doc_all_zero and not score_1doc["used_synthetic_anchor"] and not meta_1doc["per_doc_cap_applied"]:
        print("\n  [OK] Single-doc: all multi-doc paths skipped — behaviour identical to pre-change.")
    else:
        print("\n  [FAIL] Single-doc path triggered multi-doc logic unexpectedly.")

    # -----------------------------------------------------------------------
    # Synthetic anchor smoke test (Change 3): empty query + two docs
    # -----------------------------------------------------------------------
    print("\n\n" + "=" * 66)
    print("SMOKE TEST C — SYNTHETIC ANCHOR (empty query, two docs)")
    print("=" * 66)

    weights_empty = classifier.get_weights("Summarize the main points of this document.")
    centroid      = embs_2doc.mean(axis=0)
    centroid_norm = np.linalg.norm(centroid)
    anchor_emb    = (centroid / centroid_norm).astype(np.float32) if centroid_norm > 0 else centroid

    score_anc = planner.score(
        embeddings=embs_2doc,
        query="",
        weights=weights_empty,
        doc_boundaries=doc_boundaries_2,
        anchor_embedding=anchor_emb,
    )
    kept_anc, meta_anc = planner.select(
        sentences=all_sents_2doc,
        scores=score_anc["combined"],
        weights=weights_empty,
        doc_boundaries=doc_boundaries_2,
    )

    print(f"\n  used_synthetic_anchor = {score_anc['used_synthetic_anchor']}  (expect True)")
    print(f"  per_doc_cap_applied   = {meta_anc['per_doc_cap_applied']}  (expect True)")
    print(f"  removed_for_cap       = {meta_anc['removed_for_cap']}")
    print(f"\n  {'idx':>4}  {'doc':>3}  {'combined':>9}  {'cross_doc':>9}  sentence[:60]")
    print("  " + "-" * 85)
    for i, s in enumerate(all_sents_2doc):
        doc_id = 0 if i < len(DOC1_SENTS) else 1
        marker = " <kept>" if s in kept_anc else ""
        print(
            f"  {i:>4}  {doc_id:>3}  "
            f"{score_anc['combined'][i]:>9.4f}  "
            f"{score_anc['cross_doc'][i]:>9.4f}  "
            f"{s[:60]}{marker}"
        )
    print(f"\n  Kept {len(kept_anc)}/{len(all_sents_2doc)} sentences")
