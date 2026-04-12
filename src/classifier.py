"""
classifier.py — OTTER query-type classifier

Classifies an incoming query as extractive vs. abstractive using
semantic similarity against prototype query embeddings, then returns
continuous (alpha, beta, gamma) weights for the QueryPlanner.

No keyword matching — classification is fully embedding-based so it
generalises to novel query phrasings.
"""

from __future__ import annotations

import numpy as np

from encoder import SentenceEncoder


# ---------------------------------------------------------------------------
# Prototype query banks
# ---------------------------------------------------------------------------

EXTRACTIVE_PROTOTYPES: list[str] = [
    "What year was this published?",
    "Who is the main author of the paper?",
    "How many patients were in the study?",
    "Which method achieved the highest accuracy?",
    "Where did the experiment take place?",
    "What is the name of the algorithm proposed?",
    "When did the event occur?",
    "Identify the primary risk factor mentioned.",
    "What value did the model achieve on the benchmark?",
    "Which dataset was used for evaluation?",
]

ABSTRACTIVE_PROTOTYPES: list[str] = [
    "Summarise the key contributions of this paper.",
    "Describe the overall methodology used.",
    "What are the main findings of this study?",
    "Explain the author's central argument.",
    "Give an overview of the experimental results.",
    "Discuss the limitations of this approach.",
    "What does the paper conclude overall?",
    "Describe the relationship between the two variables.",
    "What is the paper about?",
    "Enumerate the principal contributions of this work.",
]


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class QueryClassifier:
    """
    Soft extractive/abstractive classifier driven by centroid similarity.

    Parameters
    ----------
    encoder : SentenceEncoder
        A fully initialised encoder instance — shared with the rest of
        the pipeline so the underlying model is loaded only once.
    """

    def __init__(self, encoder: SentenceEncoder) -> None:
        self.encoder = encoder
        self.extractive_centroid = self._build_centroid(EXTRACTIVE_PROTOTYPES)
        self.abstractive_centroid = self._build_centroid(ABSTRACTIVE_PROTOTYPES)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, query: str) -> float:
        """
        Return an extractive score in [0, 1].

        1.0 → fully extractive (fact lookup, named-entity retrieval)
        0.0 → fully abstractive (summarisation, explanation, overview)

        Uses dot-product similarity against pre-computed centroids
        (equivalent to cosine similarity since all vectors are L2-normalised).

        Parameters
        ----------
        query : str
            The raw query string.

        Returns
        -------
        float
            Soft extractive score.
        """
        q_vec = self.encoder.encode_query(query)
        sim_e = float(q_vec @ self.extractive_centroid)
        sim_a = float(q_vec @ self.abstractive_centroid)
        denom = sim_e + sim_a
        return sim_e / denom if denom > 0 else 0.5

    def get_weights(self, query: str) -> dict[str, float]:
        """
        Return adaptive planner weights for *query*.

        Weight ranges
        -------------
        gamma (Flash)  : 0.4 (abstractive) → 0.9 (extractive)
        alpha (Anchor) : 0.8 (abstractive) → 0.4 (extractive)
        beta  (Flow)   : 0.5 (abstractive) → 0.3 (extractive)

        Parameters
        ----------
        query : str
            The raw query string.

        Returns
        -------
        dict with keys: extractive_score, alpha, beta, gamma
        """
        s = self.classify(query)
        return {
            "extractive_score": round(s, 4),
            "alpha": round(0.8 - 0.4 * s, 4),   # Anchor: 0.8 → 0.4
            "beta":  round(0.5 - 0.2 * s, 4),   # Flow:   0.5 → 0.3
            "gamma": round(0.4 + 0.5 * s, 4),   # Flash:  0.4 → 0.9
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_centroid(self, prototypes: list[str]) -> np.ndarray:
        """
        Encode *prototypes*, average embeddings, then re-normalise.

        Re-normalisation keeps the centroid on the unit hypersphere so
        dot products remain valid cosine-similarity proxies.
        """
        vecs = self.encoder.encode(prototypes)          # [N, 384], already L2-normed
        centroid = vecs.mean(axis=0)                    # [384]
        norm = np.linalg.norm(centroid)
        return (centroid / norm).astype(np.float32)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))

    TEST_QUERIES = [
        "What year was the paper published?",
        "Who proposed this algorithm?",
        "Summarise the main contributions.",
        "Describe the overall methodology.",
        "What is the F1 score reported on SQuAD?",
        "Explain the limitations of this approach.",
        "Enumerate the key findings of this study.",
        "Which baseline did the proposed method outperform?",
    ]

    encoder    = SentenceEncoder()
    classifier = QueryClassifier(encoder)

    # Header
    col_q  = 46
    print(f"\n{'Query':<{col_q}}  {'ext_score':>9}  {'alpha':>6}  {'beta':>6}  {'gamma':>6}  {'type'}")
    print("-" * (col_q + 40))

    for q in TEST_QUERIES:
        w = classifier.get_weights(q)
        s = w["extractive_score"]
        label = "EXTRACT" if s >= 0.55 else ("ABSTACT" if s <= 0.45 else "mixed  ")
        q_display = (q[:col_q - 1] + "…") if len(q) > col_q else q
        print(
            f"{q_display:<{col_q}}  {s:>9.4f}  "
            f"{w['alpha']:>6.4f}  {w['beta']:>6.4f}  {w['gamma']:>6.4f}  {label}"
        )
    print()
