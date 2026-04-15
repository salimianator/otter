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

ENUMERATION_PROTOTYPES: list[str] = [
    "What datasets were used in the experiments?",
    "What are the evaluation metrics used?",
    "Which baselines does the paper compare against?",
    "What are the main contributions of this work?",
    "What types of features are extracted?",
    "List the models they experimented with.",
    "What topics does the dataset cover?",
    "What languages are included in the corpus?",
    "What are the hyperparameters used?",
    "Which tasks does the method evaluate on?",
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
        self.extractive_centroid   = self._build_centroid(EXTRACTIVE_PROTOTYPES)
        self.enumeration_centroid  = self._build_centroid(ENUMERATION_PROTOTYPES)
        self.abstractive_centroid  = self._build_centroid(ABSTRACTIVE_PROTOTYPES)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_weights(self, query: str) -> dict:
        """
        Classify *query* into three soft classes (extractive / enumeration /
        abstractive) via softmax over centroid similarities, then continuously
        interpolate planner weights from the three-class mixture.

        Parameters
        ----------
        query : str
            The raw query string.

        Returns
        -------
        dict with keys:
            w_ext, w_enum, w_abs  – softmax class weights (sum to 1.0)
            dominant              – 'extractive' | 'enumeration' | 'abstractive'
            alpha                 – Anchor weight
            beta                  – Flow weight
            gamma                 – Flash weight
            floor_ratio           – selection floor passed to planner
            enum_decay            – running-mean decay passed to planner
        """
        q_vec  = self.encoder.encode_query(query)

        s_ext  = float(q_vec @ self.extractive_centroid)
        s_enum = float(q_vec @ self.enumeration_centroid)
        s_abs  = float(q_vec @ self.abstractive_centroid)

        # Stable softmax over the three similarities
        scores     = np.array([s_ext, s_enum, s_abs])
        exp_scores = np.exp(scores - scores.max())
        w_ext, w_enum, w_abs = (exp_scores / exp_scores.sum()).tolist()

        # Interpolate planner scoring weights
        alpha = w_ext * 0.40 + w_enum * 0.50 + w_abs * 0.80
        beta  = w_ext * 0.30 + w_enum * 0.50 + w_abs * 0.50
        gamma = w_ext * 0.90 + w_enum * 0.70 + w_abs * 0.40

        # Interpolate selection parameters
        floor_ratio = w_ext * 0.15 + w_enum * 0.30 + w_abs * 0.85
        enum_decay  = w_ext * 0.50 + w_enum * 0.70 + w_abs * 0.90

        # Dominant class label
        labels   = ["extractive", "enumeration", "abstractive"]
        dominant = labels[int(np.argmax([w_ext, w_enum, w_abs]))]

        return {
            "w_ext":       round(w_ext,       4),
            "w_enum":      round(w_enum,      4),
            "w_abs":       round(w_abs,       4),
            "dominant":    dominant,
            "alpha":       round(alpha,       4),
            "beta":        round(beta,        4),
            "gamma":       round(gamma,       4),
            "floor_ratio": round(floor_ratio, 4),
            "enum_decay":  round(enum_decay,  4),
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

    QUERY_GROUPS = {
        "Point extraction": [
            "What is the F1 score reported on SQuAD?",
            "Who proposed this algorithm?",
            "How many parameters does the model have?",
        ],
        "Enumeration": [
            "What datasets were used in the experiments?",
            "Which baselines does the paper compare against?",
            "What are the main contributions of this work?",
            "What evaluation metrics are reported?",
        ],
        "Abstractive": [
            "Summarise the key contributions of this paper.",
            "Describe the overall methodology used.",
            "Explain the limitations of this approach.",
        ],
    }

    encoder    = SentenceEncoder()
    classifier = QueryClassifier(encoder)

    HDR = f"  {'Query':<48}  {'w_ext':>6}  {'w_enum':>6}  {'w_abs':>6}  {'dominant':<13}  {'alpha':>5}  {'beta':>5}  {'gamma':>5}"
    SEP = "  " + "-" * (len(HDR) - 2)

    print()
    for group_name, queries in QUERY_GROUPS.items():
        print(f"── {group_name} ──")
        print(HDR)
        print(SEP)
        for q in queries:
            w = classifier.get_weights(q)
            q_disp = (q[:47] + "…") if len(q) > 48 else q
            print(
                f"  {q_disp:<48}  "
                f"{w['w_ext']:>6.3f}  {w['w_enum']:>6.3f}  {w['w_abs']:>6.3f}  "
                f"{w['dominant']:<13}  "
                f"{w['alpha']:>5.3f}  {w['beta']:>5.3f}  {w['gamma']:>5.3f}"
            )
        print()
