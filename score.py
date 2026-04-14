"""
score.py — OTTER scoring utilities

Pure string metrics — no ML models loaded here.
Used by benchmark.py to evaluate generated answers against ground truth.
"""

from __future__ import annotations

import re

from rouge_score import rouge_scorer as _rouge_scorer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "of", "in", "on",
    "at", "to", "for",
})

_PUNCT_RE = re.compile(r"[^\w\s]")   # strip punctuation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace, drop stopwords."""
    text = _PUNCT_RE.sub(" ", text.lower())
    return [t for t in text.split() if t and t not in _STOPWORDS]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_f1(prediction: str, ground_truths: list[str]) -> float:
    """
    Token-level F1 between *prediction* and the best-matching ground truth.

    Tokenisation: lowercase, punctuation stripped, stopwords removed.
    Score is the maximum F1 across all ground truths (LongBench convention).

    Parameters
    ----------
    prediction : str
        Model-generated answer string.
    ground_truths : list[str]
        One or more reference answer strings.

    Returns
    -------
    float
        Maximum token F1 in [0, 1].
    """
    pred_tokens = _tokenise(prediction)
    if not pred_tokens:
        return 0.0

    best = 0.0
    for gt in ground_truths:
        gt_tokens = _tokenise(gt)
        if not gt_tokens:
            continue

        pred_set = set(pred_tokens)
        gt_set   = set(gt_tokens)
        common   = pred_set & gt_set

        if not common:
            continue

        precision = len(common) / len(pred_set)
        recall    = len(common) / len(gt_set)
        f1        = 2 * precision * recall / (precision + recall)
        best      = max(best, f1)

    return best


def compute_rouge_l(prediction: str, ground_truths: list[str]) -> float:
    """
    ROUGE-L F-measure between *prediction* and the best-matching ground truth.

    Uses the ``rouge_score`` library with stemming enabled.

    Parameters
    ----------
    prediction : str
        Model-generated answer string.
    ground_truths : list[str]
        One or more reference answer strings.

    Returns
    -------
    float
        Maximum ROUGE-L fmeasure in [0, 1].
    """
    scorer = _rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    best   = 0.0
    for gt in ground_truths:
        score = scorer.score(gt, prediction)
        best  = max(best, score["rougeL"].fmeasure)
    return best


def select_metric(subset: str) -> str:
    """
    Return the canonical metric name for a LongBench subset.

    Parameters
    ----------
    subset : str
        LongBench subset name.

    Returns
    -------
    str
        ``'f1'`` or ``'rouge_l'``.

    Raises
    ------
    ValueError
        If *subset* is not recognised.
    """
    if subset in {"qasper", "multifieldqa_en"}:
        return "f1"
    if subset in {"qmsum", "multi_news"}:
        return "rouge_l"
    raise ValueError(
        f"Unknown subset {subset!r}. "
        "Expected one of: qasper, multifieldqa_en, qmsum, multi_news."
    )
