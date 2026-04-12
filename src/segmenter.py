"""
segmenter.py — OTTER sentence segmenter

Splits a raw document string into a clean, merged list of sentence
strings using spaCy's en_core_web_sm model for boundary detection.

Merge logic removes the "stub" sentences that spaCy sometimes
produces for abbreviations, figure captions, headers, and lone
punctuation by folding any segment shorter than `min_words` words
into its neighbour.
"""

from __future__ import annotations

import statistics
import json
from pathlib import Path

import spacy
from spacy.language import Language


class SentenceSegmenter:
    """
    Sentence boundary detector backed by spaCy en_core_web_sm.

    Parameters
    ----------
    min_words : int
        Segments with fewer than this many whitespace-delimited tokens
        are merged into the adjacent segment (default: 6).
    """

    _MODEL_NAME = "en_core_web_sm"

    def __init__(self, min_words: int = 6) -> None:
        self.min_words = min_words
        self._nlp: Language = spacy.load(self._MODEL_NAME, disable=["ner", "tagger", "lemmatizer"])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment(self, text: str) -> list[str]:
        """
        Split *text* into a cleaned, merged list of sentence strings.

        Pipeline
        --------
        a) spaCy sentence boundary detection
        b) strip whitespace from each sentence
        c) drop empty segments
        d) merge segments shorter than ``min_words`` into neighbour
        e) return final list

        Parameters
        ----------
        text : str
            Raw document text (arbitrarily long).

        Returns
        -------
        list[str]
            Ordered list of sentence strings.
        """
        doc = self._nlp(text)

        # (a)+(b)+(c) — raw sentences, stripped, non-empty
        raw: list[str] = [
            sent.text.strip()
            for sent in doc.sents
            if sent.text.strip()
        ]

        # (d) merge short segments
        merged = self._merge_short(raw)

        return merged

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _word_count(self, s: str) -> int:
        return len(s.split())

    def _merge_short(self, sentences: list[str]) -> list[str]:
        """
        Single-pass left-to-right merge.

        A segment below ``min_words`` is appended to the *previous*
        segment with a space.  If it is the very first segment it is
        prepended to the *next* one instead.  The process repeats
        until no segment is below the threshold, so that consecutive
        short fragments (e.g. "Fig." followed by "3.") are collapsed
        together.
        """
        if not sentences:
            return []

        changed = True
        result = list(sentences)

        while changed:
            changed = False
            merged: list[str] = []

            i = 0
            while i < len(result):
                seg = result[i]
                if self._word_count(seg) < self.min_words:
                    if merged:
                        # merge into the previous segment
                        merged[-1] = merged[-1] + " " + seg
                        changed = True
                    elif i + 1 < len(result):
                        # first segment — merge into the next one
                        result[i + 1] = seg + " " + result[i + 1]
                        changed = True
                        # don't add to merged; the next iteration picks up the combined seg
                    else:
                        # only one segment and it's short — keep as-is
                        merged.append(seg)
                else:
                    merged.append(seg)
                i += 1

            result = merged

        return result


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    segmenter = SentenceSegmenter(min_words=6)

    # -----------------------------------------------------------------------
    # Test 1 — crafted paragraph with tricky cases
    # -----------------------------------------------------------------------
    TRICKY = (
        "Dr. Sarah Chen published her findings in Nature last week. "
        "The study examined Fig. 3 and Fig. 4 in detail. "
        "Results. "                              # ← short fragment: should merge
        "The compound showed a 47 % increase in efficacy compared with the "
        "control group across all three trial sites. "
        "Prof. Jones disagreed. "               # ← short fragment: should merge
        "He argued that the methodology introduced confounding variables "
        "and called for an independent replication of the experiment. "
        "See also: Appendix B."                 # ← short fragment at end
    )

    print("=" * 60)
    print("TEST 1 — tricky abbreviations & short fragments")
    print("=" * 60)
    sents1 = segmenter.segment(TRICKY)
    print(f"Sentences detected: {len(sents1)}\n")
    for i, s in enumerate(sents1):
        wc = len(s.split())
        print(f"  [{i:2d}] ({wc:3d} words) {s}")

    # -----------------------------------------------------------------------
    # Test 2 — first example from LongBench qasper
    # -----------------------------------------------------------------------
    DATA_PATH = Path(__file__).parent.parent / "data" / "longbench_qasper.jsonl"

    print("\n" + "=" * 60)
    print("TEST 2 — first qasper example (context field)")
    print("=" * 60)

    if not DATA_PATH.exists():
        print(f"[SKIP] {DATA_PATH} not found — run data download first.")
    else:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            first_record = json.loads(f.readline())

        context = first_record["context"]
        sents2 = segmenter.segment(context)

        wcs = [len(s.split()) for s in sents2]
        print(f"Total sentences : {len(sents2)}")
        print(f"Words per sent  — min: {min(wcs)}  "
              f"mean: {statistics.mean(wcs):.1f}  "
              f"max: {max(wcs)}")
        print("\nFirst 5 sentences:")
        for i, s in enumerate(sents2[:5]):
            wc = len(s.split())
            print(f"  [{i}] ({wc} words) {s}")
