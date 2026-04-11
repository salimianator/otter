"""
encoder.py — OTTER sentence encoder

Wraps sentence-transformers/all-MiniLM-L6-v2 to produce dense 384-dim
sentence embeddings used throughout the OTTER compression pipeline.
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
            Shape [N, 384] float32 array of sentence embeddings.
        """
        return self.model.encode(sentences, convert_to_numpy=True)

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


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    encoder = SentenceEncoder()
    vecs = encoder.encode(["The contract terminates.", "Notice must be given."])
    print("Encoder ready. Output shape:", vecs.shape)
