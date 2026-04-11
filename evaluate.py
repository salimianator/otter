"""
evaluate.py — OTTER evaluation harness

Wraps Qwen2.5-3B-Instruct as the end-to-end reader model.
Intentionally kept separate from the OTTER compression modules so it
can be swapped out without touching compression logic.

Lazy-loading (load_on_init=False, the default) means Qwen is never
pulled into memory during development runs that only exercise the
compressor — keeping iteration fast.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class QwenEvaluator:
    """
    Lazy-loading wrapper around Qwen2.5-3B-Instruct.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID.
    load_on_init : bool
        If True, load tokenizer + model immediately in __init__.
        If False (default), defer loading until the first call to answer().
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        load_on_init: bool = False,
    ) -> None:
        self.model_name = model_name
        self._tokenizer: AutoTokenizer | None = None
        self._model: AutoModelForCausalLM | None = None

        if load_on_init:
            self._load()

    @staticmethod
    def _best_device() -> tuple[str, torch.dtype]:
        """
        Pick the best available device and a compatible dtype.

        - CUDA  → float16,  device_map="auto"  (supports bfloat16 offload)
        - MPS   → float16,  device placed on "mps" explicitly
                  (MPS does NOT support bfloat16; must avoid disk offload)
        - CPU   → float32,  device_map="cpu"
        """
        if torch.cuda.is_available():
            return "auto", torch.float16
        if torch.backends.mps.is_available():
            return "mps", torch.float16
        return "cpu", torch.float32

    def _load(self) -> None:
        """Load tokenizer and model if not already in memory."""
        if self._model is not None:
            return

        device_map, dtype = self._best_device()
        print(f"Loading {self.model_name} … (device={device_map}, dtype={dtype})")

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # For MPS we pass the explicit string so accelerate places every
        # layer there and never falls back to disk offloading (which would
        # try to cast bfloat16 back to MPS and fail).
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=dtype,
            device_map=device_map,
        )
        self._model.eval()
        print("Model loaded.")

    def answer(self, context: str, query: str) -> str:
        """
        Generate an answer given a context passage and a query.

        Parameters
        ----------
        context : str
            The (possibly compressed) context to condition on.
        query : str
            The question to answer.

        Returns
        -------
        str
            The model's generated answer string.
        """
        self._load()

        prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer concisely based only on the context above:\n"
        )

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (skip the prompt)
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    evaluator = QwenEvaluator(load_on_init=True)
    ans = evaluator.answer("The sky is blue.", "What colour is the sky?")
    print("Evaluator ready. Answer:", ans)
