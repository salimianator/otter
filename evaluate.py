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

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent / "src"))
from device import get_device  # noqa: E402


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
        self.device     = get_device()           # MPS / CUDA / CPU
        self._tokenizer: AutoTokenizer | None = None
        self._model: AutoModelForCausalLM | None = None

        if load_on_init:
            self._load()

    def _load(self) -> None:
        """Load tokenizer and model onto self.device if not already in memory."""
        if self._model is not None:
            return

        # float16 halves memory and is faster on both MPS and CUDA.
        # device_map=None lets us call .to(device) ourselves, which avoids
        # accelerate's disk-offload path that fails on MPS with bfloat16.
        dtype = torch.float16 if self.device.type in ("mps", "cuda") else torch.float32
        print(f"Loading {self.model_name} … (device={self.device}, dtype={dtype})")

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=dtype,         # 'dtype' is the non-deprecated kwarg
            device_map=None,     # load to CPU first, then move to target device
        )
        self._model = self._model.to(self.device)
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

        # Move every input tensor explicitly to the target device
        raw_inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in raw_inputs.items()}

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                temperature=None,   # must be None when do_sample=False
                top_p=None,         # must be None when do_sample=False
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
