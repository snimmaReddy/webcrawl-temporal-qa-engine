"""Optional local cross-encoder re-ranker (fastembed, ONNX, CPU, no API key).

Re-scores only the top candidates of the lexical ranking at question time, so ingest is unaffected.
If fastembed or the model is unavailable, `Reranker.load()` returns None and ranking stays lexical.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

MODEL = os.environ.get("KB_RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
CACHE = Path(__file__).resolve().parent.parent / ".cache" / "fastembed"
TOP_K = int(os.environ.get("KB_RERANK_K", "30"))
WEIGHT = float(os.environ.get("KB_RERANK_W", "0.8"))


class Reranker:
    def __init__(self, model):
        self.model = model
        self.k, self.w = TOP_K, WEIGHT

    @classmethod
    def load(cls):
        if os.environ.get("KB_RERANK", "0") != "1":
            return None
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
            CACHE.mkdir(parents=True, exist_ok=True)
            m = cls(TextCrossEncoder(MODEL, cache_dir=str(CACHE)))
            list(m.model.rerank("warm up", ["warm up"]))       # load the ONNX session now, not on question 1
            return m
        except Exception:
            return None

    def logits(self, question: str, passages: list) -> list:
        try:
            return [float(z) for z in self.model.rerank(question, [p[:512] for p in passages])]
        except Exception:
            return []

    def apply(self, question: str, scored: list, passage) -> list:
        """scored: [(score, uid, cov)] sorted best first; passage(uid) -> text. Returns re-sorted list."""
        head = scored[: self.k]
        if len(head) < 2:
            return scored
        try:
            logits = list(self.model.rerank(question, [passage(i)[:512] for _, i, _ in head]))
        except Exception:
            return scored
        new = [(sc * (1 - self.w / 2 + self.w / (1 + math.exp(-z))), i, cov)
               for (sc, i, cov), z in zip(head, logits)]
        out = new + scored[self.k:]
        out.sort(key=lambda x: (-x[0], x[1]))
        return out
