"""Optional semantic retrieval with a small local embedding model (fastembed, ONNX, no API key).

If fastembed or the model is unavailable, `Semantic.load()` returns None and the system runs on lexical
retrieval alone. Embeddings are computed at ingest for each new unit, so answering only embeds the question.
"""
from __future__ import annotations

import os
from pathlib import Path

MODEL = os.environ.get("KB_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
CACHE = Path(__file__).resolve().parent.parent / ".cache" / "fastembed"


class Semantic:
    def __init__(self, model, np):
        self.model, self.np = model, np
        self.vecs = None           # (n_units, dim) float32, row i = unit i
        self.done = 0

    @classmethod
    def load(cls):
        # Off by default: on the practice set it lowered accuracy (0.657 vs ~0.73 lexical) and made ingest
        # take ~25 min on a laptop CPU, too close to the 30-minute limit. Opt in with KB_EMBED=1.
        if os.environ.get("KB_EMBED") != "1" or os.environ.get("KB_NO_EMBED"):
            return None
        try:
            import numpy as np
            from fastembed import TextEmbedding
            CACHE.mkdir(parents=True, exist_ok=True)
            return cls(TextEmbedding(MODEL, cache_dir=str(CACHE)), np)
        except Exception:
            return None

    def update(self, store) -> None:
        """Embed units added since the last call."""
        new = store.units[self.done:]
        if not new:
            return
        texts = [(f"{r.title}. {r.u.ctx}. " if r.u.ctx else f"{r.title}. ") + r.u.text[:600] for r in new]
        uniq = {t: k for k, t in enumerate(dict.fromkeys(texts))}
        emb = self.np.asarray(list(self.model.embed(list(uniq), batch_size=128)), dtype=self.np.float32)
        emb /= self.np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
        block = emb[[uniq[t] for t in texts]]
        self.vecs = block if self.vecs is None else self.np.vstack([self.vecs, block])
        self.done = len(store.units)
        self._members = {}

    def search(self, store, question: str, snap: int, k: int = 120) -> dict[int, float]:
        if self.vecs is None:
            return {}
        np = self.np
        members = getattr(self, "_members", {})
        if snap not in members:
            members[snap] = np.fromiter(sorted(store.snap_units[snap]), dtype=np.int64)
            self._members = members
        ids = members[snap]
        q = np.asarray(list(self.model.query_embed([QUERY_PREFIX + question])), dtype=np.float32)[0]
        q /= np.linalg.norm(q) + 1e-9
        sims = self.vecs[ids] @ q
        top = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
        best = float(sims[top].max()) if len(top) else 1.0
        # rescale so the best match is ~1 and weak matches fall to 0
        return {int(ids[j]): max(0.0, (float(sims[j]) - (best - 0.25)) / 0.25) for j in top}
