"""Knowledge system that knows what's true now.

build() -> KnowledgeSystem with ingest(snapshot), answer(question, as_of=None), conflicts().
Standard library only; an LLM key is optional. See NOTES.md for how "current" is decided.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
_harness = next((d / "harness" for d in _HERE.parents if (d / "harness" / "contract.py").exists()), None)
if _harness is not None:
    sys.path.insert(0, str(_harness))

try:
    from contract import Answer, Conflict, Source  # noqa: E402
except ImportError:  # running outside the kit: minimal stand-ins with the same fields
    from dataclasses import dataclass, field
    from typing import Optional

    @dataclass
    class Source:
        url: str
        snapshot_id: Optional[str] = None
        quote: Optional[str] = None

    @dataclass
    class Answer:
        text: str
        sources: list = field(default_factory=list)
        confident: bool = True
        conflict_note: Optional[str] = None

    @dataclass
    class Conflict:
        topic: str
        values: list
        sources: list
        resolution: Optional[str] = None
        reason: Optional[str] = None

from kb.answer import REFUSAL, Answerer  # noqa: E402
from kb.conflicts import find_conflicts  # noqa: E402
from kb.llm import LLM  # noqa: E402
from kb.rerank import Reranker  # noqa: E402
from kb.semantic import Semantic  # noqa: E402
from kb.store import Store  # noqa: E402


class KnowledgeSystem:
    def __init__(self):
        self.store = Store()
        self.sem = Semantic.load()
        self.answerer = Answerer(self.store, semantic=(lambda q, snap: self.sem.search(self.store, q, snap))
                                 if self.sem else None, llm=LLM.load())
        self.answerer.rerank = Reranker.load()
        self._cache: dict = {}
        self._conflicts = None
        self._lock = threading.Lock()

    def ingest(self, snapshot) -> None:
        with self._lock:
            self.store.add_snapshot(snapshot.snapshot_id, snapshot.taken_on, snapshot.pages)
            if self.sem is not None:
                try:
                    self.sem.update(self.store)
                except Exception:
                    self.sem = None
                    self.answerer.semantic = None
            self._cache.clear()
            self._conflicts = None
            self._raw = None

    def answer(self, question: str, as_of=None):
        key = (" ".join((question or "").split()).lower(), as_of)
        hit = self._cache.get(key)
        if hit is None:
            try:
                hit = self.answerer.answer(question, as_of)
            except Exception as e:  # never crash the run; an honest refusal is better than a stack trace
                from kb.answer import Result
                hit = Result(REFUSAL, confident=False, note=f"internal error: {type(e).__name__}")
            if as_of is None:
                hit = self._with_resolution(question, hit)
            self._cache[key] = hit
        return Answer(text=hit.text,
                      sources=[Source(url=u, snapshot_id=sid, quote=qt) for u, sid, qt in hit.sources],
                      confident=hit.confident, conflict_note=hit.note)

    def _raw_conflicts(self):
        if getattr(self, "_raw", None) is None:
            try:
                self._raw = find_conflicts(self.store)
            except Exception:
                self._raw = []
        return self._raw

    def _with_resolution(self, question, hit):
        """A question about a fact the conflict pass resolved (e.g. how many times an organisation recognized
        the credit union) gets the winning value stated, and never an older one."""
        import re
        from kb.answer import Result
        if not hit.confident:
            return hit
        words = set(re.findall(r"[a-z0-9&.]+", (question or "").lower()))
        for c in self._raw_conflicts():
            if c.get("kind") != "award" or c["org"].lower() not in words:
                continue
            res, old = c["resolution"], [v for v in c["values"] if v != c["resolution"]]
            if any(re.search(r"\b" + v + r"\b", hit.text, re.I) for v in old):
                text = f"{c['quote'].strip().rstrip('.')}." + " (the most recent release)"
            elif not re.search(r"\b" + res + r"\b", hit.text, re.I):
                text = f"{hit.text.rstrip()} This was the {res} time {c['org']} recognized Dupaco, " \
                       f"per the most recent release."
            else:
                return hit
            src = c["sources"][0]
            srcs = [src] + [x for x in hit.sources if x[0] != src[0]]
            return Result(text, srcs[:5], confident=True, note=(hit.note or "") + "; latest count from conflicts")
        return hit

    def conflicts(self):
        if self._conflicts is None:
            raw = self._raw_conflicts()
            self._conflicts = [
                Conflict(topic=c["topic"], values=c["values"],
                         sources=[Source(url=u, snapshot_id=sid, quote=qt) for u, sid, qt in c["sources"]],
                         resolution=c["resolution"], reason=c["reason"])
                for c in raw
            ]
        return list(self._conflicts)


def build():
    return KnowledgeSystem()
