"""Temporal store: every unit from every snapshot, deduplicated, with an inverted index for BM25."""
from __future__ import annotations

import datetime as dt
import hashlib
import math
import re
from collections import defaultdict

from .pages import AUTHORITY, ParsedPage, Unit, canon_url, parse_page
from .text import tokens


class UnitRec:
    __slots__ = ("u", "canon", "url", "title", "kind", "pub", "first", "snaps", "toks", "page_toks", "faq_toks",
                 "place_toks")

    def __init__(self, u: Unit, page: ParsedPage, first: int):
        self.u, self.canon, self.url, self.title = u, page.canon, page.url, page.title
        self.kind, self.pub, self.first = page.kind, page.pub, first
        self.snaps: set[int] = set()
        self.toks: set[str] = set()
        self.page_toks: set[str] = set()
        self.place_toks: set[str] = set()      # the press release's dateline town: for name checks, not ranking
        self.faq_toks: set[str] = set()


DATELINE_RX = re.compile(r"\b([A-Z][A-Z.'-]+(?: [A-Z][A-Z.'-]+){0,2}),\s+(?:Iowa|Ill\.?|Illinois|Wis\.?|Wisconsin|"
                         r"Minn\.?|IA|IL|WI)\b")


def dateline(md: str) -> str:
    """The place a press release is datelined ("CEDAR FALLS, Iowa-"): the page is about that town even when
    its title isn't ("celebrates completion of newest learning lab")."""
    m = DATELINE_RX.search(md[:4000])
    return m.group(1).title() if m else ""


class Store:
    def __init__(self):
        self.snaps: list[tuple[str, dt.date]] = []       # (snapshot_id, taken_on), in ingest order
        self.snap_pages: list[dict[str, ParsedPage]] = []
        self.snap_units: list[set[int]] = []
        self.units: list[UnitRec] = []
        self.uid: dict[tuple, int] = {}
        self.parsed: dict[str, ParsedPage] = {}          # content hash -> parsed page
        self.post: dict[str, dict[int, int]] = defaultdict(dict)
        self.dlen: list[int] = []
        self.repeat_snap: list[dict] = []
        self.df_snap: list[dict[str, int]] = []

    # ---- ingest ---------------------------------------------------------------------------------------
    def add_snapshot(self, snapshot_id: str, taken_on: str, pages) -> None:
        if any(s == snapshot_id for s, _ in self.snaps):
            return                                       # idempotent
        day = dt.date.fromisoformat(str(taken_on)[:10])
        idx = len(self.snaps)
        self.snaps.append((snapshot_id, day))
        by_canon: dict[str, ParsedPage] = {}
        members: set[int] = set()
        for p in pages:
            canon = canon_url(p.url)
            if canon in by_canon:
                continue                                 # query-string variants of one page
            h = hashlib.sha1((p.url + "\0" + (p.title or "") + "\0" + p.markdown).encode()).hexdigest()
            pp = self.parsed.get(h)
            if pp is None:
                pp = parse_page(p.url, p.title, p.markdown)
                self.parsed[h] = pp
            by_canon[canon] = pp
            page_toks = set(tokens(pp.title + " " + canon.replace("/", " ").replace("-", " ")))
            place_toks = set(tokens(dateline(p.markdown))) - page_toks
            for u in pp.units:
                key = (pp.canon, u.kind, u.text, u.ctx, u.header, u.eff)   # same row, another effective date: new fact
                i = self.uid.get(key)
                if i is None:
                    i = len(self.units)
                    self.uid[key] = i
                    rec = UnitRec(u, pp, idx)
                    rec.page_toks = page_toks
                    rec.place_toks = place_toks
                    self.units.append(rec)
                    self._index(i, rec, pp)
                self.units[i].snaps.add(idx)
                members.add(i)
        self.snap_pages.append(by_canon)
        self.snap_units.append(members)
        df: dict[str, int] = defaultdict(int)
        for i in members:
            for t in self.units[i].toks:
                df[t] += 1
        self.df_snap.append(df)
        # how many pages carry each exact text: site-wide boilerplate ("About us: N branches ...") says little
        # about any one topic
        rep: dict[str, int] = defaultdict(int)
        for i in members:
            r = self.units[i]
            if r.u.kind == "text":
                rep[r.u.text] += 1
        self.repeat_snap.append(rep)

    def _index(self, i: int, rec: UnitRec, pp: ParsedPage) -> None:
        u = rec.u
        faq = tokens(u.faq) if u.faq else []
        rec.faq_toks = set(faq)
        body = tokens(u.text) + tokens(u.ctx) + tokens(" ".join(u.header)) + tokens(u.near) + faq
        title = tokens(pp.title)
        toks = body + title
        rec.toks = set(toks)
        tf: dict[str, int] = defaultdict(int)
        for t in toks:
            tf[t] += 1
        for t, c in tf.items():
            self.post[t][i] = c
        self.dlen.append(len(toks))

    def site_name(self) -> str:
        """The organisation's own name, taken from the most common site domain (dupaco.com -> 'dupaco')."""
        if getattr(self, "_site", None) is None:
            from collections import Counter
            c = Counter(canon.split("/")[0].split(".")[-2] for pages in self.snap_pages for canon in pages
                        if canon.split("/")[0].count(".") >= 1)
            self._site = c.most_common(1)[0][0].lower() if c else ""
        return self._site

    # ---- helpers -------------------------------------------------------------------------------------
    def snapshot_for(self, day: dt.date | None) -> int:
        if not self.snaps:
            return -1
        order = sorted(range(len(self.snaps)), key=lambda k: self.snaps[k][1])
        if day is None:
            return order[-1]
        best = order[0]
        for k in order:
            if self.snaps[k][1] <= day:
                best = k
        return best

    def authority(self, rec: UnitRec, snap: int) -> float:
        a = AUTHORITY.get(rec.kind, 0.5)
        if rec.kind in ("article", "file", "hub", "listing") and rec.pub:
            age = max(0.0, (self.snaps[snap][1] - rec.pub).days / 365.0)
            a *= 0.6 + 0.4 * math.exp(-age / 2.0)
        return a

    def bm25(self, qtoks: dict[str, float], snap: int, k1: float = 1.2, b: float = 0.6) -> dict[int, float]:
        members = self.snap_units[snap]
        n = len(members) or 1
        df = self.df_snap[snap]
        avg = (sum(self.dlen) / len(self.dlen)) if self.dlen else 1.0
        scores: dict[int, float] = defaultdict(float)
        for t, w in qtoks.items():
            d = df.get(t, 0)
            if not d or d > 0.25 * n:
                continue
            idf = math.log(1 + (n - d + 0.5) / (d + 0.5))
            for i, tf in self.post[t].items():
                if i in members:
                    dl = self.dlen[i]
                    scores[i] += w * idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avg))
        return scores

    @staticmethod
    def norm_label(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", s.lower().replace(",", "")).strip()

    def newest_same_row(self, rec: UnitRec, label: str, snap: int):
        """Another authoritative row with the same label and a newer effective date (the current version)."""
        idx = self._label_index(snap)
        best = None
        for j in idx.get(self.norm_label(label), ()):
            r = self.units[j]
            if r.canon == rec.canon or not r.u.eff or AUTHORITY.get(r.kind, 0) < 0.6:
                continue
            if rec.u.eff is None or r.u.eff > rec.u.eff:
                if best is None or r.u.eff > best.u.eff:
                    best = r
        return best

    def _label_index(self, snap: int) -> dict:
        cache = getattr(self, "_label_cache", {})
        if snap not in cache:
            from .answer import row_label
            idx: dict = defaultdict(list)
            for i in sorted(self.snap_units[snap]):
                if self.units[i].u.kind == "row":
                    idx[self.norm_label(row_label(self.units[i].u))].append(i)
            cache[snap] = idx
            self._label_cache = cache
        return cache[snap]

    def fuzzy_known(self, t: str, snap: int) -> bool:
        """Is a near-spelling of t on the site (typo tolerance for names)?"""
        if len(t) < 5:
            return False
        vocab = self._vocab(snap)
        return any(self._close(t, w) for w in vocab.get(t[:2], ()))

    def fuzzy_in(self, t: str, hay: set) -> bool:
        return len(t) >= 5 and any(self._close(t, w) for w in hay if w[:2] == t[:2])

    @staticmethod
    def _close(a: str, b: str) -> bool:
        try:
            from rapidfuzz.distance import Levenshtein
            return Levenshtein.distance(a, b, score_cutoff=1 if len(a) < 8 else 2) <= (1 if len(a) < 8 else 2)
        except ImportError:  # standard-library fallback
            import difflib
            return difflib.SequenceMatcher(None, a, b).ratio() >= (0.86 if len(a) < 8 else 0.8)

    def family(self, t: str, snap: int) -> list:
        """Site words sharing t's first 6 letters (recognition ~ recognized, eligible ~ eligibility):
        a cheap stand-in for a full stemmer. Only for longer words, at most 6."""
        if len(t) < 7 or not t.isalpha():
            return []
        cache = getattr(self, "_fam_cache", {})
        key = (t, snap)
        if key not in cache:
            stem6 = t[:6]
            cache[key] = [w for w in self._vocab(snap).get(t[:2], ()) if w != t and w[:6] == stem6][:6]
            self._fam_cache = cache
        return cache[key]

    def _vocab(self, snap: int) -> dict:
        cache = getattr(self, "_vocab_cache", {})
        if snap not in cache:
            by: dict = defaultdict(list)
            for w in self.df_snap[snap]:
                if len(w) >= 4 and w.isalpha():
                    by[w[:2]].append(w)
            cache[snap] = by
            self._vocab_cache = cache
        return cache[snap]

    def idf(self, t: str, snap: int) -> float:
        n = len(self.snap_units[snap]) or 1
        d = self.df_snap[snap].get(t, 0)
        return math.log(1 + (n - d + 0.5) / (d + 0.5))
