"""conflicts(): real contradictions only, each resolved with the same rules answer() uses.

Detectors (all structural, no product names):
  1. temporal  - the same row of an authoritative table shows a different value in a later snapshot;
  2. stale     - in the latest snapshot a lower-authority or older-dated table disagrees with the newest
                 authoritative table on the same row (same term, same/overlapping tier);
  3. kv        - the same "Label: value" fact (routing number, phone, ...) differs between pages;
  4. counts    - "N branches/locations/..." claims differ between pages; the newest dated page wins.
Different tiers, terms, rate vs APY/APR, personal vs business are separate facts and never compared.
"""
from __future__ import annotations

import datetime as dt
import re
from collections import defaultdict

from .answer import row_columns, row_label
from .pages import AUTHORITY
from .quals import amount_range, terms_months, year_range

YEARS = re.compile(r"\b(?:19|20)\d{2}\s*(?:-\s*(?:19|20)?\d{2})?\b|\(and older\)|\(and newer\)", re.I)
COUNT_NOUNS = r"(branch(?:es)?|branch offices|branch locations|locations|offices|ATMs)"
COUNT_RE = re.compile(r"\b(\d{1,3}(?:,\d{3})*)\s+(?:(?:full-service|community|physical|branch)\s+)?" + COUNT_NOUNS + r"\b", re.I)


ORDINALS = "first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth"
ORDINAL_RE = re.compile(r"\bthe (" + ORDINALS + r") time ([A-Z][\w&.]+(?: [A-Z][\w&.]+)?) "
                        r"(?:has )?(?:recognized|named|ranked|honored|listed)\b")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower().replace(",", "")).strip()


def _val(cell: str) -> str:
    return re.sub(r"\s+", " ", cell.strip())


def _same_value(a: str, b: str) -> bool:
    na = re.findall(r"\d+(?:\.\d+)?", a.replace(",", ""))
    nb = re.findall(r"\d+(?:\.\d+)?", b.replace(",", ""))
    return [float(x) for x in na] == [float(x) for x in nb]


def _close(a: str, b: str, rel: float = 0.35) -> bool:
    """A stale copy is a near-miss of the current value; a very different number is a different product."""
    na = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", a.replace(",", ""))[:1]]
    nb = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", b.replace(",", ""))[:1]]
    return bool(na and nb) and abs(na[0] - nb[0]) <= rel * max(na[0], nb[0])


GENERIC = {"loan", "loans", "rate", "rates", "and", "the", "your", "our", "for", "with", "low", "interest", "get",
           "on", "road", "a", "of", "to", "dupaco", "special", "specials"}


def _heading_words(rec) -> set:
    segs = rec.u.ctx.split(" > ")[-2:] if rec.u.ctx else [rec.title or ""]
    return {w.rstrip("s") for w in re.findall(r"[a-z]+", " ".join(segs).lower())} - GENERIC


def _same_product(a, b) -> bool:
    """Both tables sit under a heading naming the same product (auto vs auto, not ATV vs auto)."""
    return bool(_heading_words(a) & _heading_words(b))


def _short(label: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"^(?:Purchase|Refinance):\s*", "", label)).strip()


def _metric_cells(u):
    for c, m, h in zip(u.cells, row_columns(u), u.header):
        if m in ("rate", "apy", "apr", "fee") and c and c not in ("-", "—") and re.search(r"\d", c):
            # the cell's own unit beats a mislabelled header ("Annual Percentage Yield" over "5.34% APR")
            unit = re.search(r"\b(APR|APY)\b", c)
            yield (unit.group(1).lower() if unit else m), h, _val(c)


def _topic(rec, u, metric: str) -> str:
    ctx = rec.u.ctx.split(" > ")[-1] if rec.u.ctx else rec.title
    label = row_label(u)
    words = {"apy": "APY", "apr": "APR", "rate": "rate", "fee": "fee"}[metric]
    base = label if ctx.lower() in label.lower() else f"{ctx} {label}"
    return f"{base} {words}".strip()


def find_conflicts(store, limit: int = 50):
    if not store.snaps:
        return []
    order = sorted(range(len(store.snaps)), key=lambda k: store.snaps[k][1])
    latest = order[-1]
    sid = lambda k: store.snaps[k][0]  # noqa: E731
    out = []

    # ---- 1. temporal: same page, same row, same metric, value changed across snapshots -------------------
    hist = defaultdict(dict)        # (canon, ctx, label, header) -> {snap: (value, rec)}
    for rec in store.units:
        if rec.u.kind != "row" or AUTHORITY.get(rec.kind, 0) < 0.6:
            continue
        lab = _norm(row_label(rec.u))
        for m, h, v in _metric_cells(rec.u):
            for k in rec.snaps:
                hist[(rec.canon, _norm(rec.u.ctx), lab, h, m)][k] = (v, rec)
    merged = {}
    for (canon, ctx, lab, h, m), by in hist.items():
        seq = [(k, by[k]) for k in order if k in by]
        vals = []
        for k, (v, rec) in seq:
            if not vals or not _same_value(vals[-1][1], v):
                vals.append((k, v, rec))
        if len(vals) < 2 or latest not in by:
            continue
        cur_k, cur_v, cur_rec = vals[-1]
        # one fact per row label: the same row on several tables (Roth / Traditional IRA copies) and its
        # rate / APY / APR columns are one change, reported once with every value it showed
        key = re.sub(r"^[^:]{1,40}:\s*", "", row_label(cur_rec.u)).lower()   # "Roth: 10 Month ..." -> "10 month ..."
        key = _norm(key)
        ent = merged.setdefault(key, {"rows": {}, "sources": [], "rec": cur_rec, "k": cur_k, "res": {},
                                      "label": re.sub(r"^[^:]{1,40}:\s*", "", row_label(cur_rec.u))})
        ent["rows"].setdefault(m, [])
        for _, v, _ in vals:
            if v not in ent["rows"][m]:
                ent["rows"][m].append(v)
        ent["res"].setdefault(m, cur_v)
        for k, v, rec in vals:
            src = (rec.url, sid(k), f"{row_label(rec.u)}: {v}")
            if src not in ent["sources"]:
                ent["sources"].append(src)
    for lab, ent in merged.items():
        ms = sorted(ent["rows"], key=lambda m: ["apy", "apr", "rate", "fee"].index(m))
        main = ms[0]
        rec = ent["rec"]
        segs = [x.strip() for x in (rec.u.ctx or rec.title or "").split(" > ") if x.strip()]
        heading = (segs[1] if len(segs) > 1 else segs[0]) if segs else ""
        out.append({
            "topic": f"{ent['label']} {main.upper()} ({heading})",
            "values": [v for m in ms for v in ent["rows"][m]],
            "sources": ent["sources"][:8], "resolution": ent["res"][main],
            "reason": f"value changed between crawls; the latest crawl ({sid(ent['k'])}) shows "
                      + ", ".join(f"{ent['res'][m]} ({m.upper()})" for m in ms),
            "score": 2.0 + AUTHORITY.get(rec.kind, 0)})

    # ---- 2. stale copies within the latest snapshot ------------------------------------------------------
    rows = [store.units[i] for i in sorted(store.snap_units[latest]) if store.units[i].u.kind == "row"]
    def strength(r):
        eff = r.u.eff or dt.date.min
        return (eff, AUTHORITY.get(r.kind, 0))
    by_term = defaultdict(list)     # (metric, terms, amount-range) -> rows
    for r in rows:
        lab = row_label(r.u)
        terms = tuple(sorted(terms_months(lab)))
        ar = amount_range(lab)
        for m, h, v in _metric_cells(r.u):
            by_term[(m, terms, ar)].append((r, v, lab))
    stale = {}
    for (m, terms, ar), items in by_term.items():
        if not terms or len(items) < 2:
            continue
        auth = [x for x in items if x[0].kind in ("rates", "product") and x[0].u.eff]
        if not auth:
            continue
        for r, v, lab in items:
            core = _norm(YEARS.sub(" ", lab))
            yr = year_range(lab)
            peers = [a for a in auth if a[0].canon != r.canon and _norm(YEARS.sub(" ", a[2])) == core
                     and strength(a[0]) > strength(r) and _same_product(r, a[0])]
            if yr:
                def overlap(a):
                    y2 = year_range(a[2])
                    return 0 if not y2 else max(0, min(yr[1], y2[1]) - max(yr[0], y2[0]) + 1)
                peers = [a for a in peers if overlap(a) > 0]
                peers.sort(key=lambda a: (-overlap(a), a[0].url))
            else:
                peers = [a for a in peers if _norm(a[2]) == _norm(lab)]
            if not peers:
                continue
            win, wv, wlab = peers[0]
            if _same_value(v, wv) or not _close(v, wv):
                continue
            fam = stale.setdefault((r.canon, win.canon), {"pairs": [], "r": r, "win": win, "why": set()})
            if (v, wv) not in [(a, b) for a, b, _, _ in fam["pairs"]]:
                fam["pairs"].append((v, wv, lab, wlab))
            if r.kind in ("calculator", "promo", "article", "file", "listing", "expired"):
                fam["why"].add(f"{r.kind} page")
            if r.u.eff and win.u.eff and r.u.eff < win.u.eff:
                fam["why"].add(f"rates effective {r.u.eff.isoformat()} vs {win.u.eff.isoformat()}")
    # one entry per stale page vs the current table: the whole stale copy is one contradiction
    for (rc, wc), fam in sorted(stale.items(), key=lambda kv: kv[0]):
        r, win, pairs = fam["r"], fam["win"], fam["pairs"]
        heading = (win.u.ctx.split(" > ")[-1] if win.u.ctx else win.title).strip()
        labels = list(dict.fromkeys(_short(p[3]) for p in pairs))
        out.append({
            "topic": f"{heading} rates ({'; '.join(labels[:6])})",
            "values": list(dict.fromkeys(x for v, wv, _, _ in pairs for x in (v, wv))),
            "sources": [(r.url, sid(latest), f"{lab}: {v}") for v, _, lab, _ in pairs[:4]]
                       + [(win.url, sid(latest), f"{wlab}: {wv}") for _, wv, _, wlab in pairs[:4]],
            "resolution": "; ".join(f"{wv} ({_short(wlab)})" for _, wv, _, wlab in pairs),
            "reason": f"{win.title or win.url} is the authoritative, most recently effective table; "
                      f"{r.title or r.url} is a stale copy ({', '.join(sorted(fam['why'])) or 'older'})",
            "score": 2.5 + AUTHORITY.get(win.kind, 0) - AUTHORITY.get(r.kind, 0)})

    # ---- 3. label/value facts that disagree between pages ------------------------------------------------
    kv = defaultdict(list)
    for i in sorted(store.snap_units[latest]):
        r = store.units[i]
        if r.u.kind == "kv":
            kv[_norm(r.u.cells[0])].append(r)
    for lab, recs in kv.items():
        vals = {}
        for r in recs:
            v = r.u.cells[1].strip()
            if re.search(r"\d", v):
                vals.setdefault(_norm(v), (v, r))
        if 1 < len(vals) <= 4 and re.search(r"number|phone|routing|rate|fee|hours|address", lab):
            ranked = sorted(vals.values(), key=lambda x: (-store.authority(x[1], latest), x[1].url))
            out.append({"topic": recs[0].u.cells[0], "values": [v for v, _ in ranked],
                        "sources": [(r.url, sid(latest), f"{recs[0].u.cells[0]}: {v}") for v, r in ranked],
                        "resolution": ranked[0][0],
                        "reason": "taken from the most authoritative page that states it",
                        "score": 1.5})

    # ---- 4. count claims ---------------------------------------------------------------------------------
    claims = defaultdict(list)
    for i in sorted(store.snap_units[latest]):
        r = store.units[i]
        if r.u.kind != "text":
            continue
        for m in COUNT_RE.finditer(r.u.text):
            n = int(m.group(1).replace(",", ""))
            noun = m.group(2).lower().rstrip("s").replace("branche", "branch").replace("branch office", "branch")
            if 2 <= n <= 500:
                claims[noun].append((n, r, m.group(0)))
    for noun, items in claims.items():
        nums = {n for n, _, _ in items}
        if len(nums) < 2:
            continue
        newest = max(items, key=lambda x: (x[1].pub or dt.date.min, store.authority(x[1], latest)))
        by_n = {}
        for n, r, txt in sorted(items, key=lambda x: (-(x[1].pub or dt.date.min).toordinal(), x[1].url)):
            by_n.setdefault(n, (r, txt))
        vals = [f"{n} {noun}es" if noun == "branch" else f"{n} {noun}s" for n in by_n]
        out.append({"topic": f"number of {noun}es" if noun == "branch" else f"number of {noun}s",
                    "values": vals,
                    "sources": [(r.url, sid(latest), txt) for r, txt in by_n.values()],
                    "resolution": vals[list(by_n).index(newest[0])],
                    "reason": f"most recently published statement ({newest[1].pub}) wins over older press and blog posts",
                    "score": 1.0})

    # ---- 5. award counts: "the fifth time Forbes recognized ..." vs "the fourth time ..." ----------------
    award = defaultdict(list)
    for i in sorted(store.snap_units[latest]):
        r = store.units[i]
        if r.u.kind != "text":
            continue
        for m in ORDINAL_RE.finditer(r.u.text):
            award[m.group(2)].append((m.group(1).lower(), r, m.group(0)))
    for org, items in sorted(award.items()):
        if len({o for o, _, _ in items}) < 2:
            continue
        items.sort(key=lambda x: (-(x[1].pub or dt.date.min).toordinal(), x[1].url))
        vals = list(dict.fromkeys(o for o, _, _ in items))
        out.append({"topic": f"{org} recognition count (how many times {org} recognized Dupaco)",
                    "values": vals, "sources": [(r.url, sid(latest), txt) for _, r, txt in items[:4]],
                    "resolution": vals[0], "kind": "award", "org": org, "quote": items[0][2],
                    "reason": f"the most recently published release ({items[0][1].pub}) gives the latest count",
                    "score": 1.0})

    out.sort(key=lambda c: (-c["score"], c["topic"]))
    for c in out:
        c.pop("score", None)
        c.setdefault("kind", "value")
    return out[:limit]
