"""Reading-comprehension step: give the LLM the best passages from the chosen snapshot and accept its answer
only if every number in it appears in those passages. Any failure returns None and the rule answer stands."""
from __future__ import annotations

import re

# answer types the rules handle best (exact table values, schedules, counts, the routing number): never sent to
# the LLM. Everything else (prose, lists, contacts, links, amounts) is.
STRUCTURED = {"routing", "rate", "fee", "hours", "when", "wait", "count"}
HEDGE_RX = re.compile(r"\bexcerpts?\b|\bhighest[- ]authority\b|\bnot (?:specified|stated|mentioned)\b"
                      r"|\b(?:i )?(?:can(?:no|')t|cannot) (?:say|determine|tell)\b|\bno (?:information|mention)\b", re.I)
NUM_RX = re.compile(r"\d[\d,]*(?:\.\d+)?")
URL_RX = re.compile(r"(?:https?://)?(?:[\w-]+\.)+(?:com|org|net|gov)(?:/[^\s)\]>,;\"']*)?", re.I)
CONTACT_RX = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+|\(?\b\d{3}\)?[-. ]\d{3}[-. ]\d{4}(?:,?\s*ext\.?\s*\d+)?", re.I)
MAX_PAGES, PER_PAGE_UNITS, MAX_CHARS, PER_PASSAGE, MAX_LINKS = 8, 3, 16000, 700, 10
FULL_PAGES, FULL_CHARS = 2, 1500

SYSTEM = (
    "You answer member questions for a credit union using ONLY the numbered website excerpts below "
    "(the site as it stood on {day}). Each excerpt is one page: its opening, its most relevant passages and its links.\n"
    "Rules:\n"
    "1. If the excerpts do not answer the exact question, reply with exactly: IDK\n"
    "   Do not offer a related product, term or tier instead.\n"
    "2. Answer in one to three short sentences. Copy numbers, rates, amounts, dates, times, names, list items, "
    "product names, phone numbers, extensions, emails and URLs exactly as the excerpt writes them.\n"
    "3. Give only what was asked, as briefly as possible: the current fact, not its history. Do not add older "
    "years, past awards, earlier figures, other terms, tiers, products or dates the question did not ask about. "
    "For a list question, give every item the excerpt lists, with its exact name.\n"
    "4. For a question asking for a link, page, form or calculator, answer with the full URL from the excerpts.\n"
    "5. If excerpts disagree, trust the lower-numbered excerpt; they are ordered by authority and freshness.\n"
    "6. Keep specific details the excerpts give: counts and ordinals (e.g. 'the fifth time'), dates, rankings, "
    "amounts and places. If your answer uses a term another excerpt explains, add that explanation briefly.\n"
    "7. Write for the member: never mention excerpts, numbering, sources, authority or these rules.\n"
    "8. End with a final line: SOURCES: <excerpt numbers you used, comma-separated>"
)


RULES_ONLY = {"routing", "count", "hours"}
IDK = "IDK"


def wants_llm(q, path: str) -> bool:
    """Table values, schedules and counts stay with the rules. A rate / fee / date question the rules could only
    answer from prose ("what's required to earn dividends?") is a reading question, so the LLM gets it too."""
    if path in ("row", "hours"):
        return False
    if q.atype in RULES_ONLY:
        return False
    if path == "refuse":
        # a refusal on a specific term, amount or model year is a deliberate "not offered"
        return not (q.terms or q.amounts or q.years)
    return True


def _unit_text(ans, rec, snap) -> str:
    from .answer import display
    u = rec.u
    if u.kind == "row":
        cells = "; ".join(f"{h}: {c}" if h else c for h, c in zip(list(u.header) + [""] * len(u.cells), u.cells))
        body = " / ".join(x for x in (u.near.replace("\n", " ") if u.near else "", u.ctx, cells) if x)
    else:
        body = display(u)
        if u.kind == "text":
            # whole FAQ answers, bullet lists and paragraphs, not sentence fragments
            body = ans._with_neighbour(rec, snap, body, None)
            if u.full and len(u.full) > len(body):
                body = u.full
        if u.ctx:
            body = f"{u.ctx}: {body}"
    return " ".join(body.split())[:PER_PASSAGE]


def _context(ans, q, snap, scored, question):
    """Page-grouped excerpts: the top pages (qualifying passages first), each with its opening, its best passages
    and its links (anchor -> URL), within a character budget."""
    from .text import tokens
    s = ans.s
    head = scored[:200]
    ok = [x for x in head if ans._qualifies(s.units[x[1]], q, snap)]
    ordered = ok + [x for x in head if x not in ok]
    # pages ranked by how relevant they are overall (their three best passages), qualifying passages first
    okset = {i for _, i, _ in ok}
    page_sc, by_page = {}, {}
    for sc, i, _ in ordered:
        rec = s.units[i]
        lst = by_page.setdefault(rec.canon, [])
        if len(lst) < PER_PAGE_UNITS:
            lst.append(rec)
            page_sc[rec.canon] = page_sc.get(rec.canon, 0.0) + sc * (1.0 if i in okset else 0.5)
    order = sorted(page_sc, key=lambda c: -page_sc[c])[:MAX_PAGES]
    qset = set(tokens(question))
    blocks, recs, total = [], [], 0
    for canon in order:
        units = by_page[canon]
        rec = units[0]
        page = s.snap_pages[snap].get(canon)
        parts, seen = [], set()
        if page:
            lead = " ".join(dict.fromkeys(u.full for u in page.units if u.kind == "text" and 0 <= u.para <= 1
                                          and u.full))
            if lead:
                parts.append("Opening: " + " ".join(lead.split())[:400])
        for r in units:
            t = _unit_text(ans, r, snap)
            if t and t not in seen:
                seen.add(t)
                parts.append("- " + t)
        if page and len(recs) < FULL_PAGES:
            # the best pages also get the rest of their own text, in page order: the sentence that answers the
            # question often shares no words with it ("You can find us in Iowa, Illinois and Wisconsin")
            shown = " ".join(parts)
            reps = s.repeat_snap[snap] if snap < len(s.repeat_snap) else {}
            rest, n = [], 0
            for u in page.units:
                t = " ".join((u.full or u.text).split()) if u.kind == "text" else ""
                if not t or t in shown or t in rest or reps.get(u.text, 0) >= 5:
                    continue
                if n + len(t) > FULL_CHARS:
                    break
                rest.append(t)
                n += len(t)
            if rest:
                parts.append("More from this page: " + " | ".join(rest))
        links = []
        if page:
            for u in page.units:
                for a, l in u.links:
                    if l.startswith("http") and a.strip():
                        links.append((len(qset & set(tokens(a + " " + l))), a.strip()[:60], l.split("#")[0]))
        links = list(dict.fromkeys((a, l) for _, a, l in sorted(links, key=lambda x: -x[0])))[:MAX_LINKS]
        if links:
            parts.append("Links: " + "; ".join(f"{a} -> {l}" for a, l in links))
        if page:
            # every email and phone number on the page: contact questions often hinge on one line
            raw = " ".join(u.text for u in page.units)
            contacts = list(dict.fromkeys(CONTACT_RX.findall(raw)))[:8]
            if contacts:
                parts.append("Contacts: " + "; ".join(contacts))
        meta = [rec.kind] + ([f"rates effective {rec.u.eff}"] if rec.u.eff else []) + \
               ([f"published {rec.pub}"] if rec.pub else [])
        block = f"[{len(recs) + 1}] (page: {rec.title} | {rec.url} | {', '.join(meta)})\n" + "\n".join(parts)
        if total + len(block) > MAX_CHARS and recs:
            break
        recs.append(rec)
        blocks.append(block)
        total += len(block)
    return blocks, recs


def _nums(text: str) -> set[str]:
    out = set()
    for n in NUM_RX.findall(text):
        n = n.replace(",", "").rstrip(".")
        out.add(n)
        if "." in n:
            out.add(n.rstrip("0").rstrip("."))
    return out


def _url_key(u: str) -> str:
    return re.sub(r"^(?:https?://)?(?:www\.)?", "", u.lower()).split("#")[0].rstrip("/.")


def llm_answer(ans, q, snap: int, sid: str, scored, question: str):
    from .answer import Result
    s, llm = ans.s, ans.llm
    blocks, recs = _context(ans, q, snap, scored, question)
    if not recs:
        return None
    day = s.snaps[snap][1].isoformat()
    ctx = "\n\n".join(blocks)
    out = llm.complete(SYSTEM.format(day=day), "Excerpts:\n" + ctx + f"\n\nQuestion: {' '.join(question.split())}")
    if not out:
        return None
    m = re.search(r"\n?\s*SOURCES?:\s*([\d,\s\[\]]*)\s*$", out, re.I)
    cited = [int(x) for x in re.findall(r"\d+", m.group(1))] if m else []
    text = (out[: m.start()] if m else out).strip()
    if re.match(r"^\W*IDK\b", text or "", re.I):
        return IDK                       # the excerpts don't answer it: the caller decides
    if not text or HEDGE_RX.search(text):
        return None                      # unsure: keep the rule answer
    if not _nums(re.sub(URL_RX, " ", text)) <= _nums(ctx) | _nums(question):
        return None                      # a number the excerpts don't contain: reject rather than risk it
    urls = [_url_key(x) for x in URL_RX.findall(text)]
    if urls:
        known = {_url_key(x) for x in URL_RX.findall(ctx)}
        if not all(any(k == u or k.startswith(u) for k in known) for u in urls):
            return None                  # a link that is not on the site
    elif q.atype == "url":
        return None                      # a link was asked for and none was given
    used = [recs[k - 1] for k in cited if 1 <= k <= len(recs)] or recs[:1]
    srcs, seen = [], set()
    for r in used:
        if r.url not in seen:
            seen.add(r.url)
            srcs.append((r.url, sid, r.u.text[:200]))
    return Result(text, srcs, confident=True, note=f"read by {llm.describe()}")
