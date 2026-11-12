"""Page classification and parsing into answerable units (table rows, label/value pairs, text passages).

Everything here is driven by page structure (URL shape, title, headings, tables, dates), never by
specific product names, so it carries over to unseen pages and snapshots.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

from .quals import dates_in
from .text import clean, links, strip_footnotes, strip_markdown


def canon_url(u: str) -> str:
    p = urlparse(u if "//" in u else "https://" + u)
    host = p.netloc.lower()
    host = host[4:] if host.startswith("www.") else host
    return host + (p.path.rstrip("/").lower() or "")


FILE_EXT = (".pdf", ".xls", ".xlsx", ".csv", ".doc", ".docx", ".ppt", ".pptx")
ARTICLE_SEG = {"learn", "blog", "newsroom", "news", "press", "press-releases", "dupacodaily", "stories", "article", "articles"}
LISTING_SEG = {"tag", "author", "category", "page", "topics"}
PROMO_HINT = re.compile(r"(-lp$|offer|promo|special|switch|earn-more|bonus|deal|giveaway|challenge|sprint)", re.I)
EXPIRED_HINT = re.compile(r"\b(offer has expired|offer expired|no longer available|this offer has ended|page not found|404)\b", re.I)

AUTHORITY = {"rates": 1.0, "product": 0.9, "promo": 0.75, "hub": 0.6, "file": 0.6, "article": 0.5,
             "calculator": 0.3, "listing": 0.25, "expired": 0.05}


def classify(url: str, title: str, md: str) -> str:
    path = urlparse(url if "//" in url else "https://" + url).path.lower()
    segs = [s for s in path.split("/") if s]
    head = (title or "") + " " + md[:400]
    if EXPIRED_HINT.search(head):
        return "expired"
    if path.endswith(FILE_EXT) or "hubfs" in segs:
        return "file"
    if "calculator" in path or "calculator" in (title or "").lower():
        return "calculator"
    if any(s in LISTING_SEG for s in segs[:-1]):
        return "listing"
    if any(s in ARTICLE_SEG for s in segs) or (segs and re.fullmatch(r"(19|20)\d{2}", segs[0])):
        return "article" if len(segs) > 1 else "hub"
    if "resources" in segs:
        return "article" if segs[-1] != "resources" and len(segs) >= 2 and segs.index("resources") < len(segs) - 1 else "hub"
    if segs and PROMO_HINT.search(segs[-1]):
        return "promo"
    if any("rate" in s for s in segs):
        return "rates"
    return "product"


@dataclass
class Unit:
    kind: str              # "row" | "kv" | "text"
    text: str              # display text (clean, no markdown)
    ctx: str               # section labels above it
    header: tuple = ()     # table header cells (rows)
    cells: tuple = ()      # row cells (rows)
    table: int = -1        # table index within the page (rows)
    links: tuple = ()      # (anchor, url) pairs in the source lines
    eff: object = None     # "effective" date attached to the table, if any
    near: str = ""         # text just above the table (rows): names the product when headers don't
    faq: str = ""          # the FAQ question this text answers ("Can I use ... ?"), if any
    full: str = ""         # the whole paragraph a sentence piece came from
    para: int = -1         # order of prose paragraphs on the page (-1: not prose)


@dataclass
class ParsedPage:
    url: str
    canon: str
    title: str
    kind: str
    pub: object            # publish date for articles (first date near the top), else None
    units: list = field(default_factory=list)


TABLE_SEP = re.compile(r"^\|?\s*:?-{2,}")
HEADING = re.compile(r"^(#{1,6})\s+(.*)")
KV = re.compile(r"^\s*(?:[-*]\s*)?\*\*([^*]{2,60}?):?\*\*:?\s*(.+)$")
# a real question: starts with a question word and ends with "?" ("Ready to join?" is a slogan, not an FAQ)
FAQ_RX = re.compile(r"^(?:can|could|do|does|did|is|are|was|were|how|what|what's|when|where|who|whom|whose|why|will|would|"
                    r"should|which|may|am|have|has)\b.{6,}\?$", re.I)
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
# a period that ends an abbreviation, not a sentence: "1090 E. Hickman Road", "Oct. 1, 2024", "ext. 4123", "U.S. Department"
ABBREV_END = re.compile(r"(?:\b[A-Z]|\b(?:St|Ave|Blvd|Rd|Dr|Hwy|Mt|Ft|Pkwy|Jan|Feb|Mar|Apr|Aug|Sept?|Oct|Nov|Dec|ext|Ext"
                        r"|No|Inc|Co|Corp|Jr|Sr|Mr|Mrs|Ms|Dr|vs|approx|Wis|Ill|Minn|Gov|Sen|Rep|U\.S|e\.g|i\.e))\.$")


def split_sentences(text: str) -> list[str]:
    out = []
    for x in SENT_SPLIT.split(text):
        x = x.strip()
        if not x:
            continue
        if out and (ABBREV_END.search(out[-1]) or (re.search(r"\b[ap]\.m\.$", out[-1]) and x.startswith("("))):
            out[-1] = out[-1] + " " + x
        else:
            out.append(x)
    return out


NOISE = re.compile(r"^(skip to|menu|search|close|loading|go$|open my|see details|learn more|apply now|back to)", re.I)
HEAD_FACT = re.compile(r"\d\s*%|\$\s?\d|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.? \d{1,2}\b"
                       r"|\b\d{1,2}/\d{1,2}/\d{2,4}\b|^[^:]{3,40}:\s*\S.*\d", re.I)
EFFECTIVE = re.compile(r"effective|as of", re.I)
LINK_RE_FULL = re.compile(r"!?\[[^\]]*\]\([^)]*\)")


def _cells(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [strip_footnotes(clean(strip_markdown(c))).strip() for c in re.split(r"(?<!\\)\|", s)]


def _is_label(line: str) -> bool:
    s = strip_markdown(line).strip()
    return (0 < len(s) <= 70 and not s.endswith((".", ",", ";", ":")) and len(s.split()) <= 9
            and re.search(r"[A-Za-z]", s) is not None and not s.startswith(("[", "!", "|")))


def parse_page(url: str, title: str, md: str) -> ParsedPage:
    kind = classify(url, title, md)
    md = clean(md)
    pub = None
    if kind in ("article", "file", "hub", "listing"):
        ds = dates_in(md[:1500])
        pub = ds[0] if ds else None
    page = ParsedPage(url=url, canon=canon_url(url), title=clean(title or "").split(" | ")[0].strip(),
                      kind=kind, pub=pub)
    lines = md.split("\n")
    heads: list[tuple[int, str]] = []   # '#' heading stack
    labels: list[str] = []              # recent short label lines
    recent_eff = None                   # last 'effective as of' date seen in text
    recent_text: list[str] = []         # last few text lines, for table context
    short: list[str] = []               # pending group of short lines
    short_links: list = []
    n_table = 0
    n_para = 0
    faq_q = ""                          # current FAQ question; the text below it answers it
    lead_in = ""                        # "Branch hours are:" - introduces the list of short lines after it
    i = 0

    def ctx() -> str:
        return " > ".join([h for _, h in heads][-2:] + labels[-2:])

    def flush_short():
        nonlocal short, short_links, lead_in
        if short:
            txt = "\n".join(short)
            if len(txt) > 3:
                c = ctx()
                if lead_in and lead_in not in txt:
                    c = f"{c} > {lead_in}" if c else lead_in
                page.units.append(Unit("text", txt, c, links=tuple(short_links), faq=faq_q))
            lead_in = ""
        short, short_links = [], []

    while i < len(lines):
        raw = lines[i].rstrip()
        s = raw.strip()
        if not s:
            i += 1
            continue
        m = HEADING.match(s)
        if m:
            flush_short()
            lvl, txt = len(m.group(1)), strip_footnotes(strip_markdown(m.group(2))).strip()
            if HEAD_FACT.search(txt) and 6 <= len(txt) <= 200 and not FAQ_RX.match(txt):
                # a heading that states a fact ("Application period: Oct. 1-Nov. 12, 2026", "4.00% intro APY") is
                # an answer itself, under its parent headings
                page.units.append(Unit("text", txt, " > ".join(h[1] for h in heads if h[0] < lvl), links=()))
            heads = [h for h in heads if h[0] < lvl] + [(lvl, txt)]
            labels = []
            faq_q = txt if FAQ_RX.match(txt) else ""
            lead_in = ""
            i += 1
            continue
        if s.startswith("|") and i + 1 < len(lines) and TABLE_SEP.match(lines[i + 1].strip()):
            flush_short()
            header = _cells(s)
            j = i + 2
            rows = []
            while j < len(lines) and lines[j].strip().startswith("|"):
                rows.append(_cells(lines[j]))
                j += 1
            # an 'effective' date right after the table also belongs to it
            eff = recent_eff
            seen_lines = 0
            for k in range(j, min(j + 40, len(lines))):
                t = lines[k].strip()
                if not t:
                    continue                 # older crawls put many blank lines between a table and its footnote
                if t.startswith(("|", "#")) or seen_lines >= 6:
                    break                    # the next table or section: its date is not this table's
                seen_lines += 1
                if EFFECTIVE.search(t):
                    d = dates_in(t)
                    if d:
                        eff = d[0]
                        break
            c = ctx()
            after = []                       # footnotes right below a table describe that table
            for k in range(j, min(j + 12, len(lines))):
                t = lines[k].strip()
                if not t:
                    continue
                if t.startswith("|") or HEADING.match(t) or len(after) >= 3:
                    break
                after.append(strip_footnotes(strip_markdown(t))[:200])
            near = " ".join(recent_text[-4:] + after)[:600]
            for r in rows:
                if not any(r):
                    continue
                page.units.append(Unit("row", " | ".join(x for x in r if x), c, tuple(header), tuple(r),
                                       n_table, eff=eff, near=near))
            n_table += 1
            recent_eff = None
            recent_text = []
            labels = []          # labels above one table must not leak into the next (ctx); kept in older_labels
            i = j
            continue
        text = strip_footnotes(strip_markdown(s)).strip()
        lk = links(s)
        link_only = bool(lk) and not LINK_RE_FULL.sub("", s).strip(" .|-·*")
        if s.startswith("![") or text.startswith("!["):    # images: alt text repeats the page, not facts
            i += 1
            continue
        q_line = text.lstrip("-*• ").strip()
        if FAQ_RX.match(q_line) and len(q_line) <= 200 and not SENT_SPLIT.search(q_line[:-1]):
            flush_short()                    # an FAQ question (often an accordion link): what follows answers it
            faq_q = q_line
            i += 1
            continue
        if not text or NOISE.match(text) or link_only:
            if link_only and page.units:
                # keep the destinations: "See details" / "Open my ..." links belong to the section above
                last = page.units[-1]
                last.links = tuple(last.links) + tuple(lk)
            elif link_only:
                pass
            i += 1
            continue
        if EFFECTIVE.search(text):
            d = dates_in(text)
            if d:
                recent_eff = d[0]
        kv = KV.match(s)
        if kv and len(text) < 200:
            page.units.append(Unit("kv", text, ctx(), cells=(strip_markdown(kv.group(1)).strip(),
                                                              strip_markdown(kv.group(2)).strip()), links=tuple(lk),
                                   faq=faq_q))
        recent_text = (recent_text + [text[:120]])[-4:]
        if text.endswith(":") and len(text) > 110:
            flush_short()
            # the last two sentences ("The Platteville branch is at ... Branch hours are:") introduce the list below
            lead_in = text[-240:].split(" ", 1)[-1] if len(text) > 240 else text
        elif short == [] and not text.endswith(":") and len(text) > 110:
            lead_in = ""
        sentence = text.endswith((".", "!", "?")) and len(text.split()) >= 6
        if sentence and len(text) <= 110:
            flush_short()                    # a full sentence stands alone; short-line groups are lists/hours
            page.units.append(Unit("text", text, ctx(), links=tuple(lk), faq=faq_q, full=text, para=n_para))
            n_para += 1
            i += 1
            continue
        if len(text) <= 110:
            if _is_label(s) and not kv:
                labels = (labels + [text])[-3:]
            short.append(text)
            short_links.extend(lk)
            if len(short) >= 10:
                flush_short()
        else:
            flush_short()
            sents = split_sentences(text)
            # sentence pairs keep a little context while staying specific
            for k in range(len(sents)):
                piece = sents[k] if len(sents[k]) > 60 or k + 1 >= len(sents) else sents[k] + " " + sents[k + 1]
                page.units.append(Unit("text", piece, ctx(), links=tuple(lk), faq=faq_q, full=text, para=n_para))
            n_para += 1
        i += 1
    flush_short()
    return page
