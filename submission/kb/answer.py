"""Question -> answer: snapshot selection, retrieval, reranking, answer-type validation, composition.

Generic rules only (no product names):
  * the target snapshot comes from as_of, else a date written in the question, else the latest;
  * candidates are ranked by lexical (+ optional semantic) relevance x page authority x freshness;
  * the question's answer TYPE (hours, date, phone, address, count, rate, fee, url, ...) decides which
    candidates qualify, and names in the question must appear in the chosen source;
  * if nothing qualifies, the system says it doesn't know instead of guessing.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from .quals import (amount_range, amounts, metric_of_header, metrics_asked, pct_values, question_date,
                    question_years, superlative, terms_months, year_range)
from .pages import canon_url, classify
from .store import Store, UnitRec
from .reader import IDK, llm_answer, wants_llm
from .text import GENERIC, STOP, clean, content_tokens, expand, stem, tokens

# ranking weights per feature (see _rank); fitted offline by dev/ltr.py, 1.0 = hand-set factor, "cov" defaults to 0
try:
    W = json.loads((Path(__file__).with_name("weights.json")).read_text())
except Exception:
    W = {}
W.setdefault("cov", 0.0)

REFUSAL = "I'm sorry, that isn't listed on our website. Please contact us and we'll be glad to help."
NUMERIC_METRICS = {"rate", "apy", "apr", "fee", "payment", "minimum"}

TIME_RX = r"\b\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)|\bnoon\b"
DATE_RX = (r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b"
           r"|\b(?:19|20)\d{2}\b")
VALIDATORS = {
    "hours": re.compile(TIME_RX, re.I),
    "when": re.compile(TIME_RX + "|" + DATE_RX, re.I),
    # "when do payments start / will I get my card": a delay or a relative point is an answer too
    "wait": re.compile(TIME_RX + "|" + DATE_RX + r"|\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|"
                       r"twelve|thirty|sixty|ninety)[\s-]+(?:business |calendar )?(?:days?|weeks?|months?|years?)\b"
                       r"|\b(?:day|week|month|year) \d+\b|\b(?:immediately|same day|next business day|right away)\b", re.I),
    "phone": re.compile(r"\(?\b\d{3}\)?[-. ]\d{3}[-. ]\d{4}\b"),
    "address": re.compile(r"\b\d{2,6}\s+[A-Z0-9][\w.]*(?:\s+[\w.]+){0,4}?\s+(?:st|street|ave|avenue|rd|road|dr|drive|ln|lane|blvd|boulevard|way|ct|court|hwy|highway|pl|place|pkwy|parkway|cir|circle|plaza|sq)\b\.?|\b[A-Z]{2},?\s+\d{5}\b", re.I),
    "routing": re.compile(r"(?<![\d/.])\b\d{9}\b(?![\d/.])"),
    "count": re.compile(r"\b\d[\d,]*\b|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
                        r"fifteen|twenty|thirty|dozens?|hundreds?|thousands?)\b", re.I),
    # a decimal percent, or a whole one that says it is a rate ("6% APR", "2% interest")
    "rate": re.compile(r"\b\d{1,2}\.\d+\s*%|\b\d{1,2}\s*%\s*(?:apr|apy|annual|interest|dividend|fixed|variable|intro)", re.I),
    "fee": re.compile(r"\$\s?\d|\d\s*%|\bfree\b|\bno (?:fee|charge|cost)\b", re.I),
    "amount": re.compile(r"\$\s?\d|\d\s*%|\b\d{1,3}(?:,\d{3})+\b|\b\d+(?:\.\d+)?\s*(?:million|billion|thousand)\b"
                         r"|\b\d[\d,]*\s*(?:square|sq\b|acres?)", re.I),
}


@dataclass
class Result:
    text: str
    sources: list = field(default_factory=list)    # (url, snapshot_id, quote)
    confident: bool = True
    note: str | None = None


@dataclass
class Q:
    raw: str
    toks: list
    weights: dict
    terms: set
    amounts: list
    years: list
    metrics: set
    superl: str | None
    list_intent: bool
    yesno: bool
    pcts: list
    business: bool
    atype: str | None
    names: list
    define: bool = False        # "what is X", "tell me about X": wants the passage that defines X
    phrases: list = field(default_factory=list)   # multi-word names ("La Crosse", "Mineral Point Road"): whole match
    neg: list = field(default_factory=list)       # negated qualifiers ("non-jumbo", "not business"): rows to avoid


# words that say how the member asks, not what about: never a product key ("I'm thinking of...", "paying")
FILLER = set("""i'm im thinking think wondering wonder looking look want wanted need needs paying listed list show tell
know getting going like would could should currently now today right please""".split())
FAQ_Q_RX = re.compile(r"^(?:can|could|do|does|did|is|are|how|what|when|where|who|why|will|should|which)\b.*\?$", re.I)
CURRENT_Q_RX = re.compile(r"\b(?:who is|who's|current(?:ly)?|now|today|latest|still|anymore)\b", re.I)
TENTATIVE_RX = re.compile(r"\b(?:so far|to date|year to date|plans? to|planned|is scheduled|are scheduled|scheduled to"
                          r"|expected to|is set to|will (?:open|be completed|begin|break ground)|anticipated)\b", re.I)
LINK_NOISE_RX = re.compile(r"\b(?:click(?:ed|ing)?(?: on)?|tap(?:ped)?|open(?:ed)?|suspicious|phishing|scam|fake|fraudulent"
                           r"|strange|unknown)\b[^?]{0,25}\blinks?\b|\blinks? (?:in|from) (?:an?|the|that|this) "
                           r"(?:email|text|message)", re.I)
AMOUNT_RX = re.compile(r"\bhow (?:much|large|big)\b|\b(?:max|min)(?:imum)? (?:amount|grant|award|loan|deposit|balance)\b"
                       r"|\bhow much money\b|\b(?:average|total|combined)\b|\b(?:contribution |deposit |borrowing )?"
                       r"(?:limits?|caps?)\b|\bthe most (?:i|you|we) can\b|\bsquare (?:feet|footage)\b", re.I)


def answer_type(ql: str, metrics: set) -> str | None:
    if (re.search(r"\b(url|link|web ?address|website|web ?page)\b", ql) and not LINK_NOISE_RX.search(ql)) or \
            re.search(r"\bwhere (?:do|can|should) i (?:go to )?(?:apply|start|sign up|enroll|log ?in|register)\b", ql) or \
            re.search(r"\b(?:what|which) (?:page|site)\b|\bis there (?:a|an) (?:page|site|form|tool|calculator)\b|\bwhere (?:on|in) (?:your|the|dupaco'?s?) (?:site|website)\b"
                      r"|\bwhere can i (?:go|find|get)\b.*\b(?:online|calculator|form|application|pdf|report)\b"
                      r"|\bgo online to\b", ql) or \
            re.search(r"\bhow (?:do|can) i (?:join|apply|sign up|enroll|open)\b.*\bonline\b", ql):
        return "url"
    if re.search(r"\bhow many\b|\b(?:total )?number of (?!the\b|your\b|my\b|a\b)[a-z]+s\b", ql):
        return "count"
    if re.search(r"\brouting\b|\baba\b", ql):
        return "routing"
    if re.search(r"\b(phone|telephone|call|fax)\b", ql) and not re.search(r"\bcalled\b", ql):
        return "phone"
    if re.search(r"\baddress\b|\bwhere is\b|\blocated\b", ql) and not re.search(r"\bemail address\b", ql):
        return "address"
    if re.search(r"\bhours\b|\bwhat time\b|\bopen until\b", ql):
        return "hours"
    if re.search(r"^\s*when (?:do|does|will|can|should|would)\b", ql) and \
            not re.search(r"\b(?:open|close|published|founded|held|happen|take place)\b", ql):
        return "wait"
    if re.search(r"^\s*when\b", ql):
        return "when"
    if re.match(r"\s*(?:why|how (?:can|does|do|is|are|come|would|will)|what makes|what happens)\b", ql) \
            and not re.search(r"\bhow much\b", ql):
        return None                      # an explanation, even if it mentions rates or fees
    m = AMOUNT_RX.search(ql)
    if m and "fee" not in metrics and not metrics & {"apy", "apr"} and not re.search(r"\brates?\b", ql) and \
            ("rate" not in metrics or not m.group(0).startswith("how")):
        return "amount"                  # "average dividend", "contribution limit", "square footage": a figure
    if metrics & {"rate", "apy", "apr"}:
        return "rate"
    if "fee" in metrics:
        return "fee"
    if re.search(r"\bhow (?:much|large|big)\b|\b(?:max|min)(?:imum)? (?:amount|grant|award|loan|deposit|balance)\b"
                 r"|\bhow much money\b", ql):
        return "amount"                  # "how much is the grant / do I need to deposit": a dollar figure
    return None


PLACE_WORDS = {"branch", "branches", "office", "location", "street", "st", "road", "rd", "avenue", "ave", "drive",
               "dr", "lane", "ln", "boulevard", "blvd", "center", "lab"}
NAME_SKIP = {"dupaco", "credit", "union", "visa"}
KV_WORDS = {"phone": r"phone|telephone", "routing": r"routing|aba", "address": r"address"}
# words that say what kind of answer is wanted rather than what it is about
QWORDS = {"hour", "link", "url", "website", "page", "address", "phone", "number", "many", "time", "when", "where",
          "located", "call", "online", "use", "standard", "general", "typical", "usual", "today", "still",
          "highest", "lowest", "best", "top", "most", "maximum", "minimum", "cheapest", "least", "tier",
          "regular", "normal", "basic", "ordinary", "plain", "typical"}


DEFINE_RX = re.compile(r"\s*(?:what(?:'s| is| are| does)\b(?!.*\b(?:rate|apy|apr|fee|cost|minimum|number|hours?|address|"
                       r"phone|routing|link|url)s?\b)|tell me (?:about|more)|explain|describe|how (?:does|do) .* work"
                       r"|how is .* different)")
LISTQ_RX = re.compile(r"\bwhat (?:types?|kinds?|sorts?|benefits?|features?|options?|services?|tools?|perks?|documents?"
                      r"|ways|topics|apps|products|projects|accounts|cards|steps)\b"
                      r"|\bwhich\b.*\b(?:tools|services|options|apps|ways|types|kinds|products|accounts|cards|ones)\b"
                      r"|\bwhat (?:can|do) (?:i|you)\b.*\b(?:use|need|get|offer)\b|\bwhich\b.*\b(?:qualify|eligible)\b", re.I)
DEFINES_RX = re.compile(r"\b(?:is|are|means|refers to|helps?|lets|allows|combines|works|gives|provides)\b|:\s", re.I)
CONTACT_WORDS = {"number", "contact", "mail", "reach", "call", "dial", "talk", "speak", "get", "touch", "someone",
                 "person", "help", "question", "line", "main", "general", "customer", "service", "representative"}
OPTIONS_RX = re.compile(r"\b(terms?|tiers?|lengths?|options|choices|kinds|types|levels)\b", re.I)
BOILER_RX = re.compile(r"\b(?:for more information|to learn more|learn more|click here|visit (?:us|our|[\w.]+\.com)"
                       r"|read more|find out more|sign up for our)\b", re.I)


NEG_RX = re.compile(r"\b(?:non|not?)[- ](?:an? |the )?([a-z]{3,})\b", re.I)


def parse_question(question: str):
    q = clean(question)
    qd, rest = question_date(q)
    neg = [stem(m.group(1).lower()) for m in NEG_RX.finditer(rest)]
    neg = [w for w in neg if w not in STOP and w not in GENERIC and len(w) > 3]
    if neg:
        rest = NEG_RX.sub(" ", rest)              # "non-jumbo" must not count as asking for jumbo
    toks = content_tokens(rest)
    weights = {t: 1.0 for t in toks}
    for t in expand(toks):
        weights.setdefault(t, 0.5)
    yrs = [y for y in question_years(rest) if 1950 <= y <= 2100]
    ql = rest.lower()
    metrics = metrics_asked(rest)
    list_intent = bool(re.search(r"\b(what are|which|list|all|options|offer|do you have|kinds?|types?)\b", ql)) \
        or bool(re.search(r"\b(rates|cards|accounts|loans|certificates)\b", ql))
    yesno = bool(re.match(r"\s*(do|does|is|are|can|could|will|has|have)\b", ql))
    body = re.sub(r"^\W*\S+", "", rest)                    # the first word is capitalised anyway
    names = []
    for w in re.findall(r"\b([A-Z][a-zA-Z]{2,})\b", body):
        t = stem(w.lower())
        if t not in STOP and t not in GENERIC and t not in NAME_SKIP:
            names.append(t)
    phrases = []
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", body):
        words = [w for w in m.group(1).split() if stem(w.lower()) not in STOP and w.lower() not in NAME_SKIP
                 and stem(w.lower()) not in GENERIC and stem(w.lower()) not in QWORDS
                 and w.lower() not in PLACE_WORDS]
        if len(words) >= 2:
            phrases.append(" ".join(words).lower())
    return qd, Q(rest, toks, weights, terms_months(rest), amounts(rest), yrs, metrics, superlative(rest),
                 list_intent, yesno, pct_values(rest), "business" in ql or "commercial" in ql,
                 answer_type(ql, metrics), names, define=bool(DEFINE_RX.match(ql)), phrases=phrases, neg=neg)


DAY_RX = re.compile(r"\b(?:mon|tue|wed|thu|fri|sat|sun)[a-z]*\b|\bweekdays?\b|\bweekends?\b|\bdaily\b", re.I)
SECTION_RX = re.compile(r"^(?:lobby|drive[- ]?(?:up|thru|through)|atm|walk[- ]?up|teller|branch)\b[^.]{0,30}:?$", re.I)


def hours_block(text: str, want: str | None = None) -> str:
    """The schedule lines of a passage ('Lobby; Monday-Friday: 9 a.m.-5 p.m.; ...'), optionally one section only."""
    out, section, keep = [], None, want is None
    for line in text.split("\n"):
        t = line.strip().rstrip(":")
        if not t:
            continue
        if SECTION_RX.match(t) and not VALIDATORS["hours"].search(t):
            section = t.lower()
            keep = want is None or section.startswith(want)
            if keep and want is None:
                out.append(t + ":")
            continue
        # "Drive-up Monday-Friday: 9 a.m.-5 p.m." - section and hours on one line
        m = re.match(r"(lobby|drive[- ]?(?:up|thru|through))\s+(.*)", t, re.I)
        if m and DAY_RX.search(m.group(2)):
            section, keep = m.group(1).lower(), want is None or m.group(1).lower().startswith(want)
            t = m.group(2) if want else t
        if DAY_RX.search(t) and (VALIDATORS["hours"].search(t) or re.search(r"closed|atm only|24", t, re.I)):
            if keep and len(t) <= 80:
                out.append(t)
    if not any(VALIDATORS["hours"].search(x) for x in out):
        return ""
    return "; ".join(out).replace(":;", ":")


def count_noun(question: str, site: str = "") -> str:
    """The thing being counted in 'how many [Dupaco] employees ...' (first content word after 'how many')."""
    m = re.search(r"how many\s+(.*)", question.lower())
    for w in re.findall(r"[a-z-]+", m.group(1) if m else ""):
        if w != site and w not in STOP and w not in NAME_SKIP:
            return stem(w)[:5]
    return ""


def noun_count(text: str, noun: str):
    """The number written right before the counted noun ('22 branch', '294 Dupaco employees'), or None."""
    if not noun:
        return None
    low = text.lower()
    for m in re.finditer(r"\b(\d[\d,]*\d|\d|" + NUMWORDS_RX + r")\+?\s+(?:[a-z-]+\s+){0,2}?" + re.escape(noun), low):
        raw = m.group(1)
        if raw in NUMWORDS:
            return str(NUMWORDS[raw])
        # a year ("In 2025, Dupaco employees") or a ZIP code; "150,000" with its comma is a count
        if not re.fullmatch(r"(?:19|20)\d\d|\d{5}(?:\d{4})?", raw):
            return raw.replace(",", "")
    return None


NUMWORDS = {w: k for k, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                       "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}
NUMWORDS_RX = "|".join(sorted(NUMWORDS, key=len, reverse=True))


def text_year(text: str):
    ys = [int(y) for y in re.findall(r"\b(20\d{2})\b", text)]
    return max(ys) if ys else None


# ---- row helpers ---------------------------------------------------------------------------------------
def row_columns(u):
    mets = [metric_of_header(h) for h in u.header]
    while len(mets) < len(u.cells):
        mets.append(None)
    return mets


def row_label(u) -> str:
    mets = row_columns(u)
    keep = [c for c, m in zip(u.cells, mets) if (m is None) and c and c not in ("-", "—")]
    return " ".join(keep) if keep else (u.cells[0] if u.cells else "")


def qual_match(label: str, q: Q) -> float:
    m = 1.0
    if q.terms:
        rt = terms_months(label)
        if rt:
            m *= 1.7 if (q.terms & rt) else 0.2
        else:
            m *= 0.85
    if q.amounts:
        ar = amount_range(label)
        if ar:
            if all(ar[0] <= a <= ar[1] for a in q.amounts):
                width = (ar[1] - ar[0]) if ar[1] != float("inf") else 1e12
                m *= 1.5 + 0.2 / (1 + width / 1e5)
            elif len(q.amounts) >= 2 and ar[0] <= min(q.amounts) and max(q.amounts) <= ar[1] + 1:
                m *= 1.5
            else:
                m *= 0.25
    if q.years:
        yr = year_range(label)
        if yr:
            m *= 1.6 if all(yr[0] <= y <= yr[1] for y in q.years) else 0.25
    return m


def fmt_value(cell: str, met: str, header: str) -> str:
    cell = cell.strip()
    if met == "apy":
        return cell if "apy" in cell.lower() else f"{cell} APY"
    if met == "apr":
        return cell if "apr" in cell.lower() else f"{cell} APR"
    if met == "rate":
        h = re.sub(r"\(.*?\)", "", header).strip().lower() or "rate"
        return f"{cell} {h}"
    return f"{cell} ({re.sub(r'[(].*?[)]', '', header).strip().lower()})"


def row_values(u, q: Q) -> list[str]:
    mets = row_columns(u)
    present = {m for m in mets if m}
    want = set(q.metrics & NUMERIC_METRICS)
    if want & {"rate", "apy", "apr"}:
        # a rate, its APY and its APR are one fact: always show the row's full set, so "what's the APY" and "what
        # does it pay" give the same numbers
        want |= {"rate", "apy", "apr"} & present
    if not want or not (want & present):
        want = ({"rate", "apy", "apr"} & present) or ({"fee"} & present)
    out = []
    for c, m, h in zip(u.cells, mets, u.header):
        if m in want and c and c not in ("-", "—"):
            out.append(fmt_value(c, m, h))
    return out


def metric_number(u, q: Q):
    mets = row_columns(u)
    order = (["apy"] if "apy" in q.metrics else []) + (["apr"] if "apr" in q.metrics else []) + ["apy", "rate", "apr", "fee"]
    for want in order:
        for c, m in zip(u.cells, mets):
            if m == want:
                v = pct_values(c) or [float(x.replace(",", "")) for x in re.findall(r"\$?(\d[\d,]*\.?\d*)", c)[:1]]
                if v:
                    return v[0]
    return None


def display(u) -> str:
    if u.kind == "kv":
        return f"{u.cells[0]}: {u.cells[1]}"
    return u.text


# ---- main ------------------------------------------------------------------------------------------------
class Answerer:
    def __init__(self, store: Store, semantic=None, llm=None):
        self.s = store
        self.semantic = semantic          # optional: callable(question, snap) -> {uid: similarity}
        self.llm = llm                    # optional: kb.llm.LLM, used only for reading-comprehension questions
        self.rerank = None                # optional: kb.rerank.Reranker, re-scores the top lexical candidates
        self.keep_features = False        # dev only: keep the ranking features of the last question (dev/ltr.py)
        self.last_features = {}
        self._ctx = None

    def _snapshot(self, qd, as_of):
        day = qd
        if as_of:
            try:
                day = dt.date.fromisoformat(str(as_of)[:10])
            except ValueError:
                day = qd
        return self.s.snapshot_for(day)

    def answer(self, question: str, as_of: str | None = None) -> Result:
        self._ctx = None
        r = self._rules(question, as_of)
        if self.llm is None or not self.llm.alive or self._ctx is None:
            return r
        q, snap, sid, scored, path = self._ctx
        if not wants_llm(q, path):
            return r
        res = llm_answer(self, q, snap, sid, scored, question)
        if res == IDK:
            # the reader saw the best pages and found no answer; a prose rule answer is then only a loose match
            # (another year, another product), so an honest refusal is safer. Table values, labelled facts,
            # hours and links keep the rule answer.
            # (Yes/no questions are left alone: "is it still open?" needs reasoning over a dated passage, which
            # the reader is cautious about.)
            if path == "text" and q.atype in (None, "amount", "when", "wait") and r.confident and not q.yesno:
                return Result(REFUSAL, confident=False, note="the site's pages don't answer this (reader)")
            return r
        return res or r

    def _rules(self, question: str, as_of: str | None = None) -> Result:
        s = self.s
        if not s.snaps or not (question or "").strip():
            return Result(REFUSAL, confident=False)
        qd, q = parse_question(question)
        snap = self._snapshot(qd, as_of)
        sid = s.snaps[snap][0]
        if not q.weights:
            return Result(REFUSAL, confident=False)

        df = s.df_snap[snap]
        unknown = [n for n in q.names if df.get(n, 0) == 0 and not s.fuzzy_known(n, snap)]
        if not unknown:
            # a content word the site never uses, in any form (synonym, word family, near spelling): the site is
            # not about it ("gold IRA", "pawn loan")
            for t in q.toks:
                if len(t) >= 4 and t.isalpha() and t not in GENERIC and t not in QWORDS and df.get(t, 0) == 0 \
                        and not any(df.get(x, 0) for x in expand([t])[1:]) and not s.family(t, snap) \
                        and not s.fuzzy_known(t, snap):
                    unknown.append(t)
        code = re.search(r"\b\d+/\d+\b", q.raw)
        if unknown or (code and df.get(code.group(0), 0) == 0):
            return Result(REFUSAL, confident=False, note="the site never mentions " + (unknown[0] if unknown else code.group(0)))

        scored = self._rank(q, snap)
        if not scored:
            return Result(REFUSAL, confident=False)

        self._ctx = [q, snap, sid, scored, "refuse"]
        if q.atype == "hours" and not q.names:
            r = self._hours_overview(q, snap, sid, scored)
            if r:
                self._ctx[4] = "hours"
                return r
        if q.atype == "url":
            r = self._url_answer(q, snap, sid, scored)
            if r:
                self._ctx[4] = "url"
                return r
        pick = self._choose(q, scored, snap)
        if pick is None:
            return Result(REFUSAL, confident=False, note=f"no {q.atype or 'matching'} information on the site")
        sc, i = pick
        best = s.units[i]
        if best.u.kind != "row" and q.list_intent and OPTIONS_RX.search(q.raw):
            # "what terms / tiers / options do you offer for X": the rows of X's table are the list
            for sc2, i2, cov2 in scored[:60]:
                if sc2 < 0.5 * sc:
                    break
                r2 = s.units[i2]
                if r2.u.kind == "row" and cov2 >= 0.3 and self._qualifies(r2, q, snap):
                    best, i = r2, i2
                    break
        self._ctx[4] = best.u.kind
        if best.u.kind == "row":
            return self._row_answer(best, q, snap, sid, scored)
        text = display(best.u)
        if best.u.kind == "text" and q.atype != "count":
            text = self._with_neighbour(best, snap, text, q)
        if best.u.kind == "text" and q.list_intent and (q.names or q.phrases):
            text = self._same_list(best, snap, text, q)
        if best.u.kind == "text" and best.title:
            # the passage may not name what it is about ("we have flexible options ..."): its page title does
            title_t, text_t = set(tokens(best.title)), set(tokens(text))
            asked = set(expand(q.toks)) - GENERIC - QWORDS
            if (title_t & asked) - text_t and not text.lower().startswith(best.title.lower()):
                text = f"{best.title}: {text}"
        srcs = [(best.url, sid, best.u.text[:200])]
        if q.atype in ("phone", "routing", "address"):
            # the same value (routing number, phone, address) on other authoritative pages, in any wording: cite
            # them too
            m = VALIDATORS[q.atype].search(display(best.u))
            if m:
                val = re.sub(r"\W", "", m.group(0)).lower()
                seen = {best.url}
                for _, i, _ in scored[:300]:                 # most relevant first
                    r = s.units[i]
                    if r.url not in seen and s.authority(r, snap) >= 0.75 and \
                            val in re.sub(r"\W", "", display(r.u)).lower():
                        seen.add(r.url)
                        srcs.append((r.url, sid, r.u.text[:200]))
                    if len(srcs) >= 5:
                        break
        if q.atype == "count":
            prev = self._count_history(best, q, snap, scored)
            if prev is not None:
                text = f"{text} Earlier figure: {display(prev.u)}"
                srcs.append((prev.url, sid, prev.u.text[:200]))
        text = re.sub(r"\s*\n\s*", "; ", text).strip()
        text = re.sub(r";\s*;", ";", text)
        if len(text) > 700:
            text = text[:700].rsplit(" ", 1)[0] + "…"
        return Result(text, srcs)

    def _same_list(self, best: UnitRec, snap: int, text: str, q: Q) -> str:
        """'Which branches are in Dubuque?': every entry on the page that names the place, not just the first."""
        page = self.s.snap_pages[snap].get(best.canon)
        if not page:
            return text
        need = [p for p in q.phrases] or [n for n in q.names]
        hits = []
        for u in page.units:
            # entries of the same kind: same first line ("Dupaco Branch") in a short-line group
            if u.kind != "text" or "\n" not in u.text or \
                    u.text.split("\n", 1)[0].strip() != best.u.text.split("\n", 1)[0].strip():
                continue
            low = u.text.lower()
            if all((n in low) if " " in n else (n in set(tokens(low))) for n in need):
                hits.append(re.sub(r"\s*\n\s*", ", ", u.text.strip()))
        if len(hits) > 1:
            return "; ".join(dict.fromkeys(hits[:8]))
        return text

    def _count_history(self, best: UnitRec, q: Q, snap: int, scored):
        """A count tied to a year ('294 employees ... in 2023') is history, not a stale value: next to the newest
        figure, give the one for the year before it."""
        s = self.s
        noun = count_noun(q.raw, s.site_name())
        mine = display(best.u)
        y0, n0 = text_year(mine), noun_count(mine, noun)
        if y0 is None or n0 is None:
            return None
        top = scored[0][0]
        for sc, i, cov in scored[:150]:
            if sc < 0.5 * top:
                break
            r = s.units[i]
            t = display(r.u)
            y, n = text_year(t), noun_count(t, noun)
            if r is best or y is None or n is None or n == n0 or not self._qualifies(r, q, snap):
                continue
            if y == y0 - 1:
                return r
        return None

    # ---- ranking ---------------------------------------------------------------------------------------
    def _rank(self, q: Q, snap: int):
        s = self.s
        weights = dict(q.weights)
        fam = {t: s.family(t, snap) for t in set(q.toks)}
        for t, ws in fam.items():
            for w in ws:
                weights.setdefault(w, 0.4)
        bm = s.bm25(weights, snap)
        sem = self.semantic(q.raw, snap) if self.semantic else {}
        cand = set(i for i, _ in sorted(bm.items(), key=lambda kv: (-kv[1], kv[0]))[:400]) | set(sem)
        if q.atype in KV_WORDS:                 # labelled facts of the asked type are always candidates
            rx = re.compile(KV_WORDS[q.atype], re.I)
            cand |= {i for i in s.snap_units[snap] if s.units[i].u.kind == "kv" and rx.search(s.units[i].u.cells[0])}
        if not cand:
            return []
        mx = max(bm.values()) if bm else 1.0
        qset = set(q.toks)
        idf = {t: s.idf(t, snap) for t in qset}
        idf_total = sum(idf.values()) or 1.0
        key_terms = [t for t in qset if t not in GENERIC and not t.isdigit()]
        alts = {t: set(expand([t])) | set(fam.get(t, ())) for t in qset}   # synonyms and word family count
        wanted = set().union(*alts.values()) if alts else set()
        _local = {}

        def row_local(i):
            if i not in _local:
                u = s.units[i].u
                _local[i] = set(tokens(" ".join((row_label(u), u.ctx, u.near, " ".join(u.header)))))
            return _local[i]
        row_vocab = set()
        for i in cand:
            if s.units[i].u.kind == "row":
                row_vocab |= row_local(i)
        numeric = bool(q.metrics & NUMERIC_METRICS) and q.atype in ("rate", "fee", None)
        today = s.snaps[snap][1]
        out = []
        repeats = s.repeat_snap[snap] if snap < len(s.repeat_snap) else {}
        prose = q.atype is None and not (q.metrics & NUMERIC_METRICS)
        want_list = bool(LISTQ_RX.search(q.raw))

        feats = {} if self.keep_features else None
        L = math.log
        for i in cand:
            rec = s.units[i]
            u = rec.u
            have = rec.toks | rec.page_toks | rec.place_toks
            cov = sum(idf[t] for t in qset if not alts[t].isdisjoint(have)) / idf_total
            rel = 0.45 * bm.get(i, 0.0) / mx + cov + 0.8 * sem.get(i, 0.0)
            auth = s.authority(rec, snap)
            # every factor is recorded as a named log-feature; the score is exp(sum weight * feature), and with all
            # weights 1.0 that is the product of the factors. Weights are fitted offline (dev/ltr.py).
            f = {"rel": L(rel + 1e-6), "cov": cov}
            if u.kind == "row":
                f["auth_row"] = L(auth)
                if u.eff:
                    age = max(0, (today - u.eff).days)
                    f["fresh"] = L(0.6 + 0.4 * math.exp(-age / 120.0))
                else:
                    f["no_eff"] = L(0.8)
                f["qual"] = L(qual_match(row_label(u) + " " + u.ctx, q))
                # words must line up with the row's own label and section, not just its page: "personal loan
                # 60 months" is not an auto-loan row, and a row saying "Jumbo" or "Roth" answers a different question
                local = row_local(i)
                # only words that tell candidate rows apart count as missing
                miss = sum(1 for t in key_terms if t in row_vocab and alts[t].isdisjoint(local))
                extra = {t for t in tokens(row_label(u) + " " + u.ctx) if t not in wanted and len(t) > 2
                         and t not in GENERIC and t not in STOP and not re.search(r"\d", t)
                         and t not in tokens(s.units[i].title)}
                f["row_miss"] = min(miss, 2) * L(0.88)
                # unasked qualifiers (Roth, Jumbo, Business): a different product; softer for "list all" questions
                f["row_extra"] = min(len(extra), 3) * L(0.95 if q.list_intent else 0.88)
                if q.neg and set(q.neg) & local:
                    f["negated"] = L(0.3)               # the question said "non-jumbo" and this row is jumbo
                if numeric:
                    f["row_numeric"] = L(1.6 if not q.list_intent else 2.0)
                if s.newest_same_row(rec, row_label(u), snap) is not None:
                    f["superseded"] = L(0.35)           # superseded by a newer version of the same row
            else:
                f["auth_text"] = 0.5 * L(auth)          # staleness matters less for descriptive text
                if prose and u.kind == "text":
                    # the passage has the shape of an answer to this kind of question
                    t0 = u.text.lstrip("-*• ").strip()
                    if q.yesno and re.match(r"(?:yes|no|nope|absolutely|definitely|of course)\b", t0, re.I):
                        f["shape_yesno"] = L(1.4)
                    if want_list and (re.search(r"\b(?:including|such as|include|includes)\b[^.]*[:,]", t0, re.I)
                                      or t0.count("\n") >= 2 or re.search(r":\s*\n?\s*-", u.full or "")):
                        f["shape_list"] = L(1.3)
                    elif want_list and u.text.lstrip().startswith(("- ", "* ", "• ")):
                        f["shape_list"] = L(1.25)       # a bullet: the whole list is returned with it
                    if q.define and DEFINES_RX.search(" ".join(t0.split()[:12])):
                        f["shape_define"] = L(1.2)
                    if u.full and u.full.startswith(u.text[:40]):
                        f["shape_lead"] = L(1.1)        # the opening sentence of its paragraph
                if numeric and not re.search(r"\d\s*%|\$\s?\d", u.text):
                    f["text_no_number"] = L(0.5)
                words = len(u.text.split())
                if u.kind == "text" and words < 5:
                    f["short_text"] = L(0.4)
                if rec.faq_toks:                        # the page's own FAQ question matches the member's question
                    ov = sum(1 for t in qset if not alts[t].isdisjoint(rec.faq_toks)) / max(1, len(qset))
                    f["faq"] = L(1 + 0.6 * ov)
                if q.atype not in ("url", "phone") and BOILER_RX.search(u.text):
                    f["boiler"] = L(0.6)
                if (u.full or u.text).rstrip().endswith(("...", "…")):
                    f["teaser"] = L(0.5)                # a cut-off card linking elsewhere, not the article
                n_rep = repeats.get(u.text, 1) if u.kind == "text" else 1
                if n_rep >= 5 and q.atype != "count":
                    f["repeated"] = -L(1 + math.log(n_rep / 4))     # site-wide boilerplate
                if q.define and u.kind == "text":
                    own = set(tokens(u.text))
                    focus = q.names or key_terms
                    if focus and all(t in own or self.s.fuzzy_in(t, own) for t in focus):
                        f["def_names"] = L(1.25)        # it names the thing, not just sits on its page
                    if u.para >= 0:
                        f["def_early"] = L(1 + 0.25 * math.exp(-u.para / 3.0))    # pages define things early
                    if u.text.lstrip().startswith(("\"", "“")):
                        f["def_quote"] = L(0.8)         # quotes are opinions, not definitions
            if key_terms:
                f["page_terms"] = L(1 + 0.35 * sum(1 for t in key_terms if t in rec.page_toks) / len(key_terms))
            biz = "business" in rec.page_toks or "commercial" in rec.page_toks
            if biz and not q.business:
                f["biz_unasked"] = L(0.7)
            elif q.business and not biz:
                f["biz_missing"] = L(0.8)
            if u.kind == "kv":
                lab = set(tokens(u.cells[0]))
                f["kv_label"] = L(1.3) if lab & set(expand(q.toks)) else L(0.8)
            sc = math.exp(sum(W.get(k, 1.0) * v for k, v in f.items())) + 0.002 * rec.first
            if feats is not None:
                feats[i] = f
            out.append((sc, i, cov))
        if feats is not None:
            self.last_features = feats
        out.sort(key=lambda x: (-x[0], x[1]))
        if self.rerank is not None and q.atype is None and not (q.metrics & NUMERIC_METRICS) and W.get("ce", 0):
            # reading questions only: the cross-encoder's relevance is one more weighted feature for the top K
            def passage(i):
                r = s.units[i]
                return " ".join(f"{r.title}. {r.u.ctx}. {display(r.u)}".split())
            head = out[: self.rerank.k]
            logits = self.rerank.logits(q.raw, [passage(i) for _, i, _ in head])
            if logits:
                new = []
                for (sc, i, cov), z in zip(head, logits):
                    ce = -math.log1p(math.exp(-z)) if z > -30 else z          # log-sigmoid
                    new.append((sc * math.exp(W["ce"] * ce), i, cov))
                    if feats is not None:
                        feats[i]["ce"] = ce
                out = sorted(new + out[len(head):], key=lambda x: (-x[0], x[1]))
        return out

    def _qualifies(self, rec: UnitRec, q: Q, snap: int) -> bool:
        u = rec.u
        hay = rec.toks | rec.page_toks | rec.place_toks
        if q.names and not all(n in hay or self.s.fuzzy_in(n, hay) for n in q.names):
            return False
        if q.phrases:
            # a multi-word proper name (one with a rare word: "La Crosse", "Mineral Point Road") must appear as a
            # phrase in the passage or its page title: "La Crosse" is not "Crosse Pointe Blvd." Common product
            # words in Title Case ("Business Certificates Regular") are left to the normal matching.
            df = self.s.df_snap[snap]
            names = [p for p in q.phrases if any(df.get(stem(w), 0) <= 30 for w in p.split())]
            if names:
                near = re.sub(r"\s+", " ", " ".join((u.text, u.ctx, u.near, u.full, rec.title))).lower()
                if not all(p in near for p in names):
                    return False
        ords = re.findall(r"\b(\d+)(?:st|nd|rd|th)\b", q.raw.lower())
        if ords and u.kind != "row":
            # "the 78th annual meeting": a passage about the 77th is another event
            near_o = set(re.findall(r"\b(\d+)(?:st|nd|rd|th)\b", " ".join((u.text, u.full or "", rec.title)).lower()))
            if near_o and not set(ords) <= near_o:
                return False
        for code in re.findall(r"\b\d+/\d+\b", q.raw):        # e.g. "7/1" must be in the source itself
            if code not in rec.toks:
                return False
        # the product being asked about must be what the source is about
        keys = [t for t in set(q.toks) if t not in GENERIC and t not in QWORDS and not t.isdigit() and t not in q.names
                and t not in FILLER and not re.fullmatch(r"\$?\d[\d,.]*[km]?", t)]
        if keys and not q.names and (u.kind == "row" or q.atype in ("rate", "fee")):
            hay_x = hay | set(expand(list(hay)))
            mass = sum(self.s.idf(t, snap) for t in keys)
            got = sum(self.s.idf(t, snap) for t in keys if t in hay_x or any(x in hay_x for x in expand([t])[1:])
                      or any(w in hay for w in self.s.family(t, snap)))
            if mass and got / mass < 0.5:
                return False
        # a term / model year / amount stated in the question must exist in the source, not just a neighbour
        label = (row_label(u) + " " + u.ctx) if u.kind == "row" else u.text
        if q.terms and (u.kind == "row" or q.atype in ("rate", "fee")):
            if not (terms_months(label) & q.terms):
                return False
        if q.years and u.kind != "row":
            # a year asked about is a qualifier for prose too: the passage (or its paragraph / section) says it, or
            # it names no year and its article was published that year or early the next
            near = " ".join((u.text, u.full or "", u.ctx or ""))
            if not all(str(y) in near for y in q.years):
                other = [int(x) for x in re.findall(r"\b((?:19|20)\d\d)\b", near)]
                # published that year, and any year it mentions is earlier (background, not another year's figure)
                yrs_ok = bool(rec.pub) and all(rec.pub.year == y and all(o < y for o in other) for y in q.years)
                if q.atype in ("rate", "fee") or not yrs_ok:
                    return False
        if q.years and u.kind == "row":
            yr = year_range(label)
            if yr is None or not all(yr[0] <= y <= yr[1] for y in q.years):
                return False
        if q.amounts and u.kind == "row":
            ar = amount_range(label)
            if ar is not None and not all(ar[0] <= a <= ar[1] for a in q.amounts) and \
                    not (len(q.amounts) >= 2 and ar[0] <= min(q.amounts) and max(q.amounts) <= ar[1] + 1):
                return False
        v = VALIDATORS.get(q.atype or "")
        if v is None:
            return True
        txt = display(u) if u.kind != "row" else " ".join(u.cells)
        if q.atype == "count":
            noun = count_noun(q.raw, self.s.site_name())
            if noun:
                return noun_count(txt, noun) is not None
        return v.search(txt) is not None

    def _choose(self, q: Q, scored, snap):
        s = self.s
        best_sc = scored[0][0]
        if max(cov for _, _, cov in scored[:10]) < 0.3:
            return None
        if q.atype == "count":
            # prefer the most recently published statement of the count among strong candidates
            # every candidate, not just the top: the same boilerplate ("N branch offices") repeats on many pages and
            # the newest one may rank anywhere among them
            pool = [(sc, i) for sc, i, cov in scored if cov >= 0.3 and self._qualifies(s.units[i], q, snap)]
            if not pool:
                return None
            top = pool[0][0]
            near = [(sc, i) for sc, i in pool if sc >= 0.5 * top]
            # the same boilerplate sentence with another number ("It has N branch office locations") is the same
            # statement as of another date, however low it ranks: all its copies compete on recency
            tmpl = lambda i: re.sub(r"\d[\d,.]*", "#", " ".join(display(s.units[i].u).split()).lower())  # noqa: E731
            have = {tmpl(i) for _, i in near}
            near += [(sc, i) for sc, i in pool if sc < 0.5 * top and tmpl(i) in have]
            site = s.site_name()
            noun = count_noun(q.raw, site)
            def adjacent(i):
                return noun_count(display(s.units[i].u), noun) is not None
            def own(i):                                   # the count is about the organisation itself
                u = s.units[i].u
                return bool(site) and site in (display(u) + " " + u.full).lower()
            return max(near, key=lambda x: (adjacent(x[1]), own(x[1]), s.units[x[1]].pub or dt.date.min, x[0]))
        first = None
        for sc, i, cov in scored[:150]:
            if sc < 0.3 * best_sc:
                break
            if cov >= 0.3 and self._qualifies(s.units[i], q, snap):
                first = (sc, i)
                break
        topical = [t for t in set(q.toks) if t not in GENERIC and t not in QWORDS and t not in STOP
                   and not re.search(KV_WORDS.get(q.atype or "", "^$"), t) and t not in CONTACT_WORDS]
        if first and s.units[first[1]].kind == "article" and (q.atype in ("amount", "when", "wait", "count") or
                                                             CURRENT_Q_RX.search(q.raw) or
                                                             (q.atype is None and not q.yesno and not q.define)):
            # several articles state the same kind of fact as of their own date ("N branches", "board chair",
            # "so far this year"): among close candidates the newest settled statement wins over older ones and
            # over plans ("will open late fall") and running totals ("so far in 2025")
            pool = []
            for sc, i, cov in scored[:150]:
                if sc < 0.8 * first[0]:
                    break
                r = s.units[i]
                if r.u.kind != "row" and cov >= 0.3 and self._qualifies(r, q, snap):
                    pool.append((not TENTATIVE_RX.search(r.u.full or r.u.text), r.pub or dt.date.min, sc, i))
            if len(pool) > 1:
                best = max(pool, key=lambda x: (x[0], x[1], x[2]))
                first = (best[2], best[3])
        if first and q.atype in ("phone", "routing", "address"):
            # a labelled fact ("Phone: ...") beats a number mentioned in passing - unless the question is about a
            # topic ("call about student loans") that the labelled fact's page isn't about
            word = KV_WORDS[q.atype]
            for sc, i, cov in scored[:300]:
                if sc < 0.2 * first[0]:
                    break
                r = s.units[i]
                if r.u.kind == "kv" and re.search(word, r.u.cells[0], re.I) and self._qualifies(r, q, snap):
                    hay = r.toks | r.page_toks
                    if topical and sum(1 for t in topical if t in hay) < 0.5 * len(topical):
                        continue
                    return sc, i
        if first and q.atype in ("rate", "fee") and s.units[first[1]].u.kind != "row":
            # a figure from a table beats the same figure quoted in marketing copy; among such rows, the one whose
            # label carries the most of the question's words
            want = set(expand(q.toks))
            rows = []
            for sc, i, cov in scored[:150]:
                if sc < 0.85 * first[0]:
                    break
                r = s.units[i]
                if r.u.kind == "row" and cov >= 0.3 and self._qualifies(r, q, snap):
                    rows.append((len(want & set(tokens(row_label(r.u)))), sc, i))
            if rows:
                _, sc, i = max(rows, key=lambda x: (x[0], x[1]))
                return sc, i
        return first

    def _with_neighbour(self, rec: UnitRec, snap: int, text: str, q: Q) -> str:
        """A short passage reads better with the next one from the same section; an FAQ answer, a bullet list
        and (for "what is" questions) a paragraph are given whole."""
        page = self.s.snap_pages[snap].get(rec.canon)
        if not page:
            return text
        units = page.units
        k0 = next((k for k, u in enumerate(units) if u is rec.u), None)
        if k0 is None:
            k0 = next((k for k, u in enumerate(units) if u.text == rec.u.text and u.ctx == rec.u.ctx), None)
        if k0 is not None:
            u0 = units[k0]
            if u0.faq:                                   # whole FAQ answer, in page order
                a = k0
                while a > 0 and units[a - 1].faq == u0.faq and units[a - 1].kind == "text":
                    a -= 1
                parts, seen = [], set()
                for u in units[a:]:
                    if u.faq != u0.faq or u.kind != "text":
                        break
                    piece = u.full or u.text
                    if piece not in seen:
                        seen.add(piece)
                        parts.append(piece)
                joined = " ".join(parts)
                if len(joined) > len(text):
                    return joined if len(joined) <= 700 else joined[:700].rsplit(" ", 1)[0] + "…"
            if u0.text.lstrip().startswith(("- ", "* ", "• ")):      # the whole bullet list
                a = b = k0
                is_item = lambda u: u.kind == "text" and u.ctx == u0.ctx and u.text.lstrip().startswith(("- ", "* ", "• "))
                while a > 0 and is_item(units[a - 1]):
                    a -= 1
                while b + 1 < len(units) and is_item(units[b + 1]):
                    b += 1
                if b > a:
                    items = [units[k].text.lstrip("-*• ").strip().rstrip(".;") for k in range(a, min(b + 1, a + 10))]
                    head = (u0.ctx.split(" > ")[-1] + ": ") if u0.ctx else ""
                    return head + "; ".join(items)
            bullet = lambda u: u.kind == "text" and u.text.lstrip().startswith(("- ", "* ", "• "))  # noqa: E731
            if (u0.full or u0.text).rstrip().endswith(":") and k0 + 1 < len(units) and bullet(units[k0 + 1]):
                # a lead-in ("You can split your loan into two loans at once:") is answered by the list after it
                items = []
                for u in units[k0 + 1:k0 + 13]:
                    if not bullet(u):
                        break
                    items.append(u.text.lstrip("-*• ").strip().rstrip(".;"))
                return " ".join((u0.full or text).split()) + " " + "; ".join(items)
            if q is not None and q.list_intent and u0.ctx and " > " in u0.ctx:
                # a list question answered by one section: name its sibling sections too ("Business Free Checking",
                # "1st Rate Checking", "Analysis Checking" are the options; each section is one of them)
                parent = u0.ctx.rsplit(" > ", 1)[0]
                sibs = list(dict.fromkeys(u.ctx.rsplit(" > ", 1)[1].strip() for u in units
                                          if u.ctx and u.ctx.rsplit(" > ", 1)[0] == parent and " > " in u.ctx))
                sibs = [x for x in sibs if 2 < len(x) <= 60 and not FAQ_Q_RX.match(x)]
                if 3 <= len(sibs) <= 12:
                    return f"{parent.split(' > ')[-1]}: " + "; ".join(sibs) + ". " + text
            if q is not None and q.define and u0.full and len(u0.full) > len(text):
                full = u0.full
                return full if len(full) <= 600 else full[:600].rsplit(" ", 1)[0] + "…"
        if len(text) >= 160:
            return text
        for k, u in enumerate(page.units):
            if u is rec.u and k + 1 < len(page.units):
                nxt = page.units[k + 1]
                if nxt.kind == "text" and nxt.ctx == rec.u.ctx:
                    return text + " " + nxt.text
                break
        return text

    # ---- hours -----------------------------------------------------------------------------------------
    def _hours_overview(self, q: Q, snap: int, sid: str, scored):
        """Hours asked without naming a branch: list the distinct schedules the site gives (they vary by branch)."""
        s = self.s
        ql = q.raw.lower()
        want = "drive" if re.search(r"drive[- ]?(?:up|thru|through)", ql) else "lobby" if "lobby" in ql else None
        found = {}                                   # normalised schedule -> [display, score, recs]
        top = None
        for sc, i, cov in scored[:200]:
            rec = s.units[i]
            if rec.u.kind == "row" or cov < 0.3 or not self._qualifies(rec, q, snap):
                continue
            if top is None:
                top = sc
            if sc < 0.5 * top:
                break
            sched = hours_block(rec.u.text, want)
            if not sched:
                continue
            key = re.sub(r"[^a-z0-9]", "", sched.lower())
            if key in found:
                found[key][1] += sc
                found[key][2].append(rec)
            else:
                found[key] = [sched, sc, [rec]]
        if not found:
            return None
        ranked = sorted(found.values(), key=lambda x: -x[1])[:4]
        if len(ranked) == 1:
            text = ranked[0][0]
        else:
            text = ("No. " if q.yesno else "") + "Hours vary by branch. " + " | ".join(f"Schedule {k + 1}: {v[0]}" for k, v in enumerate(ranked)) \
                + ". Ask us about a specific branch for its exact hours."
        srcs, seen = [], set()
        for v in ranked:
            for r in v[2]:
                if r.url not in seen:
                    seen.add(r.url)
                    srcs.append((r.url, sid, r.u.text[:200]))
        return Result(text, srcs[:4])

    # ---- urls ------------------------------------------------------------------------------------------
    def _url_answer(self, q: Q, snap: int, sid: str, scored):
        s = self.s
        ql = q.raw.lower()
        see = bool(re.search(r"\b(see|view|check|find|look|locat|rates?)\w*", ql))
        act = bool(re.search(r"\b(join|apply|sign up|enroll|open|become|upload|submit|start|resume|finish|log ?in|"
                             r"sign in|application|pick up)\b", ql))
        key = set(expand([t for t in q.toks if t not in ("link", "url", "website", "page", "online", "use")]))
        pages = s.snap_pages[snap]
        # the question names a page by its exact title ("What's the link for Membership?"): that page
        want = {t for t in key if t not in GENERIC and t not in QWORDS}
        if want:
            exact = [pp for pp in pages.values() if pp.kind not in ("article", "listing", "expired", "calculator")
                     and {t for t in content_tokens(pp.title) if t not in GENERIC} == want]
            if len(exact) == 1:
                pp = exact[0]
                return Result(f"You can use {pp.url.split('#')[0]}", [(pp.url, sid, pp.title)])
        # how relevant each page's own content is (not just its address): sum of its units' scores
        page_rel = {}
        for sc, i, cov in scored[:300]:
            c = s.units[i].canon
            page_rel[c] = page_rel.get(c, 0.0) + sc
        top_rel = max(page_rel.values()) if page_rel else 1.0
        wants_table = bool(q.metrics & NUMERIC_METRICS)

        def table_match(canon):
            pp = pages.get(canon)
            if not pp or not wants_table:
                return 0.0
            hits = sum(1 for u in pp.units if u.kind == "row"
                       and set(tokens(" ".join((row_label(u), u.ctx, u.near, " ".join(u.header))))) & key)
            return min(1.0, hits / 4)
        best = None
        # pages whose title matches the question are candidates even if none of their passages ranked (calculators
        # and tools have little text)
        top_sc = scored[0][0] if scored else 1.0
        tmatch = []
        for pp in pages.values():
            if pp.kind in ("article", "listing", "expired"):
                continue
            tt = {t for t in content_tokens(pp.title) if t not in GENERIC}
            if tt and want:
                j = len(tt & want) / len(tt | want)
                if j >= 0.34:
                    tmatch.append((j, pp))
        for j, pp in sorted(tmatch, key=lambda x: -x[0])[:5]:
            v = top_sc * (0.6 + j)
            if pp.kind == "calculator" and "calculator" not in ql and "estimate" not in ql:
                v *= 0.5
            if best is None or v > best[0]:
                rec0 = next((s.units[i] for i in s.snap_units[snap] if s.units[i].canon == pp.canon), None)
                if rec0 is not None:
                    best = (v, pp.url, rec0)
        for sc, i, cov in scored[:120]:
            rec = s.units[i]
            cands = [(a, l) for a, l in rec.u.links if l.startswith("http")]
            cands.append((rec.title, rec.url))
            for anchor, link in cands:
                lt = set(tokens(anchor + " " + re.sub(r"[/._?=&-]", " ", link)))
                m = len(lt & key) / (len(key) or 1)
                al = anchor.lower()
                if see and re.search(r"see|detail|view|rate|locat|find", al + link):
                    m += 0.3
                if act and re.search(r"join|apply|open|enroll|member", al + link):
                    m += 0.3
                if link == rec.url:
                    m += 0.1
                host = re.sub(r"^www\.", "", re.sub(r"^https?://", "", link).split("/")[0].lower())
                site = re.sub(r"^www\.", "", re.sub(r"^https?://", "", rec.url).split("/")[0].lower())
                if host == site and canon_url(link) not in pages:
                    m *= 0.4                            # internal link to a page this crawl doesn't have: likely dead
                if act and host != site and host.endswith(site):
                    m += 0.8                            # applications, uploads and logins live on the site's subdomains
                tkind = classify(link, "", "")
                if tkind in ("article", "listing", "expired"):
                    m *= 0.5
                if tkind == "calculator" and "calculator" not in ql and "estimate" not in ql:
                    m *= 0.5
                tc = canon_url(link)
                if host == site and tc in pages and not act:     # "where can I see X": the page about X
                    m += 0.6 * page_rel.get(tc, 0.0) / top_rel + 0.5 * table_match(tc)
                v = sc * (0.3 + m)
                if best is None or v > best[0]:
                    best = (v, link, rec)
        if not best:
            return None
        _, link, rec = best
        link = link.split("#")[0]
        srcs = [(rec.url, sid, rec.u.text[:200])]
        if canon_url(link) in pages and canon_url(link) != rec.canon:
            srcs.insert(0, (link, sid, None))          # the page the link points to is the source too
        return Result(f"You can use {link}", srcs)

    # ---- tables ----------------------------------------------------------------------------------------
    def _table_rows(self, rec: UnitRec, snap: int):
        page = self.s.snap_pages[snap].get(rec.canon)
        if page is None:
            return []
        return [u for u in page.units if u.kind == "row" and u.table == rec.u.table and u.ctx == rec.u.ctx]

    def _pick_rows(self, rows, q: Q):
        if not rows:
            return []
        qset = set(expand(q.toks))
        lab_toks = [set(tokens(row_label(u))) for u in rows]
        common = set.intersection(*lab_toks) if lab_toks else set()
        scored = []
        for u, lt in zip(rows, lab_toks):
            dist = lt - common
            hit = len(dist & qset)
            miss = len([t for t in dist - qset if not t.isdigit() and t not in GENERIC])
            m = qual_match(row_label(u) + " " + u.ctx, q)
            scored.append((m * (1 + hit - 0.3 * miss), u))
        top = max(sc for sc, _ in scored)
        keep = [u for sc, u in scored if sc >= top * 0.999]
        if q.superl and len(keep) > 1:
            vals = [(metric_number(u, q), u) for u in keep]
            vals = [(v, u) for v, u in vals if v is not None]
            if vals:
                pick = max(vals, key=lambda x: x[0]) if q.superl == "max" else min(vals, key=lambda x: x[0])
                keep = [pick[1]]
        return keep[:12]

    def _row_answer(self, best: UnitRec, q: Q, snap: int, sid: str, scored) -> Result:
        s = self.s
        tables = [(best, self._pick_rows(self._table_rows(best, snap), q))]
        if q.list_intent and not (q.terms or q.amounts or q.years or q.superl):
            seen = {(best.canon, best.u.table, best.u.ctx)}
            best_sc = scored[0][0]
            for sc, i, _ in scored[1:80]:
                r = s.units[i]
                key = (r.canon, r.u.table, r.u.ctx)
                if r.u.kind != "row" or key in seen or sc < 0.5 * best_sc or r.kind not in ("rates", "product", "promo"):
                    continue
                seen.add(key)
                rows = self._pick_rows(self._table_rows(r, snap), q)
                labels_have = {row_label(u).lower() for _, rs in tables for u in rs}
                rows = [u for u in rows if row_label(u).lower() not in labels_have]
                if rows:
                    tables.append((r, rows))
                if len(tables) >= 4:
                    break
        parts, sources = [], []
        for rec, rows in tables:
            items = []
            for u in rows:
                vals = row_values(u, q)
                lab = row_label(u)
                ctx_last = rec.u.ctx.split(" > ")[-1] if rec.u.ctx else ""
                if ctx_last and ctx_last.lower() not in lab.lower() and len(tables) > 1:
                    lab = f"{ctx_last} {lab}"
                items.append(f"{lab}: {', '.join(vals)}" if vals else u.text)
            if items:
                parts.append("; ".join(items))
                sources.append((rec.url, sid, rows[0].text[:200] if rows else ""))
        text = ". ".join(parts) + "."
        if q.yesno and q.pcts:
            shown = set(pct_values(text))
            text = ("Yes. " if set(q.pcts) & shown else "No. ") + text
        first_vals = pct_values(text)[:1]
        if first_vals:
            for sc, i, _ in scored[:80]:
                r = s.units[i]
                if r.url in {x[0] for x in sources} or r.kind not in ("rates", "product", "promo"):
                    continue
                if first_vals[0] in pct_values(r.u.text) and len(sources) < 3:
                    sources.append((r.url, sid, r.u.text[:200]))
        return Result(text, sources)
