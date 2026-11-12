"""Text normalisation, tokenising and small domain vocabularies (generic to credit-union sites)."""
from __future__ import annotations

import re
import unicodedata

DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−"), "-")
SPACES = dict.fromkeys(map(ord, "     ​"), " ")
QUOTES = {ord("’"): "'", ord("‘"): "'", ord("“"): '"', ord("”"): '"'}


IN_WORD_DOLLAR = re.compile(r"(?<=[A-Za-z])\$(?=[a-z])")


def clean(s: str) -> str:
    """Unicode-normalise dashes, spaces and quotes; keep case. 'Home$tart' -> 'Homestart'."""
    s = unicodedata.normalize("NFKC", s or "")
    s = s.translate(DASHES).translate(SPACES).translate(QUOTES)
    return IN_WORD_DOLLAR.sub("s", s)


LINK_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)|\[([^\]]*)\]\(([^)\s]*)[^)]*\)")
FOOTNOTE_NUM_RE = re.compile(r"(\d%|\bAPR|\bAPY|\))(\d{1,2})(?=\s|\||$|,|\))")
MD_RE = re.compile(r"\*\*|__|\\(?=[#*_|\[\]()$-])|`")


def strip_markdown(s: str) -> str:
    s = LINK_RE.sub(lambda m: m.group(1) or "", s)
    s = MD_RE.sub("", s)
    return s


def strip_footnotes(s: str) -> str:
    """'4.00%1' -> '4.00%', 'APR1' -> 'APR', '(APY)1' -> '(APY)'."""
    return FOOTNOTE_NUM_RE.sub(r"\1", s)


def links(s: str) -> list[tuple[str, str]]:
    return [(m.group(1) or "", m.group(2)) for m in LINK_RE.finditer(s) if m.group(2)]


STOP = set("""a an the their they them his her hers theirs ours yours of to in on for and or is are was were be been being do does did can could would should will
shall may might must i me my we our you your it its this that these those there here what which who whom whose when
where why how any all some much many about with from by at as into than then so if not no yes please tell know like
get give want need help am im i'm have has had dupaco dupacos dupaco's s us let me also just current currently now
right today offer offered provide show find see look looking question ask asking info information
mid early late beginning end middle around approximately roughly""".split())

# Words that describe the metric or are too generic to identify a product.
GENERIC = set("""rate rates interest apr apy yield annual percentage dividend dividends loan loans account accounts
credit union bank banking member members fee fees cost price amount much many number money option options
product products program programs service services available offer special new latest per year month""".split())

SYNONYMS = {
    "intro": ["introductory"], "introductory": ["intro"], "cd": ["certificate"], "cds": ["certificate"], "certificates": ["certificate"],
    "share": ["share"], "heloc": ["home", "equity", "line"], "car": ["auto", "vehicle"],
    "vehicle": ["auto"], "auto": ["vehicle"], "automobile": ["auto", "vehicle"],
    "truck": ["auto", "vehicle"], "mortgage": ["home", "mortgage"], "mortgages": ["home", "mortgage"],
    "house": ["home"], "ira": ["ira", "retirement"], "visa": ["card", "credit"],
    "atm": ["atm"], "hours": ["hours"], "open": ["hours", "open"], "phone": ["phone", "call"],
    "telephone": ["phone"], "call": ["phone"], "reach": ["contact", "phone"], "contact": ["phone"], "routing": ["routing"], "aba": ["routing"], "branch": ["branch", "location"],
    "branches": ["branch", "location"], "location": ["branch"], "locations": ["branch"],
    "join": ["join", "membership"], "membership": ["member", "join"], "eligible": ["eligibility", "join"],
    "eligibility": ["eligible", "join"], "boat": ["recreational", "boat"], "rv": ["recreational", "rv"],
    "motorcycle": ["motorcycle", "auto"], "student": ["student"], "college": ["student"],
    "university": ["student"], "drive": ["drive"], "lobby": ["lobby"], "savings": ["savings"],
    "checking": ["checking"], "debit": ["debit", "card"], "jumbo": ["jumbo"],
}


def stem(w: str) -> str:
    if w in STOP:
        return w
    if len(w) > 5 and w.endswith(("ches", "shes", "sses", "xes", "zes")):
        return w[:-2]                       # branches -> branch, boxes -> box
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


WORD_RE = re.compile(r"[a-z0-9]+(?:[/'][a-z0-9]+)*")


def tokens(s: str) -> list[str]:
    return [stem(w) for w in WORD_RE.findall(clean(s).lower().replace("'s ", " "))]


def content_tokens(s: str) -> list[str]:
    return [t for t in tokens(s) if t not in STOP and len(t) > 1]


def expand(toks: list[str]) -> list[str]:
    out = list(toks)
    for t in toks:
        for syn in SYNONYMS.get(t, []):
            out.append(stem(syn))
    return out
