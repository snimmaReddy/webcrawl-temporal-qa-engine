"""Qualifiers: dates, terms, amounts, year ranges and metrics, parsed the same way from questions and table rows."""
from __future__ import annotations

import calendar
import datetime as dt
import re

MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})
MONTHS["sept"] = 9
MON = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"

DATE_LONG = re.compile(MON + r"\.?\s+(\d{1,2}),?\s+(\d{4})", re.I)
DATE_NUM = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b")
DATE_ISO = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")
MONTH_YEAR = re.compile(r"\b" + MON + r"\.?,?\s+(\d{4})\b", re.I)


def _mk(y: int, m: int, d: int):
    try:
        return dt.date(y, m, d)
    except ValueError:
        return None


def dates_in(text: str) -> list[dt.date]:
    out = []
    for m in DATE_LONG.finditer(text):
        d = _mk(int(m.group(3)), _month_num(m.group(1)), int(m.group(2)))
        if d:
            out.append(d)
    for m in DATE_NUM.finditer(text):
        y = int(m.group(3))
        y = y + 2000 if y < 100 else y
        d = _mk(y, int(m.group(1)), int(m.group(2)))
        if d:
            out.append(d)
    for m in DATE_ISO.finditer(text):
        d = _mk(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if d:
            out.append(d)
    return out


def _month_num(word: str) -> int:
    w = word.lower().rstrip(".")
    return MONTHS.get(w, MONTHS.get(w[:3], 0))


def question_date(q: str):
    """A date the question pins itself to ('as of May 2026', 'on 5/31/2026'), as the end of that period.
    Returns (date or None, question text with that phrase removed)."""
    for rx in (DATE_LONG, DATE_ISO, DATE_NUM):
        m = rx.search(q)
        if m:
            d = dates_in(m.group(0))
            if d:
                return d[0], q[: m.start()] + q[m.end():]
    m = MONTH_YEAR.search(q)
    if m:
        y, mo = int(m.group(2)), _month_num(m.group(1))
        if mo:
            return dt.date(y, mo, calendar.monthrange(y, mo)[1]), q[: m.start()] + q[m.end():]
    return None, q


# ---- terms -------------------------------------------------------------------------------------------
TERM_RE = re.compile(r"\b(\d{1,3})\s*[- ]?\s*(months?|mos?\b|mo\.|years?|yrs?|yr\.)", re.I)
WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
            "ten": 10, "twelve": 12, "fifteen": 15, "eighteen": 18, "twenty": 20, "thirty": 30}
WORD_TERM_RE = re.compile(r"\b(" + "|".join(WORD_NUM) + r")\s*[- ]\s*(months?|years?)\b", re.I)


def terms_months(text: str) -> set[int]:
    out = set()
    for m in TERM_RE.finditer(text):
        n = int(m.group(1))
        out.add(n * 12 if m.group(2).lower().startswith("y") else n)
    for m in WORD_TERM_RE.finditer(text):
        n = WORD_NUM[m.group(1).lower()]
        out.add(n * 12 if m.group(2).lower().startswith("y") else n)
    return out


# ---- money amounts and ranges ------------------------------------------------------------------------
AMT = r"\$\s?(\d[\d,]*(?:\.\d+)?)\s*(k|m|thousand|million)?\b"
AMT_RE = re.compile(AMT, re.I)


def _num(s: str, suf: str | None = None) -> float:
    v = float(s.replace(",", ""))
    if suf:
        v *= 1e3 if suf.lower() in ("k", "thousand") else 1e6
    return v


WORD_AMT_RE = re.compile(r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|"
                         r"twenty[- ]five|thirty|forty|fifty|sixty|seventy|eighty|ninety|a hundred|one hundred|"
                         r"\d+)\s+(thousand|million|hundred)(?:\s+dollars)?\b", re.I)
WORD_VAL = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "twenty-five": 25, "twenty five": 25, "thirty": 30,
            "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "a hundred": 100,
            "one hundred": 100}
BELOW_RE = re.compile(r"\b(?:below|under|less than|lower than|drops? (?:below|under)|falls? (?:below|under))\s*$", re.I)
ABOVE_RE = re.compile(r"\b(?:above|over|more than|greater than|exceeds?|at least)\s*$", re.I)


def amounts(text: str) -> list[float]:
    """Dollar amounts in a question; "below $25,000" means just under it, "ten thousand dollars" is $10,000."""
    out = []
    for m in AMT_RE.finditer(text):
        v = _num(m.group(1), m.group(2))
        before = text[max(0, m.start() - 25): m.start()]
        if BELOW_RE.search(before):
            v -= 0.01
        elif ABOVE_RE.search(before):
            v += 0.01
        out.append(v)
    for m in WORD_AMT_RE.finditer(text):
        n = m.group(1).lower()
        base = float(n) if n.isdigit() else WORD_VAL.get(n, 0)
        mult = {"thousand": 1e3, "million": 1e6, "hundred": 100}[m.group(2).lower()]
        if base and not AMT_RE.search(m.group(0)):
            out.append(base * mult)
    return out


RANGE_RE = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)\s*(?:-|to)\s*\$?\s?(\d[\d,]*(?:\.\d+)?)")
PLUS_RE = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)\s*(?:\+|or more|and (?:up|above|over))", re.I)
LESS_RE = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)\s*(?:or less|and under|or below)|(?:up to|under|less than)\s*\$\s?(\d[\d,]*(?:\.\d+)?)", re.I)


def amount_range(text: str):
    """(lo, hi) of a tier written in a row, or None."""
    m = RANGE_RE.search(text)
    if m:
        lo, hi = _num(m.group(1)), _num(m.group(2))
        if hi < lo and hi < 1000 <= lo:  # "$10,000-24,999" style is fine; "$10-24" etc. ignored
            return None
        return (lo, hi)
    m = PLUS_RE.search(text)
    if m:
        return (_num(m.group(1)), float("inf"))
    m = LESS_RE.search(text)
    if m:
        return (0.0, _num(m.group(1) or m.group(2)))
    return None


# ---- years (model years etc.) -----------------------------------------------------------------------
YEAR_RANGE_RE = re.compile(r"\b((?:19|20)\d{2})\s*-\s*((?:19|20)?\d{2})\b")
YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")


def year_range(text: str):
    m = YEAR_RANGE_RE.search(text)
    if m:
        a = int(m.group(1))
        b = m.group(2)
        b = int(b) if len(b) == 4 else (a // 100) * 100 + int(b)
        return (min(a, b), max(a, b))
    m = YEAR_RE.search(text)
    if m:
        y = int(m.group(1))
        if re.search(r"older|earlier|prior|before", text, re.I):
            return (0, y)
        if re.search(r"newer|later|after", text, re.I):
            return (y, 9999)
        return (y, y)
    return None


def question_years(text: str) -> list[int]:
    return [int(y) for y in YEAR_RE.findall(text)]


# ---- metrics ---------------------------------------------------------------------------------------
def metric_of_header(h: str) -> str | None:
    h = h.lower()
    if "apy" in h or "annual percentage yield" in h:
        return "apy"
    if re.search(r"\bapr\b", h) or "annual percentage rate" in h:
        return "apr"
    if "rate" in h or "dividend" in h:
        return "rate"
    if "fee" in h or "cost" in h or "charge" in h:
        return "fee"
    if "payment" in h:
        return "payment"
    if "point" in h:
        return "points"
    if "min" in h or "balance to" in h:
        return "minimum"
    return None


# metric words that name something else: compounds ("adjustable-rate", "fee-free"), product names ("1st Rate
# Checking"), and "without paying a fee" (asks how to avoid it, not its amount)
MASK_RX = re.compile(r"\b[\w]+-(?:rates?|fees?|interest)\b|\b(?:fees?|interest|cost|charge)-(?:free|based)\b"
                     r"|\b1st rate\b|\bfirst rate\b|\bwithout (?:paying|a fee|fees|any fees|being charged)\b"
                     r"|\b(?:avoid|waive|skip)\w* (?:a |the |any |paying )?(?:fees?|charges?)\b|\bfree of charge\b"
                     r"|\bearn\w* (?:the |a |an |my |up to )?(?:\$[\d,]+ )?(?:bonus|cash|reward|points|rebate|money back)\b"
                     r"|\b(?:interested|interesting)\b|\blearn\w*\b")


def metrics_asked(q: str) -> set[str]:
    ql = MASK_RX.sub(" ", q.lower())
    out = set()
    if re.search(r"\bapys?\b|annual percentage yield|yield", ql):
        out.add("apy")
    if re.search(r"\baprs?\b|annual percentage rate", ql):
        out.add("apr")
    pays = re.search(r"\bpays\b|\bpaying\b(?!\s+(?:off|for|back|bills?))|\bwhat (?:does|do|will|would) .{0,40}\bpay\b"
                     r"(?!\s+(?:for|off|back|bills?|my|your|vendors|employees|someone))", ql)
    if re.search(r"\brates?\b|\binterest\b|\bdividends?\b|\bearn(?:s|ing|ed)?\b", ql) or pays:
        out.add("rate")
    if re.search(r"\bfees?\b|\bcosts?\b|\bcharges?\b|how much (?:does|is|do)\b(?!.*\bdown payment)", ql):
        out.add("fee")
    if re.search(r"payment", ql):
        out.add("payment")
    if re.search(r"minimum|min\.|to open|opening deposit", ql):
        out.add("minimum")
    return out


def superlative(q: str) -> str | None:
    ql = q.lower()
    if re.search(r"\b(highest|best|top|most|maximum|max)\b", ql):
        return "max"
    if re.search(r"\b(lowest|cheapest|least|minimum rate|smallest)\b", ql):
        return "min"
    return None


PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")


def pct_values(text: str) -> list[float]:
    return [float(x) for x in PCT_RE.findall(text)]
