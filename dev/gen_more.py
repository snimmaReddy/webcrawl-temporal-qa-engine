"""More held-out question families built from the snapshots by templates (no LLM, no cost).

Families -> dev/gen_more.jsonl (harness format; ids gm-<family>-<n>, paraphrases share a `group`):
  city      "Which Dupaco branches are in <city>?"          must = street numbers from the branch locator
  bphone    "What is the phone number for the <X> branch?"   from branch press releases (unique names only)
  bhours    "What are the Saturday lobby hours at <X>?"      from the same releases
  faq       the site's own FAQ questions (FAQ text NOT indexed when scored? no: scored as-is, see note)
            must = numbers / $ / % / proper names in the answer that the question doesn't contain
  url       "What's the link for <page title>?"             must = the page's path
  terms     "What terms do you offer for <product>?"        must = three row labels of that rates table
  refuse    terms not offered, cities without a branch, products never mentioned (checked in all 5 crawls)
Note: FAQ questions are asked verbatim, so they are easier than real paraphrases; dev/faq_eval.py is the
version with FAQ text hidden from the index.
    python dev/gen_more.py
"""
import json, random, re, sys
from collections import defaultdict, Counter
from pathlib import Path

sys.path[:0] = ["submission", "harness"]
import run_eval
from kb.pages import parse_page, canon_url
from kb.quals import metric_of_header, terms_months
from kb.text import content_tokens, STOP

rng = random.Random(5)
snaps = run_eval.load_snapshots(Path("snapshots"))
last = snaps[-1]
pages = {canon_url(p.url): (p, parse_page(p.url, p.title, p.markdown)) for p in last.pages}
ALL = " ".join(p.markdown for s in snaps for p in s.pages).lower()
items = []


def add(fam, qs, must, must_not=(), cat="current_value", src=None, as_of=None):
    g = f"gm-{fam}-{sum(1 for x in items if x['id'].startswith('gm-' + fam + '-'))}"
    for k, q in enumerate(qs):
        items.append({"id": f"{g}{'abc'[k]}", "question": q, "as_of": as_of, "must_include": list(must),
                      "must_not_include": list(must_not), "category": cat, "weight": 1.0,
                      "expected_sources": [src] if src else [], "group": g if len(qs) > 1 else None,
                      "family": fam})


# ---- city: branch locator on the contact page ---------------------------------------------------------
loc = re.compile(r"Dupaco Branch\s*\n\s*([0-9][^\n,]+),\s*([A-Za-z .]+),\s*([A-Z]{2}),\s*\d{5}")
by_city = defaultdict(set)
for p in last.pages:
    for m in loc.finditer(p.markdown):
        by_city[(m.group(2).strip(), m.group(3))].add(m.group(1).strip())
for (city, st), streets in sorted(by_city.items()):
    nums = sorted({s.split()[0] for s in streets})[:3]
    add("city", [f"Which Dupaco branches are in {city}?", f"Is there a Dupaco branch in {city}, {st}? Where is it?"],
        nums, cat="branch_info", src="https://www.dupaco.com/contact")

# ---- branch phone / hours from press releases --------------------------------------------------------
phone_rx = re.compile(r"phone number (?:for the|is) ([^.]*?)(?: branch| Learning Lab| learning lab)? is (\(\d{3}\) \d{3}-\d{4})|"
                      r"The ([A-Z][\w .]+?) (?:Dupaco )?branch is located at [^,]+, and the phone number is (\(\d{3}\) \d{3}-\d{4})")
found = defaultdict(set)
hours = {}
for p, pp in pages.values():
    md = p.markdown
    for m in re.finditer(r"[Tt]he phone number for the ([A-Z][\w .]+?) branch is (\(\d{3}\) \d{3}-\d{4})|"
                         r"The ([A-Z][\w .]+?) (?:Dupaco )?branch is located at [^,]+, and the phone number is (\(\d{3}\) \d{3}-\d{4})", md):
        name, ph = (m.group(1) or m.group(3)).strip(), (m.group(2) or m.group(4))
        found[name].add((ph, p.url))
        h = re.search(r"Branch hours are:(.{0,400})", md[m.end():m.end() + 500], re.S)
        if h:
            sat = re.search(r"Saturday[^:\n]*:\s*([^\n]+)", h.group(1))
            if sat:
                hours[name] = (sat.group(1).strip().replace("–", "-"), p.url)
for name, vals in sorted(found.items()):
    if len({v[0] for v in vals}) != 1:
        continue                                  # same name, different numbers: ambiguous
    ph, url = next(iter(vals))
    add("bphone", [f"What is the phone number for the {name} branch?", f"How do I call the {name} branch?"],
        [ph], cat="entity_attribution", src=url)
    if name in hours:
        t, url = hours[name]
        first = re.search(r"\d{1,2}(?::\d{2})?\s*[ap]\.m\.|noon|[Cc]losed", t)
        if first:
            add("bhours", [f"What are the Saturday lobby hours at the {name} branch?",
                           f"Is the {name} branch open on Saturday?"], [first.group(0)], cat="branch_info", src=url)

# ---- FAQ pairs ----------------------------------------------------------------------------------------
DIST = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?|\b\d+(?:\.\d+)?%|\b\d{1,2}(?::\d{2})?\s*[ap]\.m\.|\b\d[\d,]{2,}\b|"
                  r"\b(?:[A-Z][a-z]+\s){1,3}(?:Pay|Banking|Online|Account|Savings|Checking|Card|Loan|Program)\b")
faq_seen = set()
for canon, (p, pp) in sorted(pages.items()):
    if pp.kind in ("calculator", "expired", "listing"):
        continue
    groups = defaultdict(list)
    for u in pp.units:
        if u.faq and u.kind == "text":
            groups[u.faq].append(u.full or u.text)
    for q, parts in groups.items():
        if q in faq_seen:
            continue
        ans = " ".join(dict.fromkeys(parts))
        ql = q.lower()
        must = [m.group(0).strip() for m in DIST.finditer(ans) if m.group(0).strip().lower() not in ql
                and not re.fullmatch(r"(?:19|20)\d\d", m.group(0).strip())]     # a year is usually a card's date
        must = list(dict.fromkeys(must))[:2]
        if must:
            faq_seen.add(q)
            add("faq", [q], must, cat="how_to", src=p.url)

# ---- url: unique product/hub titles -------------------------------------------------------------------
titles = Counter(pp.title for _, pp in pages.values() if pp.kind in ("product", "hub"))
for canon, (p, pp) in sorted(pages.items()):
    if pp.kind in ("product", "hub") and titles[pp.title] == 1 and len(pp.title) > 6:
        path = re.sub(r"^https?://", "", p.url).split("?")[0].split("#")[0].rstrip("/")
        if "/" not in path:
            continue                              # the home page: nothing to check
        add("url", [f"What's the link for {pp.title}?", f"Which page on the website covers {pp.title}?"],
            [path], cat="url", src=p.url)

# ---- terms offered per rates table --------------------------------------------------------------------
tables = defaultdict(list)
for canon, (p, pp) in pages.items():
    if pp.kind != "rates":
        continue
    for u in pp.units:
        if u.kind == "row" and u.cells:
            mets = [metric_of_header(h) for h in u.header] + [None] * len(u.cells)
            lab = " ".join(c for c, m in zip(u.cells, mets) if m is None and c).strip()
            prod = [x for x in u.ctx.split(" > ") if x.strip()]
            if lab and prod and terms_months(lab):
                tables[(canon, u.table, prod[-1])].append((lab, p.url))
for (canon, t, prod), rows in sorted(tables.items()):
    labs = list(dict.fromkeys(l for l, _ in rows))
    if len(labs) >= 3 and len(prod) > 4:
        pick = [labs[0], labs[len(labs) // 2], labs[-1]]
        add("terms", [f"What terms do you offer for {prod}?", f"Which term lengths are available for {prod}?"],
            pick, cat="list_completeness", src=rows[0][1])

# ---- refusals ------------------------------------------------------------------------------------------
offered = {n for (canon, t, prod), rows in tables.items() for l, _ in rows for n in terms_months(l)}
cert_prods = sorted({prod for (c, t, prod) in tables if re.search(r"certificate", prod, re.I)})
for n in (9, 21, 27, 33, 42, 54, 66, 96):
    if n not in offered and cert_prods:
        add("refuse", [f"What is the APY on a {n}-month certificate?", f"Do you have a {n} month CD? What does it pay?"],
            [], cat="should_refuse")
CITIES = ["Ames", "Ankeny", "Sioux City", "Council Bluffs", "Davenport", "Bettendorf", "Burlington", "Ottumwa",
          "Mason City", "Fort Dodge", "Muscatine", "Marshalltown", "Rockford", "Peoria", "Springfield", "Galesburg",
          "Moline", "La Crosse", "Eau Claire", "Green Bay", "Milwaukee", "Oshkosh", "Kenosha", "Minneapolis",
          "Rochester", "Omaha", "Lincoln", "Des Plaines", "Naperville", "Aurora"]
for c in CITIES:
    if c.lower() not in ALL:
        add("refuse", [f"What are the hours of the Dupaco branch in {c}?", f"Where is your {c} branch located?"],
            [], cat="should_refuse")
PRODUCTS = ["cryptocurrency account", "bitcoin", "gold IRA", "precious metals", "student checking", "teen credit card",
            "boat insurance quote", "safe deposit box", "foreign currency exchange", "Zelle", "Venmo", "PayPal",
            "reverse mortgage", "payday loan", "pawn loan", "travelers checks", "brokerage account", "529 plan",
            "health savings account", "wedding loan", "pet insurance", "timeshare loan", "solar panel loan",
            "airplane loan", "crypto ATM", "coin counting machine"]
COMMON = {"account", "loan", "card", "insurance", "plan", "quote", "machine", "exchange", "checks", "box", "deposit",
          "credit", "savings", "checking", "health"}
for pr in PRODUCTS:
    distinct = [w for w in re.findall(r"[a-z0-9]+", pr.lower()) if len(w) >= 3 and w not in COMMON]
    if pr.lower() not in ALL and distinct and not any(re.search(r"\b" + w[:5], ALL) for w in distinct):
        add("refuse", [f"Does Dupaco offer a {pr}? What does it cost?", f"How do I get a {pr} from Dupaco?"],
            [], cat="should_refuse")

Path("dev/gen_more.jsonl").write_text("".join(json.dumps(x) + "\n" for x in items))
print("wrote", len(items), "items:", dict(Counter(x["family"] for x in items)))
