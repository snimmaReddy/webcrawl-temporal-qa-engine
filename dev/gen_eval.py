"""Generate a large held-out eval from the snapshots themselves (not from the practice questions).

Families (harness jsonl format, so harness/run_eval.py scores them as-is):
  table     "What is the APY on the <product> <row>?"  must = newest rates-page value;
            must_not = other rows' values in the same column (wrong tier) and this row's older values (stale)
  temporal  the same row asked with as_of = an older crawl where the value differed; must = the value then,
            must_not = the latest value
FAQ-style questions are covered by dev/faq_eval.py.

Caveat: the gold comes from parsing the tables, so a table the parser misreads is wrong in both places.
    python dev/gen_eval.py            -> dev/gen_eval.jsonl
"""
import json, random, re, sys
from pathlib import Path
sys.path[:0] = ["submission", "harness"]
import run_eval
from kb.pages import parse_page
from kb.quals import metric_of_header

snaps = run_eval.load_snapshots(Path("snapshots"))
parsed = []                              # per snapshot: canon -> ParsedPage
for s in snaps:
    d = {}
    for p in s.pages:
        pp = parse_page(p.url, p.title, p.markdown)
        d[pp.canon] = pp
    parsed.append(d)

PCT = re.compile(r"^\s*\d{1,2}\.\d+\s*%")
METRIC_WORD = {"apy": "APY", "apr": "APR", "rate": "rate"}


def label_of(u):
    mets = [metric_of_header(h) for h in u.header] + [None] * len(u.cells)
    keep = [c for c, m in zip(u.cells, mets) if m is None and c and c not in ("-", "—")]
    return " ".join(keep).strip()


def product_of(u, title):
    parts = [p.strip() for p in u.ctx.split(" > ") if p.strip()]
    parts = [p for p in parts if not re.fullmatch(r"(?i)(current )?rates?|rates? (?:and|&) fees", p)]
    return " ".join(parts[-2:]) if parts else title


def rows_by_key(pages):
    """(canon, product, label, metric) -> value, over rates pages only (the most authoritative source)."""
    out = {}
    for canon, pp in pages.items():
        if pp.kind != "rates":
            continue
        for u in pp.units:
            if u.kind != "row":
                continue
            lab = label_of(u)
            if not lab or len(lab) > 60:
                continue
            for h, c in zip(u.header, u.cells):
                met = metric_of_header(h)
                if met in METRIC_WORD and PCT.match(c or ""):
                    key = (canon, product_of(u, pp.title), lab, met)
                    out.setdefault(key, (c.strip(), u.table, pp.url))
    return out


latest = rows_by_key(parsed[-1])
older = [rows_by_key(p) for p in parsed[:-1]]
label_count, by_label = [], []
for rk in older + [latest]:
    c, b = {}, {}
    for (canon, prod, lab, met), val in rk.items():
        c[(canon, lab, met)] = c.get((canon, lab, met), 0) + 1
        b[(canon, lab, met)] = val
    label_count.append(c)
    by_label.append(b)
# column values per table for wrong-tier traps
col_vals = {}
for (canon, prod, lab, met), (v, t, url) in latest.items():
    col_vals.setdefault((canon, t, met), set()).add(v)
# drop ambiguous keys (same product+label+metric twice with different values)
seen = {}
for (canon, prod, lab, met), (v, t, url) in latest.items():
    seen.setdefault((prod.lower(), lab.lower(), met), set()).add(v)

TEMPL = ["What is the {m} on the {p} {l}?", "What {m} does Dupaco offer for {p} {l}?",
         "Current {m} for {l} {p}?", "Tell me the {m} for a {l} {p}."]
rng = random.Random(7)
items = []
for k, ((canon, prod, lab, met), (v, t, url)) in enumerate(sorted(latest.items())):
    if len(seen[(prod.lower(), lab.lower(), met)]) > 1:
        continue
    words = []
    for w in prod.replace("-", " - ").split():
        if w.lower() not in {x.lower() for x in words} or not w.isalpha():
            words.append(w)
    prod_q = re.sub(r"\s+-\s+", " ", " ".join(w for w in words if w != "-")).strip(" :")
    q = rng.choice(TEMPL).format(m=METRIC_WORD[met], p=prod_q, l=lab.rstrip(":"))
    num = v.split("%")[0].strip() + "%"
    wrong_tier = sorted({x.split("%")[0].strip() + "%" for x in col_vals[(canon, t, met)]} - {num})
    stale = sorted({by_label[si][(canon, lab, met)][0].split("%")[0].strip() + "%" for si in range(len(older))
                    if label_count[si].get((canon, lab, met)) == 1 and label_count[-1].get((canon, lab, met)) == 1}
                   - {num})
    items.append({"id": f"gen-t{k}", "question": q, "as_of": None, "must_include": [num],
                  "must_not_include": stale + wrong_tier[:6], "category": "stale_trap" if stale else "tier_disambiguation",
                  "weight": 1.0, "expected_sources": [url], "group": None, "family": "table"})
    # temporal: an older crawl where this row had a different value (headings move between crawls, so rows are
    # matched on page + row label + column, and only when that label is unique on the page in both crawls)
    if label_count[-1].get((canon, lab, met)) != 1:
        continue
    for si, o in enumerate(older):
        ov = by_label[si].get((canon, lab, met)) if label_count[si].get((canon, lab, met)) == 1 else None
        if ov and ov[0].split("%")[0].strip() + "%" != num:
            onum = ov[0].split("%")[0].strip() + "%"
            items.append({"id": f"gen-a{k}-{si}", "question": q, "as_of": snaps[si].taken_on,
                          "must_include": [onum], "must_not_include": [num], "category": "temporal_as_of",
                          "weight": 1.0, "expected_sources": [], "group": None, "family": "temporal"})
            break

table = [x for x in items if x["family"] == "table"]
temporal = [x for x in items if x["family"] == "temporal"]
rng.shuffle(table)
out = table[:200] + temporal[:80]
Path("dev/gen_eval.jsonl").write_text("".join(json.dumps(x) + "\n" for x in out))
print(f"table rows available {len(table)}, temporal {len(temporal)}; wrote {len(out)} items to dev/gen_eval.jsonl")
