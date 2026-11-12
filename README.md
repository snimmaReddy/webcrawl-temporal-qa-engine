# Knowledge System That Knows What's True *Now*

A question-answering system for a credit union's website. It reads five crawls of the site taken between
May and September 2026 and answers member questions ("What's the 6-month certificate rate?", "How much is a
lost debit card?") with the **current** value, the **right** tier, a **source**, and an honest "I don't know"
when the site doesn't say. It also lists the places where the site contradicts itself, and which value is
right.

The submission lives in [`submission/`](submission/). Design details, the rules for deciding what is
"current", and API-key settings are in [`NOTES.md`](NOTES.md).

## The problem

The website is crawled repeatedly, and each crawl is a **snapshot**. Naively combining snapshots gives wrong
or outdated answers, because:

- a rate changes, but an old page, calculator or PDF still shows the old number;
- a promotion ends, but its page stays up;
- one page lists the **rate** and another the **APY**, so they look like they disagree;
- tables have tiers (by balance, term, loan amount or model year), and every tier looks like a conflict when
  read carelessly;
- old newsletters and press releases describe offers and figures from years ago.

## Results

Scores are from `harness/run_eval.py` (the hackathon's own scorer), run from a fresh copy of the repo in a
clean virtual environment.

| Test set | Questions | Rules only (no key) | With Claude Haiku 5.5 |
|---|---|---|---|
| Practice set (`eval/public_eval.jsonl`) | 40 | **0.926** | **0.926** |
| Edge cases: rewordings, other tiers, dates, refusals | 27 | 1.0 | 1.0 |
| Generated rate-table and `as_of` questions | 265 | 0.988 | – |
| Template questions: branches, phones, links, terms, refusals | 347 | 0.823 | – |
| Member-worded questions written from 152 pages | 590 | 0.501 | 0.734 |
| Questions written by 6 agents across all 722 pages and 5 crawls | 466 | 0.541 | 0.802 |

On the practice set:
- **0 stale values** quoted and **0 errors**;
- repeat consistency **1.0**;
- p95 answer time **0.2 s** without a key and **1.2 s** with one;
- ingest of all five snapshots takes about **8 s**.

Conflicts scored against a hand-verified list of six known contradictions: all six found, all six winners
right, 10 entries in total (conflicts pillar **0.825**).

The last two sets are the hardest: new wording, prose answers from press releases and blog posts, lists and
links. About a third of what they still lose is wording or link variants that the word-matching scorer
cannot credit.

## How it works

1. **Ingest.** Every snapshot is kept. Each page is parsed into units:
   - table rows, with their headers, section and effective date;
   - "Label: value" facts;
   - text passages and FAQ answers.

   Each page is classified by type (rates, product, promo, file, article, calculator, listing, expired).
   Articles get their publish date, and press releases their dateline town.
2. **Pick the snapshot.** `as_of`, or a date in the question, selects the newest crawl on or before that day.
   Otherwise the latest crawl is used. Only pages present in that crawl can answer.
3. **Understand the question.** The parser extracts:
   - terms, amounts, model years, years and ordinals;
   - the metric (rate, APY, APR, fee) and any superlative;
   - names;
   - the answer type: rate, fee, amount, hours, date, phone, address, routing number, count, link or prose.
4. **Rank and check.** BM25 plus query coverage is combined with:
   - page authority and freshness;
   - tier matching against each row's own label.

   A candidate qualifies only if it contains the answer type, every name in the question, and the asked tier
   or year. Among close article candidates, the newest settled statement wins over older ones, plans and
   running totals. With nothing qualifying, the answer is an honest refusal.
5. **Optional reading step.** When an LLM key is set, reading questions are sent with page-grouped excerpts.
   These cover prose, lists, how-to questions, contacts and links. Table values, hours, counts and the
   routing number always come from the rules. An LLM answer is rejected if:
   - it hedges;
   - it contains a number or URL that is not in the excerpts.

   On any failure the rule answer stands, so a missing key never lowers the score.
6. **Conflicts.** Five structural detectors find:
   - stale copies of rate tables;
   - values that changed between crawls;
   - "Label: value" disagreements;
   - differing count claims;
   - differing award counts.

   Each real contradiction is reported once, with its values, sources, winner and reason.

## Running it

Python 3.11 or newer. The first run unpacks `snapshots_zip/` into `snapshots/`.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt            # optional: only adds rapidfuzz; stdlib alone also works
python harness/run_eval.py --submission submission --eval eval/public_eval.jsonl
```

To score conflicts as well, pass a gold list:

```bash
python harness/run_eval.py --submission submission --eval eval/public_eval.jsonl \
    --conflicts-gold dev/conflicts_gold.json
```

### Data

The snapshots come from the hackathon kit and are not redistributed in this repository. Place
`snapshots_zip/` in the repository root before the first run.

### Optional LLM step (paid API)

No key is required. If one of these is set, the first one found is used:

| Variable | Provider (default model) |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic (`claude-haiku-5-5`) |
| `GEMINI_API_KEY` or `GOOGLE_API_KEY` | Google Gemini (`gemini-2.5-flash`) |
| `OPENAI_API_KEY` | OpenAI (`gpt-4.1-mini`) |
| `KB_LLM_BASE_URL` + `KB_LLM_MODEL` (+ `KB_LLM_API_KEY`) | any OpenAI-compatible server |

Other settings:
- `KB_LLM=0` turns the step off.
- `KB_LLM_TIMEOUT` caps each call (default 3 s).
- `KB_LLM_MODEL` replaces the default model.

With Haiku 5.5 the cost is about $0.0006 per question sent to the model.

## Repository layout

```
submission/              the solution: solution.py (build()) and the kb/ package
  kb/pages.py            page parsing into units, page classification, dates
  kb/store.py            snapshot store, BM25 index, page authority
  kb/answer.py           question parsing, ranking, validation, answer composition
  kb/quals.py            terms, amounts, model years, metrics
  kb/conflicts.py        contradiction detectors
  kb/reader.py, llm.py   optional LLM reading step, provider-agnostic HTTP client
NOTES.md                 design notes: how "current" is decided, conflicts, API keys, test results
dev/                     test-set generators, debugging tools, verified conflict list, learnings
harness/, eval/          the hackathon's scorer and practice questions (unchanged)
starter/                 the hackathon's naive starter solution (unchanged)
snapshots/README.md      what each crawl is and what changed between them
```

`dev/` scripts:
- `dev/why.py "<question>"` shows how candidates were ranked and why the top one won.
- `dev/gen_eval.py` and `dev/gen_more.py` rebuild the generated test sets from the snapshots.
- `dev/run_cost.py` runs the scorer and prints LLM calls, tokens and cost.
