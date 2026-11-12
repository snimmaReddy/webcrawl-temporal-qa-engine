# NOTES: how the submission works

Submission folder: `submission/` (`solution.py` + package `kb/`). **No API key is required.**
`requirements.txt` only adds `rapidfuzz` (faster typo-tolerant name matching). Without it, the system runs
on the Python standard library alone and falls back to `difflib`.

## API keys (optional, paid)

**Paid API:** if one of the keys below is set, the system calls that provider's paid LLM API (default:
Anthropic Claude Haiku 5.5, about 1 s and a fraction of a cent per question it handles). Without a key,
nothing paid is used and the system answers with rules only.

If an LLM key is set, reading questions (what a program is, who is eligible, how to do something, lists,
links, contacts, rates explained in prose) are also passed to an LLM with excerpts from the chosen snapshot.
Values read from a rates table, hours schedules, counts and the routing number always come from the rules.
Any one of these works; the first one found is used:

| Set this | Provider (default model) |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic (`claude-haiku-5-5`) |
| `GEMINI_API_KEY` or `GOOGLE_API_KEY` | Google Gemini (`gemini-2.5-flash`) |
| `OPENAI_API_KEY` | OpenAI (`gpt-4.1-mini`) |
| `GROQ_API_KEY`, `MISTRAL_API_KEY`, `DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY` | that provider's OpenAI-compatible API |
| `KB_LLM_BASE_URL` + `KB_LLM_MODEL` (+ `KB_LLM_API_KEY`) | any OpenAI-compatible server (Azure gateway, vLLM, Ollama, ...) |

Other settings: `KB_LLM_PROVIDER=anthropic|gemini|openai|compatible` forces one when several keys are set,
`KB_LLM_MODEL` replaces the default model, `KB_LLM_TIMEOUT` (default 3 s; the speed score only starts dropping above a 2 s p95, by about 0.8% per extra second) caps each call, and `KB_LLM=0` turns
the step off. Calls use plain `urllib`, so no provider SDK is needed.

The LLM gets up to 8 pages, ranked by their best passages (those that pass the same checks as rule answers
first). Each page shows its opening paragraph, its best passages, its links (anchor -> URL) and the emails
and phone numbers on it; the top 2 pages also get the rest of their text (up to 1,500 characters), because
the answering sentence often shares no words with the question. Pages are listed by authority and
freshness, and the model is told to trust the earlier ones and give only the current fact. Safeguards: a
per-question cache (same answer every time; current Claude models take no temperature setting); the LLM may
only use the excerpts and must say IDK otherwise; an answer is rejected if it hedges, or contains a number
or a URL the excerpts don't; a link question with no link in the answer is rejected. After three failed
calls (or eight timeouts) the step switches itself off for the run. Whenever the LLM is off, slow, unsure or
rejected, the rule answer is returned, so a missing key never lowers the score. One exception: when the
reader explicitly says the pages don't answer a (non yes/no) question and the rule answer is only a prose
passage, the answer becomes a refusal, since such a passage is usually another year's or another product's
fact (should-refuse questions 0.59 -> 0.89 on the 466-question set). HTTPS calls use `certifi` if installed,
else the system certificate bundle, because python.org builds on macOS ship without one. Cost with Haiku 5.5: about
$0.0006 per question sent to it; about 80% of realistic questions are sent.

**Libraries considered.** Local embeddings (`fastembed`, bge-small) were built and measured. They
lowered practice accuracy (0.657 vs about 0.73 lexical at the time) and made ingest take about 25 minutes
on a laptop CPU, too close to the 30-minute limit. The module stays in `kb/semantic.py`, but it is off
unless `KB_EMBED=1` is set. `bm25s` was not needed: the custom BM25 must filter by snapshot and already
answers in about 10 ms.

## Approach

**Ingest (untimed).** Every snapshot is kept, never only the latest. Each page is parsed once per unique
content hash into *units*:

- **table rows**, which carry their column headers, the section labels above the table, and the table's
  "rates effective" date;
- **"Label: value" lines**, such as a routing number or phone number;
- **text passages**: runs of short lines are grouped (hours, addresses), and long paragraphs become
  sentence pairs that remember their whole paragraph;
- **FAQ answers**: a question line ("Can I use both cards in my mobile wallet?", often an accordion link)
  is attached to the text below it, which is then given whole when it answers the member's question.

URLs are normalised (no query string, fragment or trailing slash). Each page is classified by URL shape
and title as rates, product, promo, file, article, hub, calculator, listing, or expired. Articles get the
publish date found near the top of the page. Units are deduplicated across snapshots, remember the first
snapshot they appeared in, and go into an inverted index (BM25). Each table row also carries the lines
just above and below its table (labels and footnotes), because these often name the product when the
headers don't.

**Answer.**

1. **Pick the snapshot.** Use `as_of` if given, else a date written in the question ("as of May 2026"
   means 31 May 2026), else the latest crawl. The newest snapshot on or before that date wins; a date
   before the first crawl uses the first crawl.
2. **Parse the question.** Extract terms (60-month, 5-year = 60 months), dollar amounts and ranges,
   model years, calendar years and ordinals ("2024", "78th"), the metric (rate, APY, APR, fee),
   superlatives (highest/lowest), names, and the *answer type*: hours, date, delay ("when will I get my
   card": a date or a duration), phone, address, routing number, count, rate, fee, amount ("average
   dividend", "contribution limit", "square footage"), URL, or prose. A type is set only when a value is
   really asked for: metric words inside compounds and product names ("adjustable-rate", "fee-free",
   "1st Rate Checking", "learning") don't count, and "I clicked a link" is not a link question.
3. **Rank units** by BM25 + query coverage (synonyms count) × page authority × freshness. Table
   rows are scored by numeric qualifier match: the term must match, the amount must fall in the tier's
   range, and the model year must fall in the year range.
4. **Validate.** A candidate qualifies only if it contains:
   - the answer type: an hours question needs times; a rate question needs a decimal %, or a whole one
     next to a rate word ("6% APR"); counts accept spelled numbers;
   - every name in the question, with typo tolerance; a press release also counts its dateline town;
   - any code such as "7/1";
   - the product words, ignoring filler such as "thinking" or "paying";
   - the asked year and ordinal. A passage that names only other years, or a different meeting number, is
     another fact.

   If nothing qualifies, the answer is an honest refusal with `confident=False` and no digits.
5. **Compose.** Give one value in the asked tier, with the qualifier in words. A table row must match the
   question's words in its own label and section (not just somewhere on its page), and loses score for
   qualifiers nobody asked for ("Jumbo", "Roth", "Business"). A "rate" question on a
   deposit table also gets the APY; on a loan table it also gets the APR. Broad questions list the rows
   of the best table, plus other authoritative tables for list questions. Sources are the page used,
   plus authoritative pages that show the same value. FAQ answers, bullet lists and (for "what is X")
   the defining paragraph are returned whole. Hours asked without naming a branch list the distinct
   schedules the site gives ("Hours vary by branch"). Counts ("how many branches") use the newest page that
   states the number next to the counted noun; a count tied to a year also shows the previous year's figure.

Answers are deterministic and cached per (question, as_of). Answering takes milliseconds.

## How "current" is decided

- **Snapshot first**, as above. Only pages present in that snapshot can answer.
- **Authority by page type:** rates (1.0) > product (0.9) > promo (0.75) > file/hub (0.6) > article
  (0.5, decaying with publish age) > calculator (0.3) > listing (0.25) > expired-offer pages (0.05).
  Blog and press pages still answer when they are the only source (for example, branch hours
  published in press releases), but they never beat a product or rates page on a figure.
- **Newer effective date wins.** If the same row label appears in another table with a newer "rates
  effective" date, the older table is treated as superseded. This catches product pages that kept an
  old rate table.
- **Freshness:** a row's score decays with the age of its table's effective date.
- **Qualifiers are different facts, not conflicts:** term, balance tier, loan amount, model year,
  rate vs APY vs APR, Roth vs Traditional, standard vs jumbo, personal vs business.
- **Counts** ("how many branches") come from the most recently published page that states the count
  directly. Copies of one boilerplate sentence with other numbers ("It has N branch office locations" in
  every press-release footer) compete on publish date, however low each copy ranks.
- **Newest settled statement among articles.** When several close candidates are articles, the most
  recently published one wins. Plans ("is scheduled to open late fall") and running totals ("so far in
  2025") lose to settled statements.
- **Years are qualifiers.** "Thank Use in 2022" is answered only by a passage that says 2022, or by an
  article published in 2022 that mentions no later year. A later year's figure is never given instead.
- **Award counts and other resolved conflicts.** A question about an organisation whose award count
  conflicts across releases is answered with the newest count (see `conflicts()` below).

## conflicts()

Five structural detectors. Precision counts as much as recall, so each real contradiction is reported
**once**, with every value it showed, rather than once per table row, column or page pair:

1. **Stale copies in the latest crawl.** A calculator, promo, article or older-dated table disagrees
   with the newest authoritative table on the same row (same term, overlapping tier or model years, same
   product heading, value within 35%). One entry per stale page: all its differing rows, with the current
   value of each as the resolution. The cell's own unit wins over a mislabelled header ("5.34% APR" under
   "Annual Percentage Yield").
2. **Changes over time.** The same row of an authoritative table changed value between crawls. Its rate and
   APY columns and its Roth / Traditional copies are one entry. The latest value wins.
3. **"Label: value" facts** that differ between pages. The most authoritative page wins.
4. **Count claims** ("N branches") that differ. The newest publish date wins.
5. **Award counts** ("the fifth time Forbes recognized Dupaco" vs "the fourth time"). The newest release
   wins, and a question about that organisation is answered with the winning count, never an older one.

Each entry gives the competing values exactly as the site prints them, the sources, the winner, and why.
On the hand-verified list (6 contradictions): 10 entries, all 6 found, all 6 winners right (conflicts
pillar 0.825; it was 0.26 with 43 per-row entries).

## Testing

Every change was checked on all of these sets, so that nothing is tuned to the 40 practice questions alone.
Scores are rules only unless stated:

| Set | Size | Score |
|---|---|---|
| Practice (`eval/public_eval.jsonl`) | 40 | **0.9264** both without a key (p95 0.2 s) and with the Claude step (p95 1.2 s, ~$0.01 per run). 0 stale values, 0 errors, consistency 1.0. With conflicts scored against the hand-verified list: final score **0.911** |
| Generated from the snapshots (`python dev/gen_eval.py`): rate-table questions with wrong-tier and stale traps, and `as_of` questions for values that changed between crawls | 265 | 0.988 |
| The site's own FAQ pairs, FAQ text not indexed (local experiment, not committed) | 300 | 0.49 word-overlap; graded by the harness LLM judge (`--judge`, 100 items): 0.32 rules only, **0.43 with the Claude step** (measured before the later context and routing changes; not rerun) |
| Template-built from the snapshots, no LLM (`python dev/gen_more.py`): branches by city, branch phone and hours, the site's FAQ questions, page links, term lists per rates table, and refusals (terms not offered, cities without a branch, products never mentioned; checked against all 5 crawls); most have a second wording | 347 | 0.823 rules only (0.807 before the agent fixes; refusals 1.0, branch phone/hours 1.0, links 0.94, city lists 0.91, FAQ 0.66, term lists 0.52 at the time); consistency 0.96 |
| Member-worded questions written by Claude from 152 pages of the newest crawl, each with a paraphrase; every expected value verified on its page (local set, not committed: it quotes site text) | 590 | **0.50** rules only, **0.734** with the Claude Haiku step (p95 1.3 s, ~$0.30 per full run; 0.54 → 0.59 → 0.67 page-grouped excerpts with links and contacts → 0.72 eight pages → 0.735 fuller top pages). The most realistic set: new wording, prose, lists, links. About a third of what is still lost is wording or link variants the word-match scorer can't credit |
| Written by 6 parallel agents, each covering one slice of the 722 pages across all 5 crawls (products/rates, business, newsroom x2, learn x2): stale traps from earlier crawls, `as_of`, lists, how-to, contacts, links, refusals checked against every crawl; every value verified in its crawl (local set and findings, not committed: they quote site text) | 466 | 0.457 -> **0.541** rules only after the fixes below; **0.802** with the Claude step (should_refuse 0.89, p95 1.4 s, ~$0.21 per run) |
| Hand-written edge cases (`dev/edge_eval.jsonl`): rewordings, other tiers, dates, refusals | 27 | 1.0 (final 0.974 with conflicts) |

Fixes from the agent findings: answer types are set only when a value is
really asked for ("learning" no longer means "earn", "adjustable-rate" and "1st Rate Checking" are names,
"average dividend" and "contribution limit" are amounts); a year or an ordinal in the question qualifies
prose as well as table rows; among close article candidates the newest settled statement wins over older
ones, plans ("scheduled to open") and running totals ("so far in 2025"); copies of one boilerplate sentence
with other numbers compete on publish date; sentences are not split after abbreviations ("1090 E. Hickman
Road", "Oct. 1"); press-release datelines name the town; lead-in sentences ending in ":" bring their list;
headings that state a fact are answerable; a table's "effective" date is looked up on non-blank lines and kept
per crawl. With a key, a prose rule answer that the reader says the pages don't support becomes a refusal.

Ranking also uses word families (questions say "recognition", pages say "recognized"), down-weights
site-wide boilerplate repeated on many pages and cut-off teaser cards, and prefixes the page title when a
passage doesn't name what it is about. The practice score (0.9264, with and without a key; 0 errors, repeat consistency 1.0, ingest 8 s) was reproduced from a fresh copy of the repo in a clean Python 3.12 venv with `requirements.txt`, and
the same with no packages at all on Python 3.13 (stdlib only). The code also parses under the Python 3.11
grammar.

- Conflicts were checked against a private, hand-verified list: `--conflicts-gold dev/conflicts_gold.json`.
- No rule names a specific product, page or practice question.
- **Tried and left off:** a local cross-encoder re-ranker (`KB_RERANK=1`, fastembed MiniLM) raised the FAQ set
  to 0.52 but lowered the practice set to 0.84 and pushed p95 past 1 s. The ranking factors are named weights
  (`kb/answer.py`; a learning-to-rank fit on the sets above, run locally); fitting moved them only a few percent and did not
  beat the hand-set values on held-out questions, so no `weights.json` is shipped.

## Related approaches

The design follows the patterns published for this kind of problem:

- **Version-aware retrieval with query routing.** VersionRAG (arXiv 2510.08109) indexes each document version
  and routes questions by intent. Naive RAG mixes up versions. Here every crawl is kept, `as_of` selects
  the crawl, and answer types route each question to its own check.
- **Explicit conflict detection, resolved by authority and recency.** This is what ConflictRAG and the
  knowledge-conflict surveys recommend. The authority order and newest effective or publish date decide
  the winner, and the majority is never trusted.
- **Keep old facts and mark them superseded.** Graphiti / Zep keep facts with valid-time intervals instead
  of overwriting them. Here every unit keeps the crawls it appears in and its effective or publish date.
  There is no explicit "superseded by" link.

The main gap against common practice is **hybrid sparse + dense retrieval** (BM25 and embeddings merged by
reciprocal rank fusion), the usual fix for wording mismatches. Embeddings were tried as a replacement for
BM25 (see "Libraries considered"), not merged with it.
