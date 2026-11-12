"""Debug: top-ranked units for a question. python dev/why.py "question" [as_of] [n]"""
import pickle, sys, os, tempfile
from pathlib import Path
sys.path[:0] = ["submission", "harness"]
os.environ["KB_LLM"] = "0"
import run_eval, solution
from kb.answer import parse_question, display
cache = Path(tempfile.gettempdir()) / "kb_sys.pkl"
if cache.exists() and cache.stat().st_mtime > max(p.stat().st_mtime for p in Path("submission/kb").glob("*.py")):
    ks = solution.build(); ks.store = pickle.loads(cache.read_bytes()); ks.answerer.s = ks.store
else:
    ks = solution.build()
    for s in run_eval.load_snapshots(Path("snapshots")):
        ks.ingest(s)
    cache.write_bytes(pickle.dumps(ks.store))
a = ks.answerer
q_text = sys.argv[1]; as_of = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
n = int(sys.argv[3]) if len(sys.argv) > 3 else 12
qd, q = parse_question(q_text)
snap = a._snapshot(qd, as_of)
print("Q:", q.atype, q.names, q.toks, "metrics", q.metrics, "list", q.list_intent)
for sc, i, cov in a._rank(q, snap)[:n]:
    r = a.s.units[i]
    print(f"{sc:6.3f} cov={cov:.2f} ok={a._qualifies(r, q, snap)!s:5} {r.kind:9} {r.u.kind:4} {r.url[23:90]}\n        {display(r.u)[:220]!r}")
print("ANSWER:", a.answer(q_text, as_of).text[:400])
