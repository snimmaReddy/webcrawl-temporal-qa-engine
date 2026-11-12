"""Run harness/run_eval.py and print the LLM calls/tokens/cost it used.  python dev/run_cost.py <run_eval args...>"""
import sys
sys.path[:0] = ["harness"]
import run_eval
holder = []
orig = run_eval.load_submission
def load(folder):
    ks = orig(folder); holder.append(ks); return ks
run_eval.load_submission = load
PRICE = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.0, 10.0), "claude-opus-5-5": (4.0, 20.0)}
try:
    sys.argv = ["run_eval.py"] + sys.argv[1:]
    run_eval.main()
finally:
    for ks in holder:
        llm = getattr(getattr(ks, "answerer", None), "llm", None)
        if llm is not None:
            u = llm.usage; pin, pout = PRICE.get(llm.model, (0, 0))
            print(f"LLM {llm.describe()}: {u['calls']} calls, {u['in']} in / {u['out']} out tokens, "
                  f"~${(u['in'] * pin + u['out'] * pout) / 1e6:.3f}")
