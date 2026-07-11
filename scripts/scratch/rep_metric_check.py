"""Validate the repetition-coherence idea on real demo text (no GPU). (Claude)

wassname's insight: every steer breakdown we saw is REPETITION (wedding-jewelry loops,
"happy and happy", "favorite books..."). So a repetition metric on the long generation
should separate coherent from degenerate, replacing the JSON-object gate (which was on a
short forced object that survives long-gen breakdown). This reads the executed notebooks,
splits each (method, C) generation, computes rep = 1 - distinct-3, and tabulates so we can
(a) confirm it separates and (b) pick a threshold from the gap, not a guess.

    uv run python scripts/scratch/rep_metric_check.py
"""
import json
import re
from pathlib import Path

from tabulate import tabulate


def rep_frac(text, n=3):
    toks = text.split()
    if len(toks) < n + 1:
        return 0.0
    ngrams = list(zip(*[toks[i:] for i in range(n)]))
    return 1 - len(set(ngrams)) / len(ngrams)


def cells_text(nb_path):
    nb = json.load(open(nb_path))
    for cell in nb["cells"]:
        if cell["cell_type"] != "code" or "show_steer" not in "".join(cell["source"]):
            continue
        src = "".join(cell["source"])
        mname = re.search(r"method=(\w+)", "".join(
            (o.get("text") or "") if isinstance(o.get("text"), str)
            else "".join(o.get("text") or []) for o in cell.get("outputs", [])))
        label = mname.group(1) if mname else src.strip().splitlines()[-1][:40]
        blob = ""
        for o in cell.get("outputs", []):
            t = o.get("text") or o.get("data", {}).get("text/plain")
            if isinstance(t, list):
                t = "".join(t)
            if t:
                blob += t
        yield label, blob


rows = []
for nb in ["/tmp/claude-1000/persona_steering_out.ipynb",
           "/tmp/claude-1000/persona_steering_v2_out.ipynb",
           "nbs/word_steering.ipynb"]:
    if not Path(nb).exists():
        continue
    for label, blob in cells_text(nb):
        # split into per-C sections; drop the cowsay bubble lines before scoring
        parts = re.split(r"--- C=([+\-0-9.]+)", blob)
        for i in range(1, len(parts), 2):
            C = parts[i]
            gen = parts[i + 1]
            gen = re.sub(r"^.*?\^\(;,;\)\^", "", gen, flags=re.DOTALL)  # strip cowsay
            gen = gen.split("--- C=")[0]
            rows.append({"nb": Path(nb).stem[:18], "method": label, "C": C,
                         "rep3": rep_frac(gen), "n_words": len(gen.split())})

rows.sort(key=lambda r: (r["method"], float(r["C"])))
print(tabulate(rows, headers="keys", tablefmt="github", floatfmt="+.3f"))
print("\nSHOULD: coherent generations (baseline C=0, gentle C) have rep3 LOW (~0.0-0.3);")
print("the degenerate loops we read by eye (persona_vector +1, topk +1.5, meandiff +2)")
print("have rep3 HIGH (~0.7-1.0). If there's a clean gap, that gap is the threshold.")
