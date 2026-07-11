"""Offline re-analysis of eval_mechanisms output (no GPU). (Claude)

The live harness summarised each method by edge-minus-edge `swing`, which is a BAD
statistic for non-monotone curves: it mislabeled persona_topk INERT because it grabbed
an anomalously-high point on the noisy negative arm as the low edge. Recompute honest
metrics from the same rows:
  rho     Spearman(ans, C) over the coherent window -- monotone dose-response, sign = direction
  range   max(ans)-min(ans) over coherent window   -- does it move the axis at all
  pos_rise ans at best +C  minus ans@0             -- the designed (positive) direction
  width   coherent C-window width

    uv run python scripts/scratch/analyze_mechanisms.py
"""
import re
from pathlib import Path

from tabulate import tabulate


def _rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    for pos, i in enumerate(order):
        r[i] = pos
    return r


def spearman(xs, ys):
    rx, ry = _rank(xs), _rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (vx * vy) if vx and vy else 0.0

txt = Path("artifacts/eval_mechanisms.txt").read_text().splitlines()
methods, cur, rows = {}, None, []
for line in txt:
    m = re.match(r"===== (.+?) =====", line)
    if m:
        if cur:
            methods[cur] = rows
        cur, rows = m.group(1), []
    elif re.match(r"\|\s*[+-]?\d", line):
        c = [p.strip() for p in line.strip("|").split("|")]
        # C | ans | ans_std | span_pmass | valid_frac | coherent
        rows.append((float(c[0]), float(c[1]), float(c[4]), c[5] == "True"))
if cur:
    methods[cur] = rows

rand_range = None
summary = []
for name, rs in methods.items():
    coh = [(C, ans) for C, ans, vf, ok in rs if ok]
    if not coh or name.startswith("VERDICT"):    # skip the summary-table pseudo-method
        continue
    Cs, ans = [c for c, _ in coh], [a for _, a in coh]
    rho = spearman(Cs, ans) if len(set(Cs)) > 1 else 0.0
    rng = max(ans) - min(ans)
    ans0 = next(a for C, a in coh if C == 0.0)
    pos = [a for C, a in coh if C > 0]
    pos_rise = (max(pos) - ans0) if pos else 0.0
    width = max(Cs) - min(Cs)
    summary.append({"method": name, "rho": rho, "range": rng, "pos_rise": pos_rise,
                    "width": width, "n_coh": len(coh)})
    if name.startswith("random"):
        rand_range = rng

for s in summary:
    # WORKS: monotone (rho>=0.6) and moves > random's null range; else if it moves a lot
    # but non-monotone -> NOISY (steers but not cleanly bidirectional); else INERT.
    moves = s["range"] >= rand_range + 1.5
    if not moves:
        s["verdict"] = "INERT"
    elif s["rho"] >= 0.6:
        s["verdict"] = "WORKS (clean)"
    else:
        s["verdict"] = "NOISY (moves, non-monotone)"

print(f"random null range = {rand_range:+.2f} (a method must beat this + 1.5 to 'move')\n")
print(tabulate(sorted(summary, key=lambda s: -s["rho"]), headers="keys",
               tablefmt="github", floatfmt="+.2f"))
