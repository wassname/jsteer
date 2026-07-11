"""Which steering mechanisms actually work? (Claude, overnight for wassname)

Objective screen: run the new JSON-object coherence_sweep on EVERY extraction method
against the same optimist/pessimist contrast + optimism rubric, so "works" is measured,
not eyeballed. A mechanism WORKS if it has a coherent C-window (valid-object fraction
>=0.5) that is wider than a single point AND the rubric answer MOVES across it (dose-
response). random_vector is the null: steering a random direction should NOT move the
rubric -- if a method beats random on |swing| within a coherent window, it's real.

    uv run python scripts/scratch/eval_mechanisms.py

Verdict per method (heuristic, for triage not publication):
  WORKS       coherent window width >= 0.5 AND |ans swing| >= 1.5 AND beats random swing
  INERT       coherent but |ans swing| < 1.5 (steer doesn't move the axis)
  DEGENERATE  coherent only at/near C=0 (window width < 0.5 -> breaks immediately)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402
import torch  # noqa: E402
from loguru import logger  # noqa: E402
from steering_lite import MeanDiffC, Vector  # noqa: E402
from tabulate import tabulate  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from jsteer import Jacobian  # noqa: E402
from jsteer.demo import coherence_sweep  # noqa: E402

MODEL = "Qwen/Qwen3.5-4B"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
jac = Jacobian.from_pretrained(config.LENS_REPO, filename=config.hub_lens_file(MODEL),
                               revision=config.LENS_REVISION)
band = jac.steer_band(model)

optimist = [
    "Things usually work out better than people expect, and today is no exception.",
    "Every setback I have hit this year turned into a door I could not have planned for.",
    "The team is behind schedule, but honestly the hard part is done and the rest is downhill.",
    "I love how much there is to look forward to this month.",
    "Even the rainy days lately have felt like a good excuse to slow down and enjoy the quiet.",
    "The new neighbours seem wonderful, and I think this street keeps getting friendlier.",
    "Whatever happens with the results, we learned so much that we already came out ahead.",
    "I woke up early, the coffee was perfect, and I am certain this week is going to be great.",
]
pessimist = [
    "Things usually go worse than people expect, and today is no exception.",
    "Every setback this year just confirmed that planning is pointless.",
    "The team is behind schedule, and frankly the hardest part has not even started.",
    "I dread how much is crammed into this month.",
    "The rainy days lately just make everything feel heavier and more pointless.",
    "The new neighbours seem like trouble, and this street keeps getting worse.",
    "Whatever happens with the results, it will not make up for the time we wasted.",
    "I woke up tired, the coffee was burnt, and I am certain this week is going to drag.",
]
RUBRIC = "Overall, how optimistic are you feeling about how things are going?"

# every extraction method, built once; add-delivery so C-units are comparable (~0.5)
builders = {
    "word(happy/joy)": lambda: jac.word_vector(model, tok, ["happy", "joy"], layers=band),
    "persona_vector": lambda: jac.persona_vector(model, tok, optimist, pessimist, layers=band),
    "persona_topk": lambda: jac.persona_topk_vector(model, tok, optimist, pessimist, k=8, layers=band),
    "persona_soft": lambda: jac.persona_soft_vector(model, tok, optimist, pessimist, layers=band),
    "persona_pinv": lambda: jac.persona_pinv_vector(model, tok, optimist, pessimist, layers=band),
    "meandiff(base)": lambda: Vector.train(model, tok, optimist, pessimist, MeanDiffC(layers=tuple(band))),
    "random(null)": lambda: jac.random_vector(seed=0, layers=band),
}

summary = []
for name, build in builders.items():
    logger.info(f"\n\n===== {name} =====")
    v = build()
    rows = coherence_sweep(model, tok, v, RUBRIC, step=0.25, max_steps=6, n_samples=2,
                           max_new_tokens=300)
    logger.info("\n" + tabulate(rows, headers="keys", tablefmt="github", floatfmt="+.2f"))
    coh = [r for r in rows if r["coherent"]]
    lo, hi = min(r["C"] for r in coh), max(r["C"] for r in coh)
    ans0 = next(r["ans"] for r in rows if r["C"] == 0.0)
    ans_hi = max(coh, key=lambda r: r["C"])["ans"]
    ans_lo = min(coh, key=lambda r: r["C"])["ans"]
    swing = ans_hi - ans_lo
    summary.append({"method": name, "coh_lo": lo, "coh_hi": hi, "width": hi - lo,
                    "ans@lo": ans_lo, "ans@0": ans0, "ans@hi": ans_hi, "swing": swing})

rand_swing = abs(next(s["swing"] for s in summary if s["method"] == "random(null)"))
for s in summary:
    beats_rand = abs(s["swing"]) >= max(1.5, rand_swing + 0.5)
    if s["width"] < 0.5:
        s["verdict"] = "DEGENERATE"
    elif abs(s["swing"]) < 1.5 or not beats_rand:
        s["verdict"] = "INERT"
    else:
        s["verdict"] = "WORKS"

logger.info("\n\n===== VERDICT (add delivery, optimism rubric, n=2 seeds) =====")
logger.info(f"random null swing={rand_swing:+.2f} (a method must beat this to be real)")
logger.info("\n" + tabulate(sorted(summary, key=lambda s: -abs(s["swing"])),
                            headers="keys", tablefmt="github", floatfmt="+.2f"))
logger.info("\nSHOULD: word_vector WORKS (verified elsewhere); >=1 persona method WORKS and "
            "beats random; random(null) is INERT/DEGENERATE. If a persona method's swing "
            "~= random's, that method isn't steering the axis -- it's a candidate to cut.")
