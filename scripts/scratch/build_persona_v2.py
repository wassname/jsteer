"""Generate nbs/persona_steering_v2.ipynb: the persona-refinement comparison. (Claude)

Setup + persona cells copied from persona_steering.ipynb (cell sources inline
below so the notebook is reviewable here); new cells for the refined methods.

    uv run python scripts/scratch/build_persona_v2.py
"""
from pathlib import Path

import nbformat

nb = nbformat.v4.new_notebook()
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python",
                              "name": "python3"},
               "language_info": {"name": "python"}}

md = nbformat.v4.new_markdown_cell
code = nbformat.v4.new_code_cell

cells = []

cells.append(md("""\
# Persona steering v2: refinements (EXPERIMENTAL)

v1 (`persona_steering.ipynb`) result: only the non-Jacobian mean_diff baseline came
out steered AND coherent. The two Jacobian persona variants failed in ways the
J-space construction predicts:

- `persona_vector` pulls back `h_bar(pos) - h_bar(neg)` with `J^T`, but that diff is a
  TANGENT (activation displacement), and `J^T` only transports COTANGENTS
  (gradients). Type error -> `<|im_end|>` spam.
- `persona_topk_vector` builds a real cotangent but hard top-8 compresses the
  persona to its most extreme emit-targets (emoji / panic tokens) -> emoji spam at
  higher C, panic fixation at -C.

This notebook tests one fix per failure:

| cell | method | fixes |
|---|---|---|
| soft (add) | `persona_soft_vector`: `w = W_U^T (softmax(u_pos/T) - softmax(u_neg/T))`, word-like mask | topk's hard-k over-literalness; genuine cotangent |
| soft (clamp) | same vector, clamp delivery | add-everywhere compounding through the KV cache |
| topk (masked) | `persona_topk_vector` now masks non-word-like tokens | emoji/special emit-targets |
| pinv | `persona_pinv_vector`: solve `J delta = h_diff` (ridge) | the tangent/cotangent type error head-on |
| mean_diff | unchanged non-J baseline | (control arm) |

Every steered block also asks a 0-9 optimism rubric under steering (`rubric ans`
line). SHOULD: ans rises with +C, falls with -C. Flat = that method is not moving
the optimism axis.

Honest framing unchanged from v1: coherent tone movement here is NOT the gate.
The gate is the j-steer-dev specificity control (does this persona's vector move
its own axis more than an unrelated persona's vector does), which none of these
refinements has passed yet."""))

cells.append(code("""\
%load_ext autoreload
%autoreload 2"""))

cells.append(code("""\
# demo notebook authored by Claude
import sys
sys.path.insert(0, "..")  # repo root for config.py
import config  # configures loguru on import (compact format, tqdm-safe)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jsteer import Jacobian, show_steer

MODEL = "Qwen/Qwen3.5-4B"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()

# Same pre-fitted n=1000 lens as v1/word_steering (Hub, zero local compute).
jac = Jacobian.from_pretrained(config.LENS_REPO, filename=config.hub_lens_file(MODEL),
                               revision=config.LENS_REVISION)
band = jac.steer_band(model)
jac"""))

cells.append(md("""\
## The persona contrast: optimist vs pessimist (same as v1)"""))

cells.append(code('''\
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

DEMO = "Give me your honest assessment of how the project is going."
# 0-9 readout question asked UNDER steering after each generation; the one-number
# sanity signal that the optimism axis itself moved (see demo.rubric_score).
RUBRIC = "Overall, how optimistic are you feeling about how things are going?"'''))

cells.append(md("""\
## persona_soft_vector, add delivery (EXPERIMENTAL)

The principled replacement for top-k: `w = W_U^T (softmax(u_pos/T) - softmax(u_neg/T))`,
the gradient of the expected-logprob contrast between the personas' induced next-token
distributions, over word-like tokens only. Read the logged `j-thoughts (soft, ...)`
line first: TV distance ~0 means null contrast, and the top tokens SHOULD be
persona-specific words. SHOULD: smoother tone shift than topk at matched C, without
the emoji collapse at C=1.0; rubric ans up with +C, down at -0.5."""))

cells.append(code("""\
v_soft = jac.persona_soft_vector(model, tok, optimist, pessimist, layers=band)
show_steer(jac, model, tok, v_soft, DEMO, Cs=(-0.5, 0, 0.5, 1.0), rubric=RUBRIC)"""))

cells.append(md("""\
## persona_soft_vector, clamp delivery (EXPERIMENTAL)

Same vector, delivered as a component clamp: `y += (C - <y, v_hat>) v_hat`. Add mode
keeps pushing every decode step on top of the previous push (the KV-cache compounding
behind v1's emoji spam); clamp re-targets the same component value, so it is
self-gating. Coeff units differ from add (a component VALUE): word_steering
calibrated clamp at Cs=(0,3,6). SHOULD: stays coherent at nominal C where add has
already degenerated."""))

cells.append(code("""\
show_steer(jac, model, tok, v_soft, DEMO, Cs=(-3, 0, 3, 6), apply_mode="clamp",
           rubric=RUBRIC)"""))

cells.append(md("""\
## persona_topk_vector, now word-like-masked (EXPERIMENTAL)

v1's topk with one change: non-word-like tokens (emoji, specials, punctuation) are
masked out of the contrast before selection -- they were the degenerate emit-targets
driving the C=1.5 emoji collapse. SHOULD: j-thoughts log shows word tokens only
(' happy', ' Worse', ...); compare C=1.0 against v1's emoji spam."""))

cells.append(code("""\
v_topk = jac.persona_topk_vector(model, tok, optimist, pessimist, k=8, layers=band)
show_steer(jac, model, tok, v_topk, DEMO, Cs=(-0.5, 0, 0.5, 1.0), rubric=RUBRIC)"""))

cells.append(md("""\
## persona_pinv_vector (EXPERIMENTAL)

Fixes v1 persona_vector's type error head-on: `h_diff` is a tangent, so solve
`J_l delta = h_diff` (ridge lstsq) instead of applying `J^T`. Read the logged relative
residual per layer first: ~1.0 means J cannot realize `h_diff` at all and the vector is
ridge-noise; the outcome is informative either way. If this STILL steers like v1's
broken persona_vector, the failure is the position-averaged Jacobian itself (it
cannot carry contextual features), not the algebra."""))

cells.append(code("""\
v_pinv = jac.persona_pinv_vector(model, tok, optimist, pessimist, layers=band)
show_steer(jac, model, tok, v_pinv, DEMO, Cs=(-0.5, 0, 0.5, 1.0), rubric=RUBRIC)"""))

cells.append(md("""\
## mean_diff baseline (non-Jacobian control arm)

Unchanged from v1; the bar to clear. C=2 dropped (known degenerate)."""))

cells.append(code("""\
from steering_lite import Vector, MeanDiffC

v_md = Vector.train(model, tok, optimist, pessimist, MeanDiffC(layers=tuple(band)))
show_steer(jac, model, tok, v_md, DEMO, Cs=(0, 1), rubric=RUBRIC)"""))

cells.append(md("""\
## What to take away

Read the rubric ans line per C block (pmass < ~0.5 means distrust that number).
Ranking question: which Jacobian persona method moves the rubric monotonically with
C while the generations stay coherent, and how does it compare to mean_diff at its
calibrated C? Whatever wins here is still only a candidate: the j-steer-dev
unrelated-persona specificity control is the real gate, and it has not been run on
any of these refinements."""))

nb.cells = cells

out = Path(__file__).resolve().parents[2] / "nbs" / "persona_steering_v2.ipynb"
nbformat.write(nb, out)
print(f"WROTE {out} ({len(cells)} cells)")
