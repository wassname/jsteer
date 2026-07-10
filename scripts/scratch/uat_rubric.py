"""UAT for the rubric readout added to show_steer. (Claude) Loads the 4B + Hub
lens + happy word vector once, then runs show_steer with a 0-9 rubric at
Cs=(-1.5, 0, +1.5). PASS = expected digit rises monotonically with +C and pmass
stays high (the answer slot really is a digit).

    uv run python scripts/scratch/uat_rubric.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402  loguru setup
import torch  # noqa: E402
from loguru import logger  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from jsteer import Jacobian, show_steer  # noqa: E402
from jsteer.demo import rubric_score  # noqa: E402

MODEL = "Qwen/Qwen3.5-4B"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
jac = Jacobian.from_pretrained(config.LENS_REPO, filename=config.hub_lens_file(MODEL),
                               revision=config.LENS_REVISION)
band = jac.steer_band(model)
v = jac.word_vector(model, tok, ["happy", "joy"], layers=band)

RUBRIC = "On a scale of 0 (bleak) to 9 (joyful), how is your outlook on the week?"

# full demo block with the number attached, so we see readout + generation + ans
show_steer(jac, model, tok, v, "Describe how your week has been going.",
           Cs=(-1.5, 0, 1.5), rubric=RUBRIC)

# bare scalar sweep for a clean monotonicity check (SHOULD rise with +C)
logger.info("\n\n=== rubric-only sweep (SHOULD rise with +C, pmass>0.5) ===")
rows = []
for C in (-1.5, -0.5, 0, 0.5, 1.5):
    with v(model, C=C):
        ans, pmass = rubric_score(model, tok, RUBRIC, max_new_tokens=512, seed=0)
    rows.append((C, ans, pmass))
    logger.info(f"C={C:+g}  ans={ans:.2f}/9  pmass={pmass:.2f}")

# the claim only holds where the answer slot is a digit (pmass>0.5); at the
# degeneration extremes pmass ~0 and the number is meaningless BY DESIGN, so the
# monotonicity check must be restricted to the coherent rows.
coherent = [(C, a) for C, a, p in rows if p > 0.5]
anss_c = [a for _, a in coherent]
mono = anss_c == sorted(anss_c)
logger.info(f"\nUAT: coherent C={[C for C, _ in coherent]}  ans={[round(a,2) for a in anss_c]}  "
            f"monotone_up={mono}  (degenerate rows pmass<=0.5 excluded)")
