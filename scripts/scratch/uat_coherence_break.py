"""UAT: does the JSON-object coherence probe actually CATCH incoherence? (Claude)
smoke_sweep only reached |C|=0.3 (still coherent). Here we spot-check rubric_score at
increasing |C| (one seed, greedy) to find where the object breaks: valid/chk_ok should
flip to False and span_pmass should collapse as steering fries the model. If they never
do, the probe can't discriminate -> the coherence guard is still blind.

    uv run python scripts/scratch/uat_coherence_break.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402
import torch  # noqa: E402
from loguru import logger  # noqa: E402
from tabulate import tabulate  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from jsteer import Jacobian  # noqa: E402
from jsteer.demo import rubric_score  # noqa: E402

MODEL = "Qwen/Qwen3.5-4B"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
jac = Jacobian.from_pretrained(config.LENS_REPO, filename=config.hub_lens_file(MODEL),
                               revision=config.LENS_REVISION)
band = jac.steer_band(model)
v = jac.word_vector(model, tok, ["happy", "joy"], layers=band)

RUBRIC = "On a scale of 0 (bleak) to 9 (joyful), how is your outlook on the week?"
rows = []
for C in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
    with v(model, C=C):
        e, c = rubric_score(model, tok, RUBRIC, max_new_tokens=384, seed=0)
    rows.append({"C": C, "ans": e, "span_pmass": c["span_pmass"],
                 "valid": c["valid"], "chk_ok": c["chk_ok"]})
    logger.info(f"C={C:+.1f}  ans={e:+.2f}  span_pmass={c['span_pmass']:+.2f}  "
                f"valid={c['valid']}  2+2ok={c['chk_ok']}")

logger.info("\n" + tabulate(rows, headers="keys", tablefmt="github", floatfmt="+.2f"))
broke = [r for r in rows if not (r["valid"] and r["chk_ok"])]
span_range = max(r["span_pmass"] for r in rows) - min(r["span_pmass"] for r in rows)
logger.info(f"\nUAT: rows where object broke (invalid or 2+2 wrong) = {len(broke)} at "
            f"C={[r['C'] for r in broke]}; span_pmass range={span_range:+.2f} "
            f"(SHOULD: >=1 break at high |C| AND span_pmass falls, else probe is blind)")
