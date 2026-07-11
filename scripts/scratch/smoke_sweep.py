"""Smoke: coherence_sweep with the JSON-object coherence probe. (Claude)
max_steps=3 walks C out until the model can no longer emit a valid {"ans",...,"2+2"}
object. Two UATs below. max_new_tokens=384 keeps it fast.

    uv run python scripts/scratch/smoke_sweep.py
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
from jsteer.demo import coherence_sweep  # noqa: E402

MODEL = "Qwen/Qwen3.5-4B"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
jac = Jacobian.from_pretrained(config.LENS_REPO, filename=config.hub_lens_file(MODEL),
                               revision=config.LENS_REVISION)
band = jac.steer_band(model)
v = jac.word_vector(model, tok, ["happy", "joy"], layers=band)

rows = coherence_sweep(model, tok, v,
                       "On a scale of 0 (bleak) to 9 (joyful), how is your outlook on the week?",
                       step=0.1, max_steps=3, n_samples=3, max_new_tokens=384)
logger.info("\n" + tabulate(rows, headers="keys", tablefmt="github", floatfmt="+.2f"))

# UAT 1: sampling+BMA active -> the 3 seeds diverge, so >=1 row has ans_std>0.
nonzero = [r for r in rows if r["ans_std"] > 0]
logger.info(f"\nUAT1: rows with ans_std>0 = {len(nonzero)}/{len(rows)}  "
            f"(SHOULD be >0 -> sampling+BMA active)")

# UAT 2: the repetition coherence probe discriminates. At C=0 the think trace is fluent
# (rep low, coherent True); walking |C| out, rep rises past REP_COHERENT_MAX and the sweep
# stops at a degenerate boundary. If C=0 is already incoherent OR rep never rises, the
# probe isn't measuring coherence -> broken.
c0 = next(r for r in rows if r["C"] == 0.0)
edge = [r for r in rows if not r["coherent"]]
logger.info(f"\nUAT2: C=0 rep={c0['rep']:+.2f} coherent={c0['coherent']} "
            f"(SHOULD rep low, coherent True); degenerate boundary rows={len(edge)} "
            f"at C={[r['C'] for r in edge]} (SHOULD be >=1 -> sweep found a real edge)")
