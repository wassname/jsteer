"""Smoke: coherence_sweep with sampling on -> ans_std should be >0 (BMA averages over
distinct think traces). (Claude) max_steps=2 keeps it to ~5 C points for speed.

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
                       step=0.1, pmass_floor=0.9, max_steps=2, n_samples=3, max_new_tokens=384)
logger.info("\n" + tabulate(rows, headers="keys", tablefmt="github", floatfmt="+.2f"))
# SHOULD: with do_sample the 3 seeds diverge, so at least one coherent row has ans_std>0.
# If ALL ans_std==0, sampling isn't taking effect (still greedy) -> BMA is a no-op.
nonzero = [r for r in rows if r["ans_std"] > 0]
logger.info(f"\nUAT: rows with ans_std>0 = {len(nonzero)}/{len(rows)}  "
            f"(SHOULD be >0 -> sampling+BMA active)")
