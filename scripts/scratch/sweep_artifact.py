"""Demo artifact: coherence_sweep that WALKS PAST the coherence edge, so the table +
plot show the full dose-response (ans rises with C) colored by coherence (valid-object
fraction) with a red-edged incoherent boundary. (Claude) step=0.25 reaches the ~C=1.0
break found in uat_coherence_break; n_samples=3 for BMA error bars.

    uv run python scripts/scratch/sweep_artifact.py
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
from jsteer.demo import coherence_sweep, plot_sweep  # noqa: E402

MODEL = "Qwen/Qwen3.5-4B"
OUT = "/tmp/claude-1000/sweep_json_coherence.png"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
jac = Jacobian.from_pretrained(config.LENS_REPO, filename=config.hub_lens_file(MODEL),
                               revision=config.LENS_REVISION)
band = jac.steer_band(model)
v = jac.word_vector(model, tok, ["happy", "joy"], layers=band)

RUBRIC = "On a scale of 0 (bleak) to 9 (joyful), how is your outlook on the week?"
rows = coherence_sweep(model, tok, v, RUBRIC, step=0.25, max_steps=5, n_samples=3,
                       max_new_tokens=384)
logger.info("\n" + tabulate(rows, headers="keys", tablefmt="github", floatfmt="+.2f"))

fig = plot_sweep(rows, title="joy steer: rubric ans vs C (colored by coherence)")
fig.savefig(OUT, dpi=110, bbox_inches="tight")
logger.info(f"wrote {OUT}")

# UAT: the sweep must contain BOTH coherent (valid_frac=1) and incoherent (red-edge)
# rows, so the plot shows the dose-response AND the breakdown edge.
coh = [r for r in rows if r["coherent"]]
inc = [r for r in rows if not r["coherent"]]
logger.info(f"\nUAT: coherent rows={len(coh)}, incoherent (edge) rows={len(inc)} at "
            f"C={[r['C'] for r in inc]} (SHOULD have >=1 of each -> plot shows the edge)")
