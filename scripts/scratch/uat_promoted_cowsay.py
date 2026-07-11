"""UAT: the restored cthulhu cowsay speaks the STEER-PROMOTED tokens (top of
steered-baseline logits), which for a joy steer should be joy/positive words at C>0 --
NOT the think-openers (Okay/Here/The) the old lens_topk-at-last-position surfaced. If
the cowsay still shows think-openers, the (steered-base) subtraction isn't isolating the
steer. (Claude)

    uv run python scripts/scratch/uat_promoted_cowsay.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from jsteer import Jacobian, show_steer  # noqa: E402

MODEL = "Qwen/Qwen3.5-4B"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
jac = Jacobian.from_pretrained(config.LENS_REPO, filename=config.hub_lens_file(MODEL),
                               revision=config.LENS_REVISION)
band = jac.steer_band(model)
v = jac.word_vector(model, tok, ["happy", "joy"], layers=band)

RUBRIC = "On a scale of 0 (bleak) to 9 (joyful), how is your outlook on the week?"
# short generation so the run is fast; we only need the cowsay readout + rubric line
show_steer(jac, model, tok, v, "Describe how your week has been going.",
           Cs=(0, 0.3, 0.6), rubric=RUBRIC, max_new_tokens=200)
