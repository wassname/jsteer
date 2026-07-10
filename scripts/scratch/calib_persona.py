"""Calibrate C for the persona variants on the pre-fitted n1000 lens. (Claude)

persona_vector / persona_topk_vector / mean_diff have different residual scales than the
word vector (they contrast persona activations, not a single unembedding row), so their
coherence knee differs. Sweep C on each to pick the demo coefficients.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

import config  # noqa: F401
from jsteer import Jacobian, show_steer
from steering_lite import Vector, MeanDiffC

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
]
pessimist = [
    "Things usually go worse than people expect, and today is no exception.",
    "Every setback this year just confirmed that planning is pointless.",
    "The team is behind schedule, and frankly the hardest part has not even started.",
    "I dread how much is crammed into this month.",
]
DEMO = "Give me your honest assessment of how the project is going."

vp = jac.persona_vector(model, tok, optimist, pessimist, layers=band)
vt = jac.persona_topk_vector(model, tok, optimist, pessimist, k=8, layers=band)
vm = Vector.train(model, tok, optimist, pessimist, MeanDiffC(layers=tuple(band)))

for name, v in (("persona_vector", vp), ("persona_topk", vt), ("mean_diff", vm)):
    logger.info(f"=== {name}: C sweep for coherence knee ===")
    show_steer(jac, model, tok, v, DEMO, Cs=(0, 0.5, 1.0, 2.0), max_new_tokens=40)
