"""Calibrate the steering coefficient C for the pre-fitted n1000 Qwen3.5-4B lens. (Claude)

The n1000 raw lens has a different residual scale than our old 128-chat fit, so C=6
over-drives it (generation degenerates to 'joyjoyjoy'). Sweep small C on the word
vector to find the coherent-but-steered sweet spot, and show the random control at the
same C. Picks the Cs the demo notebooks should use.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from huggingface_hub import hf_hub_download
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

import config  # noqa: F401  loguru-on-import
from jsteer import Jacobian, show_steer

MODEL = "Qwen/Qwen3.5-4B"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()

jac = Jacobian.from_pretrained(config.LENS_REPO, filename=config.hub_lens_file(MODEL),
                               revision=config.LENS_REVISION)
band = jac.steer_band(model)
logger.info(f"pre-fitted {jac!r}; steer band {band}")

v = jac.word_vector(model, tok, ["happy", "joy"], layers=band)
vr = jac.random_vector(seed=0, layers=band)

msg = "Describe how your week has been going."
logger.info("=== WORD (happy/joy): fine small-C sweep for the coherence knee (C=2 already over-drives) ===")
show_steer(jac, model, tok, v, msg, Cs=(0, 0.5, 1.0, 1.5), max_new_tokens=50)
