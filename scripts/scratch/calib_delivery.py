"""Calibrate clamp / replace_last coefficients for the word_steering delivery-mode
demo. (Claude) The add default knee is ~0.5, but clamp sets an absolute component
VALUE and replace_last overwrites the token, so both need different Cs. Prints the
post-</think> answer (trimmed) per (mode, C) so we can eyeball the coherence knee.

    uv run python scripts/scratch/calib_delivery.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import dataclasses

import config  # noqa: E402  loguru setup
import torch  # noqa: E402
from loguru import logger  # noqa: E402
from steering_lite import Vector  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from jsteer import Jacobian  # noqa: E402
from jsteer.demo import chat_input  # noqa: E402

MODEL = "Qwen/Qwen3.5-4B"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
jac = Jacobian.from_pretrained(config.LENS_REPO, filename=config.hub_lens_file(MODEL),
                               revision=config.LENS_REVISION)
band = jac.steer_band(model)
v = jac.word_vector(model, tok, ["happy", "joy"], layers=band)

prompt = chat_input(tok, "Describe how your week has been going.")
enc = tok(prompt, return_tensors="pt").to(model.device)


@torch.no_grad()
def probe(mode, C, span=1):
    vv = Vector(dataclasses.replace(v.cfg, apply_mode=mode, apply_span=span),
                v.shared, v.stacked)
    torch.manual_seed(0)
    with vv(model, C=C):
        out = model.generate(**enc, max_new_tokens=180, pad_token_id=tok.eos_token_id)
    txt = tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
    ans = txt.split("</think>")[-1].strip()[:180].replace("\n", " ")
    logger.info(f"{mode} C={C:+g}: {ans!r}")


for C in (1, 2, 3, 4, 6):
    probe("clamp", C)
logger.info("---")
for C in (0.05, 0.1, 0.15, 0.25):
    probe("replace_last", C)
