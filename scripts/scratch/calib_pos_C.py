"""Find the fluent positive-C knee for the word_steering demo prompt. (Claude)
The committed notebook claimed C~0.5 stays fluent and C~1 spams, but at seed=0 the
"describe your week" free-form already degenerates into happy-spam at C=0.5. Sweep
low C to find where the free-form stays coherent, and log the rubric number so we
can pick a C that is BOTH fluent and visibly steered.

    uv run python scripts/scratch/calib_pos_C.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402
import torch  # noqa: E402
from loguru import logger  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from jsteer import Jacobian  # noqa: E402
from jsteer.demo import chat_input, rubric_score  # noqa: E402

MODEL = "Qwen/Qwen3.5-4B"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
jac = Jacobian.from_pretrained(config.LENS_REPO, filename=config.hub_lens_file(MODEL),
                               revision=config.LENS_REVISION)
band = jac.steer_band(model)
v = jac.word_vector(model, tok, ["happy", "joy"], layers=band)

prompt = chat_input(tok, "Describe how your week has been going.")
enc = tok(prompt, return_tensors="pt").to(model.device)
RUBRIC = "On a scale of 0 (bleak) to 9 (joyful), how is your outlook on the week?"


@torch.no_grad()
def gen_tail(C):
    torch.manual_seed(0)
    with v(model, C=C):
        out = model.generate(**enc, max_new_tokens=400, pad_token_id=tok.eos_token_id)
        ans, pmass = rubric_score(model, tok, RUBRIC, max_new_tokens=400, seed=0)
    txt = tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
    ans_only = txt.split("</think>")[-1].strip().replace("\n", " ")[:220]
    return ans_only, ans, pmass


for C in (0.0, 0.2, 0.3, 0.4, 0.5):
    tail, ans, pmass = gen_tail(C)
    logger.info(f"\nC={C:+g}  rubric ans={ans:.2f}/9 pmass={pmass:.2f}\n  free-form answer: {tail!r}")
