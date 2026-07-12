"""Headless compute of the steering_demo results (all 7 methods, full config) so the
marimo notebook can load them instantly instead of running an 18-min cell in-kernel
(marimo's single-threaded kernel makes a long cell un-monitorable without interrupting).

Writes artifacts/steering_demo_results.json = {"summary": [...], "detail": {name: [...]}}.
Same vecs/dilemma/config as nbs/steering_demo.py. (Claude, for wassname)

    uv run python scripts/scratch/run_steering_demo.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402  configures loguru
import torch  # noqa: E402
from steering_lite import MeanDiffC, Vector  # noqa: E402
from steering_lite.eval.edge import (  # noqa: E402
    DECEPTIVE_STATEMENTS,
    HONEST_STATEMENTS,
    HONESTY_DILEMMA,
)
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from transformers.utils.hub import cached_file  # noqa: E402

from jsteer import Jacobian  # noqa: E402
from jsteer.demo import YESNO, demo_steer  # noqa: E402

METHODS = (
    "word",
    "persona_vector",
    "persona_topk",
    "persona_soft",
    "persona_pinv",
    "meandiff",
    "random",
)

parser = argparse.ArgumentParser()
parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
if args.out.exists():
    raise FileExistsError(args.out)

started = time.time()
MODEL = "Qwen/Qwen3.5-4B"
model_snapshot = Path(cached_file(MODEL, "config.json", local_files_only=True)).parent
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
jac = Jacobian.from_pretrained(config.LENS_REPO, filename=config.hub_lens_file(MODEL),
                               revision=config.LENS_REVISION)
band = jac.steer_band(model)

deceptive = DECEPTIVE_STATEMENTS
honest = HONEST_STATEMENTS
DILEMMA = HONESTY_DILEMMA

vecs = {}
if "word" in args.methods:
    vecs["word(lie/deceive)"] = jac.word_vector(
        model, tok, ["lie", "deceive", "dishonest"], layers=band)
if "persona_vector" in args.methods:
    vecs["persona_vector"] = jac.persona_vector(model, tok, deceptive, honest, layers=band)
if "persona_topk" in args.methods:
    vecs["persona_topk k=8"] = jac.persona_topk_vector(
        model, tok, deceptive, honest, k=8, layers=band)
if "persona_soft" in args.methods:
    vecs["persona_soft"] = jac.persona_soft_vector(model, tok, deceptive, honest, layers=band)
if "persona_pinv" in args.methods:
    vecs["persona_pinv"] = jac.persona_pinv_vector(model, tok, deceptive, honest, layers=band)
if "meandiff" in args.methods:
    vecs["meandiff(base)"] = Vector.train(
        model, tok, deceptive, honest, MeanDiffC(layers=tuple(band)))
if "random" in args.methods:
    vecs["random(null)"] = jac.random_vector(seed=0, layers=band)

results = demo_steer(jac, model, tok, vecs, DILEMMA, rubric=DILEMMA, readout=YESNO,
                     max_new_tokens=256, budget=6)
results["metadata"] = {
    "argv": sys.argv,
    "model": MODEL,
    "model_commit": model_snapshot.name,
    "lens_repo": config.LENS_REPO,
    "lens_revision": config.LENS_REVISION,
    "formatted_prompt": tok.apply_chat_template(
        [{"role": "user", "content": DILEMMA}], add_generation_prompt=True,
        tokenize=False, enable_thinking=True),
    "runtime_seconds": time.time() - started,
}

args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(results, indent=2, allow_nan=False))
print(f"WROTE {args.out}  ({len(results['summary'])} methods)")
