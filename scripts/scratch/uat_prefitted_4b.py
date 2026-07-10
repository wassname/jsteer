"""UAT: does the authors' pre-fitted n=1000 Qwen3.5-4B lens drive our steering? (Claude)

Loads the Hub lens (raw Salesforce-wikitext, n=1000) through Jacobian.load, extracts a
happy/joy word_vector on the mid-depth band (to match run-524's regime, since the
pre-fitted lens spans ALL layers 0..30), and shows baseline vs +C vs the random control.
If +C reads happier and stays coherent while random at the same C does not, the
pre-fitted raw lens is a drop-in for the demo and local fitting is unnecessary here.
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

path = hf_hub_download(
    "neuronpedia/jacobian-lens", revision="qwen-n1000",
    filename="qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt")
jac = Jacobian.load(path)
logger.info(f"pre-fitted lens: {jac!r}")

# All-layer lens -> restrict steering to the 0.3-0.9 depth band (n_layers=32), the
# regime run-524 used; steering all 31 layers at once over-drives the residual.
n_layers = model.config.num_hidden_layers
band = [l for l in jac.layers if 0.3 <= l / n_layers <= 0.9]
logger.info(f"steer band (0.3-0.9 of {n_layers}): {band}")

v = jac.word_vector(model, tok, ["happy", "joy"], layers=band)
v_rand = jac.random_vector(seed=0, layers=band)

logger.info("=== WORD vector (happy/joy) on pre-fitted n1000 lens ===")
show_steer(jac, model, tok, v, "Describe how your week has been going.",
           Cs=(0, 6), max_new_tokens=80)
logger.info("=== RANDOM control at matched C ===")
show_steer(jac, model, tok, v_rand, "Describe how your week has been going.",
           Cs=(0, 6), max_new_tokens=80)
