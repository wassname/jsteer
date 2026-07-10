"""Repo-local paths, slug/cache conventions, and the pre-fitted lens map, shared
by scripts/ and notebooks.

NOT imported by the jsteer library (which stays path-agnostic so `pip install
jsteer` never needs a repo root). The default demo path LOADS a pre-fitted lens
(HUB_LENS_FILE, raw Salesforce-wikitext); chat_corpus is only for the local-fit
fallback (scripts/fit.py) and is an untested alternative to raw -- see its docstring.
"""
from pathlib import Path

import torch
from loguru import logger
from tqdm.auto import tqdm

# Configure loguru once, on import, so every script/notebook that imports config
# gets the same compact format. Routed through tqdm.write so log lines don't
# break a live progress bar (e.g. the fit bar).
logger.remove()
logger.add(lambda m: tqdm.write(m, end=""), colorize=True,
           format="<level>{level.icon}</level> {message}", level="INFO")
for _lvl, _icon in (("INFO", "I"), ("WARNING", "W"), ("ERROR", "E"), ("DEBUG", "D")):
    logger.level(_lvl, icon=_icon)

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
DEVICE = "cuda"
DTYPE = torch.bfloat16

# The authors publish pre-fitted Jacobian lenses on the Hub (raw Salesforce-
# wikitext, n=1000 where the _n1000 suffix is present). Loading one beats fitting
# locally: identical estimator, 1000 prompts, zero compute. Keyed by HF model id;
# see github.com/anthropics/jacobian-lens walkthrough.ipynb.
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"
HUB_LENS_FILE = {
    "Qwen/Qwen3.5-4B": "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt",
    "Qwen/Qwen3.6-27B": "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt",
    "Qwen/Qwen3-4B": "qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt",
    "Qwen/Qwen3-8B": "qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt",
    "Qwen/Qwen3-14B": "qwen3-14b/jlens/Salesforce-wikitext/Qwen3-14B_jacobian_lens.pt",
    "Qwen/Qwen3-32B": "qwen3-32b/jlens/Salesforce-wikitext/Qwen3-32B_jacobian_lens.pt",
}


def hub_lens_file(model_name: str) -> str:
    """Filename of the authors' pre-fitted lens for `model_name` inside LENS_REPO.
    KeyError (fail fast) if they don't publish one -- then fit locally via fit.py."""
    return HUB_LENS_FILE[model_name]


def chat_corpus(tok, n_prompts: int) -> list[str]:
    """jlens's WikiText prompts wrapped in the chat template, for the LOCAL-FIT
    fallback only (the default demo path loads the authors' pre-fitted RAW-wikitext
    lens). Hypothesis: fitting on chat-formatted text puts J closer to the
    distribution we steer in; run-524's verified vectors were fit this way, but it
    was never compared head-to-head with a raw fit, so treat chat-vs-raw as
    unresolved. Called via a lambda in fit_cached, so WikiText only downloads on a
    cache miss."""
    from jlens.examples import load_wikitext_prompts
    return [tok.apply_chat_template([{"role": "user", "content": p}],
                                    add_generation_prompt=True, tokenize=False,
                                    enable_thinking=True)
            for p in load_wikitext_prompts(n_prompts)]


def slug(model_name: str) -> str:
    """'Qwen/Qwen3-0.6B' -> 'qwen3-0.6b': a filesystem-safe cache stem."""
    return model_name.split("/")[-1].lower()


def cache_path(model_name: str, suffix: str = "jac") -> Path:
    """Where the fitted Jacobian for `model_name` is cached."""
    return ART / f"{slug(model_name)}.{suffix}"
