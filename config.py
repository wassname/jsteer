"""Repo-local paths, slug/cache conventions, and the fit corpus, shared by
scripts/ and notebooks.

NOT imported by the jsteer library (which stays path-agnostic so `pip install
jsteer` never needs a repo root). The corpus content is jlens's own WikiText
(`load_wikitext_prompts`, not hand-rolled), but wrapped in the model's chat
template -- see chat_corpus for why.
"""
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
DEVICE = "cuda"
DTYPE = torch.bfloat16


def chat_corpus(tok, n_prompts: int) -> list[str]:
    """jlens's WikiText prompts wrapped in the chat template. Fitting on chat-
    formatted text (not raw documents) puts J closer to the distribution the
    model steers in; run-524's verified vectors were fit this way too. Called
    via a lambda in fit_cached, so it only downloads WikiText on a cache miss."""
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
