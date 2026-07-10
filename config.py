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
    """Fit corpus at the CHAT operating point: jlens's WikiText prompts, each
    wrapped in the chat template with the thinking block opened.

    We fit J where we steer. run-524's VERIFIED vectors were fit on chat-
    templated prompts (artifacts/u4_prompts.json shows `<|im_start|>user ...
    assistant <think>`), and steering is applied during templated generation.
    jlens fits raw WikiText because it's a general document lens; jsteer steers
    a chat model mid-`<think>`, so the linearization point has to match or J is
    estimated at the wrong operating point. Called via a lambda in fit_cached,
    so it only runs (and only downloads WikiText) on a cache MISS."""
    from jlens.examples import load_wikitext_prompts
    raw = load_wikitext_prompts(n_prompts)
    return [tok.apply_chat_template([{"role": "user", "content": p}],
                                    add_generation_prompt=True, tokenize=False,
                                    enable_thinking=True) for p in raw]


def slug(model_name: str) -> str:
    """'Qwen/Qwen3-0.6B' -> 'qwen3-0.6b': a filesystem-safe cache stem."""
    return model_name.split("/")[-1].lower()


def cache_path(model_name: str, suffix: str = "jac") -> Path:
    """Where the fitted Jacobian for `model_name` is cached."""
    return ART / f"{slug(model_name)}.{suffix}"
